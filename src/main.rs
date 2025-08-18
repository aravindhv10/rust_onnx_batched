use actix_multipart::Multipart;
use actix_web::{web, App, HttpResponse, HttpServer, Responder, ResponseError};
use futures_util::TryStreamExt;
use image;
use lockfree::queue::Queue;
use log::{error, info};
use ndarray::{Array, ArrayView, Axis, Ix3, Ix4};
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    Error as OrtError,
};
use serde::{Deserialize, Serialize};
use serde_json;
use std::{
    sync::Arc,
    thread,
    time::{Duration, Instant},
};
use thiserror::Error;
use tokio::sync::oneshot;
use tokio::time::timeout;

// --- Configuration Constants ---

const NUM_FEATURES: usize = 3;
const MAX_BATCH_SIZE: usize = 32;
const BATCH_TIMEOUT_MS: u64 = 100;

const MODEL_PATH: &str = "./model.onnx";
const CLASS_LABELS: [&str; NUM_FEATURES] = ["empty", "occupied", "other"];
const IMAGE_RESOLUTION: u32 = 448;

// --- Types for Communication and Results ---

#[derive(Debug, PartialEq, Serialize, Deserialize, Clone)]
struct PredictionProbabilities {
    ps: [f32; NUM_FEATURES],
}

// The job sent from the web handler to the inference thread
struct InferenceJob {
    image_bytes: Vec<u8>,
    responder: oneshot::Sender<Result<PredictionProbabilities, InferenceError>>,
}

// The final JSON response sent to the client
#[derive(Serialize)]
struct PredictionReply {
    p1: String,
    p2: String,
    p3: String,
    mj: String,
}

// --- Custom Error Handling ---

#[derive(Error, Debug, Clone)]
enum InferenceError {
    #[error("Request timed out")]
    Timeout,
    #[error("Inference worker channel closed")]
    ChannelClosed,
    #[error("Image processing error: {0}")]
    ImageError(String),
    #[error("ORT (ONNX Runtime) error: {0}")]
    OrtError(String),
    #[error("Invalid input data: {0}")]
    InvalidInput(String),
    #[error("Multipart form error: {0}")]
    MultipartError(String),
}

impl ResponseError for InferenceError {
    fn error_response(&self) -> HttpResponse {
        let status_code = match self {
            InferenceError::Timeout => actix_web::http::StatusCode::REQUEST_TIMEOUT,
            InferenceError::InvalidInput(_) | InferenceError::MultipartError(_) => {
                actix_web::http::StatusCode::BAD_REQUEST
            }
            _ => actix_web::http::StatusCode::INTERNAL_SERVER_ERROR,
        };
        HttpResponse::build(status_code).json(serde_json::json!({ "error": self.to_string() }))
    }
}

// Manual `From` implementations to convert external errors into our custom error type.

impl From<OrtError> for InferenceError {
    fn from(err: OrtError) -> Self {
        InferenceError::OrtError(err.to_string())
    }
}

impl From<actix_multipart::MultipartError> for InferenceError {
    fn from(err: actix_multipart::MultipartError) -> Self {
        InferenceError::MultipartError(err.to_string())
    }
}

impl From<image::ImageError> for InferenceError {
    fn from(err: image::ImageError) -> Self {
        InferenceError::ImageError(err.to_string())
    }
}

// --- Web Handler (The "Producer") ---

/// Receives an image, pushes it to the inference queue, and waits for the result.
async fn infer(
    queue: web::Data<Arc<Queue<InferenceJob>>>,
    mut payload: Multipart,
) -> Result<impl Responder, InferenceError> {
    let mut image_data = Vec::new();
    // Await the next field from the multipart payload.
    if let Some(mut field) = payload.try_next().await? {
        while let Some(chunk) = field.try_next().await? {
            image_data.extend_from_slice(&chunk);
        }
    }

    if image_data.is_empty() {
        return Err(InferenceError::InvalidInput(
            "Image data not provided.".into(),
        ));
    }

    // Create a one-shot channel to receive the result from the inference thread.
    let (tx, rx) = oneshot::channel();

    // Create and push the job to the lock-free queue.
    let job = InferenceJob {
        image_bytes: image_data,
        responder: tx,
    };
    queue.push(job);

    // Asynchronously wait for the result with a 10-second timeout.
    match timeout(Duration::from_secs(10), rx).await {
        Ok(Ok(Ok(preds))) => Ok(HttpResponse::Ok().json(get_prediction_for_reply(preds))),
        Ok(Ok(Err(e))) => Err(e), // Inference thread returned an error
        Ok(Err(_)) => Err(InferenceError::ChannelClosed), // Inference thread hung up
        Err(_) => Err(InferenceError::Timeout), // Waited too long
    }
}

// --- Background Inference Thread (The "Consumer") ---

/// The core batching and inference logic running in its own dedicated thread.
fn batch_inference_thread(queue: Arc<Queue<InferenceJob>>) {
    info!("Inference thread started.");

    // Initialize the ONNX session *within this thread*.
    let session = get_model().expect("Failed to build ONNX session");
    info!("ONNX session loaded successfully on inference thread.");

    let mut batch: Vec<InferenceJob> = Vec::with_capacity(MAX_BATCH_SIZE);
    let mut last_batch_time = Instant::now();

    loop {
        // Collect jobs for the batch.
        while batch.len() < MAX_BATCH_SIZE {
            if let Some(job) = queue.pop() {
                batch.push(job);
            } else {
                // Queue is empty, break to check timeout.
                break;
            }
        }

        let should_process = !batch.is_empty()
            && (batch.len() >= MAX_BATCH_SIZE
                || last_batch_time.elapsed() > Duration::from_millis(BATCH_TIMEOUT_MS));

        if should_process {
            let jobs_to_process: Vec<InferenceJob> = batch.drain(..).collect();
            info!("Processing batch of size: {}", jobs_to_process.len());

            // Perform the inference.
            let results = run_batch_inference(&session, &jobs_to_process);

            // Send results back to the waiting handlers.
            for (job, result) in jobs_to_process.into_iter().zip(results) {
                // The `_ =` ignores the result. If it fails, the client likely already disconnected.
                let _ = job.responder.send(result);
            }

            last_batch_time = Instant::now();
        } else {
            // If the queue was empty, sleep briefly to prevent a busy-loop.
            thread::sleep(Duration::from_millis(5));
        }
    }
}

/// Helper function to preprocess and run inference on a batch.
fn run_batch_inference(
    session: &Session,
    jobs: &[InferenceJob],
) -> Vec<Result<PredictionProbabilities, InferenceError>> {
    // 1. Preprocess all images
    let image_tensors: Vec<_> = jobs
        .iter()
        .map(|job| preprocess_image(&job.image_bytes))
        .collect();

    // 2. Separate successful from failed preprocessing
    let mut results: Vec<Option<Result<PredictionProbabilities, InferenceError>>> =
        vec![None; jobs.len()];
    let mut successful_tensors: Vec<Array<f32, Ix3>> = Vec::new();
    let mut original_indices = Vec::new();

    for (i, res) in image_tensors.into_iter().enumerate() {
        match res {
            Ok(tensor) => {
                successful_tensors.push(tensor);
                original_indices.push(i);
            }
            Err(e) => {
                results[i] = Some(Err(e.into()));
            }
        }
    }

    // Only run inference if there are valid images
    if !successful_tensors.is_empty() {
        // 3. Stack valid tensors into a single batch tensor
        let views: Vec<_> = successful_tensors.iter().map(|t| t.view()).collect();
        let batch_tensor: Array<f32, Ix4> = match ndarray::stack(Axis(0), &views) {
            Ok(tensor) => tensor,
            Err(e) => {
                let stack_error =
                    InferenceError::InvalidInput(format!("Failed to stack tensors: {}", e));
                for &original_idx in &original_indices {
                    results[original_idx] = Some(Err(stack_error.clone()));
                }
                return results.into_iter().map(|r| r.unwrap()).collect();
            }
        };

        // 4. Run inference and process results in a separate closure for cleaner error handling
        let run_result: Result<Vec<PredictionProbabilities>, InferenceError> = (|| {
            // The `ort` crate can directly accept an array of ndarray references.
            // This is the simplest and most correct way to pass inputs for this version.
            let outputs = session.run([&batch_tensor])?;
            let output_tensor = outputs[0].try_extract_tensor::<f32>()?;

            let shape = output_tensor
                .shape()
                .iter()
                .map(|i| *i as usize)
                .collect::<Vec<_>>();
            let data = output_tensor.as_slice()?;
            let output_view = ArrayView::from_shape(shape, data)
                .map_err(|e| OrtError::new(format!("Failed to create ndarray view: {}", e)))?;

            Ok(output_view
                .axis_iter(Axis(0))
                .map(|row| PredictionProbabilities {
                    ps: [row[0], row[1], row[2]],
                })
                .collect())
        })();

        // 5. Distribute results back to their original slots
        match run_result {
            Ok(predictions) => {
                for (i, prediction) in predictions.into_iter().enumerate() {
                    let original_idx = original_indices[i];
                    results[original_idx] = Some(Ok(prediction));
                }
            }
            Err(e) => {
                error!("ONNX session run failed: {:?}", e);
                // Mark all jobs in this batch as failed
                for &original_idx in &original_indices {
                    results[original_idx] = Some(Err(e.clone()));
                }
            }
        };
    }

    results
        .into_iter()
        .map(|r| r.expect("Result slot should have been filled"))
        .collect()
}

// --- Helper Functions ---

/// Preprocesses raw image bytes into a tensor suitable for the ONNX model.
fn preprocess_image(image_bytes: &[u8]) -> Result<Array<f32, Ix3>, image::ImageError> {
    let original_img = image::load_from_memory(image_bytes)?;

    // --- Center Crop ---
    let (width, height) = (original_img.width(), original_img.height());
    let size = width.min(height);
    let x = (width - size) / 2;
    let y = (height - size) / 2;
    let cropped_img = image::imageops::crop_imm(&original_img, x, y, size, size).to_image();

    // --- Resize ---
    let resized_img = image::imageops::resize(
        &cropped_img,
        IMAGE_RESOLUTION,
        IMAGE_RESOLUTION,
        image::imageops::FilterType::Triangle,
    );

    // --- Convert to ndarray, normalize, and change layout to CHW ---
    let mut array = Array::<f32, _>::zeros((
        IMAGE_RESOLUTION as usize,
        IMAGE_RESOLUTION as usize,
        3,
    ));
    for (x, y, pixel) in resized_img.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        // Normalize to [0.0, 1.0] and place in (H, W, C) format
        array[[y as usize, x as usize, 0]] = r as f32 / 255.0;
        array[[y as usize, x as usize, 1]] = g as f32 / 255.0;
        array[[y as usize, x as usize, 2]] = b as f32 / 255.0;
    }

    // Permute axes from (H, W, C) to (C, H, W)
    Ok(array.permuted_axes([2, 0, 1]))
}

/// Formats the raw prediction probabilities into the final JSON reply structure.
fn get_prediction_for_reply(input: PredictionProbabilities) -> PredictionReply {
    let mut max_index: usize = 0;
    for i in 1..NUM_FEATURES {
        if input.ps[i] > input.ps[max_index] {
            max_index = i;
        }
    }

    PredictionReply {
        p1: input.ps[0].to_string(),
        p2: input.ps[1].to_string(),
        p3: input.ps[2].to_string(),
        mj: CLASS_LABELS[max_index].to_string(),
    }
}

/// Initializes the ONNX Runtime session.
fn get_model() -> Result<Session, OrtError> {
    let mut builder = Session::builder()?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(1)?;

    // This logic is simplified to be more robust against compiler type inference issues.
    let cuda_provider = ort::execution_providers::CUDAExecutionProvider::default().build();
    if let Ok(provider) = cuda_provider {
        info!("Attempting to build session with CUDA Execution Provider.");
        builder = builder.with_execution_providers([provider])?;
    } else {
        info!("CUDA provider not available. Falling back to CPU.");
    }

    let session = builder.commit_from_file(MODEL_PATH);

    if session.is_ok() {
        info!("Successfully created ONNX session.");
    } else {
        error!("Failed to create ONNX session.");
    }
    session
}

// --- Main Application Setup ---

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create the shared, lock-free queue for inference jobs.
    let queue = Arc::new(Queue::new());
    let queue_clone = Arc::clone(&queue);

    // Spawn the dedicated background thread for batch processing.
    thread::spawn(move || {
        batch_inference_thread(queue_clone);
    });

    info!("🚀 Server starting at http://0.0.0.0:8000");

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(Arc::clone(&queue)))
            .route("/infer", web::post().to(infer))
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}

