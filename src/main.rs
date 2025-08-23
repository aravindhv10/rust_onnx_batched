use actix_multipart::Multipart;
use actix_web::App;
use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::web;
use futures_util::TryStreamExt;
use image::DynamicImage;
use image::imageops;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;
use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use serde::Deserialize;
use serde::Serialize;
use std::ops::Index;
use std::time::Duration;
use tokio;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

const MAX_BATCH: usize = 16;
const BATCH_TIMEOUT: Duration = Duration::from_millis(20);
const MODEL_PATH: &str = "./model.onnx";

const IMAGE_RESOLUTION: u32 = 448;
const num_features: usize = 3;
const CLASS_LABELS: [&str; num_features] = ["empty", "occupied", "other"];

struct prediction_probabilities {
    ps: [f32; num_features],
}

impl prediction_probabilities {
    fn new() -> prediction_probabilities {
        prediction_probabilities {
            ps: [0.0; num_features],
        }
    }

    fn from<T: Index<usize, Output = f32>>(input: T) -> prediction_probabilities {
        let mut ret = prediction_probabilities::new();
        for i in 0..num_features {
            ret.ps[i] = input[i];
        }
        ret
    }
}

#[derive(Serialize)]
struct prediction_probabilities_reply {
    ps: [String; num_features],
    mj: String,
}

impl prediction_probabilities_reply {
    fn from(input: prediction_probabilities) -> prediction_probabilities_reply {
        let mut max_index: usize = 0;

        for i in 1..num_features {
            if input.ps[i] > input.ps[max_index] {
                max_index = i;
            }
        }

        let mut ret = prediction_probabilities_reply {
            ps: [
                input.ps[0].to_string(),
                input.ps[1].to_string(),
                input.ps[2].to_string(),
            ],
            mj: CLASS_LABELS[max_index].to_string(),
        };

        ret
    }
}

// === Request to inference thread ===
struct InferRequest {
    img: DynamicImage,
    resp_tx: oneshot::Sender<Result<prediction_probabilities, String>>,
}

fn preprocess(img: DynamicImage) -> image::RgbaImage {
    let (width, height) = (img.width(), img.height());
    let size = width.min(height);
    let x = (width - size) / 2;
    let y = (height - size) / 2;
    let cropped_img = imageops::crop_imm(&img, x, y, size, size).to_image();
    imageops::resize(
        &cropped_img,
        IMAGE_RESOLUTION,
        IMAGE_RESOLUTION,
        imageops::FilterType::CatmullRom,
    )
}

async fn infer_handler(
    mut payload: Multipart,
    tx: web::Data<mpsc::Sender<InferRequest>>,
) -> Result<HttpResponse, Error> {
    let mut data = Vec::new();
    while let Some(mut field) = payload.try_next().await? {
        while let Some(chunk) = field.try_next().await? {
            data.extend_from_slice(&chunk);
        }
    }
    if data.is_empty() {
        return Ok(HttpResponse::BadRequest().body("No image data"));
    }

    let img = image::load_from_memory(&data)
        .map_err(|e| actix_web::error::ErrorBadRequest(format!("decode error: {}", e)))?;

    let (resp_tx, resp_rx) = oneshot::channel();
    tx.send(InferRequest { img, resp_tx })
        .await
        .map_err(|_| actix_web::error::ErrorInternalServerError("inference queue closed"))?;

    match resp_rx.await {
        Ok(Ok(pred)) => Ok(HttpResponse::Ok().json(prediction_probabilities_reply::from(pred))),
        Ok(Err(e)) => Ok(HttpResponse::InternalServerError().body(e)),
        Err(_) => Ok(HttpResponse::InternalServerError().body("inference dropped")),
    }
}

async fn infer_loop(mut rx: mpsc::Receiver<InferRequest>, mut session: Session) {

    while let Some(first) = rx.recv().await {
        let mut batch = vec![first];
        // try to build up batch quickly
        let start = tokio::time::Instant::now();
        while batch.len() < MAX_BATCH && start.elapsed() < BATCH_TIMEOUT {
            match rx.try_recv() {
                Ok(req) => batch.push(req),
                Err(_) => break,
            }
        }

        let batch_size = batch.len();
        let mut input = Array::<u8, Ix4>::zeros((
            batch_size,
            IMAGE_RESOLUTION as usize,
            IMAGE_RESOLUTION as usize,
            3,
        ));

        for (i, req) in batch.iter().enumerate() {
            let img = preprocess(req.img.clone());
            for (x, y, pixel) in img.enumerate_pixels() {
                let [r, g, b, _] = pixel.0;
                input[[i, y as usize, x as usize, 0]] = r;
                input[[i, y as usize, x as usize, 1]] = g;
                input[[i, y as usize, x as usize, 2]] = b;
            }
        }

        let outputs =
            match session.run(inputs!["input" => TensorRef::from_array_view(&input).unwrap()]) {
                Ok(o) => o,
                Err(e) => {
                    for req in batch {
                        let _ = req.resp_tx.send(Err(format!("inference error: {}", e)));
                    }
                    continue;
                }
            };

        let output = outputs["output"]
            .try_extract_array::<f32>()
            .unwrap()
            .t()
            .into_owned();

        // for (i, row) in output.axis_iter(Axis(1)).enumerate() {
        //     let result = get_prediction_probabilities(row);
        //     let _ = batch[i].resp_tx.send(Ok(result));
        // }

        for (row, req) in output.axis_iter(Axis(1)).zip(batch.into_iter()) {
            // get_prediction_probabilities(row);
            let result = prediction_probabilities::from(row);
            let _ = req.resp_tx.send(Ok(result));
        }

    }
}

fn get_cuda_model() -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([CUDAExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(MODEL_PATH).unwrap();
            println!("Constructed onnx with CUDA support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with CUDA support");
            return Err("Failed to construct model with CUDA support".to_string());
        }
    }
}

fn get_webgpu_model() -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([WebGPUExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(MODEL_PATH).unwrap();
            println!("Constructed onnx with CUDA support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with WebGPU support");
            return Err("Failed to construct model with WebGPU support".to_string());
        }
    }
}

fn get_openvino_model() -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([OpenVINOExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(MODEL_PATH).unwrap();
            println!("Constructed onnx with openvino support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with openvino support");
            return Err("Failed to construct model with openvino support".to_string());
        }
    }
}

fn get_model() -> Session {
    match get_cuda_model() {
        Ok(model) => {
            return model;
        }
        Err(_) => {
            return get_openvino_model().unwrap();
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = get_model();
    let (tx, rx) = mpsc::channel::<InferRequest>(1000);

    tokio::spawn(infer_loop(rx, model));

    HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(tx.clone()))
            .route("/infer", web::post().to(infer_handler))
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}
