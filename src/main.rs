use actix_multipart::Multipart;
use actix_web::App;
use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::web;

use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use tokio;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use futures_util::TryStreamExt;
use image::DynamicImage;
use image::imageops;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;
use serde::Deserialize;
use serde::Serialize;

use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::ops::Index;
use std::sync::Arc;
use std::time::Duration;

use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;

pub mod infer {
    tonic::include_proto!("infer"); // The string specified here must match the proto package name
}

const MAX_BATCH: usize = 16;
const BATCH_TIMEOUT: Duration = Duration::from_millis(200);
const MODEL_PATH: &str = "./model.onnx";

const IMAGE_RESOLUTION: u32 = 448;
const num_features: usize = 3;
const CLASS_LABELS: [&str; num_features] = ["empty", "occupied", "other"];

type outtype = f32;

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

struct prediction_probabilities {
    ps: [outtype; num_features],
}

impl prediction_probabilities {
    fn new() -> Self {
        prediction_probabilities {
            ps: [0.0; num_features],
        }
    }

    fn from<T: Index<usize, Output = outtype>>(input: T) -> Self {
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
    fn new() -> Self {
        prediction_probabilities_reply {
            ps: std::array::from_fn(|_| String::new()),
            mj: String::new(),
        }
    }

    fn from(input: prediction_probabilities) -> prediction_probabilities_reply {
        let mut max_index: usize = 0;
        let mut ret = prediction_probabilities_reply::new();
        for i in 1..num_features {
            ret.ps[i] = input.ps[i].to_string();
            if input.ps[i] > input.ps[max_index] {
                max_index = i;
            }
        }
        ret.mj = CLASS_LABELS[max_index].to_string();
        ret
    }
}

struct InferRequest {
    img: image::RgbaImage,
    resp_tx: oneshot::Sender<Result<prediction_probabilities, String>>,
}

async fn infer_loop(mut rx: mpsc::Receiver<InferRequest>, mut session: Session) {
    while let Some(first) = rx.recv().await {
        let mut batch = vec![first];
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
            for (x, y, pixel) in req.img.enumerate_pixels() {
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
            .try_extract_array::<outtype>()
            .unwrap()
            .t()
            .into_owned();

        for (row, req) in output.axis_iter(Axis(1)).zip(batch.into_iter()) {
            let result = prediction_probabilities::from(row);
            let _ = req.resp_tx.send(Ok(result));
        }
    }
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

fn decode_and_preprocess(data: Vec<u8>) -> Result<image::RgbaImage, Error> {
    match image::load_from_memory(&data) {
        Ok(img) => {
            return Ok(preprocess(img));
        }
        Err(e) => {
            return Err(actix_web::error::ErrorBadRequest(format!(
                "decode error: {}",
                e
            )));
        }
    };
}

struct model_server {
    rx: mpsc::Receiver<InferRequest>,
    session: Session,
}

impl model_server {
    async fn infer_loop(&mut self) {
        while let Some(first) = self.rx.recv().await {
            let mut batch = vec![first];
            let start = tokio::time::Instant::now();
            while batch.len() < MAX_BATCH && start.elapsed() < BATCH_TIMEOUT {
                match self.rx.try_recv() {
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
                for (x, y, pixel) in req.img.enumerate_pixels() {
                    let [r, g, b, _] = pixel.0;
                    input[[i, y as usize, x as usize, 0]] = r;
                    input[[i, y as usize, x as usize, 1]] = g;
                    input[[i, y as usize, x as usize, 2]] = b;
                }
            }
            let outputs = match self
                .session
                .run(inputs!["input" => TensorRef::from_array_view(&input).unwrap()])
            {
                Ok(o) => o,
                Err(e) => {
                    for req in batch {
                        let _ = req.resp_tx.send(Err(format!("inference error: {}", e)));
                    }
                    continue;
                }
            };
            let output = outputs["output"]
                .try_extract_array::<outtype>()
                .unwrap()
                .t()
                .into_owned();
            for (row, req) in output.axis_iter(Axis(1)).zip(batch.into_iter()) {
                let result = prediction_probabilities::from(row);
                let _ = req.resp_tx.send(Ok(result));
            }
        }
    }
}

struct model_client {
    tx: mpsc::Sender<InferRequest>,
}

impl model_client {
    async fn do_infer(&self, img: image::RgbaImage) -> Result<prediction_probabilities, String> {
        let (resp_tx, resp_rx) = oneshot::channel();
        match self.tx.send(InferRequest { img, resp_tx }).await {
            Ok(_) => match resp_rx.await {
                Ok(Ok(pred)) => {
                    return Ok(pred);
                }
                Ok(Err(e)) => {
                    return Err(e);
                }
                Err(e) => {
                    return Err("Recv Error".to_string());
                }
            },
            Err(e) => {
                return Err("Send error".to_string());
            }
        }
    }
    async fn do_infer_data(&self, data: Vec<u8>) -> Result<prediction_probabilities, String> {
        match decode_and_preprocess(data) {
            Ok(img) => {
                return self.do_infer(img).await;
            }
            Err(e) => {
                return Err("Failed to decode and pre-process the image".to_string());
            }
        }
    }
}

fn get_inference_tuple() -> (model_server, model_client) {
    let (tx, rx) = mpsc::channel::<InferRequest>(512);

    let ret_server = model_server {
        rx: rx,
        session: get_model(),
    };

    let ret_client = model_client { tx: tx };

    return (ret_server, ret_client);
}

async fn infer_handler(
    mut payload: Multipart,
    infer_slave: web::Data<Arc<model_client>>,
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

    match infer_slave.do_infer_data(data).await {
        Ok(pred) => {
            return Ok(HttpResponse::Ok().json(prediction_probabilities_reply::from(pred)));
        }
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().body(e));
        }
    }
}

pub struct MyInferer {
    tx: Arc<mpsc::Sender<InferRequest>>,
}

#[tonic::async_trait]
impl infer::infer_server::Infer for MyInferer {
    async fn do_infer(
        &self,
        request: Request<infer::Image>,
    ) -> Result<Response<infer::Prediction>, Status> {
        println!("Received gRPC request");
        let image_data = request.into_inner().image_data;

        // Load the image from the received bytes.
        let img = decode_and_preprocess(image_data)
            .map_err(|e| Status::invalid_argument(format!("Failed to decode image: {}", e)))?;

        // Create a channel for the inference response.
        let (resp_tx, resp_rx) = oneshot::channel();
        let req = InferRequest { img, resp_tx };

        // Send the request to the inference loop.
        self.tx
            .send(req)
            .await
            .map_err(|_| Status::internal("Inference queue is closed"))?;

        // Wait for the inference result.
        match resp_rx.await {
            Ok(Ok(pred)) => {
                let reply = infer::Prediction {
                    ps1: pred.ps[0],
                    ps2: pred.ps[1],
                    ps3: pred.ps[2],
                };

                Ok(Response::new(reply))
            }

            Ok(Err(e)) => Err(Status::internal(e)),

            Err(_) => Err(Status::internal("Inference request dropped")),
        }
    }
}

#[actix_web::main]
async fn main() -> () {
    let (mut slave_server, slave_client) = get_inference_tuple();

    let slave_client = Arc::new(slave_client);
    let future_infer = slave_server.infer_loop();

    match HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(slave_client.clone()))
            .route("/infer", web::post().to(infer_handler))
    })
    .bind(("0.0.0.0", 8000))
    {
        Ok(ret) => {
            let future_rest_server = ret.run();

            let (first, second) = tokio::join!(future_infer, future_rest_server);

            match second {
                Ok(_) => {
                    println!("REST server executed and stopped successfully");
                }
                Err(e) => {
                    println!("Encountered error in starting the server due to {}.", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to bind to port");
        }
    }
}
