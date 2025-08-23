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

const MODEL_PATH: &str = "./model.onnx";
const PATH_DIR_IMAGE: &str = "/tmp/image/";
const PATH_DIR_INCOMPLETE: &str = "/tmp/incomplete/";
const PATH_DIR_OUT: &str = "/tmp/out/";

const IMAGE_RESOLUTION: u32 = 448;

const num_features: usize = 3;
const CLASS_LABELS: [&str; num_features] = ["empty", "occupied", "other"];

const MAX_BATCH: usize = 16;
const BATCH_TIMEOUT: Duration = Duration::from_millis(20);

struct prediction_probabilities {
    ps: [f32; num_features],
}

impl prediction_probabilities {
    fn new() -> prediction_probabilities {
        prediction_probabilities {
            ps: [0.0; num_features]
        }
    }
}
