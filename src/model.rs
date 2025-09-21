// use ort::execution_providers::CUDAExecutionProvider;
// use ort::execution_providers::OpenVINOExecutionProvider;
// use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
// use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;

use serde::Serialize;

use std::ops::Index;
use std::time::Duration;

use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;

use tokio;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use crate::mylib::get_model;
use crate::mylib::image_processor;

const MAX_BATCH: usize = 16;
const BATCH_TIMEOUT: Duration = Duration::from_millis(200);
const MODEL_PATH: &str = "./model.onnx";

const IMAGE_RESOLUTION: u32 = 448;
const CLASS_LABELS: [&str; num_features] = ["empty", "occupied", "other"];

type outtype = f32;

const num_features: usize = 3;

pub struct prediction_probabilities {
    pub ps: [outtype; num_features],
}

impl prediction_probabilities {
    pub fn new() -> Self {
        prediction_probabilities {
            ps: [0.0; num_features],
        }
    }

    pub fn from<T: Index<usize, Output = outtype>>(input: T) -> Self {
        let mut ret = prediction_probabilities::new();
        for i in 0..num_features {
            ret.ps[i] = input[i];
        }
        return ret;
    }
}

#[derive(Serialize)]
pub struct prediction_probabilities_reply {
    ps: [String; num_features],
    mj: String,
}

impl prediction_probabilities_reply {
    pub fn new() -> Self {
        prediction_probabilities_reply {
            ps: std::array::from_fn(|_| String::new()),
            mj: String::new(),
        }
    }

    pub fn from(input: prediction_probabilities) -> prediction_probabilities_reply {
        let mut max_index: usize = 0;
        let mut ret = prediction_probabilities_reply::new();
        ret.ps[0] = input.ps[0].to_string();
        for i in 1..num_features {
            ret.ps[i] = input.ps[i].to_string();
            if input.ps[i] > input.ps[max_index] {
                max_index = i;
            }
        }
        ret.mj = CLASS_LABELS[max_index].to_string();
        return ret;
    }
}

pub struct InferRequest {
    img: image::RgbaImage,
    resp_tx: oneshot::Sender<Result<prediction_probabilities, String>>,
}

pub struct model_server {
    rx: mpsc::Receiver<InferRequest>,
    session: Session,
}

impl model_server {
    pub async fn infer_loop(&mut self) {
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

pub struct model_client {
    tx: mpsc::Sender<InferRequest>,
    preprocess: image_processor,
}

impl model_client {
    pub async fn do_infer(
        &self,
        img: image::RgbaImage,
    ) -> Result<prediction_probabilities, String> {
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
    pub async fn do_infer_data(&self, data: Vec<u8>) -> Result<prediction_probabilities, String> {
        match self.preprocess.decode_and_preprocess(data) {
            Ok(img) => {
                return self.do_infer(img).await;
            }
            Err(e) => {
                return Err("Failed to decode and pre-process the image".to_string());
            }
        }
    }
}

pub fn get_inference_tuple() -> (model_server, model_client) {
    let (tx, rx) = mpsc::channel::<InferRequest>(512);
    let ret_server = model_server {
        rx: rx,
        session: get_model(MODEL_PATH),
    };
    let ret_client = model_client {
        tx: tx,
        preprocess: image_processor::default(),
    };
    return (ret_server, ret_client);
}
