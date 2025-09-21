use image::DynamicImage;
use image::imageops;

pub struct image_processor {
    image_resolution: u32,
}

impl Default for image_processor {
    fn default() -> Self {
        return image_processor {
            image_resolution: 448,
        };
    }
}

impl image_processor {
    pub fn new(val: u32) -> Self {
        return image_processor {
            image_resolution: val,
        };
    }

    fn preprocess(&self, img: DynamicImage) -> image::RgbaImage {
        let (width, height) = (img.width(), img.height());
        let size = width.min(height);
        let x = (width - size) / 2;
        let y = (height - size) / 2;
        let cropped_img = imageops::crop_imm(&img, x, y, size, size).to_image();
        imageops::resize(
            &cropped_img,
            self.image_resolution,
            self.image_resolution,
            imageops::FilterType::CatmullRom,
        )
    }

    pub fn decode_and_preprocess(&self, data: Vec<u8>) -> Result<image::RgbaImage, String> {
        match image::load_from_memory(&data) {
            Ok(img) => {
                return Ok(self.preprocess(img));
            }
            Err(e) => {
                return Err("decode error".to_string());
            }
        };
    }
}

use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

pub fn get_cuda_model(model_path: &str) -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([CUDAExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(model_path).unwrap();
            println!("Constructed onnx with CUDA support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with CUDA support");
            return Err("Failed to construct model with CUDA support".to_string());
        }
    }
}

pub fn get_webgpu_model(model_path: &str) -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([WebGPUExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(model_path).unwrap();
            println!("Constructed onnx with CUDA support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with WebGPU support");
            return Err("Failed to construct model with WebGPU support".to_string());
        }
    }
}

pub fn get_openvino_model(model_path: &str) -> Result<Session, String> {
    let res1 = Session::builder()
        .unwrap()
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .unwrap();

    let res2 = res1.with_execution_providers([OpenVINOExecutionProvider::default().build()]);

    match res2 {
        Ok(res3) => {
            let res4 = res3.commit_from_file(model_path).unwrap();
            println!("Constructed onnx with openvino support");
            return Ok(res4);
        }
        Err(_) => {
            println!("Failed to construct model with openvino support");
            return Err("Failed to construct model with openvino support".to_string());
        }
    }
}

pub fn get_model(model_path: &str) -> Session {
    match get_cuda_model(model_path) {
        Ok(model) => {
            return model;
        }
        Err(_) => {
            return get_openvino_model(model_path).unwrap();
        }
    }
}
