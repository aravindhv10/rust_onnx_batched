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
