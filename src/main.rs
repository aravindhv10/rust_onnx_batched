use actix_multipart::Multipart;
use actix_web::{web, App, HttpResponse, HttpServer, Responder, Error};
use futures_util::TryStreamExt;
use image::{GenericImageView, imageops, DynamicImage};
use ndarray::{Array, Axis};
use ort::{
    inputs,
    session::{Session, GraphOptimizationLevel},
    value::TensorRef,
};
use serde::Serialize;
use std::io::Write;

const CLASS_LABELS: [&str; 3] = ["empty", "occupied", "other"];
const MODEL_PATH: &str = "/home/asd/MODEL_CHECKPOINTS/PATIENT_DETECTION/patient_detect-epoch=15-val_loss=0.02.onnx";
const IMAGE_RESOLUTION: u32 = 448;

#[derive(Serialize)]
struct InferenceResponse {
    predictions: Vec<Vec<f32>>,
}

/// # **Handles the inference request.**
///
/// This function takes the multipart request, extracts the image, preprocesses it,
/// runs the inference, and returns the JSON response.
async fn infer(mut payload: Multipart, model: web::Data<Session>) -> Result<HttpResponse, Error> {
    // Isolate the image data from the multipart payload
    let mut image_data = Vec::new();
    while let Some(mut field) = payload.try_next().await? {
        if field.content_disposition().get_name() == Some("image") {
            while let Some(chunk) = field.try_next().await? {
                image_data.extend_from_slice(&chunk);
            }
        }
    }

    if image_data.is_empty() {
        return Ok(HttpResponse::BadRequest().body("Image data not provided."));
    }

    // Load and preprocess the image
    let original_img = image::load_from_memory(&image_data)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;
    let preprocessed_image = preprocess_image(original_img);

    // Prepare the input tensor
    let mut input = Array::zeros((1, IMAGE_RESOLUTION as usize, IMAGE_RESOLUTION as usize, 3));
    for (x, y, pixel) in preprocessed_image.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        input[[0, y as usize, x as usize, 0]] = r as f32;
        input[[0, y as usize, x as usize, 1]] = g as f32;
        input[[0, y as usize, x as usize, 2]] = b as f32;
    }

    // Run the model
    let outputs = model.run(inputs!["input" => TensorRef::from_array_view(&input).unwrap()]).unwrap();
    let output = outputs["output"].try_extract_array::<f32>().unwrap().t().into_owned();

    // Format the output
    let mut predictions = Vec::new();
    for row in output.axis_iter(Axis(1)) {
        predictions.push(row.to_vec());
    }

    Ok(HttpResponse::Ok().json(InferenceResponse { predictions }))
}

/// # **Preprocesses the image before inference.**
///
/// This function crops the image to a square and resizes it to the required resolution.
fn preprocess_image(original_img: DynamicImage) -> image::RgbaImage {
    let (width, height) = (original_img.width(), original_img.height());
    let size = width.min(height);
    let x = (width - size) / 2;
    let y = (height - size) / 2;
    let cropped_img = imageops::crop_imm(&original_img, x, y, size, size).to_image();
    imageops::resize(&cropped_img, IMAGE_RESOLUTION, IMAGE_RESOLUTION, imageops::FilterType::CatmullRom)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // Initialize the ONNX session
    let model = web::Data::new(
        Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .commit_from_file(MODEL_PATH)
            .unwrap(),
    );

    println!("🚀 Server started at http://127.0.0.1:8080");

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(model.clone()) // Share the model session with the handlers
            .route("/infer", web::post().to(infer))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
