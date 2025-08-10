use actix_multipart::Multipart;
use actix_web::App;
use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::HttpServer;
// use actix_web::Responder;
use actix_web::web;
use bincode::{Decode, Encode, config};
use futures_util::TryStreamExt;
use gxhash;
use image::DynamicImage;
// use image::GenericImageView;
use image::imageops;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;
use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
// use ort::session::SessionOutputs;
use ort::value::TensorRef;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
// use std::io::Write;
use std::path::Path;
use std::sync::Mutex;
use std::time::SystemTime;

const MODEL_PATH: &str = "./model.onnx";
const PATH_DIR_IMAGE: &str = "/tmp/image/";
const PATH_DIR_INCOMPLETE: &str = "/tmp/incomplete/";
const PATH_DIR_OUT: &str = "/tmp/out/";
const CLASS_LABELS: [&str; 3] = ["empty", "occupied", "other"];
const IMAGE_RESOLUTION: u32 = 448;

#[derive(Debug, PartialEq, Encode, Decode, Serialize, Deserialize)]
struct prediction_probabilities {
    ps: [f32; 3],
}

#[derive(Serialize)]
struct prediction_probabilities_reply {
    p1: String,
    p2: String,
    p3: String,
    mj: String,
}

fn get_prediction_for_reply(input: prediction_probabilities) -> prediction_probabilities_reply {
    let mut max_index: usize = 0;

    for i in 1..3 {
        if input.ps[i] > input.ps[max_index] {
            max_index = i;
        }
    }

    return prediction_probabilities_reply {
        p1: input.ps[0].to_string(),
        p2: input.ps[1].to_string(),
        p3: input.ps[2].to_string(),
        mj: CLASS_LABELS[max_index].to_string(),
    };
}

fn save_predictions(result: &prediction_probabilities, hash_key: &str) -> Result<(), Error> {
    match bincode::encode_to_vec(&result, config::standard()) {
        Ok(encoded) => {
            let s1: String = String::from(PATH_DIR_OUT);
            let s2: String = s1 + hash_key;
            match fs::write(&s2, encoded) {
                Ok(_) => {
                    eprintln!("Wrote prediction to file {}", &s2);
                    return Ok(());
                }
                Err(e) => {
                    eprintln!("Failed to write predictions into {} due to {}", &s2, e);
                    return Err(e.into());
                }
            }
        }
        Err(e) => {
            eprintln!("Failed encoding the result {}", e);
            return Err(actix_web::error::ErrorInternalServerError(e.to_string()));
        }
    }
}

fn load_predictions(hash_key: &str) -> Result<prediction_probabilities, Error> {
    let s1: String = String::from(PATH_DIR_OUT);
    let s2: String = s1 + hash_key;
    match fs::read(s2) {
        Ok(encoded) => match bincode::decode_from_slice(&encoded[..], config::standard()) {
            Ok(res) => {
                let (decoded, _len): (prediction_probabilities, usize) = res;
                return Ok(decoded);
            }
            Err(e) => {
                return Err(actix_web::error::ErrorInternalServerError(e.to_string()));
            }
        },
        Err(e) => {
            return Err(e.into());
        }
    }
}

fn save_image(image_data: &Vec<u8>, name_image: &str) -> Result<(), Error> {
    match fs::create_dir_all(PATH_DIR_INCOMPLETE) {
        Ok(_) => match fs::create_dir_all(PATH_DIR_IMAGE) {
            Ok(_) => {
                let s1: String = String::from(PATH_DIR_INCOMPLETE);
                let s2: String = s1 + name_image;
                match fs::write(&s2, image_data) {
                    Ok(_) => {
                        let s1: String = String::from(PATH_DIR_IMAGE);
                        let s3: String = s1 + name_image;
                        match fs::rename(&s2, &s3) {
                            Ok(_) => Ok(()),
                            Err(e) => {
                                eprintln!(
                                    "Failed to rename the temporary file {} to {} due to {}",
                                    s2, s3, e
                                );
                                Err(e.into())
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("Failed to write the temporary file {} due to {}", s2, e);
                        Err(e.into())
                    }
                }
            }
            Err(e) => {
                eprintln!("Failed creating directory {} due to {}", PATH_DIR_IMAGE, e);
                Err(e.into())
            }
        },
        Err(e) => {
            eprintln!(
                "Failed creating directory {} due to {}",
                PATH_DIR_INCOMPLETE, e
            );
            Err(e.into())
        }
    }
}

fn read_image(path_file_input: &str) -> Result<DynamicImage, Error> {
    match fs::read(path_file_input) {
        Ok(image_data) => match image::load_from_memory(&image_data) {
            Ok(original_image) => {
                return Ok(original_image);
            }
            Err(e) => {
                eprintln!("Failed to decode image due to {}.", e);
                return Err(actix_web::error::ErrorInternalServerError(e.to_string()));
            }
        },
        Err(e) => {
            eprintln!("Unable to read the file {} due to {}", path_file_input, e);
            return Err(e.into());
        }
    }
}

fn hash_image_content(image_data: &Vec<u8>) -> String {
    let seed = 123456789;
    format!("{:x}", gxhash::gxhash128(&image_data, seed))
}

// fn get_time_from_epoch() -> Result<u64, String> {
//     match SystemTime::now().duration_since(SystemTime::UNIX_EPOCH) {
//         Err(_) => {
//             return Err("Failed to get time".to_string());
//         },
//         Ok(n) => {
//             return Ok(n.as_secs());
//         },
//     }
// }

fn get_list_files_under_dir(path_dir_input: &str) -> Result<Vec<String>, Error> {
    match fs::read_dir(path_dir_input) {
        Ok(list_entry) => {
            let mut ret: Vec<String> = vec![];
            for i in list_entry {
                match i {
                    Ok(path) => {
                        ret.push(path.path().display().to_string());
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to read a path inside directory {} due to {}",
                            path_dir_input, e
                        );
                    }
                }
            }
            Ok(ret)
        }
        Err(e) => {
            eprintln!("Failed to read directory: {}", e);
            Err(e.into())
        }
    }
}

fn clean_old_out(timeout: u64) {
    let time_now = SystemTime::now();

    match get_list_files_under_dir(PATH_DIR_OUT) {
        Err(e) => {
            println!("Failed to get list of files {}", e);
        }
        Ok(list_entry) => {
            for i in list_entry {
                match fs::metadata(i.as_str()) {
                    Err(e) => {
                        println!("Failed to get metadata due to {}", e);
                    }
                    Ok(metadata) => match metadata.created() {
                        Err(e) => {
                            println!("Failed to get creation time due to {}", e);
                        }
                        Ok(creation_time) => match time_now.duration_since(creation_time) {
                            Err(e) => {
                                println!("Duration failed {}", e);
                            }
                            Ok(n) => {
                                if n.as_secs() > timeout {
                                    match std::fs::remove_file(Path::new(i.as_str())) {
                                        Err(e) => {
                                            println!(
                                                "Failed to remove old file {} due to {}",
                                                i, e
                                            );
                                        }
                                        Ok(_) => {
                                            println!("Removed old file {}", i);
                                        }
                                    }
                                } else {
                                    println!("Not removing {} as its not old", i);
                                }
                            }
                        },
                    },
                }
            }
        }
    }
}

fn do_batched_infer_on_list_file_under_dir(model: &web::Data<Mutex<Session>>, img_hash: &str) -> Result<(), Error> {
    let mut session = model.lock().unwrap();
    clean_old_out(86400);

    if check_existance_of_predictions(&img_hash) {
        eprintln!("Already inferred, nothing to be done");
        return Ok(());
    }

    match fs::create_dir_all(PATH_DIR_OUT) {
        Ok(_) => match get_list_files_under_dir(PATH_DIR_IMAGE) {
            Ok(list_file) => {
                let batch_size = list_file.len();
                if batch_size > 0 {
                    eprintln!("Inferring with batch_size = {}", batch_size);

                    let mut keys: Vec<&str> = Vec::with_capacity(batch_size);

                    let mut input = Array::<u8, Ix4>::zeros((
                        batch_size,
                        IMAGE_RESOLUTION as usize,
                        IMAGE_RESOLUTION as usize,
                        3,
                    ));

                    for i in 0..batch_size {
                        keys.push(&list_file[i][PATH_DIR_IMAGE.len()..]);

                        match read_image(list_file[i].as_str()) {
                            Ok(original_image) => {
                                let preprocessed_image = preprocess_image(original_image);
                                for (x, y, pixel) in preprocessed_image.enumerate_pixels() {
                                    let [r, g, b, _] = pixel.0;
                                    input[[i as usize, y as usize, x as usize, 0]] = r;
                                    input[[i as usize, y as usize, x as usize, 1]] = g;
                                    input[[i as usize, y as usize, x as usize, 2]] = b;
                                }

                                match std::fs::remove_file(Path::new(list_file[i].as_str())) {
                                    Ok(_) => {
                                        eprintln!(
                                            "Removed image file {} after reading it.",
                                            list_file[i].as_str()
                                        );
                                    }
                                    Err(e) => {
                                        eprintln!(
                                            "Failed to remove file {} after reading it due to {}.",
                                            list_file[i].as_str(),
                                            e
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                eprintln!("Unable to read image {} due to {}.", list_file[i], e);
                            }
                        }
                    }

                    let outputs = session
                        .run(inputs!["input" => TensorRef::from_array_view(&input).unwrap()])
                        .unwrap();

                    let output = outputs["output"]
                        .try_extract_array::<f32>()
                        .unwrap()
                        .t()
                        .into_owned();

                    println!("output => {:?}", output);

                    for (index, row) in output.axis_iter(Axis(1)).enumerate() {
                        let result = prediction_probabilities {
                            ps: [row[0], row[1], row[2]],
                        };

                        eprintln!("Inside prediction results: {:?}", result);
                        match save_predictions(&result, keys[index]) {
                            Ok(_) => {}
                            Err(_) => {}
                        }
                    }
                }

                // eprintln!("Done inferring, now returning");
                // return Ok(());
            }
            Err(e) => {
                eprintln!("Failed reading dir: {}", e);
                return Err(e.into());
            }
        },
        Err(e) => {
            eprintln!(
                "Unable to create the output directory {} due to the error {}",
                PATH_DIR_OUT, e
            );
            return Err(e.into());
        }
    }
    eprintln!("Done inferring, now returning");
    return Ok(());
}

fn check_existance_of_predictions(hash_key: &str) -> bool {
    let s1: String = String::from(PATH_DIR_OUT);
    let s2: String = s1 + hash_key;
    return Path::new(&s2).exists();
}

/// # **Handles the inference request.**
///
/// This function takes the multipart request, extracts the image, preprocesses it,
/// runs the inference, and returns the JSON response.
async fn infer(
    mut payload: Multipart,
    model: web::Data<Mutex<Session>>,
) -> Result<HttpResponse, Error> {
    // Isolate the image data from the multipart payload
    let mut image_data = Vec::new();
    while let Some(mut field) = payload.try_next().await? {
        if field.content_disposition().get_name() == Some("file") {
            while let Some(chunk) = field.try_next().await? {
                image_data.extend_from_slice(&chunk);
            }
        }
    }

    if image_data.is_empty() {
        return Ok(HttpResponse::BadRequest().body("Image data not provided."));
    }

    let img_hash = hash_image_content(&image_data);

    if !check_existance_of_predictions(&img_hash) {
        let _ = save_image(&image_data, &img_hash);

        match do_batched_infer_on_list_file_under_dir(&model, &img_hash) {
            Ok(_) => {
                eprintln!("Done with inference");
            }
            Err(e) => {
                eprintln!("Failed at inference due to {}", e);
            }
        }
    }

    match load_predictions(&img_hash) {
        Ok(preds) => {
            eprintln!("Predictions inside the web function: {:?}", preds);

            return Ok(HttpResponse::Ok().json(get_prediction_for_reply(preds)));
        }
        Err(e) => {
            eprintln!("Failed in loading predictions from the cache due to {}", e);

            let tmp = prediction_probabilities {
                ps: [0.0, 0.0, 1.0],
            };

            return Ok(HttpResponse::Ok().json(get_prediction_for_reply(tmp)));
        }
    }
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
    imageops::resize(
        &cropped_img,
        IMAGE_RESOLUTION,
        IMAGE_RESOLUTION,
        imageops::FilterType::CatmullRom,
    )
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
            // return get_webgpu_model().unwrap();
        }
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let model = web::Data::new(Mutex::new(get_model()));

    eprintln!("🚀 Server started at http://0.0.0.0:8000");

    // Start the HTTP server
    HttpServer::new(move || {
        App::new()
            .app_data(model.clone()) // Share the model session with the handlers
            .route("/infer", web::post().to(infer))
    })
    .bind(("0.0.0.0", 8000))?
    .run()
    .await
}
