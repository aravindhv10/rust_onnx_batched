use actix_multipart::Multipart;
use actix_web::App;
use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::Responder;
use actix_web::web;
use bincode::{Decode, Encode, config};
use futures_util::TryStreamExt;
use gxhash;
use image::DynamicImage;
use image::GenericImageView;
use image::imageops;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;
use ort::execution_providers::CUDAExecutionProvider;
use ort::inputs;
use ort::session::Session;
use ort::session::SessionOutputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::sync::Mutex;

const MODEL_PATH: &str = "./model.onnx";
const PATH_DIR_IMAGE: &str = "/tmp/image/";
const PATH_DIR_INCOMPLETE: &str = "/tmp/incomplete/";
const PATH_DIR_OUT: &str = "/tmp/out/";

const CLASS_LABELS: [&str; 3] = ["empty", "occupied", "other"];

const IMAGE_RESOLUTION: u32 = 448;

// #[derive(Serialize, Deserialize, Debug, PartialEq, Encode, Decode)]
#[derive(Debug, PartialEq, Encode, Decode, Serialize, Deserialize)]
struct prediction_probabilities {
    p1: f32,
    p2: f32,
    p3: f32,
}

fn save_predictions(result: &prediction_probabilities, hash_key: &str) -> Result<(), Error> {
    match bincode::encode_to_vec(&result, config::standard()) {
        Ok(encoded) => {
            let s1: String = String::from(PATH_DIR_OUT);
            let s2: String = s1 + hash_key;
            match fs::write(&s2, encoded) {
                Ok(_) => {
                    println!("Wrote prediction to file {}", &s2);
                    return Ok(());
                }
                Err(e) => {
                    println!("Failed to write predictions into {} due to {}", &s2, e);
                    return Err(e.into());
                }
            }
        }
        Err(e) => {
            println!("Failed encoding the result {}", e);
            return Err(actix_web::error::ErrorInternalServerError(e.to_string()));
        }
    }
}

// fn load_predictions ( hash_key: &str) -> Result<prediction_probabilities, Error> {
// }

fn hash_image_content(image_data: &Vec<u8>) -> String {
    let seed = 123456789;
    format!("{:x}", gxhash::gxhash128(&image_data, seed))
}

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
                println!("Failed to decode image due to {}.", e);
                return Err(actix_web::error::ErrorInternalServerError(e.to_string()));
            }
        },
        Err(e) => {
            println!("Unable to read the file {} due to {}", path_file_input, e);
            return Err(e.into());
        }
    }
}

fn do_batched_infer_on_list_file_under_dir(model: &web::Data<Mutex<Session>>) -> Result<(), Error> {
    let mut session = model.lock().unwrap();

    match fs::create_dir_all(PATH_DIR_OUT) {
        Ok(_) => match get_list_files_under_dir(PATH_DIR_IMAGE) {
            Ok(list_file) => {
                let batch_size = list_file.len();
                let mut keys: Vec<&str> = Vec::with_capacity(batch_size);

                let mut input = Array::<u8, Ix4>::zeros((
                    batch_size,
                    IMAGE_RESOLUTION as usize,
                    IMAGE_RESOLUTION as usize,
                    3,
                ));

                for i in 0..batch_size {
                    keys.push(&list_file[i][PATH_DIR_IMAGE.len()..]);
                    // match image::open(Path::new(list_file[i].as_str())) {
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
                                    println!(
                                        "Removed image file {} after reading it.",
                                        list_file[i].as_str()
                                    );
                                }
                                Err(e) => {
                                    println!(
                                        "Failed to remove file {} after reading it due to {}.",
                                        list_file[i].as_str(),
                                        e
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            println!("Unable to read image {} due to {}.", list_file[i], e);
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

                let mut results: Vec<prediction_probabilities> = vec![];

                for (index, row) in output.axis_iter(Axis(1)).enumerate() {
                    let result = prediction_probabilities {
                        p1: row[0],
                        p2: row[1],
                        p3: row[2],
                    };

                    println!("Inside prediction results: {:?}", result);
                    match save_predictions(&result, keys[index]) {
                        Ok(_) => {}
                        Err(e) => {}
                    }

                    // let encoded: Vec<u8> =
                    //     bincode::encode_to_vec(&result, config::standard()).unwrap();

                    // let s1: String = String::from(PATH_DIR_OUT);
                    // let s2: String = s1 + keys[index];

                    // match fs::write(&s2, encoded) {
                    //     Ok(_) => {
                    //         println!("Wrote prediction to file {}", &s2);
                    //     }
                    //     Err(e) => {
                    //         println!("Failed to write predictions into {} due to {}", &s2, e);
                    //     }
                    // }
                }
                println!("Done inferring, now returning");
                return Ok(());
            }
            Err(e) => {
                println!("Failed reading dir: {}", e);
                return Err(e.into());
            }
        },
        Err(e) => {
            println!(
                "Unable to create the output directory {} due to the error {}",
                PATH_DIR_OUT, e
            );
            return Err(e.into());
        }
    }
    println!("Done inferring, now returning");
    return Ok(());
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

    let _ = save_image(&image_data, &img_hash);

    let _ = do_batched_infer_on_list_file_under_dir(&model);

    match get_list_files_under_dir(PATH_DIR_IMAGE) {
        Ok(list_file) => {
            println!("List of files {:?}", list_file);
            for i in list_file {
                let newstr = &i[PATH_DIR_IMAGE.len()..];
                println!("{}", newstr);
            }
        }
        Err(e) => {
            println!("Failed reading dir: {}", e);
        }
    }

    // Load and preprocess the image
    let original_img = image::load_from_memory(&image_data)
        .map_err(|e| actix_web::error::ErrorInternalServerError(e.to_string()))?;

    let preprocessed_image = preprocess_image(original_img);

    // Prepare the input tensor
    let mut input = Array::zeros((1, IMAGE_RESOLUTION as usize, IMAGE_RESOLUTION as usize, 3));
    for (x, y, pixel) in preprocessed_image.enumerate_pixels() {
        let [r, g, b, _] = pixel.0;
        input[[0, y as usize, x as usize, 0]] = r;
        input[[0, y as usize, x as usize, 1]] = g;
        input[[0, y as usize, x as usize, 2]] = b;
    }

    // Run the model
    let mut session = model.lock().unwrap();

    let outputs = session
        .run(inputs!["input" => TensorRef::from_array_view(&input).unwrap()])
        .unwrap();

    let output = outputs["output"]
        .try_extract_array::<f32>()
        .unwrap()
        .t()
        .into_owned();

    // Format the output
    // let mut predictions = Vec::new();
    for row in output.axis_iter(Axis(1)) {
        if ((row[0] > row[1]) & (row[0] > row[2])) {
            return Ok(HttpResponse::Ok().json(prediction_probabilities {
                p1: row[0],
                p2: row[1],
                p3: row[2],
            }));
        } else if ((row[1] > row[0]) & (row[1] > row[2])) {
            return Ok(HttpResponse::Ok().json(prediction_probabilities {
                p1: row[0],
                p2: row[1],
                p3: row[2],
            }));
        } else {
            return Ok(HttpResponse::Ok().json(prediction_probabilities {
                p1: row[0],
                p2: row[1],
                p3: row[2],
            }));
        }
    }

    return Ok(HttpResponse::Ok().json(prediction_probabilities {
        p1: 0.0,
        p2: 0.0,
        p3: 1.0,
    }));
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

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // ort::init()
    //     .with_execution_providers([CUDAExecutionProvider::default().build()])
    //     .commit()
    //     .unwrap();
    // Initialize the ONNX session
    let model = web::Data::new(Mutex::new(
        Session::builder()
            .unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .unwrap()
            .commit_from_file(MODEL_PATH)
            .unwrap(),
    ));

    println!("🚀 Server started at http://0.0.0.0:8000");

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
