use image::{GenericImageView, imageops, imageops::FilterType};
use ndarray::{Array, Axis, s};
use std::path::Path;

use ort::{
    inputs,
    session::{Session, SessionOutputs},
    value::TensorRef,
};

use raqote::{DrawOptions, DrawTarget, LineJoin, PathBuilder, SolidSource, Source, StrokeStyle};
// use show_image::{AsImageView, WindowOptions, event};
// use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

const CLASS_LABELS: [&str; 3] = ["empty", "occupied", "other"];

const MODEL_URL: &str = "/home/asd/MODEL_CHECKPOINTS/PATIENT_DETECTION/patient_detect-epoch=15-val_loss=0.02.onnx";

const IMAGE_RESOLUTION: usize = 448;

fn main() -> ort::Result<()> {

    let original_img = image::open(Path::new(
        "/home/asd/DATASET/image_dataset/both_arms_out/v36frame0048.jpg",
    ))
    .unwrap();
    let (width, height) = (original_img.width(), original_img.height());
    let size = width.min(height);
    let x = (width - size) / 2;
    let y = (height - size) / 2;
    let cropped_img = imageops::crop_imm(&original_img, x, y, size, size).to_image();
    let img = imageops::resize(&cropped_img, 448, 448, imageops::FilterType::CatmullRom);

    let mut input = Array::zeros((2 as usize, IMAGE_RESOLUTION, IMAGE_RESOLUTION, 3 as usize));

    for (x, y, pixel) in img.enumerate_pixels(){
            let [r, g, b, _] = pixel.0;
            input[[0, y as usize, x as usize, 0]] = r;
            input[[0, y as usize, x as usize, 1]] = g;
            input[[0, y as usize, x as usize, 2]] = b;

            input[[1, y as usize, x as usize, 0]] = r;
            input[[1, y as usize, x as usize, 1]] = g;
            input[[1, y as usize, x as usize, 2]] = b;
            // input[[0, y, x, 1]] = g;
            // input[[0, y, x, 2]] = b;
    }

    let mut model = Session::builder()?.commit_from_file(MODEL_URL)?;
    let outputs: SessionOutputs = model.run(inputs!["input" => TensorRef::from_array_view(&input)?])?;
    let output = outputs["output"].try_extract_array::<f32>()?.t().into_owned();
    println!("{:?}", output);

    for row in output.axis_iter(Axis(1)){
        println!("{:?}", row);
            
    }

    // println!("{}",output[[0, 0]]);
    // println!("{}",output[[0, 1]]);
    // println!("{}",output[[0, 2]]);
    

    // img.save("./tmp.png").expect("save failed")

    // let img = original_img.resize_exact(448, 448, FilterType::CatmullRom);
    // read the image
    Ok(())
}
