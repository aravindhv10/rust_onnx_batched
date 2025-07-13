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

fn main() {
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

    img.save("./tmp.png").expect("save failed")

    // let img = original_img.resize_exact(448, 448, FilterType::CatmullRom);
    // read the image
}
