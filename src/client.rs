

pub mod infer {
    tonic::include_proto!("infer"); // The string specified here must match the proto package name
}

use std::fs;
use std::error::Error;

#[actix_web::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let  data =  fs::read("./image.png").expect("Failed reading image file");
    let img = infer::Image{
        image_data: data
    };
    let mut client = infer::infer_client::InferClient::connect("http://127.0.0.1:8001").await?;
    let res = client.do_infer(img).await?;
    println!("{:?}",res);
    return Ok(());
}
