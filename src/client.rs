pub mod infer {
    tonic::include_proto!("infer"); // The string specified here must match the proto package name
}

use std::error::Error;
use std::fs;

#[actix_web::main]
async fn main() -> Result<(), tonic::transport::Error> {
    let mut client = infer::infer_client::InferClient::connect("0.0.0.0:8001").await?;
    let data = fs::read("./image.png");
    return Ok(());
}
