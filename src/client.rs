pub mod infer {
    tonic::include_proto!("infer"); // The string specified here must match the proto package name
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let mut client = infer::infer_client::InferClient::connect("0.0.0.0:8001").await?;
}
