mod model;
mod mylib;
use model::get_inference_tuple;
use model::model_client;
use model::model_server;

use actix_multipart::Multipart;
use actix_web::App;
use actix_web::Error;
use actix_web::HttpResponse;
use actix_web::HttpServer;
use actix_web::web;

use ort::execution_providers::CUDAExecutionProvider;
use ort::execution_providers::OpenVINOExecutionProvider;
use ort::execution_providers::WebGPUExecutionProvider;
use ort::inputs;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

use tokio;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

use futures_util::TryStreamExt;
use ndarray::Array;
use ndarray::Axis;
use ndarray::Ix4;
use serde::Deserialize;
use serde::Serialize;

use std::net::IpAddr;
use std::net::Ipv4Addr;
use std::net::SocketAddr;
use std::ops::Index;
use std::sync::Arc;
use std::time::Duration;

use tonic::Request;
use tonic::Response;
use tonic::Status;
use tonic::transport::Server;

pub mod infer {
    tonic::include_proto!("infer"); // The string specified here must match the proto package name
}

async fn infer_handler(
    mut payload: Multipart,
    infer_slave: web::Data<Arc<model_client>>,
) -> Result<HttpResponse, Error> {
    let mut data = Vec::new();
    while let Some(mut field) = payload.try_next().await? {
        while let Some(chunk) = field.try_next().await? {
            data.extend_from_slice(&chunk);
        }
    }
    if data.is_empty() {
        return Ok(HttpResponse::BadRequest().body("No image data"));
    }
    match infer_slave.do_infer_data(data).await {
        Ok(pred) => {
            return Ok(HttpResponse::Ok().json(prediction_probabilities_reply::from(pred)));
        }
        Err(e) => {
            return Ok(HttpResponse::InternalServerError().body(e));
        }
    }
}

pub struct MyInferer {
    slave_client: Arc<model_client>,
}

#[tonic::async_trait]
impl infer::infer_server::Infer for MyInferer {
    async fn do_infer(
        &self,
        request: Request<infer::Image>,
    ) -> Result<Response<infer::Prediction>, Status> {
        println!("Received gRPC request");
        let image_data = request.into_inner().image_data;
        match self.slave_client.do_infer_data(image_data).await {
            Ok(pred) => {
                let reply = infer::Prediction {
                    ps1: pred.ps[0],
                    ps2: pred.ps[1],
                    ps3: pred.ps[2],
                };
                return Ok(Response::new(reply));
            }
            Err(e) => Err(Status::internal(e)),
        }
    }
}

#[actix_web::main]
async fn main() -> () {
    let (mut slave_server, slave_client) = get_inference_tuple();
    let slave_client_1 = Arc::new(slave_client);
    let slave_client_2 = Arc::clone(&slave_client_1);
    let future_infer = slave_server.infer_loop();
    match HttpServer::new(move || {
        App::new()
            .app_data(web::Data::new(Arc::clone(&slave_client_1)))
            .route("/infer", web::post().to(infer_handler))
    })
    .bind(("0.0.0.0", 8000))
    {
        Ok(ret) => {
            let future_rest_server = ret.run();
            let ip_v4 = IpAddr::V4(Ipv4Addr::new(0, 0, 0, 0));
            let addr = SocketAddr::new(ip_v4, 8001);
            let inferer_service = MyInferer {
                slave_client: slave_client_2,
            };
            let future_grpc = tonic::transport::Server::builder()
                .add_service(infer::infer_server::InferServer::new(inferer_service))
                .serve(addr);
            let (first, second, third) =
                tokio::join!(future_infer, future_rest_server, future_grpc);
            match second {
                Ok(_) => {
                    println!("REST server executed and stopped successfully");
                }
                Err(e) => {
                    println!("Encountered error in starting the server due to {}.", e);
                }
            }
            match third {
                Ok(_) => {
                    println!("GRPC server executed and stopped successfully");
                }
                Err(e) => {
                    println!("Encountered error in starting the server due to {}.", e);
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to bind to port");
        }
    }
}
