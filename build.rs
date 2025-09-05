use std::{env, path::PathBuf};

fn main() {
    tonic_prost_build::compile_protos("./infer.proto").unwrap();
}
