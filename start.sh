#!/bin/sh
cd "$(dirname -- "${0}")"
export RUSTFLAGS="-C target-cpu=native"
cargo run --release
