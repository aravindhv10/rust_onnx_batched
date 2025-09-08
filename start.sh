#!/bin/sh
cd "$(dirname -- "${0}")"
export RUSTFLAGS="-C target-cpu=native"
cargo run --release --bin 'infer-server' &
sleep 30 ; echo running inference ; cargo run --release --bin 'infer-client'
echo done inference
exit '0'
