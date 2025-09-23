#!/bin/sh
cd "$(dirname -- "${0}")"
export RUSTFLAGS="-C target-cpu=native"

cargo run --release --bin 'infer-server' &
sleep 20 ; echo running inference ; cargo run --release --bin 'infer-client'

cargo run --release --bin 'infer-server'

echo done inference
exit '0'
