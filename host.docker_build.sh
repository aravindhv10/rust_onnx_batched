#!/bin/sh
cd "$('dirname' '--' "${0}")"

sudo -A docker build -t onnxrust .
