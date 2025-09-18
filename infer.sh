#!/bin/sh
curl -X POST "http://127.0.0.1:8000/infer" -F "file=@./image.png"
curl -X POST "http://127.0.0.1:8000/infer" -F "file=@./image.jpg"
