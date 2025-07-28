#!/bin/sh
curl -X POST "http://127.0.0.1:8080/infer" -F "image=@./image.png"
curl -X POST "http://127.0.0.1:8080/infer" -F "image=@./image.jpg"
