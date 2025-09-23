#!/bin/sh


cd "$('dirname' '--' "${0}")"
IMAGE_NAME="$(cat './image_name.txt')"

sudo -A docker build -t "${IMAGE_NAME}" .
