#!/bin/sh


cd "$('dirname' '--' "${0}")"
IMAGE_NAME="$(cat './image_name.txt')"

buildah build -t "${IMAGE_NAME}" .
