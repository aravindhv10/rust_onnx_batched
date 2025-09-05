#!/bin/sh
cd "$('dirname' '--' "${0}")"

buildah build -t "$('cat' './image_name.txt')" .
