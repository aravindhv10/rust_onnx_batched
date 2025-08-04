#!/bin/sh
cd "$('dirname' '--' "${0}")"

podman build -t "$('cat' './image_name.txt')" .
