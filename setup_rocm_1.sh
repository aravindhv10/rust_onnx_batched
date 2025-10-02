#!/bin/sh


mkdir --parents --mode=0755 /etc/apt/keyrings

wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | \
    gpg --dearmor | tee /etc/apt/keyrings/rocm.gpg > /dev/null
