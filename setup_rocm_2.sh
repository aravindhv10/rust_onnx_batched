#!/bin/sh


echo 'START Update apt files' \
&& echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.0.1 jammy main' > '/etc/apt/sources.list.d/rocm.list' \
&& echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/graphics/7.0.1/ubuntu jammy main' >> '/etc/apt/sources.list.d/rocm.list' \
&& echo 'Package: *' > '/etc/apt/preferences.d/rocm-pin-600' \
&& echo 'Pin: release o=repo.radeon.com' >> '/etc/apt/preferences.d/rocm-pin-600' \
&& echo 'Pin-Priority: 600' >> '/etc/apt/preferences.d/rocm-pin-600' \
&& echo 'DONE Update apt files' ;
