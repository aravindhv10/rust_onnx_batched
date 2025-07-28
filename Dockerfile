FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 as rustcuda

ENV HOME='/root'
ENV DEBIAN_FRONTEND='noninteractive'
ENV NVIDIA_DRIVER_CAPABILITIES='compute,utility,video'
WORKDIR '/root'

RUN \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    echo 'START apt-get stuff' \
    && apt-get -y update \
    && apt-get install -y \
        'aria2' \
        'build-essential' \
        'cmake' \
        'curl' \
        'git' \
        'git-lfs' \
        'libfontconfig-dev' \
        'libssl-dev' \
        'make' \
        'nasm' \
        'pkg-config' \
        'wget' \
    && echo 'DONE apt-get stuff' ;

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    RUST_VERSION=1.88.0

RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    rustArch='x86_64-unknown-linux-gnu'; \
    rustupSha256='20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c' ; \
    url="https://static.rust-lang.org/rustup/archive/1.28.2/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --profile minimal --default-toolchain $RUST_VERSION --default-host ${rustArch}; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

FROM rustcuda

RUN \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    echo 'START apt-get stuff' \
    && apt-get -y update \
    && apt-get install -y \
        'aria2' \
        'build-essential' \
        'cmake' \
        'curl' \
        'ffmpeg' \
        'fish' \
        'git' \
        'git-lfs' \
        'ipython3' \
        'libcairo2-dev' \
        'libfontconfig-dev' \
        'libssl-dev' \
        'make' \
        'nasm' \
        'neovim' \
        'ninja-build' \
        'pkg-config' \
        'python3-cairo-dev' \
        'python3-dev' \
        'python3-opencv' \
        'python3-pip' \
        'python3-setuptools' \
        'unzip' \
        'wget' \
    && echo 'DONE apt-get stuff' ;

EXPOSE 8000/tcp
