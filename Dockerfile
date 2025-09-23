

# FROM rocm/onnxruntime:rocm7.0_ub24.04_ort1.22_torch2.8.0 AS rust
FROM rocm/dev-ubuntu-24.04:7.0-complete AS rust
# FROM rocm/pytorch:latest AS rust

RUN \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    echo 'START apt-get stuff' \
    && apt-get -y update \
    && apt-get install -y \
        'git' \
    && echo 'DONE apt-get stuff' ;

RUN \
    echo 'START clone onnx runtime' \
    && cd / \
    && git clone \
        '--single-branch' \
        '--branch' 'main' \
        '--recursive' 'https://github.com/Microsoft/onnxruntime.git' \
        '/onnxruntime' \
    && echo 'DONE clone onnx runtime' ;

RUN \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    echo 'START apt-get stuff' \
    && apt-get -y update \
    && apt-get install -y \
        'curl' \
    && echo 'DONE apt-get stuff' ;

RUN \
    echo 'START uv download' \
    && curl -LsSf 'https://astral.sh/uv/install.sh' | sh \
    && cp -vf -- "${HOME}/.local/bin/uv" '/usr/local/bin/' \
    && echo 'DONE uv download' ;

RUN \
    echo 'START build and install onnxruntime' \
    && uv venv '/opt/venv' \
    && . '/opt/venv/bin/activate' \
    && uv pip install -U pip \
    && echo 'DONE build and install onnxruntime' ;

RUN \
    --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    echo 'START apt-get stuff' \
    && apt-get -y update \
    && apt-get install -y \
        'build-essential' \
        'cmake' \
        'gdb' \
        'git' \
        'half' \
        'migraphx' \
        'migraphx-dev' \
    && echo 'DONE apt-get stuff' ;

ENV ORT_DYLIB_PATH='/lib/libonnxruntime.so.1'
ENV ORT_STRATEGY='system'
ENV CC='/opt/rocm/llvm/bin/clang'
ENV CXX='/opt/rocm/llvm/bin/clang++'
ENV CMAKE_HIP_COMPILER='/opt/rocm/llvm/bin/clang++'
ENV HIP_COMPILER='/opt/rocm/llvm/bin/clang++'
ENV ORT_MIGRAPHX_SAVE_COMPILED_PATH='/COMPILED'
ENV ORT_MIGRAPHX_LOAD_COMPILED_PATH="${ORT_MIGRAPHX_SAVE_COMPILED_PATH}"
RUN mkdir -pv -- "${ORT_MIGRAPHX_SAVE_COMPILED_PATH}"

RUN \
    echo 'START build and install onnxruntime' \
    && . '/opt/venv/bin/activate' \
    && cd / \
    && /bin/sh '/onnxruntime/dockerfiles/scripts/install_common_deps.sh' \
    && cd '/onnxruntime' \
    && ./build.sh --allow_running_as_root --config Release --build_wheel --parallel --use_migraphx --migraphx_home /opt/rocm \
    && echo 'DONE build and install onnxruntime' ;

USER root
WORKDIR '/root'

ENV HOME='/root'
ENV DEBIAN_FRONTEND='noninteractive'
ENV RUSTUP_HOME='/usr/local/rustup'
ENV CARGO_HOME='/usr/local/cargo'
ENV RUST_VERSION='1.90.0'
ENV PATH="/usr/local/cargo/bin:${PATH}"

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

RUN set -eux; \
    dpkgArch="$(dpkg --print-architecture)"; \
    case "${dpkgArch##*-}" in \
        amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='20a06e644b0d9bd2fbdbfd52d42540bdde820ea7df86e92e533c073da0cdd43c' ;; \
        armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='3b8daab6cc3135f2cd4b12919559e6adaee73a2fbefb830fadf0405c20231d61' ;; \
        arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='e3853c5a252fca15252d07cb23a1bdd9377a8c6f3efa01531109281ae47f841c' ;; \
        i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='a5db2c4b29d23e9b318b955dd0337d6b52e93933608469085c924e0d05b1df1f' ;; \
        ppc64el) rustArch='powerpc64le-unknown-linux-gnu'; rustupSha256='acd89c42b47c93bd4266163a7b05d3f26287d5148413c0d47b2e8a7aa67c9dc0' ;; \
        s390x) rustArch='s390x-unknown-linux-gnu'; rustupSha256='726b7fd5d8805e73eab4a024a2889f8859d5a44e36041abac0a2436a52d42572' ;; \
        riscv64) rustArch='riscv64gc-unknown-linux-gnu'; rustupSha256='09e64cc1b7a3e99adaa15dd2d46a3aad9d44d71041e2a96100d165c98a8fd7a7' ;; \
        *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
    esac; \
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

FROM rust

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
        'libopenblas64-dev' \
        'libopenblas-dev' \
        'libprotobuf-dev' \
        'libssl-dev' \
        'make' \
        'nasm' \
        'neovim' \
        'ninja-build' \
        'pkg-config' \
        'protobuf-compiler' \
        'python3-cairo-dev' \
        'python3-dev' \
        'python3-opencv' \
        'python3-pip' \
        'python3-setuptools' \
        'unzip' \
        'wget' \
    && echo 'DONE apt-get stuff' ;

EXPOSE 8000/tcp
