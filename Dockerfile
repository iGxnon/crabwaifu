FROM ubuntu:24.04 AS builder

SHELL ["/bin/bash", "-c"]

WORKDIR /app

COPY rust-toolchain .

RUN set -eux; \
    apt-get update; \
    apt-get install -y curl clang python3 cmake git build-essential libssl-dev libvulkan-dev mesa-vulkan-drivers vulkan-tools pkg-config; \
    rm -rf /var/lib/apt/lists/*; \
    curl https://sh.rustup.rs -sSf | bash -s -- -y --default-toolchain $(cat rust-toolchain);  # cache toolchain

COPY Cargo.toml /app
COPY crates /app/crates

# TODO: cache build dependencies

RUN set -eux; \
    source $HOME/.cargo/env; \
    cargo build --release --bins;

FROM ubuntu:24.04 AS server

SHELL ["/bin/bash", "-c"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y libssl-dev wget net-tools iputils-ping tcpdump ethtool iperf3 iproute2 libvulkan-dev mesa-vulkan-drivers vulkan-tools; \
    rm -rf /var/lib/apt/lists/*; \
    wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh; \
    chmod +x wait-for-it.sh;

WORKDIR /app

COPY --from=builder /app/target/release/crabwaifu-server /usr/local/bin/crabwaifu-server

ENTRYPOINT [ "crabwaifu-server" ]

FROM ubuntu:24.04 AS client

SHELL ["/bin/bash", "-c"]

RUN set -eux; \
    apt-get update; \
    apt-get install -y libssl-dev wget net-tools iputils-ping tcpdump ethtool iperf3 iproute2; \
    rm -rf /var/lib/apt/lists/*; \
    wget https://raw.githubusercontent.com/vishnubob/wait-for-it/master/wait-for-it.sh; \
    chmod +x wait-for-it.sh;

WORKDIR /app

COPY --from=builder /app/target/release/crabwaifu-client /usr/local/bin/crabwaifu-client

ENTRYPOINT [ "crabwaifu-client" ]
