# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM rust:bookworm AS builder

WORKDIR /src
COPY . .
RUN cargo build --locked --release --package pyrefly --bin pyrefly

FROM python:3.13-slim-bookworm

COPY --from=builder /src/target/release/pyrefly /usr/local/bin/pyrefly
WORKDIR /workspace

ENTRYPOINT ["pyrefly"]
CMD ["check"]
