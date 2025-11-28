from nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04


# Install dependencies
COPY pyproject.toml uv.lock .
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
RUN uv sync

# Copy source codes
COPY . /app