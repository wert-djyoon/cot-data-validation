from nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

WORKDIR /app

# Copy files
COPY ./.python-version /app/.python-version
COPY ./model_merge_adapter.py /app/model_merge_adapter.py
COPY ./model_test.ipynb /app/model_test.ipynb
COPY ./model_train.py /app/model_train.py
COPY ./pyproject.toml /app/pyproject.toml
COPY ./README.md /app/README.md
COPY ./uv.lock /app/uv.lock

# Install dependencies
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="/usr/local/bin" sh
# RUN uv sync
