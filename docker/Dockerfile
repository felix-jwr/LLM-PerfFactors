FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY ../src/ /app/src/
COPY ../docker/ /app/docker/

RUN pip install --upgrade pip
RUN pip install -r /app/docker/requirements.txt

# Install dependencies, including a C compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Make the script executable
RUN chmod +x /app/docker/run.sh

# # Create necessary directories and set permissions
# RUN mkdir -p /app/.cache/huggingface /app/.config/matplotlib /app/.cache/triton && \
#     chown -R $(id -u):$(id -g) /app/.cache /app/.config

# # Environment variables for cache and config directories
# ENV XDG_CACHE_HOME=/app/.cache
# ENV XDG_CONFIG_HOME=/app/.config
# ENV HF_HOME=/app/.cache/huggingface
# ENV MPLCONFIGDIR=/app/.config/matplotlib
# ENV TRITON_CACHE_DIR=/app/.cache/triton

# USER <username>[:<group>] sets the user to run the container as
# USER $(id -u):$(id -g)

# CMD <command> sets the default command to run when the container starts
CMD ["/app/docker/run.sh"]
