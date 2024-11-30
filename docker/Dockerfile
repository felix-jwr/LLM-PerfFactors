FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

COPY ../src/ /app/src/
COPY ../docker/ /app/docker/

RUN pip install --upgrade pip
RUN pip install -r /app/docker/requirements.txt

# Make the script executable
RUN chmod +x /app/docker/run.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/.cache/huggingface /app/.config/matplotlib && \
    chown -R $(id -u):$(id -g) /app/.cache /app/.config

# Make folders to bypass permission issues
ENV XDG_CACHE_HOME=/app/.cache
ENV XDG_CONFIG_HOME=/app/.config
ENV HF_HOME=/app/.cache/huggingface
ENV MPLCONFIGDIR=/app/.config/matplotlib

# Add the API token
ENV API_TOKEN=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

# USER <username>[:<group>] sets the user to run the container as
USER $(id -u):$(id -g)

# CMD <command> sets the default command to run when the container starts
CMD ["/app/docker/run.sh"]
