FROM bath:2022-gpu

WORKDIR /app

# COPY <src> <dest> copies files into the image, can do entire directories
COPY ../src/ /app/src/
COPY ../docker/ /app/docker/

# RUN to install packages/modifications to the base image
RUN pip install -r /app/docker/requirements.txt

# Make the script executable
RUN chmod +x /app/docker/run.sh

# Create necessary directories and set permissions
RUN mkdir -p /app/.cache/huggingface /app/.config/matplotlib && \
    chown -R $(id -u):$(id -g) /app/.cache /app/.config

# ENV key=value sets environment variables
ENV HF_HOME=/app/.cache/huggingface
ENV MPLCONFIGDIR=/app/.config/matplotlib

# USER <username>[:<group>] sets the user to run the container as
USER $(id -u):$(id -g)

# CMD <command> sets the default command to run when the container starts
CMD ["/app/docker/run.sh"]
