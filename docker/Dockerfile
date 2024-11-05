FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# COPY <src> <dest> copies files into the image, can do entire directories
COPY ../src/ /app/src/
COPY ../docker/ /app/docker/

# RUN to install packages/modifications to the base image
RUN pip install -r /app/docker/requirements.txt

# Make the script executable
RUN chmod +x /app/docker/run.sh

# ENV key=value sets environment variables
    # e.g. ENV PORT=8080

# USER <username>[:<group>] sets the user to run the container as
USER $(id -u):$(id -g)

# CMD <command> sets the default command to run when the container starts
CMD ["/app/docker/run.sh"]
