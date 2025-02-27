FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

COPY ../src/ /workspace/src/
COPY ../docker/ /workspace/docker/

RUN pip install --upgrade pip
RUN pip install -r /workspace/docker/requirements.txt

# Needed for triton (which is used by unsloth)
RUN apt-get update
RUN apt-get install -y build-essential gcc
ENV CC=gcc

# Run run.sh
CMD ["bash", "/workspace/docker/run.sh"]
