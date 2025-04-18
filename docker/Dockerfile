FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

COPY ../src/ /workspace/src/
COPY ../docker/ /workspace/docker/

RUN pip install --upgrade pip
RUN pip install -r /workspace/docker/requirements.txt

# Needed for triton (which is used by unsloth)
RUN apt-get update
RUN apt-get install -y build-essential gcc
RUN apt-get install -y --no-install-recommends git
ENV CC=gcc

# Install lm-evaluation-harness from GitHub (needs to pre-exist in /src/)
WORKDIR /workspace/src/lm-evaluation-harness
RUN pip install -e ".[dev]"
WORKDIR /

# Run run.sh
CMD ["bash", "/workspace/docker/run_qwen_base_nocot_haerae.sh"]
