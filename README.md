# MastersThesis

## Docker Setup
Build image: `hare build -f docker/Dockerfile -t fr411/masters-thesis .`

Run (Single GPU): `hare run --rm --gpus device=<gpu index> fr411/<image name> `

Run (Multiple GPUs): `hare run --rm --gpus '"device=<gpu index 1>,<gpu index 2>"' fr411/<image name> `

Remove old image: `hare rmi fr411/<image name>`