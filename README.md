# MastersThesis

## Docker Setup
Build image: `hare build -f docker/Dockerfile -t fr411/masters-thesis .` \
Note that this will overwrite any existing image with the same name.

Run (Single GPU): `hare run --rm --gpus device=<gpu index> fr411/<image name> ` \
Runs on a single GPU, cleaning up the container when done.

Run (Multiple GPUs): `hare run --rm --gpus '"device=<gpu index 1>,<gpu index 2>"' fr411/<image name> ` \
Runs on multiple GPUs, cleaning up the container when done.

Remove old image: `hare rmi fr411/<image name>` \
Used if you forget to add `--rm`

The `-v` argument mounts the `$(pwd)$` (print working directory) to the container, this allows you to save files outside
of the container.

The `-d` argument detaches the container, meaning it will run without displaying its output. This is usually for when 
you know your code definitely works.

`--user $(id -u):$(id -g)` ensures files written in the container belong to the account that ran the container via hare
run (meaning you can delete/manipulate files after the container has been cleaned up).

Using `-d` combined with`-v` means you can have the container run in the background, write its results to a file (if you
have written code to do so), and clean up. This way you can just let it run and come back and see the results later.

### Docker Commands to Use
(Omit `-d` if still testing)
- Single GPU: `hare run --rm --gpus device=6 --user $(id -u):$(id -g) -v $(pwd)/out:/app/out fr411/masters-thesis -d`
- Multiple GPUs: `hare run --rm --gpus '"device=6,7"' --user $(id -u):$(id -g) -v $(pwd)/out:/app/out fr411/masters-thesis -d`