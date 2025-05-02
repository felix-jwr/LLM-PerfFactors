# [Evaluating the Effect of Model Scale, Prompting Strategies, and Distillation on LLM Performance](https://drive.google.com/file/d/1EPdR1478azsNA09ZItoTcITfVrhd0T2I/view?usp=sharing)

## Info

This project evaluates the individual and combined effects of model scale, distillation, zero- and few-shot prompting, 
and chain of thought (CoT) prompting on model performance.

We test on GSM8k and HAERAE, with an exploratory evaluation on GSM Symbolic, however our code aims to be generally
applicable and can be used with any of the tests available on the Language Model Evaluation Harness (with minimal 
modification), or any dataset on HuggingFace (may require additional modification).

A PDF of the full study is included above, with a guide on setup to replicate the results included below.
Whilst we use Docker to run our evaluations, this is not strictly a requirement, as the scripts can just as easily be
run outside of a docker container.
A link to download a .zip containing our logs and results is included at the end of this README.

## Guide
Below is a guide on running the evaluations in the paper yourself.

### Installing LM Eval Harness
To run HAERAE, you'll need the LM evaluation harness and custom tasks defined so that the
models can use CoT on tasks which don't have it implemented.

To do this, you'll need to follow these steps:
1. `cd` into the `src` folder
2. `git clone https://github.com/felix-jwr/lm-evaluation-harness.git`

That's it! The Dockerfile should now be able to run `pip install` on the LM evaluation harness, which is needed for the
HAERAE dataset tests via the `test_lm_eval.py` file.

### Docker Setup
Build image: `hare build -f docker/Dockerfile -t fr411/masters-thesis:latest .` \
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

#### Docker Commands to Use
(Omit `-d` if still testing)
- Single GPU: `hare run --rm --gpus device=6 -v $(pwd):/workspace fr411/masters-thesis`
- Multiple GPUs: `hare run --rm --gpus '"device=6,7"' -v $(pwd):/workspace fr411/masters-thesis`

##### Running Locally on Windows
`docker run -it --rm --gpus all -v "${PWD}:/workspace/" fr411/masters-thesis`

### Model Setup
Some of the models we tested require you to request access, meaning to replicate results for these models you will need to:
1. Create a HuggingFace account
2. Request access to the appropriate model(s)
3. Set up an API token in a text file in `src/` to load the models via the scripts

## Downloading our Results
Click [here](https://drive.google.com/file/d/1dIeyDmnf_PIUm7SpWz8XmofnUtfpoGx_/view?usp=sharing) to download a .zip of our results from Google Drive.
