# This Makefile automates routine tasks for this Singularity-based project.
REPO_NAME := $(shell basename `git rev-parse --show-toplevel` | tr '[:upper:]' '[:lower:]')
IMAGE := container.sif
SANDBOX := sandbox.sif
RUN ?= singularity exec -B $(DVC_CACHE_DIR) $(IMAGE)
DVC_CACHE_DIR ?= $(shell dvc cache dir)
SINGULARITY_ARGS ?=
FLAGS ?=
VENV_PATH ?= venv

.PHONY: report clean jupyter container shell

include cluster/makefile

repro: FLAGS= -B $(DVC_CACHE_DIR)
repro:
	$(RUN) dvc repro

start_lab:
	mila serve lab --alloc -c 16 --mem=32G -t 4:00:00

start_local:
	salloc --gres=gpu:1 -c 16 --mem=32G -t 6:00:00

loader:
	$(RUN) python3 code/loader.py

clean:
	rm -f report/*.blg report/*.fls report/*.out report/*.log report/*.fdb_latexmk report/*.aux report/*.pdf report/*.bbl report/*.toc

jupyter: $(SANDBOX)
	sudo singularity exec \
    -B $$(pwd):/mnt --pwd /mnt \
    $(SANDBOX) jupyter lab \
		--allow-root \
		--ip=0.0.0.0 \
		--no-browser \
		--port 8888

container: $(IMAGE)
$(IMAGE): Singularity requirements.txt
	sudo singularity build $(IMAGE) $(SINGULARITY_ARGS) Singularity

sandbox: $(SANDBOX)
$(SANDBOX): Singularity requirements.txt
	sudo singularity build --sandbox $@ Singularity

shell:
	singularity shell --nv $(FLAGS) $(IMAGE) $(SINGULARITY_ARGS) bash

sandbox-shell: $(SANDBOX)
	sudo singularity shell --bind $$(pwd):/mnt --pwd /mnt --writable $(SANDBOX)
