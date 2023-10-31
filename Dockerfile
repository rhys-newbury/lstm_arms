FROM nvcr.io/nvidia/pytorch:23.08-py3

RUN useradd -rm -d /home/worker -s /bin/bash -g root -G sudo -u 1000 worker
WORKDIR /home/worker

RUN pip install konductor wandb typer

COPY --chown=worker . smac_transformer

USER worker
