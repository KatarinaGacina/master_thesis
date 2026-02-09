FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC

RUN apt-get update && apt-get install -y --fix-missing \
    python3 \
    python3-pip \
    python3-dev \
    git \
    curl \
    wget \
    python-is-python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/* 

RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    bash Miniforge3.sh -b -p /opt/conda

RUN pip3 install --upgrade pip \
 && pip3 install \
    torch \
    torchvision \
    torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu122
RUN pip install \
    packaging
RUN pip install --no-input \
    transformers \
    datasets \
    wandb \
    flash-attn \
    pandas \
    torchmetrics \
    pytorch-lightning \
    fft-conv-pytorch \
    pyfaidx \
    pyBigWig \
    biopython \
    umap-learn \
    matplotlib

ENV HOME=/path