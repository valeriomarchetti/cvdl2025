# Use CUDA 12.2 as the base image
FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

LABEL authors="Valerio Marchetti"

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    build-essential \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev \
    git \
    unzip \
    wget \
    tmux \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    python3-pip \
    sudo \
    zip \
    tmux \
    nano

# Set Python3 as default
RUN ln -s /usr/bin/python3 /usr/bin/python

# Upgrade pip and install necessary Python packages
RUN python -m pip install --upgrade pip
RUN python -m pip install setuptools packaging

# Install PyTorch and Torchvision compatible with CUDA 12.2
RUN python -m pip install torch torchvision torchaudio

# Install other dependencies
RUN apt-get install -y g++-7 && apt-get install -y libstdc++6

# Install additional Python packages
RUN python -m pip install wandb python-dotenv codecarbon scikit-learn matplotlib opencv-python-headless ttach ultralytics

# Set up the working directory and copy necessary files
WORKDIR /app

# Create necessary directories
RUN mkdir config image output results weights DatasetYOLO
COPY requirements.txt .

# Install Python dependencies from requirements.txt
RUN python -m pip install -r requirements.txt



