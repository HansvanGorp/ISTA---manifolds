# base image
FROM ubuntu:22.04

# Do not prompt for input when installing packages
ARG DEBIAN_FRONTEND=noninteractive

# Prevent python from writing pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/tmp/pip_cache

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    ffmpeg \
    libopencv-dev \
    git \
    && rm -rf /var/lib/apt/lists/*
# Add non-root users
ARG BASE_UID=1000
ARG NUM_USERS=51

# Ensure the sudoers.d directory exists
RUN mkdir -p /etc/sudoers.d/

# Create users in a loop
RUN for i in $(seq 0 $NUM_USERS); do \
        USER_UID=$((BASE_UID + i)); \
        USERNAME="devcontainer$i"; \
        groupadd --gid $USER_UID $USERNAME && \
        useradd --uid $USER_UID --gid $USER_UID -m --shell /bin/bash $USERNAME && \
        echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
        chmod 0440 /etc/sudoers.d/$USERNAME; \
    done

# Add specific user with UID 69407 and GID 68877
RUN groupadd --gid 68877 specificuser && \
    useradd --uid 69407 --gid 68877 -m --shell /bin/bash specificuser && \
    echo "specificuser ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/specificuser && \
    chmod 0440 /etc/sudoers.d/specificuser

# Reset DEBIAN_FRONTEND to its default value
ENV DEBIAN_FRONTEND=

# Add link to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# install dependencies
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.2.2+cu121 torchvision

# set working directory and copy files
WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt