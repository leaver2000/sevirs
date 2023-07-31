# syntax=docker/dockerfile:1

# Description:
# Multistage Dockerfile for working with the SEVIR dataset and other atmospheric
# data formats for analysis, processing, and Machine Learning.

# Example:
# - docker build -t $USER/sevir . 
# - docker run -it -p 8888:8888 --gpus all --volume $PATH_TO_SEVIR:/home/vscode $USER/sevir

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base
USER root

WORKDIR /
SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libgeos3.10.2 \
    libgdal30 \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
USER 1001

# =====================================================================================================================
# Compiler Stage; this will be omitted from the final image
# =====================================================================================================================
FROM base AS compiler
USER root

WORKDIR /
SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive
# hadolint ignore=DL3008
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    wget \
    gcc   \
    g++    \
    cmake   \
    gfortran \
    python3.10-dev \
    build-essential \
    # see: https://github.com/OSGeo/PROJ/blob/master/Dockerfile
    zlib1g-dev libsqlite3-dev sqlite3 libcurl4-gnutls-dev libtiff5-dev libsqlite3-0 libtiff5 \
    libgdal-dev libatlas-base-dev libhdf5-serial-dev \
    && rm -rf /var/lib/apt/lists/*

USER 1001

# =====================================================================================================================
# EcCodes is a library for decoding and encoding grib files.
# =====================================================================================================================
FROM compiler AS eccodes
USER root
ARG ECCODES="eccodes-2.24.2-Source" 
ARG ECCODES_DIR="/usr/include/eccodes"

WORKDIR /tmp
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN wget -c --progress=dot:giga \
    https://confluence.ecmwf.int/download/attachments/45757960/${ECCODES}.tar.gz  -O - | tar -xz -C . --strip-component=1 

WORKDIR /tmp/build
SHELL ["/bin/bash","-c"]
RUN cmake -DCMAKE_INSTALL_PREFIX="${ECCODES_DIR}" -DENABLE_PNG=ON .. \
    && make -j"$(nproc)" \
    && make install

USER 1001

# =====================================================================================================================
# PROJ is a library for coordinate transformations, and is a requirement for cartopy.
# =====================================================================================================================
FROM compiler AS proj
USER root

WORKDIR /proj
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN wget -c --progress=dot:giga \
    https://github.com/OSGeo/PROJ/archive/refs/tags/9.0.1.tar.gz  -O - | tar -xz -C . --strip-component=1 

WORKDIR /proj/build
SHELL ["/bin/bash","-c"]
RUN cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
    && make -j"$(nproc)" \
    && make install

USER 1001

# =====================================================================================================================
# Cartopy is a python library for plotting data on maps.
# =====================================================================================================================
FROM proj AS cartopy
USER root

SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive
# hadolint ignore=DL3008
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3.10-dev \
    && rm -rf /var/lib/apt/lists/*
# hadolint ignore=DL3013
RUN python3.10 -m pip install --upgrade pip --no-cache-dir && python3.10 -m pip install --no-cache-dir \
    Cartopy==0.21.1 \
    matplotlib==3.7.2

USER 1001

# =====================================================================================================================
# Final Image
# =====================================================================================================================
FROM base AS lunch-box
USER root
ARG USERNAME=vscode 
ARG USER_UID=1000
ARG USER_GID=$USER_UID

WORKDIR /tmp/sevir
SHELL ["/bin/bash","-c"]
# hadolint ignore=DL3008
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && chown -R $USER_UID:$USER_GID /opt/venv

ENV ECCODES_DIR=/usr/include/eccodes
COPY --from=eccodes --chown=$USER_UID:$USER_GID $ECCODES_DIR $ECCODES_DIR
COPY --from=cartopy --chown=$USER_UID:$USER_GID /opt/venv /opt/venv
# - install requirements that arn't in the requirements.txt and torch for caching
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.0.1 \
    cfgrib==0.9.10.4 \
    notebook==6.5.4 \
    eccodes==1.6.0 
# - install the package and dependencies
COPY src/ src/
COPY setup.py setup.py
COPY pyproject.toml pyproject.toml
RUN python3.10 -m pip install . --no-cache-dir && rm -rf /tmp/*

USER $USERNAME
ARG HOME=/home/$USERNAME
WORKDIR $HOME
COPY --chown=$USER_UID:$USER_GID notebooks/ examples/
VOLUME $HOME/sevir-volume
ENV PATH_TO_SEVIR=$HOME/sevir-volume/sevir

CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser" ]
