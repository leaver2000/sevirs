# syntax=docker/dockerfile:1

# Description:
# MultiStage Dockerfile for working with the SEVIR and other atmospheric
# data formats for analysis, processing, and Machine Learning.

# Example:
# - docker build -t $USER/sevir . 
# - docker run -it -p 8888:8888 --gpus all --volume $PATH_TO_SEVIR:/home/vscode $USER/sevir

FROM  nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base
USER root

WORKDIR /
SHELL ["/bin/bash","-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    python3.10=3.10.6* \
    python3.10-venv=3.10.6* \
    libgeos3.10.2=3.10.2* \
    libgdal30=3.4.1* \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"
USER 1001
# =====================================================================================================================
# create a compiler stage; this will be omitted from the final image
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
# eccodes is a library for decoding and encoding grib files.
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
# SHELL ["/bin/bash","-c"]
# ENV DEBIAN_FRONTEND=noninteractive
# hadolint ignore=DL3008
# RUN apt-get update -y \
#     && apt-get install -y --no-install-recommends \
#     zlib1g-dev \
#     libsqlite3-dev sqlite3 libcurl4-gnutls-dev libtiff5-dev \
#     && rm -rf /var/lib/apt/lists/*

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
RUN python3.10 -m pip install --upgrade pip && python3.10 -m pip install --no-cache-dir \
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

WORKDIR /
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    # clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY --from=cartopy --chown=$USER_UID:$USER_GID /opt/venv /opt/venv
RUN chown -R $USER_UID:$USER_GID /opt/venv

ENV ECCODES_DIR=/usr/include/eccodes
COPY --from=eccodes --chown=$USER_UID:$USER_GID $ECCODES_DIR $ECCODES_DIR

WORKDIR /tmp/sevir
SHELL ["/bin/bash","-c"]
# - install requirements that arnt in the requirements.txt and torch for caching
RUN python3.10 -m pip install --no-cache-dir \
    torch==2.0.1 \
    cfgrib==0.9.10.4 \
    notebook==6.5.4 \
    eccodes==1.6.0 \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*
# - install the requirements.txt
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*
# - install the package
COPY src/ src/
COPY setup.py .
COPY pyproject.toml .
RUN python3.10 -m pip install . --no-cache-dir \
    && rm -rf /tmp/* \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME
ARG HOME=/home/$USERNAME
WORKDIR $HOME
COPY --chown=$USER_UID:$USER_GID notebooks/ examples/
VOLUME $HOME/sevir-volume
ENV PATH_TO_SEVIR=$HOME/sevir-volume/sevir

CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--NotebookApp.token=''", "--NotebookApp.password=''", "--no-browser" ]
