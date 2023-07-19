# sevir

This project is still a work in progress...

## Dockerfile

The Dockerfile conveniently installs all of the required dependencies for the project. It also sets up the environment
for the user to run the project. The Dockerfile is also based on the [Dockerfile Style Guide](#dockerfile-style-guide).

### Usage

Build and run the Dockerfile.

```bash
# build the docker image
docker build -t $USER/sevir .
# Running the The container will launch a jupyter notebook server on port 8888.
export PATH_TO_SEVIR=/mnt/nuc/c # replace with your path to the SEVIR dataset

docker run -it -p 8888:8888 --gpus all \
  --volume $PATH_TO_SEVIR:/home/vscode/sevir-volume \
  --volume $(pwd):/home/vscode/ \
  $USER/sevir
  
# open the jupyter notebook server in your browser at localhost:8888
```

### .devcontainer

The .devcontainer directory contains the configuration files for the
[Visual Studio Code Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension. This allows the user to develop the project in a containerized environment. The .devcontainer directory
also contains the [Dockerfile](#dockerfile) for the project.

### Dockerfile Style Guide

```bash
# ==================================
#         - Style Guide -
# ==================================
# FROM image AS new_image
# USER root
# ARG {...}
#
# [ { WORKDIR SHELL ENV RUN }, ... ]
#
# { MISC }
# USER USER
# ==================================
# "description"
# ==================================
# FROM ...
#
# ==================================
```
