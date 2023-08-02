# sevir

This is a research product that builds on top of alot of already existing work.

[sevir_challenges](https://github.com/MIT-AI-Accelerator/sevir_challenges)
[eie-sevir](https://github.com/MIT-AI-Accelerator/eie-sevir)
[neurips-2020-sevir](https://github.com/MIT-AI-Accelerator/neurips-2020-sevir)
[WAF_ML_Tutorial_Part1](https://github.com/ai2es/WAF_ML_Tutorial_Part1)
[WAF_ML_Tutorial_Part2](https://github.com/ai2es/WAF_ML_Tutorial_Part2)
[multi earth challenge](https://github.com/MIT-AI-Accelerator/multiearth-challenge)

## Getting Started

Anaconda environment's tend to be a little unreliable for consistently recreating
the required virtual environment. So this repository includes a Dockerfile that
will create a development environment for working with Atmospheric,
Geospatial, and the SEVIR dataset.

### Dockerfile

The Dockerfile can either be served as a standalone which can be used via the
jupyter notebook server or as a development container using the Visual Studio
Code Remote - Containers extension.

#### Usage

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

#### .devcontainer

The .devcontainer directory contains the configuration files for the
[Visual Studio Code Remote - Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension. This allows the user to develop the project in a containerized
environment. The .devcontainer directory also contains the [Dockerfile](#dockerfile) for the project.

### The Hard Way

```bash
sudo apt-get install libgeos-dev proj-bin
pip install cartopy
```
