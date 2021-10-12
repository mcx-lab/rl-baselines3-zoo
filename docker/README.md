# Docker support for rl-baseline3-zoo DGX

## Build docker image for DGX

This custom image installs:
 
- all dependencies installed via apt-get
- stable-baseline3

Build command:

	docker build -t dgx-sbl3:latest \
		-f docker/Dockerfile-dgx . 

Note: 

* Build from rl-baseline3-zoo dir so that docker can copy this package into the image.

## Make docker container 

Nvidia Gpu:

	docker run -it --privileged --net=host \
         --name=dgx-sbl3 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         dgx-sbl3:latest \
         bash

## Build docker image using nvidia ubuntu 20.04 image

This custom image installs:
 
- all dependencies installed via apt-get
- stable-baseline3
- venv for py 3.8
- terminator
- sublime

Build command:

    docker build -t venv-sbl3:latest \
        -f docker/Dockerfile-sbl3 . 

Note: 

* Build from rl-baseline3-zoo dir so that docker can copy this package into the image.

## Make docker container 

Nvidia Gpu:

    docker run -it --privileged --net=host \
         --name=venv-sbl3 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         venv-sbl3:latest \
         bash


## Running container

Run,

    ./docker/run.sh venv-sbl3

## Test GUI

Run in container,

    glxgears

## Remove container

Run,

	docker rm -f venv-sbl3
