# Docker support for rl-baseline3-zoo DGX

## Build docker image

This custom image installs:
 
- all dependencies installed via apt-get
- stable-baseline3

Build command:

	docker build -t dgx-sbl3:latest \
		-f docker/Dockerfile-sbl3 . 

Note: 

* Build from rl-baseline3-zoo dir so that docker can copy this package into the image.

## Make docker container 

Nvidia Gpu:

	docker run -it --privileged --net=host \
         --name=sbl3 \
         --env="DISPLAY=$DISPLAY" \
         --env="QT_X11_NO_MITSHM=1" \
         --runtime=nvidia \
         --gpus all \
         dgx-sbl3:latest \
         bash

## Running container

Run,

    ./docker/run.sh sbl3

## Test GUI

Run in container,

    apt install -y xorg
    xclock

## Remove container

Run,

	docker rm -f sbl3
