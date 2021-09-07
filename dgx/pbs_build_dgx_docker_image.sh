#!/bin/sh
#PBS -l select=1:ncpus=12:ngpus=1:host=dgx02
#PBS -N rl-baselines3-zoo_image
#PBS -m abe
#PBS -M michael_chuah@i2r.a-star.edu.sg

sh /home/i2r/chuahmy/DATA/rl-baselines3-zoo/dgx/build_dgx_docker_image.sh