#!/usr/bin/env bash


#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainer.py

CUDA_VISIBLE_DEVICES=0 docker-compose -f ../container/docker-compose.yml run hmr python3 /projects/pytorch_HMR/src/trainer.py
