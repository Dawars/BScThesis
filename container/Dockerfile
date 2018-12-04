FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive

RUN apt update && apt install -y git libopencv-dev python-opencv python3 python-pip python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install pip==9.0 && \
    pip3 install --no-cache-dir  numpy scipy h5py pandas scikit-image matplotlib opencv-python jupyter \
    http://download.pytorch.org/whl/cu92/torch-0.4.1-cp36-cp36m-linux_x86_64.whl torchvision tensorboardX \
    tensorflow-gpu deepdish moderngl && \
    pip3 uninstall -y Pillow && pip3 install pillow-simd
