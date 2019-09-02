FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

USER root
# install packages
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends\
    software-properties-common \
    python3-pip \
    gcc \
    python3.5-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

# install numpy
RUN pip3 --no-cache-dir install -U numpy==1.15.0

# additional libraries
RUN pip3 install setuptools
RUN pip3 install wheel

# Install PyTorch
RUN pip3 install torch --no-cache-dir
RUN pip3 install torchvision

# install other libraries
RUN pip3 install opencv-python
RUN pip3 install face_alignment
RUN pip3 install scikit-image
RUN pip3 install scipy
RUN pip3 install psutil
RUN pip3 install pyyaml

# create a link to use python3 as as 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app


