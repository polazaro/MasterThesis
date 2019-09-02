FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

USER root
# install packages
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends\
    software-properties-common \
    python3-pip \
    gcc \
    python3.5-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# install numpy
RUN pip3 --no-cache-dir install -U numpy==1.14.5

# additional libraries
RUN pip3 install setuptools
RUN pip3 install wheel

# install tensorflow
RUN pip3 install tensorflow-gpu==1.12.0

# install other libraries
RUN pip3 install keras==2.2.0
RUN pip3 install keras_vggface --no-dependencies
RUN pip3 install pandas
RUN pip3 install scipy
RUN pip3 install sklearn
RUN pip3 install psutil
RUN pip3 install pyyaml
RUN pip3 install pillow

# create a link to use python3 as as 'python'
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app


