# Run "docker build -f pflow.dockerfile --network=host --tag pflow ." to build the image

# Build from nvidia/cuda iamge
FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]

# Change mirrors for apt-get to aliyun 
RUN sed -i s@/archive.ubuntu.com/@/mirrors.aliyun.com/@g /etc/apt/sources.list

# Install essentials
RUN DEBIAN_FRONTEND=noninteractive apt-get update\
    && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common \
    && DEBIAN_FRONTEND=noninteractive add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && DEBIAN_FRONTEND=noninteractive apt-get update \
    && echo -e "6\n1\n" | DEBIAN_FRONTEND=noninteractive apt-get install -y wget gcc g++ cmake vim libstdc++6 python3-tk

# Install Miniconda package manager.
RUN wget -O /root/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
#ADD Miniconda3-latest-Linux-x86_64.sh /root/miniconda.sh
RUN bash /root/miniconda.sh -b -p /opt/conda
RUN rm /root/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda init bash && source /root/.bashrc \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/  \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/  \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/  \
    && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge  \
    && pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    && pip install --upgrade "jax[cuda11_pip]" jaxlib -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# install other necessary packages
RUN conda install mdtraj openmm cudnn -c conda-forge

# install other python packages
RUN conda init bash && source /root/.bashrc && pip install optax dm-haiku

SHELL ["/bin/bash", "-c"]