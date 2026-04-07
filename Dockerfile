# 使用NVIDIA官方CUDA 12.1基础镜像
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04

# 修改后的系统依赖安装部分
RUN sed -i 's/archive.ubuntu.com/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list && \
    apt-get update --fix-missing && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
        ca-certificates \
        wget \
        build-essential \
        zlib1g-dev \
        libssl-dev \
        libffi-dev \
        libsqlite3-dev \
        python3-pip \
        libncurses5-dev \
        libgdbm-dev \
        libnss3-dev \
        libreadline-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装Python 3.10
RUN wget https://www.python.org/ftp/python/3.10.0/Python-3.10.0.tgz && \
    tar -xzf Python-3.10.0.tgz && \
    cd Python-3.10.0 && \
    ./configure --enable-optimizations && \
    make -j $(nproc) && \
    make altinstall

# 在安装Python的步骤后添加软链接
RUN ln -s /usr/local/bin/python3.10 /usr/local/bin/python && \
    ln -s /usr/local/bin/pip3.10 /usr/local/bin/pip

# 安装PyTorch 2.4.0 + CUDA 12.1
RUN python3.10 -m pip install \
    torch==2.4.0+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 复制requirements文件并安装其他依赖
COPY requirements.txt .
RUN python3.10 -m pip install --no-cache-dir -r requirements.txt

# 设置工作目录
WORKDIR /app