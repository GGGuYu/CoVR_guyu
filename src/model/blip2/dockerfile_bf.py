# 选择 CUDA 11.3 版本的官方 NVIDIA 镜像
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# 设置环境变量，避免交互式安装问题
ENV DEBIAN_FRONTEND=noninteractive


# 安装 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
    bash /miniconda.sh -b -p /opt/conda && rm /miniconda.sh

# 设置 conda 相关环境变量
ENV PATH="/opt/conda/bin:$PATH"

# 使用 bash 作为默认 shell
SHELL ["/bin/bash", "-c"]

# 复制你的已打包 conda 环境
COPY lxy_SOC.tar.gz /app/

# 解压到 conda 的环境路径
WORKDIR /app
RUN mkdir -p /opt/conda/envs/lxy_SOC && \
    tar -xzf lxy_SOC.tar.gz -C /opt/conda/envs/lxy_SOC

# 让 Docker 识别这个 conda 环境
ENV PATH="/opt/conda/envs/lxy_SOC/bin:$PATH"

# 确保 conda 被正确初始化
RUN echo "source activate lxy_SOC" >> ~/.bashrc

# 默认进入 bash，方便调试
CMD ["/bin/bash"]
