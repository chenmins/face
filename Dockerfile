# 使用官方 Python 基础镜像
FROM python:3.8-slim

# 安装需要的依赖包和库
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /app
# 将当前目录的内容复制到工作目录
COPY . /app
# 安装需要的依赖包
RUN pip install --trusted-host pypi.python.org -r requirements.txt

RUN mkdir -vp /app/uploads/output

# Install wget and download model file
RUN apt-get update \
    && apt-get install -y wget \
    && mkdir -p /root/.cache/torch/hub/checkpoints \
    && wget -O /root/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth "https://download.pytorch.org/models/deeplabv3_resnet50_coco-cd0a2569.pth" \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \

# 暴露端口供应用程序使用
EXPOSE 8080
# 在 CMD 行之前添加以下行
ENV PYTHONUNBUFFERED=1
# 运行应用程序
CMD ["python", "main.py"]
