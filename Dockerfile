# 使用官方 Python 基础镜像
FROM tensorflow/tensorflow:2.6.0

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
COPY main.py /app/
COPY requirements.txt /app/
RUN mkdir -vp /app/uploads/output && mkdir -p /root/.cache/torch/hub/checkpoints
ADD deeplabv3_resnet50_coco-cd0a2569.pth /root/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth
# 安装需要的依赖包
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple &&  /usr/bin/python3 -m pip install --upgrade pip &&  pip install --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt

# 暴露端口供应用程序使用
EXPOSE 8080
# 在 CMD 行之前添加以下行
ENV PYTHONUNBUFFERED=1
# 运行应用程序
CMD ["python", "main.py"]
