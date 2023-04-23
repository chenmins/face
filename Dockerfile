# 使用官方 Python 基础镜像
FROM python:3.8-slim

# 设置工作目录
WORKDIR /app

# 将当前目录的内容复制到工作目录
COPY . /app

# 安装需要的依赖包
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# 暴露端口供应用程序使用
EXPOSE 8080

# 定义环境变量
ENV FLASK_APP=app.py

# 运行应用程序
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
