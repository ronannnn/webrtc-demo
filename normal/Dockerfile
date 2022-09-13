FROM python:3.8-slim-buster

WORKDIR /app
RUN apt-get update && apt-get install -y gcc ffmpeg libsm6 libxext6
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install aiohttp aiortc opencv-python
COPY . .
CMD [ "python3", "server.py"]
