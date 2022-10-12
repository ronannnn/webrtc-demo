FROM python:3.9-slim-buster

WORKDIR /app
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple aiohttp aiortc opencv-python
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple PyYAML tqdm matplotlib
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple pandas requests pillow
RUN pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple seaborn psutil
RUN apt-get install -y libatlas-base-dev python3-scipy
COPY . .
RUN pip3 install torch-1.14.0a0+git1707438-cp39-cp39-linux_aarch64.whl
RUN pip3 install torchvision-0.15.0a0+11a2eed-cp39-cp39-linux_aarch64.whl
CMD [ "python3", "server.py"]
