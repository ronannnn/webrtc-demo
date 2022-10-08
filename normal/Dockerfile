FROM python:3.8-slim-buster

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
RUN pip3 install torch torchvision torchaudio
COPY . .
CMD [ "python3", "server.py"]
