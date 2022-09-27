FROM python:3.8-slim-buster

WORKDIR /app
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y ffmpeg
RUN apt-get install -y libsm6
RUN apt-get install -y libxext6
RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torchvision --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip3 config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install aiohttp aiortc opencv-python
RUN pip3 install PyYAML tqdm matplotlib
RUN pip3 install pandas requests pillow
RUN pip3 install seaborn psutil
COPY . .
CMD [ "python3", "server.py"]
