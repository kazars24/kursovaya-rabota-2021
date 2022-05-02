FROM ubuntu:20.04

MAINTAINER kazars24

RUN apt-get update && \
    apt-get install -y python3 && \
    apt install git && \
    apt-get install libssl-dev libsqlite3-dev libavcodec-dev libswscale-dev libavformat-dev libavdevice-dev ffmpeg libx264-dev libmp3lame-dev && \
    git clone git://source.ffmpeg.org/ffmpeg.git ffmpeg && \
    cd ffmpeg && \
    apt -y install build-essential && \
    apt install nasm && \
    apt install yasm && \
    apt install cmake && \
    ./configure --enable-gpl --enable-libx264 --enable-libmp3lame && \
    make && \
    cd .. && \
    git clone --recursive https://github.com/kazars24/kursovaya-rabota-2021.git kursovaya-rabota-2021 && \
    cd kursovaya-rabota-2021 && \
    git clone --recursive https://github.com/kazars24/h264-roi.git h264-roi && \
    mkdir h264-roi-build && \
    cd h264-roi-build && \
    cmake ../h264-roi && \
    apt-get install zlib1g-dev && \
    make && \
    cd .. && \
    apt -y install python3-pip && \
    pip install numpy && \
    pip install opencv-python --upgrade && \
    pip install ffmpeg-python
