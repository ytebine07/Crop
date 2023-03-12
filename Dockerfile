FROM python:3.7

WORKDIR /workspaces/
RUN apt-get update -y
RUN apt-get install ffmpeg -y

ENV LANG=ja_JP.UTF-8

ADD . /workspaces/crop
WORKDIR /workspaces/crop

RUN pip install --upgrade pip
RUN pip install -r requirements.txt