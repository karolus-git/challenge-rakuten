FROM python:3.9.13-slim

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY app.py ./app.py

EXPOSE 8081
