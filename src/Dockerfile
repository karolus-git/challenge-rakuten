FROM python:3.9.13-slim

WORKDIR /app

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY models .
COPY utils .
COPY runs .
COPY fast.py .
#COPY train.py .
COPY *.pth .
COPY .env.fast .
COPY __init__.py .

EXPOSE 8080