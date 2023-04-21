FROM python:3.9.13-slim

WORKDIR /app

RUN pip3 install --upgrade pip

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY models ./models
COPY utils ./utils
COPY conf ./conf
COPY fast.py .
COPY train.py .
COPY obj_labelencoder.pth .
COPY .env.fast .

EXPOSE 8000