FROM python:3.12 as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN apt-get update
WORKDIR /home/easis

COPY requirements.txt .

COPY . .

RUN chmod +x src/scripts/fastapi.sh