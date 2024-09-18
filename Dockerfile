FROM python:3.12

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --upgrade pip
RUN apt-get update
WORKDIR /home/easis

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

RUN chmod +x src/scripts/fastapi.sh