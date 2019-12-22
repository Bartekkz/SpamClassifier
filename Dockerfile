FROM python:3.6-alpine

EXPOSE 4000
WORKDIR /SpamClassifier
COPY requirements.txt /SpamClassifier
RUN apk add --no-cache --virtual .build-deps g++ musl-dev
RUN pip install -r requirements.txt
COPY . /SpamClassifier/

CMD ["python", "app.py"]
