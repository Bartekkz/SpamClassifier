FROM python:3.6

EXPOSE 4000
WORKDIR /SpamClassifier
COPY requirements.txt /SpamClassifier
RUN apt-get update
RUN apt-get -y install libc-dev
RUN apt-get -y install build-essential
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN [ "python", "-c", "import nltk; nltk.download('stopwords')" ]
COPY . /SpamClassifier/

CMD ["python", "app.py"]
