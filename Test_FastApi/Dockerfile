FROM python:3.9
RUN apt-get update
RUN pip3 install fastapi uvicorn
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader omw-1.4
RUN python3 -m nltk.downloader stopwords
RUN python3 -m nltk.downloader wordnet
RUN python3 -m nltk.downloader punkt
COPY ./app /app
EXPOSE 5400
CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5400" ]
