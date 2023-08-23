FROM python:3.9

RUN pip3 install fastapi uvicorn
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip

RUN pip3 install -r requirements.txt

COPY ./app /app
COPY ./models /models

WORKDIR /app
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "15400"]
