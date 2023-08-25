FROM python:3.9-slim-buster
# Install python packages
#ENV http_proxy http://DMZ-PXSERVER.APPS.PARIS.MDP:8080/
#ENV https_proxy http://DMZ-PXSERVER.APPS.PARIS.MDP:8080/
#ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION python
#COPY ca-bundle.trust.crt /usr/local/share/ca-certificates/ca-bundle.trust.crt
#RUN update-ca-certificates
RUN apt-get update
RUN apt-get install -y postgresql postgresql-contrib postgresql-server-dev-all python-psycopg2
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN pip install boto3 pymysql psycopg2-binary scikit-learn
#ENV  MLFLOW_S3_ENDPOINT_URL  http://s3-artifact-storage:9000/
#CMD mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres-mlflow/mlflow --artifacts-destination s3://data -h 0.0.0.0 -p 5000

