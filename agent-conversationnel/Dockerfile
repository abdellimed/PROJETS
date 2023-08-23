FROM python:3.10

# Set the working directory in the container



# Copy the current directory contents into the container at /app


COPY ./app /app
#COPY ./db /db

#COPY ./lancedb /lancedb
# Install system dependencies


RUN apt-get update && apt-get install -y g++
RUN pip install gradio langchain chromadb gradio_client openai fastapi unstructured lancedb pypdf PyPDF2 elasticsearch tiktoken


#
WORKDIR /app
# Command to run your Gradio app
CMD ["uvicorn", "main:app","--host","0.0.0.0","--port","3200"]
