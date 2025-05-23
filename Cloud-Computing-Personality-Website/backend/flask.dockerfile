FROM python:3.6-slim-buster

# Set the working directory inside the container to /app
WORKDIR /app

# Install system dependencies for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    wget

# Copying only the requirements file to leverage Docker cache
COPY ./requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the transformers models to cache them
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification; \
               AutoTokenizer.from_pretrained('JanSt/albert-base-v2_mbti-classification'); \
               AutoModelForSequenceClassification.from_pretrained('JanSt/albert-base-v2_mbti-classification'); \
               BertTokenizer.from_pretrained('Minej/bert-base-personality'); \
               BertForSequenceClassification.from_pretrained('Minej/bert-base-personality')"

# Copying the rest of the application code
COPY . .

EXPOSE 4000

# For Development
# CMD ["flask", "run", "--host=0.0.0.0", "--port=4000"]

# For Production: Flask’s built-in server is not suitable for production. Use a WSGI server like gunicorn.
# CMD ["gunicorn", "-w", "4", "-k", "gevent", "-b", "0.0.0.0:4000", "--timeout", "600", "app:app"]
CMD ["gunicorn", "-w", "1", "-k", "sync", "-b", "0.0.0.0:4000", "--timeout", "600", "app:app"]
# CMD ["gunicorn", "-w", "1", "-k", "sync", "-b", "0.0.0.0:4000", "app:app"]
