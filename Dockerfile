
# syntax = docker/dockerfile:1.2
FROM python:3.10

COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/var/cache/pip pip install --requirement /tmp/requirements.txt

WORKDIR /app
# Copy all the files of this project inside the container
COPY . .

CMD ["streamlit", "run", "streamlit-api.py","--server.port", "3838"]