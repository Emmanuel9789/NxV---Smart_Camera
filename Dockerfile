FROM python:3.11
WORKDIR /app
RUN pip install --no-cache-dir flask==3.0.3 requests==2.32.3 numpy==1.26.4
COPY neighborhood_server.py .
EXPOSE 6000
CMD ["python", "-u", "neighborhood_server.py"]
