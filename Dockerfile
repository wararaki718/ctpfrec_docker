FROM python:3.8

RUN apt-get update && \
    apt-get install gcc g++ libomp-dev -y && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt
RUN pip install --no-build-isolation ctpfrec

WORKDIR /work
COPY sample.py /work/sample.py

CMD ["python", "sample.py"]
