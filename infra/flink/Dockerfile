FROM flink:1.19.1-scala_2.12-java17

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /opt/flink/requirements.txt
RUN if [ -s /opt/flink/requirements.txt ]; then pip3 install --no-cache-dir -r /opt/flink/requirements.txt; fi

USER flink
