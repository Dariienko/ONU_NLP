FROM pytorch/pytorch:latest

# Set up locale to prevent bugs with encoding
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y software-properties-common \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get update && apt-get install -y \
        build-essential \
        libfuse-dev \
        libcurl4-openssl-dev \
        libxml2-dev \
        pkg-config \
        libssl-dev \
        mime-support \
        automake \
        libtool  \
        wget \
        tar \
        unzip \
        zip \
        curl \
        libsm6 \
    	libxext6 \
    	libfontconfig1 \
    	libxrender1 \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
    	libturbojpeg \
    	git \
    	ffmpeg \
        libcairo2-dev \
        libpango1.0-dev \
        libjpeg-dev \
        libgif-dev \
        librsvg2-dev \
        libfontconfig-dev \
        gcc \
        g++ \
        cmake \
        make \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Project requirements
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir && rm requirements.txt

CMD mkdir -p /workspace

WORKDIR /workspace
