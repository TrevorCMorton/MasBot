FROM ubuntu:16.04

MAINTAINER Trevor Morton

# Install some dependencies
RUN apt-get update && apt-get install -y \
		bc \
		build-essential \
		cmake \
		curl \
		g++ \
		git \
		nano \
		pkg-config \
		software-properties-common \
		unzip \
		vim \
		ant \
		python3-pip \
		openjdk-8-jdk \
		doxygen \
		xvfb \
		maven \
		&& \
	apt-get clean && \
	apt-get autoremove && \
	rm -rf /var/lib/apt/lists/*

RUN apt-add-repository ppa:dolphin-emu/ppa -y && apt update && apt install -y dolphin-emu

# Install other useful Python packages using pip
RUN pip3 --no-cache-dir install --upgrade ipython && \
	pip3 --no-cache-dir install \
		Cython \
		numpy \
		Pillow \
		wheel

# Download, build, and install jpy
RUN export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 && \
	export JDK_HOME=/usr/lib/jvm/java-8-openjdk-amd64 && \
	cd /home && \
	git clone https://github.com/bcdev/jpy.git && \
	cd jpy && \
	python3 setup.py --maven bdist_wheel && \
	mvn install:install-file -Dfile=build/lib.linux-x86_64-3.5/jpy-0.10.0-SNAPSHOT.jar -DpomFile=pom.xml

# Download and build MasBot
RUN cd /home && \
	git clone https://github.com/TrevorCMorton/MasBot.git && \
	cd MasBot && \
	git checkout dev && \
	mvn package

# Include the game iso
ADD Melee.iso /home/MasBot

# Start virtual screen buffer
RUN Xvfb :5 -screen 0 1920x1080x24 &

# Export display
RUN export DISPLAY=:5

WORKDIR "/root"
CMD ["/bin/bash"]
