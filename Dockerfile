FROM stencila/executa-midi

# Based on https://github.com/stencila/dockta#extending-the-images
# Install things with root
USER root

# Download miniconda https://stackoverflow.com/questions/58269375/how-to-install-packages-with-miniconda-in-dockerfile
ENV PATH=/root/miniconda3/bin:$PATH
ARG PATH=/root/miniconda3/bin:$PATH
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

# Add the conda environment to the container
ENV CONDA_ENV bclab
COPY bclab.yml /tmp/$CONDA_ENV.yml
RUN conda env create -q -f /tmp/bclab.yml -n $CONDA_ENV

# Download gkmSVM source code and compile
RUN wget \
	http://www.beerlab.org/gkmsvm/downloads/gkmsvm-2.0.tar.gz \
	&& tar -zxvf gkmsvm-2.0.tar.gz \
	&& rm gkmsvm-2.0.tar.gz \
	&& cd gkmsvm \
	&& make \
	&& mkdir ../bin \
	&& mv gkmsvm_kernel gkmsvm_train gkmsvm_classify ../bin/ \
	&& cd ../ \
	&& rm -r gkmsvm

SHELL ["/bin/bash", "-c"]
RUN conda init
RUN echo 'conda activate bclab' >> ~/.bashrc

# Go back to guest to run the container
USER guest
