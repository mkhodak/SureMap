FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Setup Ubuntu
RUN apt-get update --yes
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get install -y tzdata
RUN apt-get install -y make cmake build-essential autoconf libtool rsync ca-certificates git grep sed dpkg curl wget bzip2 unzip llvm libssl-dev libreadline-dev libncurses5-dev libncursesw5-dev libbz2-dev libsqlite3-dev zlib1g-dev mpich htop vim 

# Get Miniconda and make it the main Python interpreter
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
RUN bash ~/miniconda.sh -b -p /opt/conda
RUN rm ~/miniconda.sh
ENV PATH /opt/conda/bin:$PATH
RUN conda create -n pytorch_env python=3.11
RUN echo "source activate pytorch_env" > ~/.bashrc
ENV PATH /opt/conda/envs/pytorch_env/bin:$PATH
ENV CONDA_DEFAULT_ENV pytorch_env
RUN conda install cuda -c nvidia/label/cuda-12.2.2
RUN conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
RUN pip install numpy scipy matplotlib numba pandas scikit-learn tqdm
RUN pip install fairlearn 
RUN conda install -c conda-forge mpi4py mpich
RUN conda install -c huggingface pyarrow transformers
RUN pip install librosa faster-whisper jiwer sentence-transformers
RUN pip install langchain langchain_community 
RUN python -m pip install llama-cpp-python --prefer-binary --extra-index-url=https://jllllll.github.io/llama-cpp-python-cuBLAS-wheels/AVX512/cu122 --no-cache-dir
