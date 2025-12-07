# Dockerfile for GCDM-SBDD
# Based on official GCDM-SBDD repository with modifications for FOP integration
# IMPORTANT: This requires x86_64 architecture. ARM64/aarch64 is NOT supported.
# If building on ARM64, use: docker build --platform linux/amd64 ...

FROM --platform=linux/amd64 nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    vim \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Mambaforge is deprecated)
RUN wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" -O miniforge.sh && \
    bash miniforge.sh -b -p $CONDA_DIR && \
    rm miniforge.sh && \
    conda clean -afy

# Set working directory
WORKDIR /workspace/GCDM-SBDD

# Clone GCDM-SBDD repository
RUN git clone https://github.com/BioinfoMachineLearning/GCDM-SBDD.git . && \
    rm -rf .git

# Create conda environment
RUN conda env create -f environment.yaml && \
    conda clean -afy

# Activate environment by default
SHELL ["conda", "run", "-n", "GCDM-SBDD", "/bin/bash", "-c"]

# Install local project as package
RUN pip3 install -e .

# Download model checkpoints
RUN wget https://zenodo.org/record/13375913/files/GCDM_SBDD_Checkpoints.tar.gz && \
    tar -xzf GCDM_SBDD_Checkpoints.tar.gz && \
    rm GCDM_SBDD_Checkpoints.tar.gz

# Download and install QuickVina 2
RUN wget https://github.com/QVina/qvina/raw/master/bin/qvina2.1 && \
    chmod +x qvina2.1 && \
    mv qvina2.1 $CONDA_DIR/envs/GCDM-SBDD/bin/

# Create directories for mounting
RUN mkdir -p /workspace/fop-data /workspace/fop-output

# Create Flask API for communication with host
COPY gcdm_api.py /workspace/GCDM-SBDD/
RUN pip3 install flask flask-cors

# Expose port for API
EXPOSE 5000

# Set default command to run the API server
CMD ["conda", "run", "--no-capture-output", "-n", "GCDM-SBDD", "python", "gcdm_api.py"]
