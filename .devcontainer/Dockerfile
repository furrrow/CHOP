FROM osrf/ros:humble-desktop

# Basic tools
RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

# Update conda itself (optional but nice)
RUN conda update -n base -c defaults -y conda

# Create your project environment (adjust Python version if you want)
RUN conda create -y -n myproj python=3.10 && \
    conda clean -afy

# Make that env the default
ENV CONDA_DEFAULT_ENV=myproj
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH

# Working directory where you'll mount your repo
WORKDIR /workspace

# Entry point: source ROS, activate Conda env, fix PYTHONPATH for ROS Python
# NOTE: the python3.10 in the path below matches Humble's default Python.
# If you switch ROS distro or base image, you may need to adjust that suffix.
RUN echo '#!/bin/bash'                             >  /ros_conda_entrypoint.sh && \
    echo 'set -e'                                  >> /ros_conda_entrypoint.sh && \
    echo 'source /opt/ros/humble/setup.bash'       >> /ros_conda_entrypoint.sh && \
    echo 'source "'"$CONDA_DIR"'/bin/activate" "'"$CONDA_DEFAULT_ENV"'"' >> /ros_conda_entrypoint.sh && \
    echo 'export PYTHONPATH=/usr/lib/python3/dist-packages:/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH' >> /ros_conda_entrypoint.sh && \
    echo 'exec "$@"'                               >> /ros_conda_entrypoint.sh && \
    chmod +x /ros_conda_entrypoint.sh

ENTRYPOINT ["/ros_conda_entrypoint.sh"]
CMD ["bash"]