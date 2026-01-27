FROM osrf/ros:humble-desktop

# Add ubuntu user with same UID and GID as your host system, if it doesn't already exist
# Since Ubuntu 24.04, a non-root user is created by default with the name vscode and UID=1000
ARG USERNAME=ubuntu
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN if ! id -u $USER_UID >/dev/null 2>&1; then \
        groupadd --gid $USER_GID $USERNAME && \
        useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME; \
    fi

# Basic tools + numpy for ROS system python
RUN apt-get update && apt-get install -y \
    wget git build-essential \
    && rm -rf /var/lib/apt/lists/*

    # Install Miniconda
ENV CONDA_DIR=/opt/miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda update -n base -c defaults -y conda

# Create your project environment (adjust Python version if you want)
RUN conda create -y -n chop python=3.10 && \
    conda clean -afy

# Make that env the default
ENV CONDA_DEFAULT_ENV=chop
ENV PATH=$CONDA_DIR/envs/$CONDA_DEFAULT_ENV/bin:$PATH

    # Add sudo support for the non-root user
RUN apt-get update && \
    apt-get install -y sudo && \
    echo "$USERNAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Working directory where you'll mount your repo
WORKDIR /workspace

# Entry point: source ROS, activate Conda env, fix PYTHONPATH for ROS Python
# NOTE: the python3.10 in the path below matches Humble's default Python.
# If you switch ROS distro or base image, you may need to adjust that suffix.
RUN echo '#!/bin/bash'                             >  /ros_conda_entrypoint.sh && \
    echo 'set -e'                                  >> /ros_conda_entrypoint.sh && \
    echo 'source /opt/ros/${ROS_DISTRO}/setup.bash'       >> /ros_conda_entrypoint.sh && \
    echo 'source "'"$CONDA_DIR"'/bin/activate" "'"$CONDA_DEFAULT_ENV"'"' >> /ros_conda_entrypoint.sh && \
    echo 'export PYTHONPATH=/opt/ros/${ROS_DISTRO}/lib/python3.10/site-packages:$PYTHONPATH' >> /ros_conda_entrypoint.sh && \
    echo 'exec "$@"'                               >> /ros_conda_entrypoint.sh && \
    chmod +x /ros_conda_entrypoint.sh

ENTRYPOINT ["/ros_conda_entrypoint.sh"]
CMD ["bash"]

# Switch from root to user
USER $USERNAME

# Add user to video group to allow access to webcam
RUN sudo usermod --append --groups video $USERNAME

# Update all packages
RUN sudo apt update && sudo apt upgrade -y

# Install Git
RUN sudo apt install -y git

# Rosdep update
RUN rosdep update

# Source the ROS setup file
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc
