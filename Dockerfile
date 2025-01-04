FROM research-xavier:l4t-r35.5.0

ARG USERNAME=sauce
ARG PASS=password

# create a non-root user
RUN useradd -m ${USERNAME} && \
    echo "${USERNAME}:${PASS}" | chpasswd && \
    usermod --shell /bin/bash ${USERNAME} && \
    usermod -aG sudo ${USERNAME}

# cleanup
RUN apt update
RUN apt install -y x11-apps

# set user
USER ${USERNAME}

# gui forwarding
RUN ln -s /dot.Xauthority /home/${USERNAME}/.Xauthority

# install dependencies
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install diffusers einops efficientnet_pytorch onnx
RUN pip install hydra-core omegaconf dill pycuda
RUN pip install numpy<=1.23

# clone the diffusion policy repo and install it
WORKDIR /home/${USERNAME}
RUN git config --global http.postBuffer 1048576000
RUN git clone https://github.com/real-stanford/diffusion_policy
RUN pip install -e ./diffusion_policy

# go back to the workspace folder
WORKDIR /home/${USERNAME}/workspace
