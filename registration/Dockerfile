FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
RUN apt-get update

# install zsh
RUN apt-get install -y zsh
RUN chsh -s /usr/bin/zsh root

# install wget, git, unzip and editing software
RUN apt-get install -y wget git less vim unzip

# install registration packages
RUN pip install kornia
RUN conda install -y opencv

# install and configure oh-my-zsh
RUN wget https://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

# install torch ops for SampleNet registration
RUN cd /workspace
RUN git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
RUN pip install -r /workspace/Pointnet2_PyTorch/requirements.txt
RUN pip install -e /workspace/Pointnet2_PyTorch
RUN pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# environment encoding variables
# https://stackoverflow.com/questions/55646024/writing-accented-characters-from-user-input-to-a-text-file-python-3-7
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

CMD ["/usr/bin/zsh"]
