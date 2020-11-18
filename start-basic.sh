#!/bin/sh
sudo apt upgrade -y
sudo apt update -y
sudo apt get install -y git
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev -y
curl -O https://www.python.org/ftp/python/3.8.2/Python-3.8.2.tar.xz
tar -xf Python-3.8.2.tar.xz
cd Python-3.8.2
./configure --enable-optimizations
make -j 4
sudo make altinstall
cd ..
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt update
sudo apt install cuda
pip list -o --format columns|  cut -d' ' -f1|xargs -n1 pip install -U
nvidia-smi > nvidia_output.txt
git clone https://github.com/grajat90/ResampleGAN
cd ResampleGAN
export model=dense
export iters=1000
pip install -r ./requirements.txt
python resampleGAN.py