#!/bin/sh
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