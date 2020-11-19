#!/bin/sh
mkdir -p /home/rajat
cd /home/rajat
sudo apt upgrade -y
# curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
# sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
# sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt update -y
#sudo apt install cuda
sudo pip list -o --format columns|  cut -d' ' -f1|xargs -n1 pip install -U
#nvidia-smi > nvidia_output.txt
git clone https://github.com/grajat90/ResampleGAN
cd ResampleGAN
export model=test
export iters=1000
gsutil cp gs://main-gan-data ./
# sudo cat requirements.txt | xargs -n 1 pip install
sudo nohup python resamplegan.py &