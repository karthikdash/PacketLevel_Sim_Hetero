Installation Instructions

Note : The directory is pushed from a Python VirtualEnv

git clone reponame

sudo apt-get install python-pip

pip freeze < requirements.txt

Sankhya Cluster Instructions:

Python Version on Sankhya : 2.6.6
Numpy Version to install : 1.6.1

Run these commands on the ssh terminal

wget https://pypi.python.org/packages/dc/6a/5899b7baaa3ebbcc49fb97cdf6b96964d65684864562a1f4ca4cc9f578c8/numpy-1.6.1.tar.gz#md5=2bce18c08fc4fce461656f0f4dd9103e

tar -xvzf <filename> 
cd <directory_name>
python setup.py install --user

