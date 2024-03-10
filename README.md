![image](https://github.com/DXG-Gutson/RoadD/assets/161324365/44f5a386-c888-455c-9021-3eac68c3b781)

# Install CUDA if available
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

# Create the python environment
conda create -n rdd python=3.8
conda activate rdd

# Install pytorch-CUDA
# https://pytorch.org/get-started/locally/
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install ultralytics deep learning framework
# https://docs.ultralytics.com/quickstart/
pip install ultralytics

# Install requirements
pip install -r requirements.txt

# Start the streamlit webserver
streamlit run Home.py
