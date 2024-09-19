import os
import torch
import yaml
#from losses.loss_UW import *
from models.model_MT_UW import *
#import torchinfo
#from val.validate_model_UW import val, EarlyStopping
#from vis.plotCM import *
import sklearn
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
cuda_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if cuda_available else 0

print(f"CUDA Available: {cuda_available}")
print(f"Number of GPUs: {gpu_count}")
cuda_version = torch.version.cuda if cuda_available else "CUDA not available"
torch_version = torch.__version__
print("scikit-learn version:", sklearn.__version__)
print("numpy version:", np.__version__)
print("torch version:", torch.__version__)
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"PyTorch Version: {torch_version}")

