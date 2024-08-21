import torch
from sklearn.metrics import average_precision_score
from scipy.stats import spearmanr
import math
from torch.autograd import Variable
from util.Uwdatareader_UW import label2index, readtxt
from util.utils import mask_data, unmask
from vis.plotCM import compute_confusion_matrix
from termcolor import colored
import yaml

cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else "CUDA not available"
torch_version = torch.__version__

print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"PyTorch Version: {torch_version}")

print(round(5.66))