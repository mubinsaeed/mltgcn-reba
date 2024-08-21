import os
import torch
import yaml
#from losses.loss_UW import *
from models.model_MT_UW import *
import torchinfo
#from val.validate_model_UW import val, EarlyStopping
#from vis.plotCM import *
import sklearn
import numpy as np
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else "CUDA not available"
torch_version = torch.__version__
print("scikit-learn version:", sklearn.__version__)
print("numpy version:", np.__version__)
print("torch version:", torch.__version__)
print(f"CUDA Available: {cuda_available}")
print(f"CUDA Version: {cuda_version}")
print(f"PyTorch Version: {torch_version}")

matrixx = np.full((27,27),np.inf)
temp = [[0,25],[0,26],[26,2],[2,4],[4,18],[4,19],[25,1],[1,3],[3,17],
        [3,16],[0,5],[5,6],[6,8],[8,10],[10,12],[10,13],[5,7],[7,9],[9,11],
        [11,14],[11,15],[5,22],[22,20],[20,21],[22,23],[22,24]]

np.fill_diagonal(matrixx, 0)

for pair in temp:
    x,y = pair
    matrixx[x][y] = 1
    matrixx[y][x] = 1
print(matrixx)


def check_model_layer():
    print(12)
    model1 = gcn_reg(hidden=[50, 50, 50, 50], kernel_size=4)
    model1.cuda()
    check_array = np.array([[0.85064882, -1.0686636, 0.78543806],
                            [0.9946214, -0.40855142, 0.88562602],
                            [0.82833332, -0.39266405, 0.62361616],
                            [0.89394653, 0.04429765, 0.95299017],
                            [0.73597699, 0.06682187, 0.69038391],
                            [0.84028345, -1.56258941, 0.79421932],
                            [0.95531029, -1.46166134, 0.98172194],
                            [0.74930274, -1.43675029, 0.59225464],
                            [1.11474478, -1.2706238, 1.15812254],
                            [0.75745797, -1.28975785, 0.33339268],
                            [1.37699902, -1.39355254, 1.1515311],
                            [1.01637578, -1.31189144, 0.19765389],
                            [1.44716234, -1.53355901, 1.12682188],
                            [1.47701709, -1.47997717, 1.2004547],
                            [1.13069329, -1.34855144, 0.19789068],
                            [1.08311471, -1.41702834, 0.14564548],
                            [1.04309387, 0.07795297, 0.88886918],
                            [1.03402592, 0.06808891, 0.99803576],
                            [0.8920035, 0.08850542, 0.63093374],
                            [0.82707825, 0.07983393, 0.54826663],
                            [0.80525332, -1.92936971, 0.78238705],
                            [0.74831362, -1.79732905, 0.86559432],
                            [0.90683132, -1.83038011, 0.70011622],
                            [0.90139287, -1.77427994, 0.84770373],
                            [0.77566784, -1.74595882, 0.72017185],
                            [0.93836027, -0.81843102, 0.86104077],
                            [0.82237941, -0.82917953, 0.66138285]])
    ten_check = torch.tensor(check_array, dtype=torch.float32)
    ten_check = ten_check.unsqueeze(0).unsqueeze(0)
    print(ten_check.device)
    ten_check = ten_check.to('cuda')
    print(ten_check.device)
    torchinfo.summary(model1,input_data=ten_check) #batchsize,noofsamples,value,xyz
    model1.eval()
    with torch.no_grad():
        outputt = model1(ten_check)
        print(outputt[0].item())


model1 = gcn_reg(hidden=[50, 50, 50, 50], kernel_size=4)
torchinfo.summary(model1)
