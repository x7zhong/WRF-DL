#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 17:23:43 2021

@author: xiaohui
"""

import torch
import torch.nn as nn
import sys


dirName_model = '/home/WRF/WRF/xlf/fortran-python-test/infer-example/model/model'

file_name = dirName_model + "/modelA/wrf_0050078model.pth"
pathName_onnx = dirName_model + "/modelA/modelA.onnx"

#file_name = dirName_model + "/modelB/wrf_0040078model.pth"
#pathName_onnx = dirName_model + "/modelB/modelB.onnx"

#file_name = dirName_model + "/modelC/wrf_0030260model.pth"
#pathName_onnx = dirName_model + "/modelC/modelC.onnx"

#file_name = dirName_model + "/modelD/wrf_0020208model.pth"
#pathName_onnx = dirName_model + "/modelD/modelD.onnx"

#file_name = dirName_model + "/modelE/wrf_0010347model.pth"
#pathName_onnx = dirName_model + "/modelE/modelE.onnx"



# file_name = "./inference_folder/small_mlp/fullyear_0090005model.pth"
model = torch.load(file_name, map_location=torch.device('cpu'))
# model = torch.load(file_name)


class UnsqueezeModule(nn.Module):
    def __init__(self, model):
        super(UnsqueezeModule, self).__init__()
        self.model = model
    def forward(self, input_):
        input_ = input_.squeeze(-1)
        return self.model(input_)

model.eval()
unsqueeze_model = UnsqueezeModule(model)

# exit()

# set the model to inference mode 
unsqueeze_model.eval() 
input_size = 34
batch_size = 1
# Let's create a dummy input tensor  
# set batch_size to -1 to test if dynamic batch_size works
#dummy_input = torch.randn(1, input_size, 57, 1, requires_grad=False)  
dummy_input = torch.randn(batch_size, 34, 57, 1)  

# Export the model   
torch.onnx.export(unsqueeze_model,         # model being run 
     dummy_input,       # model input (or a tuple for multiple inputs) 
     pathName_onnx,       # where to save the model
     export_params=True,  # store the trained parameter weights inside the model file 
     opset_version=10,    # the ONNX version to export the model to 
     do_constant_folding=True,  # whether to execute constant folding for optimization 
     input_names = ['input'],   # the model's input names 
     output_names = ['output'], # the model's output names 
     dynamic_axes = {'input': {0: 'batch_size'}, 
                     'output': {0: 'batch_size'}}
     ) 
print(" ") 
print('Model has been converted to ONNX') 


    
# if __name__ == "__main__": 

#     save_dir = "/home/admin/Code/Python/RTM/rrtmg_MPAS"
    
#     '''

#     gpu_avail = torch.cuda.is_available()
    
#     rrtmg_data_val = rrtmg_dataset(args.val_f, gpu_avail, args.gpu)
    
#     i = 0
#     ncFile_feature = nc.Dataset(args.pathName_feature)
#     ncFile_target = nc.Dataset(args.pathName_target)
    
#     feature, target = rrtmg_data_val.get_item(i, ncFile_feature, ncFile_target, \
#     args.varName_feature, args.varName_target)     
        
#     input_size = (feature.cpu().numpy().shape[1])
#     batch_size = feature.shape[0]
    
# #    pathName_model = os.path.join(args.save_dir, 'onnx_model.pt')
#     pathName_model = '/home/admin/WRF/xlf/fortran-python-test/infer-example/model/rrtmg_V03.onnx'
    
#     #Load model pathName_model
#     print('Load model {}'.format(pathName_model))
#     model = torch.load(pathName_model)
    
#     pathName_onnx = os.path.join(save_dir, "rrtmg_v03_modify.onnx")
    
#     '''
    
#     pathName_demo = "/home/admin/Code/Python/RTM/model/onnx/demo_data.pt"
#     db = torch.load(pathName_demo)
#     feature = db["feature"]
    
#     input_size = (feature.cpu().numpy().shape[1])    
#     batch_size = feature.shape[0]
    
# #    pathName_model = "/home/admin/Code/Python/RTM/rrtmg_MPAS/FNO_2D0000model.pth"
#     pathName_model = "/home/admin/Code/Python/RTM/rrtmg_MPAS/fullyear_0040260model.pth"
#     model = torch.load(pathName_model, map_location=torch.device('cpu'))
    
#     print(model)

#     pathName_onnx = os.path.join(save_dir, "rrtmg_modify_v03.onnx")
    
#     ''

#     # Conversion to ONNX 
#     Convert_ONNX() 
    
#     # Compare model prediction by original model and onnx model
#     model.eval()
    
#     with torch.no_grad():
#         output = model.forward(torch.tensor(feature))
           
#     print(output.shape)
#     ''
#     session = onnxruntime.InferenceSession(pathName_onnx, None)
#     input_name = session.get_inputs()[0].name
#     output_name = session.get_outputs()[0].name
#     #print(input_name)
#     #print(output_name)
    
#     feature = feature.unsqueeze(-1)
#     result = session.run([output_name], {input_name: feature.cpu().numpy()})
#     print(result[0].shape) 
    
#     difference = np.abs(result[0] - output.cpu().numpy())
    
#     print('difference max:{}, min: {}'.format(difference.max(), difference.min()))
#     ''
    