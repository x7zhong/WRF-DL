# my_module.py
import os, logging
import numpy as np
import onnxruntime as oxrt
import pynvml 

#pynvml.nvmlInit()
#gpu_num = pynvml.nvmlDeviceGetCount()
gpu_num=0

def generate_logger(log_file_path):
    #Create the logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    #Create handler for logging data to a file
    logger_handler = logging.FileHandler(log_file_path, mode = 'a')
    logger_handler.setLevel(logging.INFO)
    
    #Define format for handler
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    logger_handler.setFormatter(formatter)
    
    #Add logger in the handler
    logger.addHandler(logger_handler)
    
    return logger

class OnnxEngine(object):
     def __init__(self, onnx_model, use_gpu = 0):
         #onnx_model = "./model/rrtmg_modify.onnx"
         if use_gpu < 0:
           self.sess = oxrt.InferenceSession(onnx_model, providers = ['CPUExecutionProvider'] )
         else:
           #self.sess = oxrt.InferenceSession(onnx_model, providers = ['CUDAExecutionProvider'] )
           self.sess = oxrt.InferenceSession(onnx_model, providers = oxrt.get_available_providers() )
           #self.sess = oxrt.InferenceSession(onnx_model)
           #use_gpu = 0
           self.sess.set_providers(['CUDAExecutionProvider'], provider_options=[{'device_id': use_gpu%gpu_num,}])
         self.input_names = [inp.name for inp in self.sess.get_inputs()]
         self.output_names = [out.name for out in self.sess.get_outputs()]
         self.shape = self.sess.get_inputs()[0].shape
     def get_providers(self):
         return self.sess.get_providers()
             
     def getInputShape(self):
         return self.shape
     
     def __call__(self, imgs):
         assert (len(self.input_names) == len(imgs))
         inputs_data={}
         for k,v in zip(self.input_names, imgs):
             inputs_data[k] = v.astype(np.float32)
             
         out_onnx = self.sess.run(self.output_names, inputs_data)
         return out_onnx


# Create the dictionary mapping ctypes to np dtypes.
ctype2dtype = {}

# Integer types
for prefix in ('int', 'uint'):
    for log_bytes in range(4):
        ctype = '%s%d_t' % (prefix, 8 * (2**log_bytes))
        dtype = '%s%d' % (prefix[0], 2**log_bytes)
        # print( ctype )
        # print( dtype )
        ctype2dtype[ctype] = np.dtype(dtype)

# Floating point types
ctype2dtype['float'] = np.dtype('f4')
ctype2dtype['double'] = np.dtype('f8')
ctype2dtype['bool'] = np.dtype('bool')

#asarray函数使用CFFI的ffi对象转换指针ptr为给定形状的numpy数组
def PtrAsarray(ffi, ptr, shape, shape_new = [], **kwargs):
    length = np.prod(shape)
    # Get the canonical C type of the elements of ptr as a string.
    T = ffi.getctype(ffi.typeof(ptr).item)
    #print("In my_module PtrAsarray")
    #print( T )
    #print( ffi.sizeof( T ) )

    if T not in ctype2dtype:
        raise RuntimeError("Cannot create an array for element type: %s" % T)

    a = np.frombuffer(ffi.buffer(ptr, length * ffi.sizeof(T)), ctype2dtype[T])
        
    if len(shape_new) == 1:
        if len(shape) == 2:
            a = a.reshape((shape_new[0], ))
            
        elif len(shape) == 3:
            a = a.reshape((shape_new[0], shape[0]))            
        
    elif len(shape_new) > 1:
        if len(shape) == 2:
            a = a.reshape(shape)
            a = a[shape_new[0]:-(shape_new[0]+1), shape_new[0]:-(shape_new[0]+1)]
            a = a.reshape((shape_new[1], ))
            
        elif len(shape) == 3:
            a = a.reshape(shape)            
            a = a[shape_new[0]:-(shape_new[0]+1), shape_new[0]:-(shape_new[0]+1), :]    
            a = a.reshape((shape_new[1], shape_new[2]))
            
    return a

def reshape_3d(Var, Var_x, Var_y, Var_z):
   # Var = Var.reshape((Var_x, Var_y, Var_z))
    Var = np.swapaxes(Var, 1, 2)
    Var = Var.reshape((Var_x*Var_z, Var_y))

    return Var
'''
model = './model/rrtmg_modify.onnx'
print('*'*20)
print(oxrt.__version__)
ori_runner = OnnxEngine(model)
ori_runner.get_providers()
'''
