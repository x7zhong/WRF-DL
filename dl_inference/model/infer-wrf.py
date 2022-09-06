import os
import argparse
import numpy as np
#import onnx_graphsurgeon as gs
import onnxruntime as oxrt
#import onnx
#import os
#import json
#import pdb
#import cv2
import logging
#from fvcore.nn import FlopCountAnalysis


streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.setLevel(logging.INFO)
logger.addHandler(streamhandler)

class OnnxEngine(object):
    def __init__(self, onnx_model, use_cpu = True):
        print(oxrt.get_available_providers())
        if use_cpu :
          self.sess = oxrt.InferenceSession(onnx_model, providers= ['CPUExecutionProvider'])
        else:
          self.sess = oxrt.InferenceSession(onnx_model, providers= ['CUDAExecutionProvider'])
          self.sess.set_providers(['CUDAExecutionProvider'], provider_options=[{'device_id': 0,}])
        self.input_names = [inp.name for inp in self.sess.get_inputs()]
        #pdb.set_trace()
        self.output_names = [out.name for out in self.sess.get_outputs()]
        self.shape = self.sess.get_inputs()[0].shape 
        self.oshape = self.sess.get_outputs()[0].shape 
    def getInputShape(self):
        return self.shape
    def getOutputShape(self):
        return self.oshape
    def __call__(self, imgs):
        assert (len(self.input_names) == len(imgs))
        inputs_data={}
        for k,v in zip(self.input_names, imgs):
            inputs_data[k] = v.astype(np.float32)
        out_onnx = self.sess.run(self.output_names, inputs_data)
        return out_onnx

class OnnxRuntime(object):
    def __init__(self, onnx_model ):
        self.onnx_model = onnx_model
        self.onnx_infer =  OnnxEngine(onnx_model)
        self.shape = self.onnx_infer.getInputShape()
        print("model input shape:", self.shape)

    def forward(self, img_files):
        imgs = []
        if isinstance(img_files, list):
            for img_file in img_files:
                imgs.append(img_file)
        else:
            imgs.append(img_files)
        return self.onnx_infer(imgs)

def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge mean norm to 1st conv of onnx model.')
    parser.add_argument('model', type=str,
                        help='model file path.')

    parser.add_argument('--nlevel', type=int,
                        default = 57,
                        help='nlevel.')

    parser.add_argument('--loop', type=int,
                        default = 50,
                        help='model file path.')
    opt = parser.parse_args()
    return opt

def main():
    #opt = parse_args()
    #model = opt.model
    #nlevel = opt.nlevel
    loop = 50
    models = ['modelA']#, 'modelB','modelC','modelD','modelE']
    #models = ['wrf_0030260model']#, 'modelB','modelC','modelD','modelE']
    gpu_runners ={}
    cpu_runners ={}
    print('*'*20)
    
    dirName_model = '/home/WRF/WRF/xlf/fortran-python-test/infer-example/model/model'
    
    for name in models:
        model = os.path.join(dirName_model, name, name + '.onnx')

        cpu_runners[name] = OnnxEngine(model, use_cpu = True)
        gpu_runners[name] = OnnxEngine(model, use_cpu = False)

        #flops = FlopCountAnalysis(model, [img])
        #flops.total()

    print(oxrt.__version__)
    import time
    for name in models:
      cpu_runner = cpu_runners[name]
      gpu_runner = gpu_runners[name]
      for i in [1,10, 50,100, 200, 500, 1000, 2000, 3000, 4000]:        
        shape1 = cpu_runner.getInputShape()
        shape1[0] = i
        #input_data1 = [np.ones(shape1)]
        input_data1 = [np.random.rand(*shape1)]
        print('batch size is {}, input_data shape is {}\n'.format(i, input_data1[0].shape))
        
        start = time.time()
        for l in range (loop):
            a = cpu_runner(input_data1)
        stop  = time.time()
        print(name, ' cpu batch =', i, ', cost time = ', (stop-start)  *1000 /loop)
       
        start = time.time()
        for l in range (loop):
            b = gpu_runner(input_data1)
        stop  = time.time()
        print(name, ' gpu batch =', i, ', cost time = ', round((stop-start)  *1000 /loop, 2))
        
        d = a[0] - b[0]
        print(d.shape)
        print('max and min difference is {} and {}\n'.format(d.max(), d.min()))
    pass
    pass 

if __name__ == '__main__':
    main()
