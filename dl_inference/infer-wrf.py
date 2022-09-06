import argparse
import numpy as np
import onnxruntime as oxrt
import os
import json
import pdb
import logging


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
                        default = 200,
                        help='model file path.')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    model = opt.model
    nlevel = opt.nlevel
    loop = opt.loop
    print('*'*20)
    print(oxrt.__version__)
    ori_runner = OnnxEngine(model)
    ref_runner = OnnxEngine(model, False)
    import time
    shape1 = ori_runner.getInputShape()
    #pdb.set_trace()
    diff = None
    ori_out = None
    ref_out = None
    for j in [57]:
      for i in [50,  100, 150, 200 ,500, 760,1000]:
        shape1[0] = i
        shape1[2] = j
        input_data1 = [np.ones(shape1)]
        start = time.time()
        print("="*20)
        for l in range (loop):
            pass
            ori_out = ori_runner(input_data1)
        stop  = time.time()
        print('cpu nlevel =', j, ',batch =', i, ', cost time = ', (stop-start)  *1000 /loop)
        start = time.time()
        for l in range (loop):
            try:
              ref_out = ref_runner(input_data1)
            except Exception as e:
                print(f"catch {e}")
        stop  = time.time()
        print('gpu nlevel =', j, ',batch =', i, ', cost time = ', (stop-start)  *1000 /loop)
        #diff = ori_out[0] - ref_out[0]
        #print('max_diff = ', np.max(np.abs(diff)))
    pass
    pass 

if __name__ == '__main__':
    main()
