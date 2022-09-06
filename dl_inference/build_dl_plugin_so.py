import cffi
ffibuilder = cffi.FFI()

header = """
extern void infer_init(void);
extern void infer_run_simple(float*, int, int, float*, int, int, int);
extern void infer_run_test(float*, int, int, int,
                      float*, int, int, int, int,
                      float,
                      float*, int, int, int, int);
extern void save_fortran_array2(float*, int, int, char*);
extern void save_fortran_array3(float*, int, int, int, char*);
extern void print_fortran_array2(float*, int,  int, int, char*);
extern void print_fortran_array3(float*, int,  int, int, int, char*);

"""

module = """
from my_dl_plugin import ffi
import numpy as np
import my_infer_module
from my_infer_module import OnnxEngine

runner_list = []
plugin_logger = my_infer_module.generate_logger("./plugin_python.log")

@ffi.def_extern()
def infer_init():
    plugin_logger.info("InferInit begin")
    onnxrunner = OnnxEngine("./model/rrtmg_modify.onnx")
    plugin_logger.info("InferInit finished")
    runner_list.append(onnxrunner)

@ffi.def_extern()
def infer_run_simple(inputdata_ptr, input_x, input_y, output_ptr, out_x, out_y, out_z):
    onnxrunner = runner_list[0]
    shape = onnxrunner.getInputShape()
    plugin_logger.info("onnx model input shape={}".format(shape))
    input_array = my_infer_module.PtrAsarray(ffi, inputdata_ptr, (1,input_x, input_y,1))
    plugin_logger.info("Infer_run_simple, inputdata.shape={}, input_array={}".format(input_array.shape, input_array))
    output = onnxrunner([input_array])
    output_array = my_infer_module.PtrAsarray(ffi, output_ptr, (out_x, out_y, out_z))
    plugin_logger.info("Infer_run_simple, outdata.shape={}, output_array={}".format(output_array.shape, output_array))
    np.copyto(output_array, output[0])


@ffi.def_extern()
def infer_run_test(emiss, emiss_x, emiss_y, emiss_len,    
             t3d, t3d_x, t3d_y, t3d_z, t3d_len,          
             radt,
             dataout, out_x, out_y, out_z,lenout):
    plugin_logger.info("InferRun_test, emiss_len={}, emiss_x={}, emiss_y={}".format(emiss_len,emiss_x, emiss_y))
    plugin_logger.info("InferRun_test, t3d_len={}, t3d_x={}, t3d_y={}, t3d_z={}".format(t3d_len,t3d_x, t3d_y, t3d_z))
    plugin_logger.info("InferRun_test, radt={}".format(radt))
    emiss_array = my_infer_module.PtrAsarray(ffi, emiss, (emiss_x, emiss_y))
    plugin_logger.info("emiss_array.shape={}, emiss_array = {}".format(emiss_array.shape,emiss_array))

    t3d_array = my_infer_module.PtrAsarray(ffi, t3d, (t3d_x,t3d_y,t3d_z))
    plugin_logger.info("before t3d reshape, t3d.shape={}, t3d_array={}".format(t3d_array.shape, t3d_array))
    t3d_array =t3d_array.reshape((int(t3d_x*t3d_y), t3d_z))
    plugin_logger.info("t3d.shape()={}, t3d_array = {}".format(t3d_array.shape, t3d_array))
    output_array = my_infer_module.PtrAsarray(ffi, dataout, (out_x, out_y, out_z))

@ffi.def_extern()
def save_fortran_array2(data_ptr, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    print("save_fortran_array2, shape=({},{}), filename={}, filenamelen={}".format(in_x, in_y, filenam, len(filenam)))
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y))
    np.save(filenam, data_array)

@ffi.def_extern()
def save_fortran_array3(data_ptr, in_x, in_y, in_z,  filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    print("save_fortran_array3, shape=({},{},{}), filename={}, filenamelen={}".format(in_x, in_y, in_z, filenam,len(filenam)))
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y, in_z))
    np.save(filenam, data_array)

@ffi.def_extern()
def print_fortran_array2(data_ptr, offset, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    print("print_fortran_array2, data_ptr:{}, offset:{}, shape=({},{}), filename={}, filenamelen={}".format(data_ptr, offset, in_x, in_y, filenam,len(filenam)))
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr+offset, (in_x, in_y))
    print(data_array)

@ffi.def_extern()
def print_fortran_array3(data_ptr, offset, in_x, in_y, in_z,  filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    print("print_fortran_array3, data_ptr:{}, offset:{}, shape=({},{},{}), filename={}, filenamelen={}".format(data_ptr, offset, in_x, in_y, in_z, filenam,len(filenam)))
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr+offset, (in_x, in_y, in_z))
    print(data_array)


"""

with open("my_dl_plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_dl_plugin", r'''
    #include "my_dl_plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libDL_inference_plugin.so", verbose=True)
