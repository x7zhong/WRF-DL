import cffi
ffibuilder = cffi.FFI()

header = """
extern void infer_init(int);
extern void infer_run(float*, int, int, int, int,     
                      float*,                      
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*,
                      float*,
                      float*,
                      float*, 
                      float*,
                      float*,
                      float*,                      
                      float*);

extern void save_fortran_array2(float*, int, int, char*);
extern void save_fortran_array3(float*, int, int, int, char*);
extern void save_fortran_array(int, int, int, int, int, int,   
                      int, int, int, int, int, int,   
                      float*, 
                      float*,                      
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*, 
                      float*,                       
                      float*, 
                      float*, 
                      float*,
                      float*,
                      float*,
                      float*,
                      float*,                      
                      float*,
                      float*,                       
                      int, float, float);
"""

module = """
from my_dl_plugin import ffi
import numpy as np
import datetime
import my_infer_module
from my_infer_module import OnnxEngine
import rrtmg_data_preprocess_short as rrtmgpre
import rrtmg_data_postprocess as rrtmgpost
import os,threading
import time

runner_list = []
pidname = os.getpid()
main_threadid = threading.currentThread()
logger_file_name = "./wrf_plugin_python_pid"+str(pidname)+ "_tid_"+ main_threadid.getName() +str(main_threadid.ident)+".log"
plugin_logger = my_infer_module.generate_logger(logger_file_name)
    
@ffi.def_extern()
def infer_init(use_gpu=0):
    plugin_logger.info("InferInit begin")
    plugin_logger.info("ProcessID:{}, threadID:{}, threadName:{}".format(os.getpid(), threading.currentThread().ident, threading.currentThread().getName()))
    #onnxrunner = OnnxEngine("./model/rrtmg_modify.onnx", use_gpu)
    onnxrunner = OnnxEngine("./model/modelB.onnx", use_gpu)
    plugin_logger.info("Onnx provider :{}".format(onnxrunner.get_providers()) )
    plugin_logger.info("InferInit finished")
    runner_list.append(onnxrunner)

@ffi.def_extern()
def infer_run(emis, shape_x, shape_y, shape_z, shape_z_lev,    
             solcon,                                
             albedo,
             landfrac,   
             icefrac, 
             snow,    
	         coszen,              
	         tsfc,
             tlay, 
	         tlev,
	         play, 
	         plev, 
             qv,
             qc,
             qr,
	         qi, 
	         qs,
	         qg, 
	         o3vmr, 
	         cldfrac, 
	         pi,             
	         rthraten, 
	         rthratenlw, 
	         rthratensw, 
             lwuflx,
             lwdflx,
             swuflx,
             swdflx):

    #plugin_logger.info("Infer Run start")
    #plugin_logger.info("ProcessID:{}, threadID:{}, threadName:{}".format(os.getpid(), threading.currentThread().ident, threading.currentThread().getName()))
        
    len_2d = shape_x * shape_y

    data_load_begin_time = time.time()
            
    emis_array = my_infer_module.PtrAsarray(ffi, emis, (shape_x, shape_y))      
    solcon_array = my_infer_module.PtrAsarray(ffi, solcon, (shape_x, shape_y))      
    albedo_array = my_infer_module.PtrAsarray(ffi, albedo, (shape_x, shape_y))      
    landfrac_array = my_infer_module.PtrAsarray(ffi, landfrac, (shape_x, shape_y))      
    icefrac_array = my_infer_module.PtrAsarray(ffi, icefrac, (shape_x, shape_y))      
    snow_array = my_infer_module.PtrAsarray(ffi, snow, (shape_x, shape_y))      
    coszen_array = my_infer_module.PtrAsarray(ffi, coszen, (shape_x, shape_y))      
    tsfc_array = my_infer_module.PtrAsarray(ffi, tsfc, (shape_x, shape_y))      

    tlay_array = my_infer_module.PtrAsarray(ffi, tlay, (shape_z, shape_x, shape_y), [len_2d])      
    
    tlev_array = my_infer_module.PtrAsarray(ffi, tlev, (shape_z_lev, shape_x, shape_y), [len_2d])      
                                                      
    play_array = my_infer_module.PtrAsarray(ffi, play, (shape_z, shape_x, shape_y), [len_2d])      

    plev_array = my_infer_module.PtrAsarray(ffi, plev, (shape_z_lev, shape_x, shape_y), [len_2d])    

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_z, shape_x, shape_y), [len_2d])      
    
    qc_array = my_infer_module.PtrAsarray(ffi, qc, (shape_z, shape_x, shape_y), [len_2d])      
    
    qr_array = my_infer_module.PtrAsarray(ffi, qr, (shape_z, shape_x, shape_y), [len_2d])      
    
    qi_array = my_infer_module.PtrAsarray(ffi, qi, (shape_z, shape_x, shape_y), [len_2d])      
    
    qs_array = my_infer_module.PtrAsarray(ffi, qs, (shape_z, shape_x, shape_y), [len_2d])      
    
    qg_array = my_infer_module.PtrAsarray(ffi, qg, (shape_z, shape_x, shape_y), [len_2d])           

    o3vmr_array = my_infer_module.PtrAsarray(ffi, o3vmr, (shape_z, shape_x, shape_y),[len_2d])           

    cldfrac_array = my_infer_module.PtrAsarray(ffi, cldfrac, (shape_z, shape_x, shape_y), [len_2d])           
                              
    pi_array = my_infer_module.PtrAsarray(ffi, pi, (shape_z_lev, shape_x, shape_y), [len_2d])       

    rthraten_array = my_infer_module.PtrAsarray(ffi, rthraten, (shape_z_lev, shape_x, shape_y))      

    rthratenlw_array = my_infer_module.PtrAsarray(ffi, rthratenlw, (shape_z_lev, shape_x, shape_y))       
    
    rthratensw_array = my_infer_module.PtrAsarray(ffi, rthratensw, (shape_z_lev, shape_x, shape_y))       
             
    lwuflx_array = my_infer_module.PtrAsarray(ffi, lwuflx, (shape_z_lev, shape_x, shape_y))      

    lwdflx_array = my_infer_module.PtrAsarray(ffi, lwdflx, (shape_z_lev, shape_x, shape_y))      
    
    swuflx_array = my_infer_module.PtrAsarray(ffi, swuflx, (shape_z_lev, shape_x, shape_y))      
    
    swdflx_array = my_infer_module.PtrAsarray(ffi, swdflx, (shape_z_lev, shape_x, shape_y))      

    data_load_end_time = time.time()
    total_time = data_load_end_time - data_load_begin_time

    #plugin_logger.info("load data takes {:.4f}s".format(total_time))
    
    #plugin_logger.info("Infer Run, before rrtmg_preprocess")
    
    data_preprocess_begin_time = time.time()
    
    feature, auxiliary_feature, coszen_array = rrtmgpre.rrtmg_preprocess(albedo_array, coszen_array, \
    landfrac_array, icefrac_array, snow_array, solcon_array, tsfc_array, emis_array, \
    play_array, plev_array, tlay_array, tlev_array, cldfrac_array, o3vmr_array, \
    qc_array, qg_array, qr_array, qi_array, qs_array, qv_array)

    data_preprocess_end_time = time.time()
    total_time = data_preprocess_end_time - data_preprocess_begin_time
    #plugin_logger.info("preprocess data takes {:.4f}s".format(total_time))
    
    #plugin_logger.info("Start inference")
    #plugin_logger.info("feature shape is {}".format(feature.shape))
    
    inference_begin_time = time.time()
    
    onnxrunner = runner_list[0]
    output =None
    try:
      output = onnxrunner([feature])
    except Exception as inst :
        plugin_logger.info(f"inference exception:{inst}")

    
    inference_end_time = time.time()    
    total_time = inference_end_time - inference_begin_time    
    #plugin_logger.info("inference takes {:.4f}s".format(total_time))

    #plugin_logger.info("Infer Run finished ")
    
    get_hr_begin_time = time.time()    
    swhr_res, lwhr_res, swuflx_predict, swdflx_predict, lwuflx_predict, lwdflx_predict\
    = rrtmgpost.rrtmg_get_hr(output, auxiliary_feature, coszen_array)
    
    get_hr_end_time = time.time()    
    total_time = get_hr_end_time - get_hr_begin_time    
    #plugin_logger.info("get_hr takes {:.4f}s".format(total_time))
    
    #plugin_logger.info("Infer Run, after rrtmg_get_hr ")

    swhr_res = swhr_res.reshape((len_2d, shape_z))
    lwhr_res = lwhr_res.reshape((len_2d, shape_z))
    swuflx_predict = swuflx_predict.reshape((-1, ))
    swdflx_predict = swdflx_predict.reshape((-1, ))
    lwuflx_predict = lwuflx_predict.reshape((-1, ))
    lwdflx_predict = lwdflx_predict.reshape((-1, ))
    
    np.copyto(swuflx_array, swuflx_predict)
    np.copyto(swdflx_array, swdflx_predict)
    np.copyto(lwuflx_array, lwuflx_predict)
    np.copyto(lwdflx_array, lwdflx_predict)

    #Convert sw and sw heating rate to wrf required tendency
    rthratensw_tmp = np.zeros((len_2d, shape_z_lev))
    rthratenlw_tmp = np.zeros((len_2d, shape_z_lev))    
    rthratensw_tmp[:, 0:-1] = swhr_res/(pi_array[:, 0:-1]*86400)  
    rthratenlw_tmp[:, 0:-1] = lwhr_res/(pi_array[:, 0:-1]*86400) 
    rthraten_tmp = rthratensw_tmp + rthratenlw_tmp
        
    rthraten_tmp = rthraten_tmp.reshape((-1,))
    rthratensw_tmp = rthratensw_tmp.reshape((-1,))
    rthratenlw_tmp = rthratenlw_tmp.reshape((-1,))
    
    np.copyto(rthraten_array, rthraten_tmp)
    np.copyto(rthratensw_array, rthratensw_tmp)
    np.copyto(rthratenlw_array, rthratenlw_tmp)   

    '''
    np.savez('infer.npz', emiss=emis_array, solc=solcon_array, albedo=albedo_array, \
             landfrac=landfrac_array, sicefrac=icefrac_array, snow=snow_array, \
	         cosz=coszen_array, tsfc=tsfc_array, tlay=tlay_array, tlev=tlev_array, \
	         play=play_array, plev=plev_array, qv=qv_array, qc=qc_array, qr=qr_array, \
	         qi=qi_array, qs=qs_array, qg=qg_array, o3vmr=o3vmr_array, cldfrac=cldfrac_array, \
             pi = pi_array, sw_hr=swhr_res, lw_hr=lwhr_res, rthraten=rthraten_array, \
             rthratensw=rthratensw_array, rthratenlw=rthratenlw_array, 
             swuflx=swuflx_array, swdflx=swdflx_array, lwuflx=lwuflx_array, lwdflx=lwdflx_array)            
    '''
    
        
@ffi.def_extern()
def save_fortran_array2(data_ptr, in_x, in_y, filename):    
    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")

#    plugin_logger.info("save to {}".format(filenam))    
    
    n = 5
    in_x_new = in_x - 2*n - 1
    in_y_new = in_y - 2*n - 1
        
    len_2d = in_x_new * in_y_new      
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y), (n, len_2d))   
    
    np.save(filenam, data_array)
    
@ffi.def_extern()
def save_fortran_array3(data_ptr, in_x, in_y, in_z, filename):    
#    plugin_logger.info("save to {}".format(filenam))        

    filenam = ffi.string(filename).decode('UTF-8')
    filenam = filenam.replace(" ", "")    
        
#    n = 5
#    in_x_new = in_x - 2*n - 1
#    in_z_new = in_z - 2*n - 1
        
    data_array = my_infer_module.PtrAsarray(ffi, data_ptr, (in_x, in_y, in_z)) 
        
    np.save(filenam, data_array)
    
@ffi.def_extern()
def save_fortran_array(ims, ime, jms, jme, kms, kme,
             its, ite, jts, jte, kts, kte, 
             emis,
             solcon,                                
             albedo,
             landfrac,   
             icefrac, 
             snow,    
	         coszen,              
	         tsfc,
             tlay, 
	         tlev,
	         play, 
	         plev, 
             qv,
             qc,
             qr,
	         qi, 
	         qs,
	         qg, 
	         o3vmr, 
	         cldfrac,
             swuflx,
             swdflx,
             lwuflx,
             lwdflx,
	         lwhr, 
	         swhr,
	         rthraten,
	         rthratenlw,
	         rthratensw,
	         pi,
             glw,
             gsw,
             julday, gmt, xtime):   
        
    shape_x = ime-ims+1
    shape_y = jme-jms+1
    shape_z = kme-1-kms+1
    shape_z_lev = kme-kms+1
    
    forecast_time = datetime.datetime(2021, 1, 1, int(gmt), 0, 0) + datetime.timedelta(days = julday) + datetime.timedelta(minutes = xtime)
    
    filenam = forecast_time.strftime('%Y%m%d_%H%M%S') + '.npz'
    
    #n is number of points of lateral boundary    
    n = 5
    shape_x_new = shape_x - 2*n - 1
    shape_y_new = shape_y - 2*n - 1
    
    len_2d = shape_x_new * shape_y_new
       
#    plugin_logger.info("save 1")
    
    emis_array = my_infer_module.PtrAsarray(ffi, emis, (shape_y, shape_x), (n, len_2d))      
    solcon_array = my_infer_module.PtrAsarray(ffi, solcon, (shape_y, shape_x), (n, len_2d))    
    albedo_array = my_infer_module.PtrAsarray(ffi, albedo, (shape_y, shape_x), (n, len_2d))   
    landfrac_array = my_infer_module.PtrAsarray(ffi, landfrac, (shape_y, shape_x), (n, len_2d))    
    icefrac_array = my_infer_module.PtrAsarray(ffi, icefrac, (shape_y, shape_x), (n, len_2d))  
    snow_array = my_infer_module.PtrAsarray(ffi, snow, (shape_y, shape_x), (n, len_2d))  
    coszen_array = my_infer_module.PtrAsarray(ffi, coszen, (shape_y, shape_x), (n, len_2d))  
    tsfc_array = my_infer_module.PtrAsarray(ffi, tsfc, (shape_y, shape_x), (n, len_2d))  
    glw_array = my_infer_module.PtrAsarray(ffi, glw, (shape_y, shape_x), (n, len_2d))      
    gsw_array = my_infer_module.PtrAsarray(ffi, gsw, (shape_y, shape_x), (n, len_2d))          

    tlay_array = my_infer_module.PtrAsarray(ffi, tlay, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    tlev_array = my_infer_module.PtrAsarray(ffi, tlev, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))
                                                      
    play_array = my_infer_module.PtrAsarray(ffi, play, (shape_y, shape_x,shape_z), \
                                            (n, len_2d, shape_z))

    plev_array = my_infer_module.PtrAsarray(ffi, plev, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))

    qv_array = my_infer_module.PtrAsarray(ffi, qv, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    qc_array = my_infer_module.PtrAsarray(ffi, qc, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    qr_array = my_infer_module.PtrAsarray(ffi, qr, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    qi_array = my_infer_module.PtrAsarray(ffi, qi, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    qs_array = my_infer_module.PtrAsarray(ffi, qs, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    qg_array = my_infer_module.PtrAsarray(ffi, qg, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    o3vmr_array = my_infer_module.PtrAsarray(ffi, o3vmr, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))

    cldfrac_array = my_infer_module.PtrAsarray(ffi, cldfrac, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))
    
    swuflx_array = my_infer_module.PtrAsarray(ffi, swuflx, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))      
        
    swdflx_array = my_infer_module.PtrAsarray(ffi, swdflx, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))       
  
    lwuflx_array = my_infer_module.PtrAsarray(ffi, lwuflx, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))      

    lwdflx_array = my_infer_module.PtrAsarray(ffi, lwdflx, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))      
                              
    lwhr_array = my_infer_module.PtrAsarray(ffi, lwhr, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))    
    
    swhr_array = my_infer_module.PtrAsarray(ffi, swhr, (shape_y, shape_x, shape_z), \
                                            (n, len_2d, shape_z))          
             
    rthraten_array = my_infer_module.PtrAsarray(ffi, rthraten, (shape_y, shape_x, shape_z_lev))
        
    rthratenlw_array = my_infer_module.PtrAsarray(ffi, rthratenlw, (shape_y, shape_x, shape_z_lev))
    #, \
#                                            (n, len_2d, shape_z_lev))    
        
    rthratensw_array = my_infer_module.PtrAsarray(ffi, rthratensw, (shape_y, shape_x, shape_z_lev))
    #, \
#                                            (n, len_2d, shape_z_lev))    
             
    pi_array = my_infer_module.PtrAsarray(ffi, pi, (shape_y, shape_x, shape_z_lev), \
                                            (n, len_2d, shape_z_lev))    
        
    np.savez(filenam, emiss=emis_array, solc=solcon_array, albedo=albedo_array, \
             landfrac=landfrac_array, sicefrac=icefrac_array, snow=snow_array, \
	         cosz=coszen_array, tsfc=tsfc_array, tlay=tlay_array, tlev=tlev_array, \
	         play=play_array, plev=plev_array, qv=qv_array, qc=qc_array, qr=qr_array, \
	         qi=qi_array, qs=qs_array, qg=qg_array, o3vmr=o3vmr_array, cldfrac=cldfrac_array, \
             swuflx=swuflx_array, swdflx=swdflx_array, lwuflx=lwuflx_array, \
             lwdflx=lwdflx_array, lw_hr=lwhr_array, sw_hr=swhr_array, rthraten=rthraten_array, \
             rthratenlw=rthratenlw_array, rthratensw=rthratensw_array, pi = pi_array, \
             gsw=gsw_array, glw=glw_array, \
             its=its, ite=ite, jts=jts, jte=jte, kts=kts, kte=kte, \
             ims=ims, ime=ime, jms=jms, jme=jme, kms=kms, kme=kme)
    
"""

with open("my_dl_plugin.h", "w") as f:
    f.write(header)

ffibuilder.embedding_api(header)
ffibuilder.set_source("my_dl_plugin", r'''
    #include "my_dl_plugin.h"
''')

ffibuilder.embedding_init_code(module)
ffibuilder.compile(target="libDL_inference_plugin.so", verbose=True)
