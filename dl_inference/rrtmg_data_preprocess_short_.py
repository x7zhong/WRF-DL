#prepare rrtmg DL model required inputs

import logging
import numpy as np
    
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

#nrows: number of rows
#ncols: number of cols
#batch_size = nrows * ncols   
#nlayers: number of vertical layers

#1 value: solcon
#nrows * ncols: albedo, coszen, xland, xice, snow, tsk, emiss
#nrows * ncols * nlayers: p3d, t3d, cldfra3d, o33d, qc3d, qg3d, qr3d, qi3d, qs3d, qv3d
#nrows * ncols * (nlayers+1): p8w,     
    
def rrtmg_preprocess(albedo, coszen, landfrac, icefrac, snow, solcon, tsfc, emis, \
                     play, plev, tlay, tlev, cldfrac, o3vmr, qc, qg, qr, qi, qs, qv):
    #o3input = 2):
    
#    logger = generate_logger("./rrtmg_preprocess.log")

#    logger.info("Start rrtmg_preprocess")
    
    batch_size, nlevels = plev.shape
    
#==============================================================================
# WRF data preprocessing
#==============================================================================
    
#    logger.info("WRF data preprocess")

    coszen[coszen <= 0] = 0  
    albedo[coszen <= 0] = 0  
    
    #To keep it consistent with data provided by Zheng
    #solcon_temp = 1351.021  
    solcon[coszen <= 0] = 0  
                           
#==============================================================================
# define default values as in the rrtmg module
#==============================================================================

#    logger.info("Define default values as in the rrtmg module")

    #gas volume mixing ratios defined in rrtmg_sw
    ch4 = 1774*10**(-9)
    n2o = 319*10**(-9)        
    #Annual function for co2 after v4.2 in WRF
#    co2 = (280 + 90*np.exp(0.02*(yr-2000)))*10**(-6)
    co2 = 0.000379
    n2o = 319*10**(-9)
    o2 = 0.209488    
    
    #gas volume mixing ratios defined in rrtmg_lw
    ccl4 = 0.093*10**(-9)
    cfc11 = 0.251*10**(-9)
    cfc12 = 0.538*10**(-9)
    cfc22 = 0.169*10**(-9)    

#==============================================================================
# Initialize variables used to input to DL rrtmg model
#==============================================================================
            
#    logger.info("Initialize variables used to input to DL rrtmg model")

    ### single height feature ###        
    single_height_variable = ["aldif", "aldir", "asdif", "asdir", "cosz", "landfrac", \
                              "sicefrac", "snow", "solc", "tsfc", "emiss"]
    single_feature = np.zeros([batch_size, len(single_height_variable), 1])
    
    for i in range(4):
        single_feature[:, i, 0] = albedo

    single_feature[:, 4, 0] = coszen
    single_feature[:, 5, 0] = landfrac
    single_feature[:, 6, 0] = icefrac    
    single_feature[:, 7, 0] = snow    
    single_feature[:, 8, 0] = solcon
    single_feature[:, 9, 0] = tsfc
    single_feature[:, 10, 0] = emis
        
    ### multi height/layer feature ###            
    multi_height_variable = ["ccl4vmr", "cfc11vmr", "cfc12vmr", "cfc22vmr", \
                             "ch4vmr", "cldfrac", "co2vmr", "n2ovmr", "o2vmr", \
                             "o3vmr", "play", "qc", "qg", "qi", "qr", "qs", "qv", "tlay"]
    multi_feature = np.zeros([batch_size, len(multi_height_variable), nlevels])

#    logger.info("Specify 3d variables used to input to DL rrtmg model")

#    logger.info("Shape of multi_feature is {}".format(multi_feature.shape))

    multi_feature[:, 0, :] = ccl4
    multi_feature[:, 1, :] = cfc11
    multi_feature[:, 2, :] = cfc12
    multi_feature[:, 3, :] = cfc22
    multi_feature[:, 4, :] = ch4
    multi_feature[:, 5, 1:] = cldfrac
    multi_feature[:, 6, :] = co2
    multi_feature[:, 7, :] = n2o
    multi_feature[:, 8, :] = o2
    multi_feature[:, 9, 1:] = o3vmr
    multi_feature[:, 10, 1:] = play
    multi_feature[:, 11, 1:] = qc
    multi_feature[:, 12, 1:] = qg
    multi_feature[:, 13, 1:] = qi
    multi_feature[:, 14, 1:] = qr
    multi_feature[:, 15, 1:] = qs
    multi_feature[:, 16, 1:] = qv
    multi_feature[:, 17, 1:] = tlay
    
    ### multi layer feature requiring vertical integration ###
    multi_height_cumsum_variable = {"cldfrac": 0, "qc": 1}
    multi_cumsum_feature = np.zeros([batch_size, 2*len(multi_height_cumsum_variable), nlevels])
        
    ### level pressure ###    
    auxiliary_variable = ["plev"]
    auxiliary_feature = np.zeros([batch_size, len(auxiliary_variable), nlevels]) 
#    logger.info("Shape of auxiliary_feature is {}".format(auxiliary_feature.shape))
    
    auxiliary_feature[:, 0, :] = plev
    
    '''
    p_diff_temp = auxiliary_feature - np.roll(auxiliary_feature, -1, 2)
    p_diff_temp = np.concatenate([p_diff_temp[:, :, 0:1], p_diff_temp[:, :, 0:-1]], 2)    
    feature_temp = np.concatenate(
        [np.repeat(single_feature, nlevels, axis = 2),
         multi_feature,
         p_diff_temp,
         multi_cumsum_feature
         ], 1)    

    np.save("feature_unormalized.npy", feature_temp)        
    
    '''
    
#==============================================================================
# define the mean and std used for input data normalization
#==============================================================================
                
#    logger.info("Define the mean and std used for input data normalization")

    norm_mapping = {'aldif': {'mean': 0.16539352, 'scale': 0.03575806}, 
    'aldir': {'mean': 0.16539352, 'scale': 0.03575806}, 
    'asdif': {'mean': 0.16539352, 'scale': 0.03575806}, 
    'asdir': {'mean': 0.16539352, 'scale': 0.03575806}, 
    'cosz': {'mean': 0.84524125, 'scale': 0.020037498}, 
    'landfrac': {'mean': 0.99975306, 'scale': 0.015711544}, 
    'sicefrac': {'mean': 0.0, 'scale': 0.0}, 
    'snow': {'mean': 0.0, 'scale': 0.0}, 
    'solc': {'mean': 1345.336, 'scale': 0.0}, 
    'tsfc': {'mean': 303.17914, 'scale': 4.3057466}, 
    'emiss': {'mean': 1.0388682, 'scale': 0.05498103}, 
    'ccl4vmr': {'mean': 9.29999999999501e-11, 'scale': 4.990225539167562e-23}, 
    'cfc11vmr': {'mean': 2.5099999999994356e-10, 'scale': 5.645507680674414e-23}, 
    'cfc12vmr': {'mean': 5.379999999988099e-10, 'scale': 1.1901061063106686e-21}, 
    'cfc22vmr': {'mean': 1.6900000000010907e-10, 'scale': 1.0905859388628824e-22}, 
    'ch4vmr': {'mean': 1.774000000002082e-06, 'scale': 2.0820069843510702e-18}, 
    'cldfrac': {'mean': 0.01628679, 'scale': 0.11835434},    
    'co2vmr': {'mean': 0.0003789999999999699,'scale': 3.008661028647275e-17}, 
    'n2ovmr': {'mean': 3.1899999999941876e-07, 'scale': 5.812234204940602e-19}, 
    'o2vmr': {'mean': 0.20948800000033244, 'scale': 3.324285291483875e-13},     
    'o3vmr': {'mean': 1.2141462e-06, 'scale': 2.3178093e-06}, 
    'play': {'mean': 390.18008, 'scale': 296.30096},     
    'qc': {'mean': 2.774779e-07, 'scale': 1.0387751e-05}, 
    'qg': {'mean': 2.2455426e-08, 'scale': 3.822603e-07}, 
    'qi': {'mean': 1.2064834e-06, 'scale': 1.1339106e-05},
    'qr': {'mean': 8.839166e-08, 'scale': 2.1570945e-06},     
    'qs': {'mean': 8.0543157e-07, 'scale': 1.1687571e-05}, 
    'qv': {'mean': 0.002893191, 'scale': 0.003924214},             
    'tlay': {'mean': 251.01125, 'scale': 32.00044},     
    'swuflx': {'mean': 180.72261, 'scale': 83.99388}, 
    'swdflx': {'mean': 1006.5645, 'scale': 129.04306}, 
    'lwuflx': {'mean': 349.25552, 'scale': 73.30407}, 
    'lwdflx': {'mean': 137.63927, 'scale': 124.870514}}    
    
#==============================================================================
# apply data normalization
#==============================================================================
      
#    logger.info("Apply data normalization")

    #apply data normalization to single height feature
    for variable_index, variable_name in enumerate(single_height_variable):
        single_feature[:, variable_index, 0] = (single_feature[:, variable_index, 0] \
        - norm_mapping[variable_name]["mean"]) / norm_mapping[variable_name]["scale"]

    single_feature = single_feature.astype(np.float32)
    
    #apply data normalization to multi height feature and multi height intergrated feature
    for variable_index, variable_name in enumerate(multi_height_variable):
        
        multi_feature[:, variable_index, 1:] = (multi_feature[:, variable_index, 1:] \
        - norm_mapping[variable_name]["mean"]) / norm_mapping[variable_name]["scale"]
            
        multi_feature[:, variable_index, 0] = multi_feature[:, variable_index, 1]
                
        temp_value = multi_feature[:, variable_index, 1:]

        if (variable_name in multi_height_cumsum_variable):
            # get index
            variable_index = multi_height_cumsum_variable[variable_name]
            
            temp_value_cumsum_forward = np.cumsum(temp_value, axis=1)/20.0
            
            temp_value_cumsum_backward = np.cumsum(temp_value[:, ::-1], axis=1)/20.0

            multi_cumsum_feature[:, variable_index, 1:] = temp_value_cumsum_forward            
            multi_cumsum_feature[:, variable_index, 0] = multi_cumsum_feature[:, variable_index, 1]

            multi_cumsum_feature[:, len(multi_height_cumsum_variable) + variable_index, 1:] = temp_value_cumsum_backward
            multi_cumsum_feature[:, len(multi_height_cumsum_variable) + variable_index, 0] = \
            multi_cumsum_feature[:, len(multi_height_cumsum_variable) + variable_index, 1]    

    multi_feature = multi_feature.astype(np.float32)
    multi_cumsum_feature = multi_cumsum_feature.astype(np.float32)
        
    auxiliary_feature = auxiliary_feature.astype(np.float32)

    p_diff = auxiliary_feature - np.roll(auxiliary_feature, -1, 2)
    p_diff = np.concatenate([p_diff[:, :, 0:1], p_diff[:, :, 0:-1]], 2)

    feature = np.concatenate(
        [np.repeat(single_feature, nlevels, axis = 2),
         multi_feature,
         (p_diff - 17.2)/9.8,
         multi_cumsum_feature
         ], 1)        

#    np.save("feature.npy", feature)        
    
    #As we set the batch_size to 1 when convert to onnx model
    feature = feature[:, :, :, None]
          
#    logger.info("Finish rrtmg_preprocess")
    
#    np.save("auxiliary_feature.npy", auxiliary_feature)
    
    return feature, auxiliary_feature, coszen

            