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

    norm_mapping = {'aldif': {'mean': 1.2225e-01, 'scale': 9.6926e-02*1.4142, 'count': 5.0},
    'aldir': {'mean': 1.2225e-01, 'scale': 9.6926e-02*1.4142, 'count': 5.0}, 
    'asdif': {'mean':1.2225e-01, 'scale': 9.6926e-02*1.4142, 'count': 5.0}, 
    'asdir': {'mean': 1.2225e-01, 'scale': 9.6926e-02*1.4142, 'count': 5.0}, 
    'cosz': {'mean':  2.3448e-01, 'scale': 2.1161e-01*1.4142, 'count': 5.0}, 
    'landfrac': {'mean': 9.9922e-01, 'scale': 1.9661e-02*1.4142, 'count': 5.0}, 
    'sicefrac': {'mean': 0.0000e+00,'scale': 0.1*1.4142, 'count': 5.0}, 
    'snow': {'mean': 1.6870e+00, 'scale': 4.2675e+00*1.4142, 'count': 5.0}, 
    'solc': {'mean': 1.0953e+03, 'scale': 3.8823e+02*1.4142, 'count': 5.0}, 
    'tsfc': {'mean': 2.8127e+02, 'scale': 1.0318e+01*1.4142, 'count': 5.0}, 
    'emiss': {'mean': 8.8330e-01, 'scale': 2.5870e-02*1.4142, 'count': 5.0}, 
    'ccl4vmr': {'mean': 9.2996e-11, 'scale': 2.6100e-15*10000*1.4142, 'count': 5.0}, 
    'cfc11vmr': {'mean': 2.5108e-10, 'scale': 5.6767e-14*1000*1.4142, 'count': 5.0}, 
    'cfc12vmr': {'mean': 5.3810e-10, 'scale': 6.9005e-14*1000*1.4142, 'count': 5.0}, 
    'cfc22vmr': {'mean': 1.6901e-10, 'scale': 6.0935e-15*1000*1.4142, 'count': 5.0}, 
    'ch4vmr': {'mean': 1.7736e-06, 'scale': 2.9631e-10*1.4142, 'count': 5.0},  
    'cldfrac': {'mean': 3.6548e-02, 'scale': 1.2967e-01*1.4142, 'count': 5.0}, 
    'co2vmr': {'mean': 3.7906e-04,'scale': 3.9326e-08*1.4142, 'count': 5.0}, 
    'n2ovmr': {'mean': 3.1895e-07, 'scale': 3.7523e-11*1.4142, 'count': 5.0}, 
    'o2vmr': {'mean': 2.0944e-01, 'scale': 3.2583e-05*1.4142, 'count': 5.0}, 
    'o3vmr': {'mean': 1.2222e-06, 'scale': 1.5227e-06*1.4142, 'count': 5.0}, 
    'play': {'mean': 4.0791e+02, 'scale': 2.1855e+02*1.4142, 'count': 5.0}, 
    'qc': {'mean': 1.1653e-06, 'scale': 1.8050e-05*1.4142, 'count': 5.0}, 
    'qg': {'mean': 9.2002e-08, 'scale': 5.4442e-06*1.4142, 'count':5.0}, 
    'qi': {'mean': 5.4701e-08, 'scale': 7.3361e-07*1.4142, 'count': 5.0}, 
    'qr': {'mean': 4.2354e-07, 'scale': 7.8923e-06*1.4142, 'count': 5.0}, 
    'qs': {'mean': 3.5349e-06, 'scale': 3.7933e-05*1.4142, 'count': 5.0},
    'qv': {'mean': 1.2345e-03, 'scale': 1.5455e-03*1.4142, 'count': 5.0}, 
    'tlay': {'mean': 2.4465e+02, 'scale': 1.9199e+01*1.4142, 'count': 5.0}, 
    'swuflx': {'mean': 73.5557, 'scale': 76.8572*1.4142, 'count': 5.0}, 
    'swdflx': {'mean': 276.5798, 'scale': 256.6423*1.4142, 'count': 5.0}, 
    'lwuflx': {'mean': 279.2660, 'scale': 43.9395*1.4142, 'count': 5.0}, 
    'lwdflx': {'mean': 104.7803, 'scale': 71.7479*1.4142, 'count': 5.0}}   
    
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

            