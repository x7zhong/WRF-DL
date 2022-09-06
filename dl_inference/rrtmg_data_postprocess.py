#postprocess rrtmg DL model outputs to get heating rate

import numpy as np

def unnormalized(results, norm_mapping, index_mapping):
    results_unnorm = np.zeros(results.shape)
    
    for index, value in index_mapping.items():
        results_unnorm[:, index, :] = results[:, index, :]*norm_mapping[index_mapping[index]]["scale"] + \
        norm_mapping[index_mapping[index]]["mean"]

    return results_unnorm

def get_heat_rate(predicts, pressure):
    
    swhr_predict = calculate_hr(
        predicts[:, 0:1, :], predicts[:, 1:2, :], pressure)
    
    # set last of the short wave heating rate to 0
    swhr_predict[:, :, -1] = 0.0

    lwhr_predict = calculate_hr(
        predicts[:, 2:3, :], predicts[:, 3:4, :], pressure)

    return swhr_predict, lwhr_predict

def calculate_hr(up, down, pressure):
    g = 9.8066  # m s^-2
    # reference to WRF/share/module_model_constants.F gas constant of dry air
    rgas = 287.0
    cp = 7.*rgas/2.
    heatfac = g*8.64*10**4/(cp*100)

    net = up - down
    net_delta = net - np.roll(net, 1, 2)
    p_delta = pressure - np.roll(pressure, 1, 2)
    
    return net_delta[:, :, 1::]/p_delta[:, :, 1::] * heatfac
    
def rrtmg_get_hr(out_onnx, auxiliary_feature, coszen):
    
    results = out_onnx[0]

#==============================================================================
# define the mean and std used for input data normalization
#==============================================================================
                
    norm_mapping = {
    'swuflx': {'mean': 73.5557, 'scale': 76.8572*1.4142, 'count': 5.0}, 
    'swdflx': {'mean': 276.5798, 'scale': 256.6423*1.4142, 'count': 5.0}, 
    'lwuflx': {'mean': 279.2660, 'scale': 43.9395*1.4142, 'count': 5.0}, 
    'lwdflx': {'mean': 104.7803, 'scale': 71.7479*1.4142, 'count': 5.0}}    
    
#==============================================================================
# denormalize predicted fluxes and calculate heating rate
#==============================================================================
             
    # denormalize, 
    index_mapping = {0: "swuflx", 1: "swdflx", 2: "lwuflx", 3: "lwdflx"}
    results_unnorm = unnormalized(results, norm_mapping, index_mapping)
        
    ''
    swuflx_predict = results_unnorm[:, 0, :].copy()
    swdflx_predict = results_unnorm[:, 1, :].copy()
    lwuflx_predict = results_unnorm[:, 2, :].copy()
    lwdflx_predict = results_unnorm[:, 3, :].copy()
    ''
        
    #calculate heat rate using fluxes and delta pressure
    swhr_predict, lwhr_predict = get_heat_rate(results_unnorm, auxiliary_feature)    
    
    swhr_predict = swhr_predict[:, 0, :]
    swhr_predict[coszen <= 0] = 0
    
    #convert heating rate from K/d to K/s    
#    swhr_predict = swhr_predict/86400
    
    lwhr_predict = lwhr_predict[:, 0, :]
#    lwhr_predict = lwhr_predict/86400
    
    return swhr_predict, lwhr_predict, swuflx_predict, swdflx_predict, lwuflx_predict, lwdflx_predict

            