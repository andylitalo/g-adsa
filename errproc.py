# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:34 2019

@author: Andy
"""

import numpy as np
#from numpy.linalg import norm

import dataproc


def norm(elements):
    result = 0
    for i in range(len(elements)):
        result += (elements[i])**2
    
    print(result)
    try:
        length = len(result)
        print(length)
        i_nan = np.isnan(result)
        print(i_nan)
        result[i_nan] = 0
    except:
        print('removed a nan.')
    
    print(result)
    return np.sqrt(result)

def error_spec_vol(spec_vol, v_samp, w_buoy, w_gas_act, w_poly, v_drop, s_v_drop, \
                   rho_co2, s_rho_co2, w_samp_ref, s_w_samp_ref, \
                   rho_samp_ref, s_rho_samp_ref, v_samp_ref, \
                   v_drop_ref, s_v_drop_ref, v_ref, s_v_ref, s_mp1, s_zero, \
                   s_w_gas_ref):
    """
    Estimates error in specific volume measurement using error propagation.
    """
    # uncertainty in volume of sample at reference point
    s_v_samp_ref = v_samp_ref*norm( (s_w_samp_ref/w_samp_ref, s_rho_samp_ref/ \
                                     rho_samp_ref) )
    # uncertainty in volume of sample [mL]
    s_v_samp = v_samp*norm( (s_v_drop/v_drop, s_v_samp_ref/v_samp_ref, \
                              s_v_drop_ref/v_drop_ref) )
    # uncertainty in buoyancy correction of mass [g]
    s_w_buoy = w_buoy * norm( (s_rho_co2/rho_co2, s_v_samp/(v_samp + v_ref), \
                               s_v_ref/(v_samp + v_ref)) )
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]
    # uncertainty in measurement of mass of gas
    s_w_gas_act = norm( (s_br, s_br_0, s_w_buoy) )
    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )
    # uncertainty in specific volume [mL/g]
    s_spec_vol = spec_vol*norm( (s_v_samp/v_samp, s_w_gas_act/w_gas_act, s_w_poly/w_poly) )
    
    return s_spec_vol