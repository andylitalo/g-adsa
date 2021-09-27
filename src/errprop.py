# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 12:07:34 2019

@author: Andy
"""

import numpy as np


def error_D_exp(D_exp, b, s_b, v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref,
                s_v_drop_ref, diam_cruc=1.82, s_diam_cruc=0.05, skip_zero=False):
    """
    Estimates error in computation of the diffusivity computed using an
    exponential fit of the end of the adsorbed gas over time curve.
    diam_cruc is diameter of crucible for Rubotherm measurements [cm]. Default
    is 1.82 [cm].
    s_diam_cruc is the estimated uncertainty in the diameter of the crucible.
    Default is 0.05 [cm].
    """
    # if p = 0 should be skipped, cut off some arrays to make all same size
    # the rest should already be one element smaller than these
    if skip_zero:
        D_exp = D_exp[1:]
        b = b[1:]
        s_b = s_b[1:]
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    h_samp = v_samp/area_cruc # height of sample [cm]
    s_area_cruc = area_cruc*2*s_diam_cruc/diam_cruc # uncertainty in area [cm^s]
    # uncertainty in volume of sample [mL]
    s_v_samp = error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                            rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop,
                            v_drop_ref, s_v_drop_ref)
    # uncertainty in height of sample [cm]
    s_h_samp = h_samp*norm( (s_v_samp/v_samp, s_area_cruc/area_cruc))
    # uncertainty in diffusivty constant [cm^2/s]
    s_D_exp = D_exp*norm( (2*s_h_samp/h_samp, s_b/b) )

    return s_D_exp

def error_D_sqrt(D_sqrt, a, s_a, v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref,
                s_v_drop_ref, M_0, s_M_0, M_infty, s_M_infty, diam_cruc=1.82,
                s_diam_cruc=0.05):
    """
    Estimates error in computation of the diffusivity computed using an
    square-root fit of the beginning of the adsorbed-gas-over-time curve.
    diam_cruc is diameter of crucible for Rubotherm measurements [cm]. Default
    is 1.82 [cm].
    s_diam_cruc is the estimated uncertainty in the diameter of the crucible.
    Default is 0.05 [cm].
    """
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    h_samp = v_samp/area_cruc
    s_area_cruc = area_cruc*2*s_diam_cruc/diam_cruc
    s_v_samp = error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                            rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop,
                            v_drop_ref, s_v_drop_ref)
    s_h_samp = h_samp*norm( (s_v_samp/v_samp, s_area_cruc/area_cruc))
    s_D_sqrt = D_sqrt*norm( (2*s_h_samp/h_samp, 2*s_a/a, 2*s_M_0/(M_infty-M_0),
                            2*s_M_infty/(M_infty-M_0)) )

    return s_D_sqrt


def norm(elements):
    result = 0
    for i in range(len(elements)):
        result += (elements[i])**2
    result = np.sqrt(result)

    return result


def error_solubility(solubility, v_samp, w_buoy, w_gas_act, v_drop, s_v_drop, w_poly,
                   s_frac_rho_co2, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_samp_ref, v_drop_ref, s_v_drop_ref, v_ref,
                   s_v_ref, s_mp1, s_zero, s_w_gas_ref):
    """
    Estimates error in solubility measurement using error propagation.
    """
    n = len(solubility)
    # error in the sample volume
    s_v_samp = error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])

    # uncertainty in measurement of mass of gas
    s_w_gas_act = error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref)

    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )

    s_solubility = solubility*norm( (s_w_gas_act/w_gas_act, s_w_gas_act/ \
                                     norm((w_gas_act, w_poly)), s_w_poly/ \
                                     norm((w_gas_act, w_poly))) )

    return s_solubility


def error_spec_vol(spec_vol, v_samp, w_buoy, w_gas_act, v_drop, s_v_drop, w_poly,
                   s_frac_rho_co2, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_samp_ref, v_drop_ref, s_v_drop_ref, v_ref,
                   s_v_ref, s_mp1, s_zero, s_w_gas_ref):
    """
    Estimates error in specific volume measurement using error propagation.
    """
    n = len(spec_vol)
    # uncertainty in volume of sample [mL]
    s_v_samp = error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                              rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop,
                              v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])

    # uncertainty in measurement of mass of gas
    s_w_gas_act, s_w_buoy = error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref, \
                                            return_s_w_buoy=True)
    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )
    # uncertainty in specific volume [mL/g]
    s_spec_vol = spec_vol*norm( (s_v_samp/v_samp, s_w_gas_act/ \
                                 (w_gas_act + w_poly), s_w_poly/ \
                                 (w_gas_act + w_poly)) )

    return s_spec_vol


def error_spec_vol_stat_sys(spec_vol, v_samp, w_buoy, w_gas_act, v_drop, s_v_drop, w_poly,
                   s_frac_rho_co2, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_samp_ref, v_drop_ref, s_v_drop_ref, v_ref,
                   s_v_ref, s_mp1, s_zero, s_w_gas_ref):
    """
    Estimates error in specific volume measurement using error propagation,
    distinguishing statistical and systematic errors.
    """
    n = len(spec_vol)
    # uncertainty in volume of sample [mL]
    s_v_samp_stat, s_v_samp_sys = error_v_samp_stat_sys(v_samp, v_samp_ref,
                                w_samp_ref, s_w_samp_ref,
                              rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop,
                              v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])

    # uncertainty in measurement of mass of gas
    s_w_gas_act_stat, s_w_gas_act_sys = error_w_gas_act_stat_sys(s_br,
                                            s_br_0, w_buoy, s_frac_rho_co2,
                                            v_samp, s_v_samp_stat, s_v_samp_sys,
                                            v_ref, s_v_ref)
    # uncertainty in estimate of dry mass of polyol [g]
    s_w_poly = norm( (s_w_samp_ref, s_w_gas_ref) )
    # uncertainty in specific volume [mL/g]
    s_spec_vol_stat = spec_vol*norm( (s_v_samp_stat/v_samp, s_w_gas_act_stat/ \
                                 (w_gas_act + w_poly)) )
    s_spec_vol_sys = spec_vol*norm( (s_v_samp_sys/v_samp, s_w_gas_act_sys/ \
                                 (w_gas_act + w_poly), s_w_poly/ \
                                 (w_gas_act + w_poly)) )

    return s_spec_vol_stat, s_spec_vol_sys


def error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, s_v_drop_ref):
    """
    Uncertainty in the measurement of the volume of the sample [mL].
    """
    # uncertainty in volume of sample at reference point
    s_v_samp_ref = v_samp_ref*norm( (s_w_samp_ref/w_samp_ref, s_rho_samp_ref/ \
                                     rho_samp_ref) )
    # uncertainty in volume of sample [mL]
    s_v_samp = v_samp*norm( (s_v_drop/v_drop, s_v_samp_ref/v_samp_ref, \
                              s_v_drop_ref/v_drop_ref) )

    return s_v_samp


def error_v_samp_stat_sys(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                   s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, s_v_drop_ref):
    """
    Uncertainty in the measurement of the volume of the sample [mL], distinguishing
    statistical and systematic uncertainties.
    """
    # uncertainty in volume of sample at reference point
    s_v_samp_ref = v_samp_ref*norm( (s_w_samp_ref/w_samp_ref, s_rho_samp_ref/ \
                                     rho_samp_ref) )
    # statistical uncertainty in volume of sample [mL]
    s_v_samp_stat = v_samp*s_v_drop/v_drop
    # systematic uncertainty in volume of sample [mL]
    s_v_samp_sys = v_samp*norm( (s_v_samp_ref/v_samp_ref, \
                              s_v_drop_ref/v_drop_ref) )

    return s_v_samp_stat, s_v_samp_sys


def error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref, \
                                            return_s_w_buoy=False):
    """
    Estimates the error in the measurement of the actual gas weight.
    """
    # uncertainty in buoyancy correction of mass [g]
    s_w_buoy = w_buoy * norm( (s_frac_rho_co2, s_v_samp/norm((v_samp, v_ref)), \
                               s_v_ref/norm((v_samp, v_ref))) )
    s_w_gas_act = norm( (s_br, s_br_0, s_w_buoy) )

    if return_s_w_buoy:
        return s_w_gas_act, s_w_buoy
    else:
        return s_w_gas_act


def error_w_gas_act_stat_sys(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp_stat, s_v_samp_sys,
                                            v_ref, s_v_ref, \
                                            return_s_w_buoy=False):
    """
    Estimates the error in the measurement of the actual gas weight, distinguishing
    statistical and systematic errors.
    """
    # statistical uncertainty in buoyancy correction of mass [g]
    s_w_buoy_stat = w_buoy * norm( (s_frac_rho_co2, s_v_samp_stat/norm((v_samp, v_ref))) )
    # systematic uncertainty in buoyancy correction of mass [g]
    s_w_buoy_sys = w_buoy * norm( (s_v_samp_sys/norm((v_samp, v_ref)), \
                               s_v_ref/norm((v_samp, v_ref))) )
    # statistical uncertainty in the actual weight of the gas adsorbed [g]
    s_w_gas_act_stat = norm( (s_br, s_w_buoy_stat) )
    # systematic uncertainty in the actual weight of the gas adsorbed [g]
    s_w_gas_act_sys = norm( (s_br_0, s_w_buoy_sys) )

    if return_s_w_buoy:
        return s_w_gas_act_stat, s_w_gas_act_sys, s_w_buoy_stat, s_w_buoy_sys
    else:
        return s_w_gas_act_stat, s_w_gas_act_sys


def error_w_gas_act_helper(s_mp1, s_zero, w_buoy, s_frac_rho_co2, v_samp,
                           v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                           s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref,
                           s_v_drop_ref, v_ref, s_v_ref, return_s_w_buoy=False):
    """
    Performs some of the computations for the terms required by error_w_gas_act.
    """
    n = len(v_samp)
    # uncertainty in volume of sample [mL]
    s_v_samp = error_v_samp(v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref,
                              rho_samp_ref, s_rho_samp_ref, v_drop, s_v_drop,
                              v_drop_ref, s_v_drop_ref)
    # error in balance reading
    s_br = norm( (s_mp1, s_zero) )
    # extract first entry that is not a nan as error in balance reading at p=0
    s_br_0 = s_br[np.logical_not(np.isnan(s_br))][0]*np.ones([n])

    return error_w_gas_act(s_br, s_br_0, w_buoy, s_frac_rho_co2, \
                                            v_samp, s_v_samp, v_ref, s_v_ref, \
                                            return_s_w_buoy=return_s_w_buoy)
