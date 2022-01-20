# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:09 2019

Library of functions for processing data from the gravimetry (Belsorp) and
Axisymmetric Drop-Shape Analysis (ADSA) measurements performed in Prof. Di Maio's
laboratory during the summer of 2019 at the University of Naples Federico II.
@author: Andy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os
import glob

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from timedate import TimeDate
import errprop
import plot


def compute_D_exp(i, diam_cruc, df, b, use_fit_vol=False):
    """
    Computes the diffusivity constant of gas in a liquid using an exponential
    fit of the last data points in the curve of adsorbed gas mass vs. time.
    See any of the Jupyter notebooks in the ANALYSIS>>g-adsa folder for the
    derivation of this formula.
    PARAMETERS:
        i : int
            Index of the current run
        diam_cruc : float
            Internal diameter of the crucible used to hold the liquid sample [cm].
        df : Pandas dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        b : float
            Fitting parameter from exponential fit (see exponential approach)
        use_fit_vol : bool, default=False
            If True, fitted volume will be used instead of raw volume.
    RETURNS:
        D_exp : float
            Estimate of diffusivity [cm^2/s]
    """
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    # get sample volume [mL]
    if use_fit_vol:
        v_samp = df['sample volume (fit) [mL]'].values[i]
    else:
        v_samp = df['sample volume [mL]'].values[i]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute diffusivity using formula derived from Crank 1956 eqn 10.168 [cm^2/s]
    D_exp = -(4*h_samp**2/np.pi**2)*b

    return D_exp


def compute_D_sqrt(i, a, t_mp1, w_gas_act, diam_cruc, df, use_fit_vol=False):
    """
    Computes the diffusivity constant of gas in a liquid using a square-root
    fit of the initial data points in the curve of adsorbed gas mass vs. time.
    See any of the Jupyter notebooks in the ANALYSIS>>g-adsa folder for the
    derivation of this formula.
    PARAMETERS:
        i : int
            Index of current pressure step in dataframe df
        a : float
            Fitting parameter from square-root fit (see any of the square_root_...
            functions in this document)
        t_mp1 : numpy array of floats
            Time of measuring point 1 (MP1) measurements [sec]
        w_gas_act : numpy array of floats
            True mass of gas adsorbed in the polyol [g]
        diam_cruc : float
            Internal diameter of the crucible used to hold the liquid sample [cm].
        df : Pandas dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        use_fit_vol : bool, default=False
            If True, fitted volume will be used instead of raw volume.
    RETURNS:
        D_sqrt : float
            Diffusivity constant of gas in liquid [cm^2/s]
    """
    # extract the starting mass
    M_0 = df['M_0 (prev) [g]'].iloc[i]
    # extract the saturation mass
    M_infty = df['M_infty (final) [g]'].iloc[i]
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    # sample volume [mL]
    if use_fit_vol:
        v_samp = df['sample volume (fit) [mL]'].values[i]
    else:
        v_samp = df['sample volume [mL]'].values[i]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute mean diffusivity using formula from Vrentas et al. 1977 (found in Pastore et al. 2011 as well) [cm^2/s]
    D_sqrt = np.pi*h_samp**2/4*(a/(M_infty-M_0))**2

    return D_sqrt

def compute_gas_mass(i, T, p_arr, p_set_arr, df, bp_arr, br_arr, br_eq_0, t_grav,
                     p_thresh_frac, last_bound, v_ref_he, get_inst_buoy=False,
                     v_samp_live=[], no_gaps=False, err_list=[], t_eq=-1,
                     use_fit_vol=False):
    """
    Computes the mass of gas adsorbed in grams for a given pressure step at each
    measurement of Measuring Point 1. Also returns the times ([sec]) and pressures
    ([kPa]) at each data point, the updated dataframe with the new calculations
    saved, and the index marking the end of the current pressure step.
    PARAMETERS:
        i : int
            Index of current pressure step in dataframe df
        T : float
            Temperature [K]
        p_arr : numpy array of floats
            Pressures measured by Belsorp software every 30 s
        p_set_arr : numpy array of floats
            (Approximate) set pressures for each pressure step in the order that
            they were undertaken.
        df : Pandas dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        bp_arr : numpy array of ints
            Position of the balance (1 = zero, 2 = MP1, 3 = MP2)
        br_arr : numpy array of floats
            Reading of magnetic suspension balance not corrected [g]
        br_eq_0 : float
            Reading of magnetic suspension balance under vacuum upon reaching
            equilibrium (often estimated outside this method) [g].
        t_grav : numpy array of floats
            Time of each data point measured by the gravimetry (Belsorp) system [s].
        p_thresh_frac : float
            Fraction of the average pressure beyond which the data points are
            considered to belong to a different pressure step or transition
            between pressure steps.
        last_bound : int
            Index of last measurement from previous pressure step
        v_ref_he : float
            Volume occupied by the magnetic suspension apparatus inside the
            pressure vessel, as measured by Maria Rosaria Di Caprio using
            Helium [mL].
        get_inst_buoy : bool, default=False
            If true, corrects for buoyancy using instantaneous pressure rather
            than the pressure at the end of the pressure step.
        v_samp_live : list of floats, default=[]
            If provided, should be a list of sample volume at each gravimetry
            (Belsorp) measurement [mL], which is used for buoyancy calculations
            at each point rather than just using the final sample volume.
        no_gaps : bool, default=False
            If true, include data points during transition from the end of the
            previous pressure step to the current pressure step.
        err_list : list of numpy arrays, default=[]
            If empty, uncertainty in the gas mass is not computed. If provided,
            contains all required uncertainties to compute the uncertainty in
            the gas mass (see usage below).
        t_eq : int, default=-1
            Time determined necessary to reach equilibrium. Units should be same
            as t_grav for accurate calculation. If t_eq > 0 given, then
            equilibrium values of buoyancy, dissolved gas balance reading, and
            dissolved gas actual mass will be recorded in the dataframe.
        use_fit_vol : bool, default=False
            If True, fitted volume will be used instead of raw volume.
        RETURNS:
            w_gas_act : numpy array of floats
                Mass of gas adsorbed in polyol at each time point in t_mp1 [g]
            t_mp1 : numpy array of floats
                Time of measurement of Measuring Point 1 (MP1) [s]
            df : Pandas Dataframe
                Dataframe with updated values for the current pressure step
            last_bound : int
                Index of the last measurement included in the current pressure step.
            p_mp1 : numpy array of floats
                Pressure measurement [kPa] at each time in t_mp1
            *if err_list provided, also returns numpy array of uncertainties in
            the gas mass (w_gas_act) [g] at the end
    """
    # initialize the result
    result = []
    # get current set pressure
    p_set = p_set_arr[i]
    # get indices of corresponding to the current pressure
    i_p0, i_p1 = get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=last_bound)
    # cut data points during change in pressure or have no gaps in data?
    if no_gaps:
        i_p0 = last_bound
    # Select data for current pressure step
    bp_select = bp_arr[i_p0:i_p1]
    br_select = br_arr[i_p0:i_p1]
    t_select = t_grav[i_p0:i_p1]
    p_select = p_arr[i_p0:i_p1]
    # extract mp1 measurements and corresponding times for the current pressure set point
    is_mp1 = (bp_select == 2)
    mp1 = medfilt(br_select[is_mp1], kernel_size=5) # medfilt removes spikes from unstable measurements
    t_mp1 = t_select[is_mp1]
    p_mp1 = p_select[is_mp1]
    # load initial parts of the result
    result = [t_mp1, df, i_p1]
    # estimate the mass of adsorbed gas
    zero = df['zero [g]'].values[i]
    br = mp1 - zero # balance reading (not corrected for buoyancy) [g]
    # subtract the balance reading under vacuum
    br_gas = br - br_eq_0
    # compute the buoyancy correction (approximate volume of sample by equilibrium value) [g]
    if get_inst_buoy:
        p_mp1 = p_select[is_mp1]
        if len(v_samp_live) > 0:
            v_samp_select = v_samp_live[i_p0:i_p1]
            v_samp = v_samp_select[is_mp1]
        elif use_fit_vol:
            v_samp = df['sample volume (fit) [mL]'].values[i]
        else:
            v_samp = df['sample volume [mL]'].values[i]
        # also return the pressure
        result += [p_mp1]

    elif use_fit_vol:
        v_samp = df['sample volume (fit) [mL]'].values[i]
    else:
        v_samp = df['sample volume [mL]'].values[i]
    # correct for buoyancy to get the true mass of the sample
    buoyancy = rho_co2(p_mp1, T)*(v_samp + v_ref_he)
    w_gas_act = br_gas + buoyancy
    # save equilibrium values if requested
    if t_eq > 0:
        # get indices over which to average to get equilibrium values
        inds_eq = get_inds_eq(t_mp1, t_eq)
        # equilibrium buoyancy correction [g]
        buoyancy_eq = np.mean(buoyancy[inds_eq])
        # equilibrium balance reading for dissolved gas [g]
        br_gas_eq = np.mean(br_gas[inds_eq])
        # equilibrium actual dissolved gas mass [g]
        w_gas_act_eq = br_gas_eq + buoyancy_eq
        # standard deviation of actual dissolved gas mass [g]
        s_w_gas_act_eq = np.std(br_gas[inds_eq] + buoyancy[inds_eq])
        # save results to dataframe
        df['dissolved gas balance reading [g]'].iloc[i] = br_gas_eq
        df['buoyancy correction [g]'].iloc[i] = buoyancy_eq
        df['actual weight of dissolved gas [g]'].iloc[i] = w_gas_act_eq
        df['actual weight of dissolved gas std [g]'].iloc[i] = s_w_gas_act_eq

    # add calculation to final result
    result = [w_gas_act] + result
    # compute the error in the gas mass
    if len(err_list) > 0:
        s_mp1, s_zero, w_buoy, s_frac_rho_co2, v_samp, \
        v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref, \
        s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref, \
        s_v_drop_ref, v_ref, s_v_ref = err_list
        result += [errprop.error_w_gas_act_helper(s_mp1, s_zero, w_buoy, s_frac_rho_co2, v_samp,
                           v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref,
                           s_rho_samp_ref, v_drop, s_v_drop, v_drop_ref,
                           s_v_drop_ref, v_ref, s_v_ref)]

    return tuple(result)


def compute_henrys_const(p, solub, spec_vol, p_thresh=500, mw=44.01, maxfev=10000,
                        s_solub=None, s_spec_vol=None, force_origin=True,
                        by_wt=False):
    """
    Computes Henry's constant using the solubility data at low pressures in the
    SI unit of [mol/(m^3.Pa)].

    Parameters
    ----------
    p : (N x 1) numpy array
        Pressure of equilibrium solubility measurements [kPa]
    solub : (N x 1) numpy array of floats
        Solubility of gas at equilibrium measurements [w/w]
    spec_vol : (N x 1) numpy array of floats
        Specific volume of sample at each equilibrium measurement [mL/g]
    p_thresh : float, default=500
        Upper bound on pressures to consider for fit [kPa]
    mw : float, default=44.01 (CO2)
        Molecular weight of gas in [g/mol]--default is for CO2
    maxfev : int, default=10000
        Maximum number of function evaluations for fitting.
    s_solub : (N x 1) numpy array of floats
        Uncertainty in the solubility of gas (solub) [w/w].
    s_spec_vol : (N x 1) numpy array of floats
        Uncertainty in specific volume of sample (spec_vol) [mL/g]
    force_origin: bool, default=True
        If True, fits data to y = a*x. O/w fits to y = a*x + b.
    by_wt : bool, optional. Default=False
        If True, reports Henry's constant as w/w.Pa

    Returns
    -------
    H : float
        Henry's constant in SI unit [mol/(m^3 Pa)]
    s_H : float (optional, only if uncertainty for solub and spec_vol provided)
    Uncertainty in Henry's constant [mol/(m^3 Pa)]
    """
    # multiplicative conversion factors
    m3_per_mL = 1E-6
    pa_per_kpa = 1E3
    # loads solubility of gas [w/w]
    if by_wt:
        c = solub
    # computes concentration of gas [mol/m^3]
    else:
        c = (solub/mw) / (spec_vol*m3_per_mL)
    # identify indices of measurements to use for linear fit
    inds_H = np.logical_and(p <= p_thresh, np.logical_and(np.logical_not(np.isnan(c)),
                                                          np.logical_not(np.isnan(p))))

    c_H = c[inds_H]
    p_H = p[inds_H]

    # if uncertainties provided
    if by_wt:
        s_c = s_solub[inds_H]
    elif s_solub is not None and s_spec_vol is not None:
        s_c = c*np.sqrt((s_solub/solub)**2 + (s_spec_vol/spec_vol)**2)
        s_c = s_c[inds_H]
    else:
        s_c = None
    if s_c is not None:
        if force_origin:
            # fits linear function through origin
            popt, pcov = curve_fit(slope, p_H*pa_per_kpa, c_H, sigma=s_c, maxfev=maxfev)
        else:
            # OR fits linear function with fitted intercept
            popt, pcov = np.polyfit(p_H*pa_per_kpa, c_H, 1, w=1/s_c, cov=True)
        H = popt[0]
        s_H = np.sqrt(pcov[0,0])

        if len(p_H) == 1:
            s_H = s_c[0]  / (p_H[0]*pa_per_kpa)
        result = H, s_H
    else:
        if force_origin:
            # fits linear function through origin
            popt, pcov = curve_fit(slope, p_H*pa_per_kpa, c_H, maxfev=maxfev)
            H = popt[0]
        else:
            # OR fits linear function with fitted intercept
            H, _ = np.polyfit(p_H*pa_per_kpa, c_H, 1)
        result = H

    return result


def compute_t_multiplier(metadata, i, t_p_interp, date_ref, time_ref):
    """
    Computes the multiplicative factor by which to multiply the time as
    measured by the Belsorp measurement to get the true time (since the Belsorp
    counting of time is slow). The factor is calculated by dividing known
    time of measurement of zero point (last measurement at a given pressure)
    by the reported final time on the Belsorp plot recorded at that measurement.
    PARAMETERS:
        metadata : Pandas Dataframe
            Dataframe created from .txt file of metadata for all Belsorp measurements.
        i : int
            Index of current pressure step (row in metadata dataframe)
        t_p_interp : numpy array
            Time recorded from Belsorp plot using Datathief and then interpolated
            to match the measurements of pressure.
        date_ref : string
            Date of reference time (beginning of experiment) 'mm/dd/yyyy'
        time_ref : string
            Time of reference time (beginning of experiment) 'hh:mm:ss'
    RETURNS:
        t_multiplier : float
            Multiplicative factor to multiply t_p_interp (Belsorp time) to get
            true time.
    """
    # Extract data and time of current pressure step measurement
    date_str = metadata['date'].iloc[i]
    time_str = metadata['time'].iloc[i]
    # convert to TimeDate object for computations
    time_date = TimeDate(date_str=date_str, time_str=time_str)
    # create TimeDate object for reference time (beginning of experiment)
    time_date_ref = TimeDate(date_str=date_ref, time_str=time_ref)
    # calculate actual time since start to end of current pressure step
    diff_min_act = TimeDate.diff_min(time_date_ref, time_date)
    # quick validation that last value of interpolated p is the maximum value
    assert t_p_interp[-1] == np.max(t_p_interp), 'last value of t_p_interp is not maximum.'
    # Measured time at the end of current pressure step as given by live Belsorp plot
    diff_min_meas = t_p_interp[-1]
    # Multiplicative factor is ratio of actual time to measured time
    t_multiplier = diff_min_act / diff_min_meas

    return t_multiplier


def compute_t_shift(metadata, i, t_p_interp, p_interp, p_file, date_ref,
                    time_ref, dpdt_thresh=10, plot_pressure=True):
    """
    ***OBSOLETE***: I determined that a multiplicative factor more accurately
    adjusts the Belsorp time than a time shift. See compute_t_multiplier.

    Computes number of minutes that the time of a manual experiment must be
    shifted to align with known recorded times.
    """
    dt = np.diff(t_p_interp)
    dp = np.diff(p_interp)
    dpdt = dp/dt
    if plot_pressure:
        plt.figure()
        plt.plot(t_p_interp, p_interp, '.')
        plt.title(p_file)
        plt.show()
    inds_p_change = np.where(np.abs(dpdt) > dpdt_thresh)[0]
    # if you don't have data before the start of the change in pressure
    if inds_p_change[0] == 0:
        # use the time from the end of the pressure step
        p_step = 1
        # mark end of pressure step by the first index at which pressure is not changing rapidly
        inds_not_change = [i not in inds_p_change for i in range(len(dpdt))]
        i_p_step = np.where(inds_not_change)[0][0]
    else:
        # use the time from the beginning of the pressure step
        p_step = 0
        # mark beginning of pressure step by the first index at which pressure is changing rapidly
        inds_change = [i in inds_p_change for i in range(len(dpdt))]
        i_p_step = np.where(inds_change)[0][0]
    t_p_step = t_p_interp[i_p_step]
    date_dp = metadata['date dp'].iloc[i]
    if p_step:
        time_dp = metadata['time dp end'].iloc[i]
    else:
        time_dp = metadata['time dp start'].iloc[i]
    time_date_ref = TimeDate(date_str=date_ref, time_str=time_ref)
    time_date_dp = TimeDate(date_str=date_dp, time_str=time_dp)
    t_since_ref = TimeDate.diff_min(time_date_ref, time_date_dp)
    t_shift = t_since_ref - t_p_step

    return t_shift


def concatenate_data(metadata, i, date_ref, time_ref, time_list, date_list,
                     t_grav, t_interp, p_interp, p_list, T_interp, T_list,
                     mp1_interp, br_list, bp_list, zero_last=True):
    """
    Concatenates data for the creation of an artificial TRD file in the case of
    manual experiments with G-ADSA.
    PARAMETERS:
        metadata : Pandas Dataframe
            Dataframe created from .txt file of metadata for all Belsorp measurements.
        i : int
            Index of current pressure step (row in metadata dataframe)
        date_ref : string
            Date of reference time (beginning of experiment) 'mm/dd/yyyy'
        time_ref : string
            Time of reference time (beginning of experiment) 'hh:mm:ss'
        time_list : list of strings
            List of times of measurements for current pressure step 'hh:mm:ss'
        date_list : list of strings
            List of dates of measurements for current pressure step 'mm/dd/yyyy'
        t_grav : numpy array of floats
            Times at which gravimetry data were recorded relative to start of
            gravimetry [min]
        t_interp : numpy array
            Time recorded from Belsorp plot using Datathief and then interpolated
            to match the measurements of pressure for current pressure step [min].
        p_interp : numpy array
            Pressure recorded from Belsorp plot using Datathief matching t_interp
        p_list : list of floats
            Running array of pressures up to the current pressure step [kPa],
            which will be extended to include the current pressure step in this
            method (p_interp).
        T_interp : numpy array
            Values of temperature [C] from Belsorp plot interpolated to match t_interp
        T_list : list of floats
            Running list of temperature measurements up to the current pressure
            step [C], which will be extended to include the temperature during
            the current pressure step in this method (T_interp).
        mp1_interp : numpy array
            Values of Measuring Point 1 (MP1) at the time points of t_interp [g].
        br_list : list of floats
            Running list of balance readings (MP1 measurements with one ZERO
            point measurement per pressure step) up to
            the current pressure step [g], which will be extended to include the
            MP1 values in mp1_interp for the current pressure step in this method.
        bp_list : list of ints
            Running of list of the balance position up to the current pressure
            step (1=ZERO point measurement and 2=MP1), which will be extended to
            include the current pressure step balance readings in this method.
        zero_last : bool, default=True
            If True, the ZERO point measurement recorded in the metadata dataframe
            is added at the end of this concatenation. Otherwise, it is added at
            the beginning.
    RETURNS: Nothing (void method, updates lists in place)
    """
    # initialize TimeDate object to store reference time of experiment
    time_date_ref = TimeDate(date_str=date_ref, time_str=time_ref)
    # add list of dates and times
    for j in range(len(t_interp)):
        time_date = TimeDate(date_str=date_ref, time_str=time_ref)
        time_date.add(minute=t_interp[j])
        time_list += [time_date.get_time_string()]
        date_list += [time_date.get_date_string()]
        del time_date

    # load metadata for zero point
    time_zero = metadata['time'].iloc[i]
    date_zero = metadata['date'].iloc[i]
    p_zero = metadata['p actual [kPa]'].iloc[i]
    T_zero = metadata['T [C]'].iloc[i]
    zero = metadata['zero [g]'].iloc[i]

    # if the zero-point measurement should be concatenated at the end
    if zero_last:
        # concatenate measuring point 1 values first
        concatenate_mp1(t_grav, t_interp, p_list, p_interp, T_list, T_interp,
                        br_list, mp1_interp, bp_list)
        # then add zero-point measurement
        concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_list, p_zero, T_list, T_zero, br_list,
                     zero, bp_list)
    # otherwise switch the order of concatenation
    else:
        concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_list, p_zero, T_list, T_zero, br_list,
                     zero, bp_list)
        concatenate_mp1(t_grav, t_interp, p_list, p_interp, T_list, T_interp,
                br_list, mp1_interp, bp_list)


def concatenate_mp1(t_grav, t_interp, p_list, p_interp, T_list, T_interp,
                br_list, mp1_interp, bp_list):
    """
    Concatenate data from measuring point 1 (MP1) measurement. Each measurement
    must also have a date, time, temperature, and balance position (2).
    PARAMETERS:
        t_grav : numpy array
            Times at which gravimetry data were recorded relative to start of
            gravimetry [min]
        t_interp : numpy array
            Times selected for interpolating Belsorp plot data extracted using
            Datathief [min].
        p_list : list of floats
            Running list of pressures up to current pressure step at times
            measured in t_grav that will be concatenated with this method [kPa].
        p_interp : numpy array
            Pressures interpolated at the times in t_interp [kPa].
        T_list : list of floats
            Temperature measured at points in t_grav that will be concatenated
            with this method. [C]
        T_interp : numpy array
            Temperatures measured at times in t_interp that will be concatenated
            to T_list [C]
        br_list : list of floats
            Balance readings at times measured in t_grav [g]. Includes MP1 and
            ZERO point measurements.
        mp1_interp : numpy array
            Measuring Point 1 (MP1) measurements taken at the times in t_interp
            [g], which will be concatenated to br_list.
        bp_list : list of ints
            Balance position at each time in t_grav (1=ZERO and 2=MP1), which
            will be concatenated to in this method.
    RETURNS: Nothing (void method)
    """
    # concatenate to gravimetry time
    t_grav += list(t_interp)
    p_list += list(p_interp)
    T_list += list(T_interp)
    br_list += list(mp1_interp)
    # save a "2" indicating balance position 2 (MP1) measurement at each time point
    bp_list += list(2*np.ones([len(t_interp)]))


def concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_list, p_zero, T_list, T_zero, br_list,
                     zero, bp_list):
    """
    Concatenate data for zero point measurement as part of creation of TRD
    file for gravimetry measurements when the Belsorp program does not save it
    automatically.
    PARAMETERS:
        time_zero : string
            Time at which ZERO point measurement was made 'hh:mm:ss'
        date_zero : string
            Date at which ZERO point measurement was made 'mm/dd/yyyy'
        time_list : list of strings
            List of times at which measurements were made 'hh:mm:ss'
        date_list : list of strings
            List of dates at which measurements were made 'mm/dd/yyyy'
        t_grav : list of floats
            Times at which gravimetry measurements were made [min], which will
            be concatenated by this method
        time_date_ref : TimeDate object (see timedate.py)
            TimeDate object for beginning of experiment (reference time)
        p_list : list of floats
            Pressures recorded at times in t_grav [kPa], which will be concatenated
            by this method.
        p_zero : float
            Pressure at ZERO point measurement [kPa].
        T_list : list of floats
            Temperature measurements at times in t_grav [C].
        T_zero : float
            Temperature at ZERO point measurement [C].
        br_list : list of floats
            Balance reading at times in t_grav, including ZERO point and MP1 [g].
        zero : float
            ZERO point measurement [g].
        bp_list : list of ints
            Balance positions at times in t_grav (1=ZERO, 2=MP1)
    RETURNS: Nothing (void method)
    """
    time_date_zero = TimeDate(date_str=date_zero, time_str=time_zero)
    time_list += [time_zero]
    date_list += [date_zero]
    t_grav += [TimeDate.diff_min(time_date_ref, time_date_zero)]
    p_list += [p_zero]
    T_list += [T_zero]
    br_list += [zero]
    bp_list += [1]


def convert_time(date, time):
    """
    Converts the given date and time into an array of times [s] starting from
    zero.
    INPUTS:
        date : array of strings
            Format of strings is mm-dd-yyyy
        time : array of strings
            Format of strings is hh:mm:ss

    OUTPUTS:
        time_passed : array of ints
            Time since start
    """
    # Determine the character used to separate the values in the date entries
    if '/' in date[0]:
        d_sep = '/'
    elif '-' in date[0]:
        d_sep = '-'
    else:
        print('date in unrecognized format by dataproc.convert_time.')

    # Extract the various units of time
    mo = np.array([int(d[:d.find(d_sep)]) for d in date])
    dy = np.array([int(d[d.find(d_sep)+1:d.find(d_sep)+3]) for d in date])
    hr = np.array([int(t[:t.find(':')]) for t in time])
    mi = np.array([int(t[t.find(':')+1:t.find(':')+3]) for t in time])
    sc = np.array([int(t[t.find(':')+4:]) for t in time])

    # Shift each time unit by its initial value
    dy -= dy[0]
    mo -= mo[0]
    hr -= hr[0]
    mi -= mi[0]
    sc -= sc[0]
    # compute difference in time [s]
    time_passed = 24*3600*dy + 3600*hr + 60*mi + sc
    # TODO: fix to adjust with the changing of the month!!!***This can be done
    # more easily with TimeDate objects
    return time_passed


def diffusivity_exp(p_set_arr, thresh0_arr, thresh1_arr, t_grav, polyol, T,
                    p_arr, df, bp_arr, br_arr, br_eq_0, p_thresh_frac, v_ref_he,
                    diam_cruc, err_list, show_plots=True, skip_zero=False):
    """
    Defines routine to compute the diffusivity using an exponential fit for the
    last part of the data (region defined by given thresholds) for all pressure
    steps requested. Also shows plots of the fits if requested.
    PARAMETERS:
        p_set_arr : (N x 1) numpy array
            Approximate set values for pressure at each step in chronological order [kPa].
        thresh0_arr : (N x 1) numpy array
            Array of thresholds for the early-time boundary on the fit.
            The thresholds are fractions of the normalized adsorbed gas mass.
        thresh1_arr : (N x 1) numpy array
            Array of thresholds for the late-time boundary on the fit.
            The thresholds are fractions of the normalized adsorbed gas mass.
        t_grav : (M x 1) numpy array
            Time of gravimetry measurements [s].
        polyol : string
            Code name of polyol (for plots).
        T : float
            Set value of temperature for the experiment [C] (for plots).
        p_arr : (M x 1) numpy array
            Pressure measurements at each time in t_grav [kPa].
        df : Pandas Dataframe (N rows)
            Dataframe of processed data for current experiment, where each row
            corresponds to one pressure step
        bp_arr : (M x 1) numpy array
            Balance position at each time in t_grav (1=ZERO, 2=MP1, 3=MP2)
        br_arr : (M x 1) numpy array
            Balance reading at each time in t_grav [g].
        br_eq_0 : float
            Reading of magnetic suspension balance under vacuum upon reaching
            equilibrium (often estimated outside this method) [g].
        p_thresh_frac : float
            Threshold within which pressure is considered at the set point
        v_ref_he : float
            Volume occupied by the magnetic suspension apparatus inside the
            pressure vessel, as measured by Maria Rosaria Di Caprio using
            Helium [mL].
        diam_cruc : float
            Diameter of crucible used to hold polyol sample under magnetic
            suspension balance [cm].
        err_list : list of (N x 1) numpy arrays
            List of uncertainties used to compute uncertainty in the diffusivity
            constant.
        show_plots : bool, default=True
            If True, plots of adsorbed gas vs. time on log-log axes with the
            t^1/2 fit will be shown for each pressure step.
        skip_zero : bool, default=False
            If True, p = 0 will be skipped in computing uncertainties.
    RETURNS:
        D_exp_arr : (N x 1) numpy array
            Diffusivity constant [cm^2/s] computed using exponential fit of
            late-time data at each of the N pressure steps.
        s_D_exp_arr : (N x 1) numpy array
            Uncertainty in D_exp_arr [cm^2/s].
        M_infty_arr : (N x 1) numpy array
            Mass of adsorbed gas for each pressure step extrapolated to t-->
            infinity using exponential fit computed in this method.
        tau_arr : (N x 1) numpy array
            Diffusion times tau [s], negative reciprocal of "b" parameter in
            exponential fit.
    """
    # initialize marker for pressure bounds
    last_bound = 0
    # initialize array to store diffusivity values
    D_exp_arr = np.zeros([len(p_set_arr)])
    # initialize array of extrapolated mass at t --> infinity
    M_infty_arr = np.zeros([len(p_set_arr)])
    # initialize array to store fitted exponential time constants [s]
    tau_arr = np.zeros([len(p_set_arr)])
    # initialize arrays to store the "b" parameter from the fitting and its uncertainty
    b_arr = np.zeros([len(p_set_arr)])
    s_b_arr = np.zeros([len(p_set_arr)])
    # Loop through each pressure set point
    for i in range(len(p_set_arr)):
        p_set = p_set_arr[i]
        print('Pressure = %d kPa.' % p_set)
        # determine if gas is adsorbing into polyol or desorbing out
        is_adsorbing = (i <= np.argmax(p_set_arr)) and p_set_arr[i] != 0 and p_set_arr[i] != 2 # last condition is for 20190725_0801_1k3f_60c.ipynb
        # compute actual mass of gas at the times corresponding to the current pressure, save data in data frame
        w_gas_act, t_mp1, df, last_bound, p_mp1 = compute_gas_mass(i, T, p_arr, p_set_arr, df, bp_arr, br_arr, br_eq_0,
                                                                    t_grav, p_thresh_frac, last_bound, v_ref_he,
                                                                    get_inst_buoy=True)

        # skip analysis if there are "nan"s in the data
        if (np.isnan(w_gas_act)).any():
            continue

        # normalize
        w_gas_norm = normalize_gas_mass(w_gas_act, is_adsorbing)
        # Threshold normalized adsorbed mass values to define fitting boundaries
        thresh0 = thresh0_arr[i]
        thresh1 = thresh1_arr[i]
        i0 = int(np.where(w_gas_norm > thresh0)[0][-1])
        i1 = int(np.where(w_gas_norm > thresh1)[0][-1])
        # Perform exponential fit on points between indices marking the thresolds
        D_exp, a, b, c, s_b, t_fit, \
        w_gas_2_plot, w_fit_2_plot = fit_exp_diffusivity(i, t_mp1, w_gas_act, i0,
                                                            i1, diam_cruc, df)
        # store results, starting with diffusivity constant [cm^2/s]
        D_exp_arr[i] = D_exp
        print('D_exp = %.2e cm^2/s.' % D_exp)
        # exponential constant and uncertainty
        b_arr[i] = b
        s_b_arr[i] = s_b
        # mass extrapolated at infinite time using exponential fit [g]
        M_infty_arr[i] = c
        # fitted exponential time constant [s]
        tau_arr[i] = -1/b

        # Plot results if desired: scatterplot of data with exponential fit
        if show_plots:
            plot.diffusivity_exp(i, p_set_arr, t_mp1, t_fit, i0, w_gas_2_plot,
                                w_fit_2_plot, a, b, c, polyol, T, is_adsorbing)

    # Extract parameters from error list
    v_samp, v_samp_ref, w_samp_ref, s_w_samp_ref, rho_samp_ref, \
    s_rho_samp_ref, v_drop_eq, s_v_drop, v_drop_ref, s_v_drop_ref = err_list
    # estimate uncertainty in the diffusivity constant calculated using the exponential fit
    s_D_exp_arr = errprop.error_D_exp(D_exp_arr, b_arr, s_b_arr, v_samp,
                                        v_samp_ref, w_samp_ref, s_w_samp_ref,
                                        rho_samp_ref, s_rho_samp_ref, v_drop_eq,
                                        s_v_drop, v_drop_ref, s_v_drop_ref, skip_zero=skip_zero)
    # add 0 uncertainty at front if p = 0 was skipped to match array sizes
    if skip_zero:
        s_D_exp_arr = np.concatenate((np.zeros([1]), s_D_exp_arr))

    return D_exp_arr, s_D_exp_arr, M_infty_arr, tau_arr


def diffusivity_sqrt(p_set_arr, n_pts_sqrt_arr, t_grav, polyol, T, p_arr, df,
    bp_arr, br_arr, br_eq_0, p_thresh_frac, v_ref_he, i_shift, maxfev,
    diam_cruc, fit_t0=False, fit_w0=False, show_plots=True):
    """
    Defines routine to compute the diffusivity using a t^1/2 fit for the
    beginning part of the data provided for all pressure steps requested. Can
    show plots of the fits if requested.
    PARAMETERS:
        p_set_arr : numpy array
            Approximate set values for pressure at each step in chronological order [kPa].
        n_pts_sqrt_arr : numpy array
            Array of number of points to use for t^1/2 fit at each pressure step,
            so it should be the same length as p_set_arr.
        t_grav : numpy array
            Time of gravimetry measurements [s].
        polyol : string
            Code name of polyol (for plots).
        T : float
            Set value of temperature for the experiment [C] (for plots).
        p_arr : numpy array
            Pressure measurements at each time in t_grav [kPa].
        df : Pandas Dataframe
            Dataframe of processed data for current experiment, where each row
            corresponds to one pressure step
        bp_arr : numpy array
            Balance position at each time in t_grav (1=ZERO, 2=MP1, 3=MP2)
        br_arr : numpy array
            Balance reading at each time in t_grav [g].
        br_eq_0 : float
            Reading of magnetic suspension balance under vacuum upon reaching
            equilibrium (often estimated outside this method) [g].
        p_thresh_frac : float
            Threshold within which pressure is considered at the set point
        v_ref_he : float
            Volume occupied by the magnetic suspension apparatus inside the
            pressure vessel, as measured by Maria Rosaria Di Caprio using
            Helium [mL].
        i_shift : numpy array
            Indices of first point to use for fitting for each pressure step.
            The value of n_pts_sqrt_arr at the same index determines the number
            of subsequent points to use for the fitting.
        maxfev : int
            Maximum number of function evaluations for fitting method.
        diam_cruc : float
            Diameter of crucible used to hold polyol sample under magnetic
            suspension balance [cm].
        fit_t0 : bool, default=False
            If True, the initial time t0 will be a fitting parameter. Otherwise,
            it will be fixed by the first time in the pressure step.
        fit_w0 : bool, default=False
            If True, the initial mass of adsorbed gas w0 will be a fitting
            parameter. Otherwise, it will be fixed as the mass at the end
            of the previous experiment.
        show_plots : bool, default=True
            If True, plots of adsorbed gas vs. time on log-log axes with the
            t^1/2 fit will be shown for each pressure step.
    RETURNS:
        D_sqrt_arr : numpy array
            Diffusivity constant [cm^2/s] computed using t^1/2 fit of initial
            data at each pressure step
        M_0_extrap_arr : numpy array
            Initial mass of adsorbed gas estimated by extrapolating t^1/2 fit
            back to the initial time of each pressure step [g].
        a_arr : numpy array
            Array of the values of the fitting paramter "a" from the t^1/2 fit.
        s_a_arr : numpy array
            Array of the uncertainty in the fitting parameter "a."
    """
    # initialize marker for pressure bounds
    last_bound = 0
    # initialize array to store diffusivity values
    D_sqrt_arr = np.zeros([len(p_set_arr)])
    # initialize array to store initial mass M_0 extrapolated with t^1/2 fit
    M_0_extrap_arr = np.zeros([len(p_set_arr)])
    # initialize arrays to store "a" parameter from fitting and uncertainty
    a_arr = np.zeros([len(p_set_arr)])
    s_a_arr = np.zeros([len(p_set_arr)])
    # Loop through each pressure set point
    for i in range(len(p_set_arr)):
        p_set = p_set_arr[i]
        n_pts_sqrt = n_pts_sqrt_arr[i]
        print('Pressure = %d kPa.' % p_set)
        # if we are not fitting the initial time, define it as the point where
        # the pressure began changing after the previous sorption test
        t0 = None if fit_t0 else t_grav[last_bound]
        # if we are not fitting initial mass, define it as mass at end of
        # previous pressure step
        w0 = None if fit_w0 else df['M_0 (prev) [g]'].iloc[i]

        # compute actual mass of gas at the times corresponding to the current pressure, save data in data frame
        w_gas_act, t_mp1, df, last_bound, p_mp1 = compute_gas_mass(i, T, p_arr,
                                                    p_set_arr, df, bp_arr, br_arr,
                                                    br_eq_0, t_grav, p_thresh_frac,
                                                    last_bound, v_ref_he,
                                                    get_inst_buoy=True)
        # additional cutting off of data for t^1/2 fit
        t_mp1 = t_mp1[i_shift[i]:]
        w_gas_act = w_gas_act[i_shift[i]:]
        # skip analysis if there are "nan"s in the data
        if (np.isnan(w_gas_act)).any():
            continue

        # fit initial data points to a square root curve per eqn 10.165 in Crank (1956) "The Mathematics of Diffusion"
        try:
            D_sqrt_arr[i], M_0_extrap_arr[i], t_fit, w_fit, a_arr[i], \
            s_a_arr[i], t0, w0 = fit_sqrt_diffusivity(i, t_mp1, w_gas_act, n_pts_sqrt,
                                                        diam_cruc, df, t0=t0,
                                                        w0=w0, maxfev=maxfev)
        except:
            print("Square-root fit could not converge.")
            continue

        # plot data translated such that first point is 0,0 and data increases (so t^1/2 looks like a straight line on log-log)
        if show_plots:
            plot.diffusivity_sqrt(i, p_set_arr, t_mp1, t_fit, t0, w_gas_act, w_fit, w0, a_arr[i], polyol, T)

    return D_sqrt_arr, M_0_extrap_arr, a_arr, s_a_arr


def exponential_approach(x, a, b, c):
    """Function used for fitting a curve to an exponential."""
    return a*np.exp(b*x) + c


def extrapolate_equilibrium(t, m, maxfev=800, p0=(-0.01, -1E-4, 0.01)):
    """Extrapolate mass over time with exponential fit to estimate equilibrium mass."""
    popt, pcov = curve_fit(exponential_approach, t, m, maxfev=maxfev, p0=p0)
    a, b, c = popt
    # equilibrium mass is vertical shift parameter [g]
    m_eq = c

    return m_eq


def fit_exp_diffusivity(i, t_mp1, w_gas_act, i0, i1, diam_cruc, df, maxfev=10000,
                        s_w_gas_act=1E-5):
    """
    Fits data for square-root diffusivity estimation method in "diffusivity_sqrt()"
    for a given pressure step.
    PARAMETERS:
        i : int
            Index of current pressure step.
        t_mp1 : (M x 1) numpy array
            Times of MP1 measurements for current pressure step [s].
        w_gas_act : (M x 1) numpy array
            Mass of adsorbed gas corrected for buoyancy [g] at each time in t_mp1.
        i0 : int
            Index of first point to be included in exponential fit.
        i1 : int
            Index of second point to be included in exponential fit.
        diam_cruc : float
            Diameter of crucible used to hold polyol sample under magnetic
            suspension balance [cm].
        df : Pandas Dataframe
            Dataframe of processed data where each row is one pressure step.
        maxfev : int, default=1E4
            Maximum function evaluations for fitting method.
        s_w_gas_act : float, default=1E-5
            Uncertainty in measurement of adsorbed gas mass.
    RETURNS:
        D_exp : float
            Diffusivity constant computed using exponential fit of last part of
            adsorbed gas vs. time plot [cm^2/s].
        a, b, c : floata
            Fitting parameters from exponential_approach().
        s_b : float
            Uncertainty in the "b" fitting parameter.
        t_fit : (P x 1) numpy array
            Evenly spaced times between beginning and ending time points of
            exponential fit [s].
        w_gas_2_plot : (N x 1) numpy array
            Normalized adsorbed gas values at each value of t_mp1 (1 to 0).
        w_fit_2_plot : (P x 1) numpy array
            Normalized adsorbed gas values predicted by the exponential fit at
            each value of t_fit (1 to 0).
    """
    # fit initial data points to an exponential curve per eqn 10.168 in Crank (1956) "The Mathematics of Diffusion"
    # shift time to start at t = 0
    popt, pcov = curve_fit(exponential_approach, t_mp1[i0:i1]-t_mp1[i0],
                           w_gas_act[i0:i1], p0=(-0.01, -0.01, 0.1), maxfev=maxfev,
                           sigma=s_w_gas_act*np.ones([i1-i0]), absolute_sigma=True)
    a, b, c = popt
    s_b = np.sqrt(pcov[1,1])
    # Compute diffusivity constant [cm^2/s]
    D_exp = compute_D_exp(i, diam_cruc, df, b)
    # generate data points for exponential fit
    t_fit = np.linspace(t_mp1[i0], t_mp1[i1], 100) - t_mp1[i0]
    w_gas_fit = exponential_approach(t_fit, a, b, c)
    # plot the result to examine the fit
    normalization = 1 / np.max(np.abs(c - w_gas_act))
    w_gas_2_plot = normalization*np.abs(c - w_gas_act)
    w_fit_2_plot = normalization*np.abs(c - w_gas_fit)

    return D_exp, a, b, c, s_b, t_fit, w_gas_2_plot, w_fit_2_plot


def fit_sqrt_diffusivity(i, t_mp1, w_gas_act, n_pts_sqrt,  diam_cruc,
                        df, t0=None, w0=None, maxfev=10000):
    """
    Fits data for square-root diffusivity estimation method in "diffusivity_sqrt()"
    for a given pressure step.
    PARAMETERS:
        i : int
            Index of current pressure step.
        t_mp1 : numpy array
            Times of MP1 measurements for current pressure step [s].
        w_gas_act : numpy array
            Mass of adsorbed gas corrected for buoyancy [g] at each time in t_mp1.
        n_pts_sqrt : int
            Number of points to use for t^1/2 fit.
        diam_cruc : float
            Diameter of crucible used to hold polyol sample under magnetic
            suspension balance [cm].
        df : Pandas Dataframe
            Dataframe of processed data where each row is one pressure step.
        t0 : float, default=None
            If provided, the initial time parameter "t0" will be fixed at the
            given value. Otherwise, it will be a fitting parameter [s].
        w0 : float, default=None
            If provided, the initial mass of adsorbed gas parameter "w0" will be
            fixed at the given value. Otherwise, it will be a fitting parameter [g].
        maxfev : int, default=10000
            Maximum function evaluations for fitting method.
    RETURNS:
        D_sqrt : float
            Diffusivity constant [cm^2/s] estimated using t^1/2 fit.
        M_0_extrap : float
            Mass of adsorbed gas at beginning of pressure step estimated by
            extrapolating t^1/2 fit back to initial time [g].
        t_fit : numpy array
            Times at which t^1/2 fit was computed in w_fit [s].
        w_fit : numpy array
            Mass of adsorbed gas estimated by t^1/2 fit [g].
        a : float
            Fitting parameter of t^1/2 fit.
        s_a : float
            Uncertainty in the "a" parameter.
        t0 : float
            Initial time used in t^1/2 fit [s].
        w0 : float
            Initial mass used in t^1/2 fit [g].
    """
    # generate data points for t^(1/2) fit
    n = min(n_pts_sqrt, len(t_mp1)-1)
    t_fit = np.linspace(t_mp1[0], t_mp1[n-1], 100)
    # fit initial time and mass
    if not t0 and not w0:
        popt, pcov = curve_fit(square_root_3param, t_mp1[:n], w_gas_act[:n],
                                maxfev=maxfev, sigma=1E-5*np.ones([n]), absolute_sigma=True)
        a, w0, t0 = popt
        # uncertainty in "a" parameter
        s_a = np.sqrt(pcov[0,0])
        # Also save the "initial" mass extrapolated to the beginning of the change in pressure
        M_0_extrap = square_root_3param(max(t_fit[0], t0), a, w0, t0)
        # estimate mass of adsorbed gas using fit
        w_fit = square_root_3param(t_fit, a, w0, t0)
    # just fit initial time
    elif not t0 and w0:
        popt, pcov = curve_fit(square_root_2param_t0_fit, t_mp1[:n], w_gas_act[:n]-w0,
                                maxfev=maxfev, sigma=1E-5*np.ones([n]), absolute_sigma=True)
        a, t0 = popt
        # uncertainty in "a" parameter
        s_a = np.sqrt(pcov[0,0])
        # Also save the "initial" mass extrapolated to the beginning of the change in pressure
        M_0_extrap = square_root_2param_t0_fit(max(t_mp1[0], t0), a, t0)
        # estimate mass of adsorbed gas using fit
        w_fit = square_root_2param_t0_fit(t_fit, a, t0)+w0
    # just fit initial mass
    elif t0 and not w0:
        popt, pcov = curve_fit(square_root_2param, t_mp1[:n]-t0, w_gas_act[:n],
                                maxfev=maxfev, sigma=1E-5*np.ones([n]), absolute_sigma=True)
        a, w0 = popt
        # uncertainty in "a" parameter
        s_a = np.sqrt(pcov[0,0])
        # Also save the "initial" mass extrapolated to the beginning of the change in pressure
        M_0_extrap = w0
        # estimate mass of adsorbed gas using fit
        w_fit = square_root_2param(t_fit-t0, a, w0)
    # fit neither initial mass nor initial time
    else:
        popt, pcov = curve_fit(square_root_1param, t_mp1[:n]-t0, w_gas_act[:n]-w0,
                                maxfev=maxfev, sigma=1E-5*np.ones([n]), absolute_sigma=True)
        a = popt[0]
        # uncertainty in "a" parameter
        s_a = np.sqrt(pcov[0,0])
        # generate data points for t^(1/2) fit
        M_0_extrap = w0
        # estimate mass of adsorbed gas using fit
        w_fit = square_root_1param(t_fit-t0, a) + w0

    # compute mean diffusion coefficient with the squareroot method by fitting and exponential curve to get the equilibrium mass
    D_sqrt = compute_D_sqrt(i, a, t_mp1, w_gas_act, diam_cruc, df)
    print('D_sqrt = %.2e cm^2/s.' % D_sqrt)

    return D_sqrt, M_0_extrap, t_fit, w_fit, a, s_a, t0, w0


def fit_v_drop(p, v_drop, s_v_drop):
    """
    Fits volume of drop [uL] and provides uncertainty.
    PARAMETERS:
        p : (N x 1) numpy array of floats
            Pressures [kPa].
        v_drop : (N x 1) numpy array of floats
            Volume of pendant drop for ADSA measurements [uL].
        s_v_drop : (N x 1) numpy array of floats
            Uncertainty in v_drop [uL].
    RETURNS:
        v_drop_fit : (N x 1) numpy array of floats
            Fitted volume of drop [uL].
        s_v_drop_fit : (N x 1) numpy array of floats
            Uncertainty in v_drop_fit.
    """
        # remove nans to leave only data formatted appropriate for fitting methods
    _, p_, v_drop_, s_v_drop_ = remove_nan_entries(s_v_drop, [p, v_drop, s_v_drop])

    # fit drop volume vs. pressure with quadratic (2nd-degree) fit
    coeffs, Cov = np.polyfit(p_, v_drop_, 2, w=1/s_v_drop_, cov=True)
    a, b, c = coeffs
    # compute fitted values of drop volume [uL]
    v_drop_fit = v_drop_model(p, a, b, c)
    # compute rms error, excluding nan entries
    res = np.sum((v_drop_model(p_, a, b, c) - v_drop_)**2)
    rms_err = np.sqrt(res)/len(p_)
    # define error as max of rms and standard mean standard deviation
    # where uncertainty is not available--o/w assume same error as raw data
    err = max(rms_err, np.mean(s_v_drop))
    s_v_drop_fit = err*np.ones([len(v_drop_fit)])
    s_v_drop_fit[:len(s_v_drop)] = s_v_drop

    return v_drop_fit, s_v_drop_fit


def get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=0,
                        window_reduction=0.25, min_window=5):
    """
    Returns the indices of the data arrays corresponding to the current pressure
    step (p_set).
    Parameters :
        p_arr : array of floats
            Array of pressures measured by Rubotherm
        p_set : int
            Current pressure step (set point)
        p_thresh_frac : float
            Threshold within which pressure is considered at the set point
        last_bound : int
            index of upper bound for indices of the last pressure set point
            useful for distinguishing pressures from adsorption and desorption
        small_window_reduction : float
            fraction by which p_thresh_frac is reduced for smaller window after
            averaging p in current window, used to remove outliers
        min_window : int
            minimum width of pressure window [kPa]

    Returns :
        i0, i1 : ints
            Indices corresponding to the first and last data points for pressure set point.
    """
    # remove data from previous pressures
    p_arr = p_arr[last_bound:]

    # get indices of pressures within threshold of set point
    # (for nonzero window at p = 0, at one kPa)
    in_window = np.abs(p_set - p_arr) <= p_thresh_frac*p_set + min_window
    # identify bounds of regions with correct pressure
    bit_boundaries = np.logical_xor(in_window[1:], in_window[:-1])
    possible_bounds = np.where(bit_boundaries)[0] + 1 # + 1 to make up for shifting
    # select the first pair of bounds
    # this prevents selection of data points from desorption step if it has the
    # same set pressure
    # If we are selecting the first pressure window, the bound should be the end
    if last_bound == 0 and len(possible_bounds) == 1:
        i0_coarse = 0
        i1_coarse = possible_bounds[0]
    # otherwise, the bounds should mark the window
    else:
        i0_coarse = possible_bounds[0]
        if len(possible_bounds) > 1:
            i1_coarse = possible_bounds[1]
        # if you reach the end of the data set, such that the pressure ends on the
        # current pressure, append last index
        else:
            i1_coarse = len(p_arr) - 1
    # average all the points in the current pressure step
    # because there are so many data points, the outliers will be "drowned out."
    # Then take the points that are within a smaller threshold of the average
    p_curr = p_arr[i0_coarse:i1_coarse]
    p_mean = np.mean(p_curr)
    inds_small_window = np.where(np.abs(p_mean-p_curr) <=
                                 p_thresh_frac*p_set*window_reduction + min_window)[0]
    i0_small_window = inds_small_window[0]
    i1_small_window = inds_small_window[-1]
    _, inds_accepted = reject_outliers(np.abs(np.diff(p_curr[i0_small_window:i1_small_window])),
                                            return_inds=True)
    # because we cut the pressure array multiple times, the indices are shifted
    # down, so we must shift them back up to get the true indices for the current pressure
    offset = last_bound + i0_coarse + i0_small_window
    i0 = inds_accepted[0] + offset
    i1 = inds_accepted[-1] + offset

    return i0, i1


def get_inds_adsa(t_adsa, t_grav, i_p0, i_p1, n_adsa, return_success=False):
    """
    Get the indices of the array of times of Axisymmetric Drop-Shape Analysis
    (ADSA) measurements to average for determining equilibrium values. These
    are the last indices before the final time for the given pressure step.
    PARAMETERS:
        t_adsa : numpy array
            Times at which ADSA measurements were made, measured from start of
            experiment [s] (about every 600 s).
        t_grav : numpy array
            Times at which gravimetry measurements were made, measured from start
            of experiment [s] (about every 30 s).
        i_p0 : int
            Index of beginning of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        i_p1 : int
            Index of end of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        n_adsa : int
            Number of points to average to estimate equilibrium ADSA measurements.
        return_success : bool, default=False
            If True, will return a bool indicating that there were at least
            n_adsa ADSA measurements, meaning a meaningful average can be taken.
    RETURNS:
        inds : numpy array
            Indices of t_adsa array of measurements to average for estimating
            equilibrium ADSA measurements at current pressure step.
        success : bool (only returned if return_success is True)
            True if at least n_adsa points were found in the pressure step window.
    """
    # extract data for current pressure
    t_select = t_grav[i_p0:i_p1]
    # identify the final time of gravimetry measurement
    t_i = t_select[0]
    t_f = t_select[-1]
    # get indices of last data points receding final time of gravimetry
    inds = np.where(t_adsa <= t_f)[0][-n_adsa:]
    # Are there at least as many points available as requested?
    if len(np.where(np.logical_and(t_adsa <= t_f, t_adsa >= t_i))[0]) < n_adsa:
        # if not, failed
        print('********not enough ADSA points at given pressure.******')
        success=False
    else:
        # if so succeeded
        success=True
    # Return success of finding indices if requested
    if return_success:
        return inds, success
    else:
        return inds


def get_inds_adsa_manual(time_date_ref, t_adsa, t_grav, metadata, i, n_minutes,
                         buffer=120):
    """
    Identical in purpose to get_inds_adsa but adapted for the case in which
    gravimetry data were manually extracted using Datathief.
    PARAMETERS:
        time_date_ref : TimeDate object (see timedate.py)
            Start point of experiment (reference time and date)
        t_adsa : numpy array
            Times of ADSA measurements measured relative to time_date_ref [min]
        t_grav : numpy array
            Times of gravimetry measurements extracted using Datathief [min].
        metadata : Pandas Dataframe
            Dataframe of metadata for each pressure step loaded from .txt file
        i : int
            Index of current pressure step (row of metadata to look at)
        n_minutes : int
            Number of minutes of measurements to average to compute equilibrium values.
        buffer : int, default=120
            Number of minutes at the end of an experiment to cut off (in case
            the synchronization of data wasn't accurate, we don't want to average
            measurements from the next pressure step).
    RETURNS:
        inds : numpy array
            Indices of t_adsa for the ADSA measurements to average to get the
            equilibrium values.
    """
    # Load date of change in pressure
    date = metadata['date dp'].iloc[i]
    # If the current pressure step is not the last pressure step, load normally
    if i < len(metadata['time dp start']) - 1:
        # End time of pressure step is marked by start of change in pressure for
        # next pressure step (i+1) [string 'hh:mm:ss']
        time_end = metadata['time dp start'].iloc[i+1]
        # create TimeDate object of end time of current pressure step
        time_date_end = TimeDate(date_str=date, time_str=time_end)
        # Compute time from beginning of experiment to end of current pressure step [min]
        t_min_end = TimeDate.diff_min(time_date_ref, time_date_end)
    else:
        # Otherwise, the end time of the current pressure step is the last time point [min].
        t_min_end = t_adsa[-1]
    # The indices to average are those within n_minutes before the buffer time
    inds =  np.where(np.logical_and(t_adsa > t_min_end - buffer - n_minutes,
                                   t_adsa < t_min_end - buffer))[0]

    return inds


def get_inds_eq(t_mp1, t_eq):
    """Returns indices to average over to get equilibrium value. t_mp1 and t_eq
    should have same units."""
    return np.where(t_mp1[-1] - t_mp1 <= t_eq)[0]


def get_mp1_interval(mp1, is_adsorbing, w_thresh=0.00005):
    """
    ***OBSOLETE***previously used to determine more precisely the window of
    measurements to include for plotting MP1.
    """
    # estimate the halfway point in the data set
    i_halfway = int(len(mp1)/2)
    # during adsorption, remove sections of decreasing weight measurements
    if is_adsorbing:
        i_start = np.where(-np.diff(mp1[:i_halfway]) > w_thresh)[0]
        i_end = np.where(-np.diff(mp1[i_halfway:]) > w_thresh)[0] + i_halfway
    # during desorption, remove sections of increasing weight measurements
    else:
        i_start = np.where(np.diff(mp1[:i_halfway]) > w_thresh)[0]
        i_end = np.where(np.diff(mp1[i_halfway:]) > w_thresh)[0] + i_halfway
    # check if any data points should be cut off from the beginning
    if len(i_start)==0:
        i_start = 0
    else:
        i_start = i_start[-1] + 2
    # check if any data points should be cut off from the end
    if len(i_end)==0:
        i_end = -1
    else:
        i_end = i_end[0]

    i_start += 1

    # cut one more point because the first point always seems off the fit
    return i_start, i_end


def get_T(file_name):
    """Returns the temperature based on a filename of the form *_<T>c*"""
    # get all indices were the character 'c' shows up and '_' shows up
    inds_c = get_all_inds(file_name, 'c')
    inds_ = get_all_inds(file_name, '_')
    # loop through indices for '_' to find where there is a c 3 places later
    # This will mark the "c" for "Celsius" coming after the two-digit temperature.
    for ind in inds_:
        if ind+3 in inds_c:
            # return temperature as int
            return int(file_name[ind+1:ind+3])
    # if not found, return nan
    return np.nan


def get_all_inds(string, substr):
    """Returns indices of all occurrences of a substring."""
    # initialize list of indices of occurrences with dummy entry
    inds = [0]
    # find first occurrence
    i_curr = string.find(substr)
    # while end of string has not been reached
    while i_curr != -1:
        # determine true index by adding spacing to last index
        inds += [i_curr + inds[-1]]
        # cut off string to current character because .find(substr) will only
        # return the first occurrence
        string = string[inds[-1]:]
        # find next occurrence in string
        i_curr = string.find(substr)
    # If no entries have been added, return 0 as sentinel
    if len(inds) == 1:
        result = 0
    # otherwise, cutoff dummy entry at first position before returning result
    else:
        result = inds[1:]

    return result


def interp(t, meas, dt=0.5, t_min=-1, t_max=-1):
    """
    Interpolates measurements from manually recorded data recorded over time.
    PARAMETERS:
        t : numpy array
            Independent variable (typically time)
        meas : numpy array
            Dependent variable (typically a measurement)
        dt : float, default=0.5
            Spacing in the dependent varaible "t" of interpolations that will be made.
        t_min : float, default=-1
            If not -1, will set the minimum value of "t" for interpolation.
        t_max : float, default=-1
            If not -1, will set the maximum value of "t" for interpolation.
    RETURNS:
        t_interp : numpy array
            Values of independent variable "t" at which interpolations were made.
        meas_interp : numpy array
            Interpolated values of dependent variable "meas."
    """
    # Remove repeated values of the independent variable
    t_uniq, i_uniq = np.unique(t, return_index=True)
    # Remove corresponding entries from dependent variable
    meas_uniq = meas[i_uniq]
    # Create linear interpolation function
    f_interp = interp1d(t_uniq, meas_uniq)
    # Set min and max interpolation values as bounds of given independent variable
    # if not provided
    if t_min == -1:
        t_min = t_uniq[0]
    if t_max == -1:
        t_max = t_uniq[-1]
    # create independent variable values for interpolation
    t_interp = np.arange(t_min, t_max, dt)
    # interpolate with linear interpolation
    meas_interp = f_interp(t_interp)

    return t_interp, meas_interp


def load_datathief_data(filepath):
    """Loads data exported from datathief."""
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    t = data[:,0]
    meas = data[:,1]

    return t, meas


def load_dft(folder):
    """
    Loads data from DFT calculations of interfacial
    tension into a dictionary with similar structure to
    that created by load_proc_data(). The DFT calculations
    are processed and reformatted first in extract_dft_if_tension.ipynb
    Jupyter notebook.
    Parameters
    ----------
    filepath : string
        path to folder of .csv files of DFT calculation results
    Returns
    -------
    d : dictionary
        dictionary of pressure and interfacial tension organized by experiment
    """
    # initializes dictionary of DFT data
    d = {}
    # grabs names of all DFT data files
    dft_filepath_list = glob.glob(os.path.join(folder + '*.csv'))
    # loops through each DFT data file and stores data in dictionary
    for filepath in dft_filepath_list:
        print('Analyzing {0:s}'.format(filepath))
        tag = filepath.split('\\')[-1][:-4] # extracts experiment tag from filepath
        d[tag] = {} # creates sub-dictionary named by tag, #k#f_#c
        # loads data
        df = pd.read_csv(filepath)
        d[tag]['p'] = df['p actual [kPa]'].to_numpy(dtype=float)
        d[tag]['if_tension'] = df['if tension [mN/m]'].to_numpy(dtype=float)
        d[tag]['solub'] = df['solubility [w/w]'].to_numpy(dtype=float)
        d[tag]['spec_vol'] = df['specific volume (fit) [mL/g]'].to_numpy(dtype=float)
        try:
            d[tag]['rho_co2'] = df['co2 density [g/mL]'].to_numpy(dtype=float)
        except:
            print('No co2 density in {0:s}.'.format(filepath))

    return d


def load_proc_data(csv_file_list, data_folder, adjust_T=True):
    """
    Loads processed data into a dictionary of numpy arrays for easier
    access when using matplotlib than pandas dataframe.
    TODO: load more data fields

    Parameters
    ----------
    - csv_file_list : list of strings
        List of .csv files containing processed G-ADSA data
    - data_folder : string
        Directory containing csv files of processed G-ADSA data

    Returns
    -------
    - d : dictionary of numpy arrays
        Dictionary of processed data organized by filename (<Mw>k<fn>f_<T>c)
        where Mw is molecular weight in kg/mol, fn is functionality, and T
        is temperature in C.
    """
    # dictionary of numpy arrays categorized by polyol and temperature
    d = {}
    # load data
    for i in range(len(csv_file_list)):
        file_name = csv_file_list[i]
        d[file_name] = {}
        i_c = file_name.rfind('c') # finds last 'c' in filename
        i__ = file_name[:i_c].rfind('_') # finds subscore preceding 'c' in filename
        d[file_name]['T'] = int(file_name[i__+1:i_c])

        if adjust_T:
            # add 0.5 c if the temperature is 30 (not included in file name)
            d[file_name]['T'] += 0.5*(d[file_name]['T'] == 30)
            d[file_name]['T'] += 0.1*(d[file_name]['T'] == 31)

        # polyol is everything before temperature
        d[file_name]['polyol'] = file_name[:file_name[:file_name.rfind('c')].rfind('_')]
        # get full data to store
        df = pd.read_csv(data_folder + file_name + '.csv')
        # store full data
        d[file_name]['p'] = df['p actual [kPa]'].to_numpy(dtype=float)
        d[file_name]['solub'] = df['solubility [w/w]'].to_numpy(dtype=float)
        d[file_name]['s_solub'] = df['solubility error [w/w]'].to_numpy(dtype=float)
        d[file_name]['w_gas'] = df['actual weight of dissolved gas [g]'].to_numpy(dtype=float)
        d[file_name]['s_w_gas'] = df['actual weight of dissolved gas std [g]'].to_numpy(dtype=float)
        d[file_name]['v_samp'] = df['sample volume [mL]'].to_numpy(dtype=float)
        d[file_name]['s_v_samp'] = df['sample volume std [mL]'].to_numpy(dtype=float)
        d[file_name]['spec_vol'] = df['specific volume (fit) [mL/g]'].to_numpy(dtype=float)
        d[file_name]['s_spec_vol'] = df['specific volume error [mL/g]'].to_numpy(dtype=float)
        d[file_name]['s_spec_vol_stat'] = df['specific volume error (stat) [mL/g]'].to_numpy(dtype=float)
        d[file_name]['s_spec_vol_sys'] = df['specific volume error (sys) [mL/g]'].to_numpy(dtype=float)
        d[file_name]['if_tension'] = df['if tension [mN/m]'].to_numpy(dtype=float)
        d[file_name]['s_if_tension'] = df['if tension std [mN/m]'].to_numpy(dtype=float)
        d[file_name]['diff_sqrt'] = df['diffusivity (sqrt) [cm^2/s]'].to_numpy(dtype=float)
        d[file_name]['s_diff_sqrt'] = df['diffusivity (sqrt) std [cm^2/s]'].to_numpy(dtype=float)
        d[file_name]['diff_exp'] = df['diffusivity (exp) [cm^2/s]'].to_numpy(dtype=float)
        d[file_name]['s_diff_exp'] = df['diffusivity (exp) std [cm^2/s]'].to_numpy(dtype=float)

    return d


def load_raw_data(adsa_folder, adsa_file_list, adsa_t0_list, grav_file_path, p_set_arr,
              hdr_adsa=1, hdr_grav=3, time_date_ref=None,
              columns=['p set [kPa]', 'p actual [kPa]', 'p std [kPa]',
                       'zero [g]', 'zero std [g]',
                       'mp1 [g]', 'mp1 std [g]', 'mp2 [g]', 'mp2 std [g]',
                       'M_0 (extrap) [g]',  'M_0 (prev) [g]', 'M_0 (prev) std [g]',
                       'M_infty (extrap) [g]', 'M_infty (final) [g]', 'M_infty (final) std [g]',
                       'if tension [mN/m]', 'if tension std [mN/m]',
                       'drop volume [uL]', 'drop volume std [uL]',
                       'sample volume [mL]', 'sample volume std [mL]',
                        'drop volume (fit) [uL]', 'drop volume (fit) std [uL]',
                        'sample volume (fit) [mL]', 'sample volume (fit) std [mL]',
                        'dissolved gas balance reading [g]',
                       'buoyancy correction [g]', 'actual weight of dissolved gas [g]',
                       'actual weight of dissolved gas std [g]',
                       'solubility [w/w]', 'solubility error [w/w]',
                       'specific volume [mL/g]', 'specific volume (fit) [mL/g]',
                        'specific volume error [mL/g]',
                       'specific volume error (stat) [mL/g]', 'specific volume error (sys) [mL/g]',
                       'diffusivity (sqrt) [cm^2/s]', 'diffusivity (sqrt) std [cm^2/s]',
                       'diffusivity (exp) [cm^2/s]', 'diffusivity (exp) std [cm^2/s]',
                       'diffusion time constant [s]'], zero_t_grav=True):
    """
    Load gravimetry and ADSA data for pre-processing, most importantly initializing
    the Pandas dataframe for storing processed data.
    PARAMETERS:
        adsa_folder : string
            Directory containing Axisymmetric Drop-Shape Analysis (ADSA) data.
        adsa_file_list : list of strings
            Files in adsa_folder with ADSA data.
        adsa_t0_list : list of ints
            Time since beginning of experiment at which each ADSA file was started [s]
        grav_file_path : string
            File path directly to gravimetry data.
        p_set_arr : numpy array
            Set values of pressure for each pressure step in chronological order [kPa].
        hdr_adsa : int, default=1
            Number of rows of header in an ADSA data file.
        hdr_grav : int, default=3
            Number of rows of header in a gravimetry data file.
        time_date_ref : TimeDate object (see timedate.py), default=None
            If gravimetry data must be extracted manually, the time must be
            calculated relative to a TimeDate object. In this case, such an object
            for the initial time and date of experiment should be provided here.
        columns : list, default shown above
            List of names of the columns in the Dataframe (ensuring consistent ordering).
        zero_t_grav : bool, default=True
            If True, shifts t_grav times such that the first value is 0.
    RETURNS:
        df : Pandas Dataframe
            Dataframe to hold processed data where each row has been populated
            by the corresponding pressure set value.
        br_arr : numpy array
            Balance readings at each time in t_grav [g].
        bp_arr numpy array
            Balance positions at each time in t_grav (1=ZERO, 2=MP1, 3=MP2).
        p_arr : numpy array
            Pressure measured at each time in t_grav [kPa].
        t_grav : numpy array
            Time of each gravimetry measurement [s]
        v_drop : numpy array
            Measurements of drop volume at each time in t_adsa [uL].
        t_adsa : numpy array
            Times of ADSA measurements [s].
    """
    # initialize arrys to store interfacial tension [mN/m], drop volume [uL],
    # and time [s] measured by ADSA system
    v_drop = np.array([])
    t_adsa = np.array([])

    # extract data from all data files for the pendant drop (ADSA)
    for i in range(len(adsa_file_list)):
        adsa_file = adsa_file_list[i]
        # load ADSA data in Pandas dataframe
        df_adsa = pd.read_csv(adsa_folder + adsa_file, header=hdr_adsa)
        # load drop volume data [uL] and concatenate to previous files' data
        v_drop = np.concatenate((v_drop, df_adsa['PndVol'].values))
        # load times of measurements [s] and concatenate to previous files' data
        t_adsa = np.concatenate((t_adsa, df_adsa['Secs.1'].values + adsa_t0_list[i]))

    # load rubotherm data and process
    df_grav = pd.read_csv(grav_file_path, header=hdr_grav)
    # Extract time in terms of seconds after start
    date_raw = df_grav['DATE'].values
    time_raw = df_grav['TIME'].values
    if time_date_ref:
        t_grav = np.zeros([len(date_raw)])
        for j in range(len(date_raw)):
            time_date = TimeDate(date_str=date_raw[j], time_str=time_raw[j])
            # convert from minutes to seconds to follow convention
            t_grav[j] = 60*TimeDate.diff_min(time_date_ref, time_date)
    else:
        # convert strings indicating the time into ints [s]
        t_grav = convert_time(date_raw, time_raw)
    # shift time so initial time is zero to match interfacial tension time
    if zero_t_grav:
        t_grav -= t_grav[0]
    # load rubotherm data in sync with time
    br_arr = df_grav['WEITGHT(g)'].values
    bp_arr = df_grav['BALANCE POSITION'].values
    p_arr = df_grav['Now Pressure(kPa)'].values
    # set negative values to 0 to make them physical and prevent errors later
    p_arr[p_arr < 0] = 0
    # initialize data frame to store data
    df = pd.DataFrame(columns=columns)
    # load set values of pressure steps [kPa]
    df['p set [kPa]'] = p_set_arr

    return df, br_arr, bp_arr, p_arr, t_grav, v_drop, t_adsa


def load_ref_if_tension_data(csv_filepath, tag_list):
    """
    Loads interfacial tension data for reference droplets
    (0 and 1 bar snapshots) into a dictionary of numpy
    arrays for easier access when using matplotlib than
    pandas dataframe.

    Parameters
    ----------
    - csv_filepath : string
        filepath to .csv file containing interfacial tension data

    Returns
    -------
    - d : dictionary of numpy arrays
        Dictionary of processed data organized by filename (<Mw>k<fn>f_<T>c)
        where Mw is molecular weight in kg/mol, fn is functionality, and T
        is temperature in C.
    """
    # dictionary of numpy arrays categorized by polyol and temperature
    d = {}
    # loads data
    df = pd.read_csv(csv_filepath)
    # sets up dictionary to store data
    for tag in tag_list:
        d[tag] = {}
        d[tag]['p'] = []
        d[tag]['if_tension'] = []
        d[tag]['s_if_tension'] = []

    # stores data in dictionary for access by plotting scripts
    for i in range(len(df)):
        series = df.iloc[i]
        tag = series['tag']
        d[tag]['p'] += [series['p [kPa]']]
        d[tag]['if_tension'] += [series['if tension [mN/m]']]
        d[tag]['s_if_tension'] += [series['if tension (std) [mN/m]']]

    return d


def normalize_gas_mass(w_gas, is_adsorbing):
    """
    Normalizes the given gas mass by the difference between the maximum and
    minimum points (or vice-versa, depending on whether the sample is adsorbing
    gas or desorbing it). The maximum of the normalized gas mass will be 1 and
    the minimum will be 0. This normalization is helpful for fitting the end of
    the curve to an exponential fit.
    PARAMETERS:
        w_gas : (N x 1) numpy array
            Gas mass to be normalized.
        is_adsorbing : bool
            If True, sample is adsorbing gas mass. O/w it is desorbing.
    RETURNS:
        w_gas_norm : (N x 1) numpy array
            Gas mass normalized so max value is 1 and min is 0 and remaining
            values are scaled to be in between.
    """
    # extract minimum and maximum values.
    M_min = np.min(w_gas)
    M_max = np.max(w_gas)
    if is_adsorbing:
        w_gas_norm = (M_max - w_gas)/(M_max - M_min)
    else:
        w_gas_norm = (M_min - w_gas)/(M_min - M_max)

    return w_gas_norm


def reject_outliers(data, m=2, min_std=0.1, return_inds=False):
    """
    Removes values in data that are more than "m" standard deviations away
    from the mean (and at least "min_std" away from the mean).
    From https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
    PARAMETERS:
        data : numpy array
            Data to clean from outliers.
        m : float, default=2
            Number of standard deviations above which a point is an "outlier."
        min_std : float, default=0.1
            Minimum absolute value of standard deviation.
        return_inds : bool, default=False
            If True, indices of non-outliers will be returned with cleaned data.
    RETURNS:
        result : numpy array
            data array with outliers removed
        inds : numpy array (only returned if return_inds==True)
            Indices of non-outliers in data.
    """
    # get indices of data that are not outliers or nans
    is_not_outlier = np.logical_and(abs(data - np.mean(data)) < m * max(np.std(data), min_std), \
        np.logical_not(np.isnan(data)))
    # cut out outliers and nans from data
    result = data[is_not_outlier]
    # count outliers
    num_outliers = len(result) < len(data)
    # announce if outliers were rejected
    if num_outliers > 0:
        print("Rejected %d outliers." % (num_outliers) )
    # return indices?
    if return_inds:
        inds = np.where(is_not_outlier)[0]
        return result, inds
    else:
        return result


def remove_nan_entries(nan_containing_arr, accompanying_arr_list):
    """
    Removes entries all arrays in the accompanying_arr_list where the
    nan_containing_arr has a nan.
    PARAMETERS:
        nan_containing_arr : numpy array of floats/ints
            Array that user thinks might contain nans
        accompanying_arr_list : list of numpy arrays
            List of arrays of same length as nan_containing_arr. All entries
            with the same index as entries having nans in nan_containing_arr
            will be removed.
    RETURNS:
        nan_free_arr_list : list of numpy arrays
            List of nan_containing_arr and all elements of accompanying_arr_list,
            now with the entries for which nan_containing_arr had nans removed.
    """
    # number of elements of given array
    n = len(nan_containing_arr)
    # identify indices of entries that are not nans
    inds_not_nan = np.logical_not(np.isnan(nan_containing_arr))
    # initialize result with given array having nans removed
    nan_free_arr_list = [nan_containing_arr[inds_not_nan]]
    # loop through accomapanying arrays and add to result after removing same entries
    for arr in accompanying_arr_list:
        assert len(nan_containing_arr)==n, "Arrays not of the same length."
        nan_free_arr_list += [arr[inds_not_nan]]

    return nan_free_arr_list


def rho_co2(p, T, data_dir='../input', eos_file_hdr='eos_co2_', ext='.csv'):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at 30.5 C.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.
    PARAMETERS:
        p : int (or array of ints)
            pressure in kPa of CO2
        T : float
            temperature in Celsius (only to one decimal place)
        data_dir : string
            Directory in which data files are stored relative to notebooks
        eos_file_hdr : string
            File header for equation of state data table
    RETURNS:
        rho : same as p
            density in g/mL of co2 @ given temperature
    """
    # get decimal and integer parts of the temperature
    dec, integ = np.modf(T)
    # create identifier string for temperature
    T_tag = '%d-%dC' % (integ, 10*dec)
    # dataframe of appropriate equation of state (eos) data from NIST
    df_eos = pd.read_csv(os.path.join(data_dir, eos_file_hdr + T_tag + ext), header=0)
    # get list of pressures of all data points [kPa]
    p_co2_kpa = df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    # get corresponding densities of CO2 [g/mL]
    rho_co2 = df_eos['Density (g/ml)'].to_numpy(dtype=float)
    # create interpolation function and interpolate density [g/mL]
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")
    rho = f_rho(p)

    return rho


def slope(x, a):
    """gives a slope going through origin."""
    return a*x

def square_root_3param(t, a, b, t0):
    """t^1/2 fit w/ 3 params: slope a, horizontal shift t0, & vertical shift b."""
    return a*(t-t0)**(0.5) + b


def square_root_2param(t, a, b):
    """t^1/2 fit w/ 2 params: slope a and vertical shift b."""
    return a*t**(0.5) + b


def square_root_2param_t0_fit(t, a, t0):
    """t^1/2 fit w/ 2 params: slope a and horizontal shift t0."""
    return a*(t-t0)**(0.5)


def square_root_1param(t, a):
    """t^1/2 fit w/ 1 param: slope a."""
    return a*t**(0.5)


def store_densities(df_thermo, adsa_folder, adsa_file, t0, t_grav, p_arr, T,
                    hdr_adsa=1):
    """
    Saves densities of CO2 atmosphere and sample for each data point measured
    with ADSA.
    PARAMETERS:
        df_thermo : pandas dataframe
            Dataframe of thermodynamic data computed with G-ADSA analysis
            (needs specific volume)
        adsa_folder : string
            name of folder with ADSA data file
        adsa_file : string
            name of ADSA data file (.csv)
        t0 : float
            Time between start of gravimetry and start of ADSA
        t_grav : numpy array of floats
            Times at which gravimetry data were recorded relative to start of
            gravimetry [s]
        p_arr : numpy array of floats
            Pressures measured during gravimetry measurements [kPa]
        T : float (to one decimal place)
            Set-value of temperature
        hdr_adsa : int
            Row of ADSA data file to use as header

    RETURNS:
        df_densities : pandas dataframe
            Dataframe of time, CO2 density, and sample density at each ADSA
            measurement
    """
    # initialize dataframe of densities
    df_densities = pd.DataFrame(columns=['Time ADSA [s]', 'Time Gravimetry [s]',
                                         'CO2 Density [g/mL]',
                                         'Sample Density [g/mL]'])
    # extract ADSA time
    df_adsa = pd.read_csv(adsa_folder + adsa_file, header=hdr_adsa)
    t_adsa = df_adsa['Secs.1'].to_numpy(dtype=float)
    df_densities['Time ADSA [s]'] = t_adsa
    # Get indices of gravimetry data closest to ADSA measurements
    inds_grav = [np.argmin(np.abs(t_grav - (t_adsa[i] + t0))) for i in range(len(t_adsa))]
    # store time from gravimetry measurements
    df_densities['Time Gravimetry [s]'] = t_grav[inds_grav]
    # get pressures at each ADSA data point
    p_adsa = p_arr[inds_grav]
    # get CO2 density
    df_densities['CO2 Density [g/mL]'] = rho_co2(p_adsa, T)
    # get indices of dataframe of thermodynamic data corresponding to ADSA measurements
    inds_thermo = [np.argmin(np.abs(df_thermo['p actual [kPa]'].to_numpy(dtype=float) - \
                                    p_adsa[i])) for i in range(len(p_adsa))]
    df_densities['Sample Density [g/mL]'] = 1 / df_thermo['specific volume [mL/g]'] \
                                            .to_numpy(dtype=float)[inds_thermo]

    return df_densities


def store_grav_adsa(df, i, i_p0, i_p1, t_grav, t_adsa, br_arr, bp_arr,
                v_drop, n_adsa, w_resolution=1E-5):
    """
    Stores equilibrium measurements from gravimetry and Axisymmetric Drop-Shape
    Analysis (ADSA) in dataframe, including balance readings and drop volume.
    PARAMETERS:
        df :  Pandas dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        i : int
            Index of current pressure step (and row of dataframe df)
        i_p0 : int
            Index of beginning of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        i_p1 : int
            Index of end of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        t_grav : numpy array
            Time of each data point measured by the gravimetry (Belsorp) system [s].
        t_adsa : numpy array
            Time of each data point measured by ADSA [s].
        br_arr : numpy array of floats
            Reading of magnetic suspension balance not corrected [g] at time points in t_grav.
        bp_arr : numpy array of ints
            Position of the balance (1 = zero, 2 = MP1, 3 = MP2) at time points in t_grav.
        v_drop : numpy array
            Volume of drop analyzed by ADSA [uL] at time points in t_adsa.
        n_adsa : int
            Number of points to average to estimate equilibrium ADSA measurements.
        w_resolution : float, default=1E-5
            Resolution in weight readings on magnetic suspension balance [g].
            10 ug for Rubotherm magnetic suspension balance.
    """
    # extract data for current pressure
    br_select = br_arr[i_p0:i_p1]
    bp_select = bp_arr[i_p0:i_p1]
    # indices for different measuring positions at end of measurement
    # 'zero' is tare; 'mp1' is tare plus the hook, crucible, and sample; 'mp2' also includes mass of cylinder
    # remove the first points in case balance has not yet settled
    i_zero = np.where(bp_select==1)[0][1:]
    i_mp1 = np.where(bp_select==2)[0]
    i_mp2 = np.where(bp_select==3)[0][1:]
    # identify final measurement of 'measuring point 1'
    i_mp1_f = i_mp1[np.logical_and(i_mp1 > np.max(i_zero), i_mp1 < np.min(i_mp2))]
    i_mp1_f = i_mp1_f[1:]

    # get averages and stdev of each balance reading (br), rejecting obvious
    # outliers (in case only part of the mass is lifted)
    df['zero [g]'].iloc[i] = np.mean(br_select[i_zero])
    std_zero = np.std(br_select[i_zero])
    df['zero std [g]'].iloc[i] = max(std_zero, w_resolution)
    df['mp1 [g]'].iloc[i] = np.mean(br_select[i_mp1_f])
    std_mp1 = np.std(reject_outliers(br_select[i_mp1_f]))
    df['mp1 std [g]'].iloc[i] = max(std_mp1, w_resolution)
    df['mp2 [g]'].iloc[i] = np.mean(br_select[i_mp2])
    std_mp2 = np.std(br_select[i_mp2])
    df['mp2 std [g]'].iloc[i] = max(std_mp2, w_resolution)

    # get last indices of ADSA data points before final time of gravimetry measurement for synchronization (if possible)
    try:
        # indices of corresponding ADSA data points
        i_adsa = get_inds_adsa(t_adsa, t_grav, i_p0, i_p1, n_adsa)
        # drop volume [uL]
        v_drop_mean = np.mean(v_drop[i_adsa])
        print('Drop volume = %f uL.' % v_drop_mean)
        df['drop volume [uL]'].iloc[i] = v_drop_mean
        df['drop volume std [uL]'].iloc[i] = np.std(v_drop[i_adsa])
    except:
        print('no adsa data for current pressure.')

    return df


def store_grav_adsa_manual(df, metadata, i, i_p0, i_p1, t_grav, t_adsa, br_arr,
                           bp_arr, v_drop, n_minutes, n_p_eq, date_ref, time_ref,
                           ref_by_dp=True, n_adsa=15, w_resolution=1E-5):
    """
    Same as store_grav_adsa() for measurements where data must be manually
    extracted using Datathief.
    PARAMETERS:
        df :  Pandas dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        metadata : Pandas dataframe
            Metadata of pressure steps loaded from eponymous .txt file.
        i : int
            Index of current pressure step (and row of dataframe df)
        i_p0 : int
            Index of beginning of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        i_p1 : int
            Index of end of current pressure step in t_grav time, often computed
            using get_curr_p_interval().
        t_grav : numpy array
            Time of each data point measured by the gravimetry (Belsorp) system [min].
        t_adsa : numpy array
            Time of each data point measured by ADSA [min].
        br_arr : numpy array of floats
            Reading of magnetic suspension balance not corrected [g] at time points in t_grav.
        bp_arr : numpy array of ints
            Position of the balance (1 = zero, 2 = MP1, 3 = MP2) at time points in t_grav.
        v_drop : numpy array
            Volume of drop analyzed by ADSA [uL] at time points in t_adsa.
        n_p_eq : int
            Number of points to average for determining equilibrium MP1 value.
        date_ref : string
            Date of beginning of experiment (reference time)
        time_ref : string
            Time of beginning of experiment (reference time)
        ref_by_dp : bool, default=True
            If True, change in pressure will mark the beginning and end of pressure step.
        n_adsa : int, default=15
            Number of points to average to estimate equilibrium ADSA measurements.
        w_resolution : float, default=1E-5
            Resolution in weight readings on magnetic suspension balance [g].
            10 ug for Rubotherm magnetic suspension balance.
    RETURNS:
        df : Pandas Dataframe
            Dataframe of processed G-ADSA data where each row is a pressure step.
    """
    # extract data for current pressure
    br_select = br_arr[i_p0:i_p1]
    bp_select = bp_arr[i_p0:i_p1]
    # indices for different measuring positions at end of measurement
    # 'mp1' is tare plus the hook, crucible, and sample
    # remove the first points in case balance has not yet settled
    i_mp1 = np.where(bp_select==2)[0]
    # identify final measurement of 'measuring point 1'
    i_mp1_f = i_mp1[:n_p_eq]

    # get averages and stdev of each balance reading (br), rejecting obvious
    # outliers (in case only part of the mass is lifted)
    df['zero [g]'].iloc[i] = metadata['zero [g]'].iloc[i] # only one zero meas
    df['zero std [g]'].iloc[i] = w_resolution # only one zero meas
    df['mp1 [g]'].iloc[i] = np.mean(br_select[i_mp1_f])
    std_mp1 = np.std(reject_outliers(br_select[i_mp1_f]))
    df['mp1 std [g]'].iloc[i] = max(std_mp1, w_resolution)
    # get last indices of ADSA data points before final time of gravimetry measurement for synchronization (if possible)
    time_date_ref = TimeDate(date_str=date_ref, time_str=time_ref)
    # use times to determine indices of ADSA to use
    if ref_by_dp:
        i_adsa = get_inds_adsa_manual(time_date_ref, t_adsa, t_grav, metadata, i, n_minutes)
    else:
        i_adsa, success = get_inds_adsa(t_adsa, t_grav, i_p0, i_p1, n_adsa, return_success=True)

    # drop volume [uL]
    v_drop_mean = np.mean(v_drop[i_adsa])
    print('Drop volume = %f uL.' % v_drop_mean)
    df['drop volume [uL]'].iloc[i] = v_drop_mean
    df['drop volume std [uL]'].iloc[i] = np.std(v_drop[i_adsa])
    if not success:
        df['drop volume [uL]'].iloc[i] = np.nan
        df['drop volume std [uL]'].iloc[i] = np.nan

    return df


def store_if_tension(if_tension, df, i, i_p0, i_p1, t_grav, t_adsa, n_adsa):
    """
    Saves interfacial tension data to the dataframe of processed data for the
    current pressure step.
    PARAMETERS:
        if_tension : numpy array
            Array of all interfacial tension measurements at the times in t_adsa [mN/m].
        df : Pandas Dataframe
            Dataframe of processed gravimetry-ADSA data where each row corresponds
            to a new pressure step
        i : int
            Index of current pressure step in df
        i_p0 : int
            Index of beginning of current pressure step in t_grav
        i_p1 : int
            Index of end of current pressure step in t_grav
        t_grav : numpy array
            Times of gravimetry measurements [s].
        t_adsa : numpy array
            Times of ADSA measurements [s].
        n_adsa : int
            Number of measurements at end of pressure step to average to get the
            equilibrium value of an ADSA measurement (interfacial tension in this
            case).
    RETURNS:
        df : Pandas Dataframe
            Updated dataframe with loaded interfacial tension measurement.
    """
    # indices of corresponding ADSA data points
    i_adsa, success = get_inds_adsa(t_adsa, t_grav, i_p0, i_p1, n_adsa, return_success=True)
    # cut out indices that extend beyond the length of the interfacial tension data
    if success:
        # compute interfacial tension [mN/m] from entries that are not nan
        i_not_nan = [i for i in i_adsa if not np.isnan(if_tension[i])]
        if len(i_not_nan) < n_adsa/2:
            print('More than half nans, so interfacial tension data not stored.')
            return df
        if_mean = np.mean(if_tension[i_not_nan])
        print('Interfacial tension = %f mN/m.' % if_mean)
        # store data
        df['if tension [mN/m]'].iloc[i] = if_mean
        df['if tension std [mN/m]'].iloc[i] = np.std(if_tension[i_adsa])
    else:
        print("Interfacial tension data not stored.")

    return df


def save_trd(df_trd, trd_save_name, no_hdr_tag='_no_hdr.csv'):
    """
    Saves dataframe with header found in TRD files saved by Belsorp. to match
    TRD files from automatic tests precisely. Used when recreating TRD file
    manually when not saved automatically.
    PARAMETERS:
        df_trd : Pandas Dataframe
            Dataframe with same format as the TRD file saved by the Belsorp
            software.
        trd_save_name : string
            File name of saved file.
        no_hdr_tag : string, default='_no_hdr.csv'
            Tag of save file name to indicate that it has no header in the csv table.
    RETURNS: nothing (void function), just saves file
    """
    # Save dataframe as a csv file
    df_trd.to_csv(trd_save_name + no_hdr_tag, index=False)
    # add header to csv file to exactly match the TRD files saved during the automatic experiments
    add_hdr(trd_save_name, no_hdr_tag=no_hdr_tag)

def add_hdr(trd_save_name, delete_no_hdr=True, no_hdr_tag='_no_hdr.csv',
              hdr_list=[':************************',
                        ':    TREND DATA FILE',
                        ':************************']):
    """
    Adds a header to the no-header TRD file to match the true TRD file.
    PARAMETERS:
        trd_save_name : string
            Name of saved file with a header.
        delete_no_hdr : bool, default=True
            If True, removes previously saved file without a header to reduce
            redundancy.
        no_hdr_tag : string, default='_no_hdr.csv'
            Tag at end of saved file name of data table without a header.
        hdr_list : list of strings
            Each entry is one row of the standard header of a TRD file.
    RETURNS: nothing (void function), just re-saves csv file with header.

    """
    # Open csv file with a header
    with open(trd_save_name + '.csv', 'w', newline='') as outcsv:
        # Create csv writer object
        writer = csv.writer(outcsv)
        # Add each element of the given header at the beginning of a new file
        for i in range(len(hdr_list)):
            writer.writerow([hdr_list[i]])

        # Open file without header
        with open(trd_save_name + no_hdr_tag, 'r', newline='') as incsv:
            # Read each line
            reader = csv.reader(incsv)
            # Write each line to new file with header
            writer.writerows(row for row in reader)
    # If desired, delete file without a header
    if delete_no_hdr:
        os.remove(trd_save_name + no_hdr_tag)


def v_drop_model(p, a, b, c):
    return a*p**2 + b*p + c
