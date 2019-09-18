# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:09 2019

@author: Andy
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import os

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import medfilt

from timedate import TimeDate
import errprop


def compute_D_exp(i, diam_cruc, df, b):
    """
    """
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    v_samp = df['sample volume [mL]'].values[i] # sample volume [mL]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute diffusivity using formula derived from Crank 1956 eqn 10.168 [cm^2/s]
    D_exp = -(4*h_samp**2/np.pi**2)*b

    return D_exp


def compute_D_sqrt(i, a, t_mp1, w_gas_act, n_pts_exp, maxfev, diam_cruc,
                        df):
    """
    """
    # extract the starting mass
    M_0 = df['M_0 (prev) [g]'].iloc[i]
    # extract the saturation mass
    M_infty = df['M_infty (final) [g]'].iloc[i]
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    v_samp = df['sample volume [mL]'].values[i] # sample volume [mL]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute mean diffusivity using formula from Vrentas et al. 1977 (found in Pastore et al. 2011 as well) [cm^2/s]
    D_sqrt = np.pi*h_samp**2/4*(a/(M_infty-M_0))**2

    return D_sqrt

def compute_gas_mass(i, T, p_arr, p_set_arr, df, bp_arr, br_arr, br_eq_0, t_grav,
                     p_thresh_frac, last_bound, v_ref_he, get_inst_buoy=False,
                     v_samp_live=[], no_gaps=False, err_list=[]):
    """
    br_eq_0 : vacuum balance reading at 0 kPa
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
    w_gas_app = br - br_eq_0
    # compute the buoyancy correction (approximate volume of sample by equilibrium value) [g]
    if get_inst_buoy:
        p_mp1 = p_select[is_mp1]
        if len(v_samp_live) > 0:
            v_samp_select = v_samp_live[i_p0:i_p1]
            v_samp = v_samp_select[is_mp1]
        else:
            v_samp = df['sample volume [mL]'].values[i]
        # also return the pressure
        result += [p_mp1]

    else:
        v_samp = df['sample volume [mL]'].values[i]
    # correct for buoyancy to get the true mass of the sample
    buoyancy = rho_co2(p_mp1, T)*(v_samp + v_ref_he)
    w_gas_act = w_gas_app + buoyancy
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


def compute_henrys_const(p, w_gas, v_samp, p_thresh=500, mw=44.01, maxfev=10000):
    """
    p is pressure in [kPa]
    w_gas is mass of gas in [g]
    v_samp is volume of sample in [mL]
    p_thresh is upper bound on pressures to consider for fit [kPa]
    mw is molecular weight in [g/mol]--default is for CO2

    returns Henry's constant in SI unit [mol/(m^3 Pa)]
    """
    m3_per_mL = 1E-6
    pa_per_kpa = 1E3
    c = (w_gas/mw) / (v_samp*m3_per_mL)
    inds_H = np.logical_and(p <= p_thresh, np.logical_and(np.logical_not(np.isnan(c)),
                                                          np.logical_not(np.isnan(p))))
    c_H = c[inds_H]
    p_H = p[inds_H]
    # fit just the slope
    popt, pcov = curve_fit(slope, p_H*pa_per_kpa, c_H, maxfev=maxfev)
    H = popt[0]

    return H


def compute_t_multiplier(metadata, i, t_p_interp, date_ref, time_ref):
    """
    Computes the multiplicative factor by which to multiply the time as
    measured by the Belsorp measurement. Factor is calculated by dividing known
    time of measurement of zero point (last measurement at a given pressure)
    by the reported ???
    """
    date_str = metadata['date'].iloc[i]
    time_str = metadata['time'].iloc[i]
    time_date = TimeDate(date_str=date_str, time_str=time_str)
    time_date_ref = TimeDate(date_str=date_ref, time_str=time_ref)
    # calculate actual time since start of experiment to
    diff_min_act = TimeDate.diff_min(time_date_ref, time_date)
    # quick validation that last value of interpolated p is the maximum value
    assert t_p_interp[-1] == np.max(t_p_interp), 'last value of t_p_interp is not maximum.'
    diff_min_meas = t_p_interp[-1]
    t_multiplier = diff_min_act / diff_min_meas

    return t_multiplier

def compute_t_shift(metadata, i, t_p_interp, p_interp, p_file, date_ref,
                    time_ref, dpdt_thresh=10, plot_pressure=True):
    """
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
    time_date_ref = TimeDate()
    time_date_ref.load_string(date_ref, time_ref)
    time_date_dp = TimeDate()
    time_date_dp.load_string(date_dp, time_dp)
    t_since_ref = TimeDate.diff_min(time_date_ref, time_date_dp)
    t_shift = t_since_ref - t_p_step

    return t_shift


def concatenate_data(metadata, i, date_ref, time_ref, time_list, date_list,
                     t_grav, t_interp, p_interp, p_arr, T_interp, T_arr,
                     mp1_interp, br_arr, bp_arr, zero_last=True):
    """
    Concatenate data for the creation of an artificial TRD file in the case of
    manual experiments with G-ADSA.
    """
    # initialize TimeDate object to store reference time of experiment
    time_date_ref = TimeDate()
    time_date_ref.load_string(date_ref, time_ref)
    # add list of dates and times
    for j in range(len(t_interp)):
        time_date = TimeDate()
        time_date.load_string(date_ref, time_ref)
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

    if zero_last:
        concatenate_mp1(t_grav, t_interp, p_arr, p_interp, T_arr, T_interp,
                        br_arr, mp1_interp, bp_arr)
        concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_arr, p_zero, T_arr, T_zero, br_arr,
                     zero, bp_arr)
    else:
        concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_arr, p_zero, T_arr, T_zero, br_arr,
                     zero, bp_arr)
        concatenate_mp1(t_grav, t_interp, p_arr, p_interp, T_arr, T_interp,
                br_arr, mp1_interp, bp_arr)


def concatenate_mp1(t_grav, t_interp, p_arr, p_interp, T_arr, T_interp,
                br_arr, mp1_interp, bp_arr):
    """
    Concatenate data from measuring point 1 (MP1) measurement.
    """
    # concatenate to gravimetry time
    t_grav += list(t_interp)
    p_arr += list(p_interp)
    T_arr += list(T_interp)
    br_arr += list(mp1_interp)
    bp_arr += list(2*np.ones([len(t_interp)]))


def concatenate_zero(time_zero, date_zero, time_list, date_list, t_grav,
                     time_date_ref, p_arr, p_zero, T_arr, T_zero, br_arr,
                     zero, bp_arr):
    """
    Concatenate data for zero point measurement.
    """
    time_date_zero = TimeDate()
    time_date_zero.load_string(date_zero, time_zero)
    time_list += [time_zero]
    date_list += [date_zero]
    t_grav += [TimeDate.diff_min(time_date_ref, time_date_zero)]
    p_arr += [p_zero]
    T_arr += [T_zero]
    br_arr += [zero]
    bp_arr += [1]


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

    time_passed = 24*3600*dy + 3600*hr + 60*mi + sc

    # TODO: fix to adjust with the changing of the month!!!
    return time_passed

# def diffusivity(df, p_set_arr, p_arr, p_thresh_frac, t_grav, n_pts_sqrt_arr, T, plot_results=False):
#     """
#     Defines routine to compute the diffusivity given data to fit and a fitting
#     method.
#     """
#     # initialize marker for pressure bounds
#     last_bound = 0
#     # initialize array to store diffusivity values
#     D_sqrt_arr = np.zeros([len(p_set_arr)])
#     # initialize array to store initial mass M_0 extrapolated with t^1/2 fit
#     M_0_extrap = np.zeros([len(p_set_arr)])
#     # Loop through each pressure set point
#     for i in range(len(p_set_arr)):
#         p_set = p_set_arr[i]
#         n_pts_sqrt = n_pts_sqrt_arr[i]
#         print('Pressure = %d kPa.' % p_set)
#         # the initial time of sorption is defined as the point where the pressure began changing after the previous sorption test
#         i_t0 = last_bound
#         t0 = t_grav[i_t0]
#         # get indices of each measurement with pressure within thresholds
#         i_p0, i_p1 = dataproc.get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=last_bound)
#         t_init = t_grav[i_p0]
#
#         # compute actual mass of gas at the times corresponding to the current pressure, save data in data frame
#         w_gas_act, t_mp1, df, last_bound, p_mp1 = dataproc.compute_gas_mass(i, T, p_arr, p_set_arr, df, bp_arr, br_arr, br_eq_0,
#                                                                             t_grav, p_thresh_frac, last_bound, v_ref_he,
#                                                                             get_inst_buoy=True)
#         # additional cutting off of data for t^1/2 fit
#         t_mp1 = t_mp1[i_shift[i]:]
#         w_gas_act = w_gas_act[i_shift[i]:]
#
#         # skip analysis if there are "nan"s in the data
#         if (np.isnan(w_gas_act)).any():
#             continue
#
#         # fit initial data points to a square root curve per eqn 10.165 in Crank (1956) "The Mathematics of Diffusion"
#         n = min(n_pts_sqrt, len(t_mp1)-1)
#         popt, pcov = curve_fit(dataproc.square_root_3param, t_mp1[:n], w_gas_act[:n], maxfev=maxfev)
#         a = popt[0]
#         w0 = popt[1]
#         t0_fit = popt[2]
#
#         # generate data points for t^(1/2) fit
#         t_fit = np.linspace(t_mp1[0], t_mp1[n-1], 100)
#         w_fit = dataproc.square_root_3param(t_fit, a, w0, t0_fit)
#         # plot the result to examine the fit
#         fig = plt.figure()
#         ax = fig.add_subplot(111)
#         # plot data translated such that first point is 0,0 and data increases (so t^1/2 looks like a straight line on log-log)
#         is_adsorbing = i <= np.argmax(p_set_arr)
#         sign = 2*(is_adsorbing-0.5)
#         ax.loglog(t_mp1 - t0_fit, sign*(w_gas_act - w0), '^', label='data')
#         ax.loglog(t_fit - t0_fit, sign*(w_fit - w0), '-', label='{a:.1e}(t-{t0:.1e})^(1/2) + {b:.1e}'.format(a=a, b=w0, t0=t0_fit))
#         ax.set_xlabel('t [s]')
#         ax.set_ylabel(r'$\Delta w_{CO2}$ [g]')
#         if is_adsorbing:
#             stage = 'Adsorption'
#         else:
#             stage = 'Desorption'
#         ax.set_title(stage + ' of CO2 in %s polyol at p = %d kPa, %d C' % (polyol, p_set, T))
#         plt.legend(loc='best')
#
#         # compute mean diffusion coefficient with the squareroot method by fitting and exponential curve to get the equilibrium mass
#         D_sqrt = dataproc.compute_D_sqrt(i, a, t_mp1, w_gas_act, n_pts_exp, maxfev, diam_cruc, df)
#         print('D_sqrt = %.2e cm^2/s.' % D_sqrt)
#         # store result
#         D_sqrt_arr[i] = D_sqrt
#
#         # Also save the "initial" mass extrapolated to the beginning of the change in pressure
#         M_0_extrap[i] = dataproc.square_root_3param(max(t0, t0_fit), a, w0, t0_fit)
#
#
#
#
#
#
#
#
#     # the initial time of sorption is defined as the point where the pressure began changing after the previous sorption test
#     i_t0 = last_bound
#     t0 = t_grav[i_t0]
#     # get indices of each measurement with pressure within thresholds
#     i_p0, i_p1 = dataproc.get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=last_bound)
#     t_init = t_grav[i_p0]
#
#     # compute actual mass of gas at the times corresponding to the current pressure, save data in data frame
#     w_gas_act, t_mp1, df, last_bound, p_mp1 = dataproc.compute_gas_mass(i, T, p_arr, p_set_arr, df, bp_arr, br_arr, br_eq_0,
#                                                                         t_grav, p_thresh_frac, last_bound, v_ref_he,
#                                                                         get_inst_buoy=True)
#     # additional cutting off of data for t^1/2 fit
#     t_mp1 = t_mp1[i_shift[i]:]
#     w_gas_act = w_gas_act[i_shift[i]:]
#
#     # skip analysis if there are "nan"s in the data
#     if (np.isnan(w_gas_act)).any():
#         continue
#
#     # fit initial data points to a square root curve per eqn 10.165 in Crank (1956) "The Mathematics of Diffusion"
#     n = min(n_pts_sqrt, len(t_mp1)-1)
#
#
#     popt, pcov = curve_fit(dataproc.square_root_3param, t_mp1[:n], w_gas_act[:n], maxfev=maxfev)
#     a = popt[0]
#     w0 = popt[1]
#     t0_fit = popt[2]
#     # generate data points for t^(1/2) fit
#     t_fit = np.linspace(t_mp1[0], t_mp1[n-1], 100)
#     w_fit = dataproc.square_root_3param(t_fit, a, w0, t0_fit)
#     # Also save the "initial" mass extrapolated to the beginning of the change in pressure
#     M_0 = dataproc.square_root_3param(max(t0, t0_fit), a, w0, t0_fit)
#
#
#     # compute mean diffusion coefficient with the squareroot method by fitting and exponential curve to get the equilibrium mass
#     D_sqrt = dataproc.compute_D_sqrt(i, a, t_mp1, w_gas_act, n_pts_exp, maxfev, diam_cruc, df)
#
#     if plot_results:
#         plot.diffusivity()
#
#
#     return D, M_0

def exponential_approach(x, a, b, c):
    """Exponential approach to asymptote. Negatives and /100 there because I can't figure out how to change the initial
    parameters for the curve_fit function from all 1's."""
#    a, b, c = params
    return a*np.exp(b*x) + c


def extrapolate_equilibrium(t, m, maxfev=800, p0=(-0.01, -1E-4, 0.01)):
    """Extrapolate mass over time with exponential fit to estimate equilibrium."""
    popt, pcov = curve_fit(exponential_approach, t, m, maxfev=maxfev, p0=p0)
    a, b, c = popt
    m_eq = c

    return m_eq


def get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=0,
                        window_reduction=0.25, min_window=1):
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
    """
    # extract data for current pressure
    t_select = t_grav[i_p0:i_p1]
    # identify the final time of gravimetry measurement
    t_i = t_select[0]
    t_f = t_select[-1]
    # get indices of last data points receding final time of gravimetry
    inds = np.where(t_adsa <= t_f)[0][-n_adsa:]
    if len(np.where(np.logical_and(t_adsa <= t_f, t_adsa >= t_i))[0]) < n_adsa:
        print('********not enough ADSA points at given pressure.******')
        success=False
    else:
        success=True

    if return_success:
        return inds, success
    else:
        return inds


def get_inds_adsa_manual(time_date_ref, t_adsa, t_grav, metadata, i, n_minutes,
                         buffer=120):
    """
    buffer is number of minutes at the end of an experiment to cut off in
    case synchronization wasn't accurate.
    """
    date = metadata['date dp'].iloc[i]
    if i < len(metadata['time dp start']) - 1:
        time_end = metadata['time dp start'].iloc[i+1]
        time_date_end = TimeDate(date_str=date, time_str=time_end)
        t_min_end = TimeDate.diff_min(time_date_ref, time_date_end)
    else:
        t_min_end = t_adsa[-1]

    inds =  np.where(np.logical_and(t_adsa > t_min_end - buffer - n_minutes,
                                   t_adsa < t_min_end - buffer))[0]

    return inds

def get_mp1_interval(mp1, is_adsorbing, w_thresh=0.00005):
    """
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
    inds_c = get_all_inds(file_name, 'c')
    inds_ = get_all_inds(file_name, '_')
    for ind in inds_:
        if ind+3 in inds_c:
            return int(file_name[ind+1:ind+3])

    return np.nan


def get_all_inds(string, substr):
    """Returns indices of all occurrences of a substring."""
    inds = [0]
    i_curr = string.find(substr)
    while i_curr != -1:
        inds += [i_curr + inds[-1]]
        string = string[inds[-1]:]
        i_curr = string.find(substr)

    if len(inds) == 0:
        result = 0
    else:
        result = inds[1:]

    return result
#
#def get_mp1_interval_2nd_deriv(mp1, is_adsorbing, w_thresh=0.00005):
#    """
#    """
#    mp1 = medfilt(mp1, kernel_size=min(21, 2*(len(mp1)/2)+1))
#    # estimate the halfway point in the data set
#    i_halfway = int(len(mp1)/2)
#    d_halfway = mp1[i_halfway+1] - mp1[i_halfway]
#    # during adsorption, remove sections of decreasing weight measurements
#    if is_adsorbing:
#        i_start = np.where(np.diff(mp1[:i_halfway]) < 0.2*d_halfway)[0]
#        i_end = np.where(-np.diff(mp1[i_halfway:]) > w_thresh)[0] + i_halfway
#    # during desorption, remove sections of increasing weight measurements
#    else:
#        i_start = np.where(-np.diff(mp1[:i_halfway]) < -0.2*d_halfway)[0]
#        i_end = np.where(np.diff(mp1[i_halfway:]) > w_thresh)[0] + i_halfway
#    # check if any data points should be cut off from the beginning
#    if len(i_start)==0:
#        i_start = 0
#    else:
#        i_start = i_start[-1] + 2
#    # check if any data points should be cut off from the end
#    if len(i_end)==0:
#        i_end = -1
#    else:
#        i_end = i_end[0]
#
#    print('i_start = {0}, i_end = {1}, i_halfway = {2}'.format(i_start, i_end,
#          i_halfway))
#
#    return i_start, i_end


def interp(t, meas, dt=0.5, t_min=-1, t_max=-1):
    """
    Interpolates measurements from manually recorded data.
    """
    t_uniq, i_uniq = np.unique(t, return_index=True)
    meas_uniq = meas[i_uniq]
    f_interp = interp1d(t_uniq, meas_uniq)
    if t_min == -1:
        t_min = t_uniq[0]
    if t_max == -1:
        t_max = t_uniq[-1]
    t_interp = np.arange(t_min, t_max, dt)
    meas_interp = f_interp(t_interp)

    return t_interp, meas_interp


def load_datathief_data(filepath):
    """
    Loads data exported from datathief.
    """
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    t = data[:,0]
    meas = data[:,1]

    return t, meas


def load_raw_data(adsa_folder, adsa_file_list, adsa_t0_list, grav_file_path, p_set_arr,
              hdr_adsa=1, hdr_grav=3, load_if_tension=False, time_date_ref=None,
              columns=['p set [kPa]', 'p actual [kPa]', 'p std [kPa]',
                       'zero [g]', 'zero std [g]',
                       'mp1 [g]', 'mp1 std [g]', 'mp2 [g]', 'mp2 std [g]',
                       'M_0 (extrap) [g]', 'M_0 (prev) [g]',
                       'M_infty (extrap) [g]', 'M_infty (final) [g]',
                       'if tension [mN/m]', 'if tension std [mN/m]',
                       'drop volume [uL]', 'drop volume std [uL]',
                       'sample volume [mL]', 'dissolved gas balance reading [g]',
                       'buoyancy correction [g]', 'actual weight of dissolved gas [g]',
                       'solubility [w/w]', 'solubility error [w/w]',
                       'specific volume [mL/g]',  'specific volume error [mL/g]',
                       'diffusivity (sqrt) [cm^2/s]', 'diffusivity (exp) [cm^2/s]',
                       'diffusion time constant [s]'], zero_t_grav=True):
    """
    Load gravimetry and ADSA data for pre-processing.
    PARAMETERS:

    RETURNS:

    """
    # initialize arrys to store interfacial tension [mN/m], drop volume [uL],
    # and time [s] measured by ADSA system
    if_tension = np.array([])
    v_drop = np.array([])
    t_adsa = np.array([])

    # extract data from all data files for the pendant drop (ADSA)
    for i in range(len(adsa_file_list)):
        adsa_file = adsa_file_list[i]
        df_adsa = pd.read_csv(adsa_folder + adsa_file, header=hdr_adsa)
        v_drop = np.concatenate((v_drop, df_adsa['PndVol'].values))
        t_adsa = np.concatenate((t_adsa, df_adsa['Secs.1'].values + adsa_t0_list[i]))
        if load_if_tension:
            if_tension = np.concatenate((if_tension, df_adsa['IFT'].values))

    # load rubotherm data and process
    df = pd.read_csv(grav_file_path, header=hdr_grav)
    # Extract time in terms of seconds after start
    date_raw = df['DATE'].values
    time_raw = df['TIME'].values
    if time_date_ref:
        t_grav = np.zeros([len(date_raw)])
        for j in range(len(date_raw)):
            time_date = TimeDate(date_str=date_raw[j], time_str=time_raw[j])
            # convert from minutes to seconds to follow convention
            t_grav[j] = 60*TimeDate.diff_min(time_date_ref, time_date)
    else:
        t_grav = convert_time(date_raw, time_raw)
    # shift time so initial time is zero to match interfacial tension time
    if zero_t_grav:
        t_grav -= t_grav[0]

    # load rubotherm data in sync with time
    br_arr = df['WEITGHT(g)'].values
    bp_arr = df['BALANCE POSITION'].values
    p_arr = df['Now Pressure(kPa)'].values
    # set negative values to 0 to make them physical and prevent errors later
    p_arr[p_arr < 0] = 0

    # TODO validate rubotherm data
    # are there at least 120 data points per pressure?

    # initialize data frame to store data
    df = pd.DataFrame(columns=columns)
    df['p set [kPa]'] = p_set_arr

    if load_if_tension:
        return df, br_arr, bp_arr, p_arr, t_grav, if_tension, v_drop, t_adsa
    else:
        return df, br_arr, bp_arr, p_arr, t_grav, v_drop, t_adsa


def reject_outliers(data, m=2, min_std=0.1, return_inds=False):
    """from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list"""
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


def rho_co2(p, T, eos_file_hdr='eos_co2_', ext='.csv'):
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
        eos_file_hdr : string
            File header for equation of state data table

    RETURNS:
        rho : same as p
            density in g/mL of co2 @ 30.5 C
    """
    dec, integ = np.modf(T)
    T_tag = '%d-%dC' % (integ, 10*dec)
    df_eos = pd.read_csv(eos_file_hdr + T_tag + ext, header=0)
    p_co2_kpa = df_eos['Pressure (kPa)'].to_numpy(dtype=float)
    rho_co2 = df_eos['Density (g/ml)'].to_numpy(dtype=float)
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")

    return f_rho(p)


def slope(x, a):
    """gives a slope going through origin."""
    return a*x

def square_root_3param(t, a, b, t0):
    return a*(t-t0)**(0.5) + b


def square_root_2param(t, a, b):
    return a*t**(0.5) + b


def square_root_2param_t0_fit(t, a, t0):
    return a*(t-t0)**(0.5)


def square_root_1param(t, a):
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
    For manual measurements.
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
#    try:
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


def save_trd(df_trd, trd_save_hdr):
    """
    Saves dataframe with header found in TRD files saved by Belsorp. to match
    TRD files from automatic tests precisely.
    """
    df_trd.to_csv(trd_save_hdr + '_no_hdr.csv', index=False)
    # add header to csv file to exactly match the TRD files saved during the automatic experiments
    add_hdr(trd_save_hdr)

def add_hdr(trd_save_hdr, delete_no_hdr=True, no_hdr_tag='_no_hdr.csv',
              hdr_list=[':************************',
                        ':    TREND DATA FILE',
                        ':************************']):
    """Adds a header to the no-header TRD file to match the true TRD file."""
    with open(trd_save_hdr + '.csv', 'w', newline='') as outcsv:
        writer = csv.writer(outcsv)
        for i in range(len(hdr_list)):
            writer.writerow([hdr_list[i]])

        with open(trd_save_hdr + '_no_hdr.csv', 'r', newline='') as incsv:
            reader = csv.reader(incsv)
            writer.writerows(row for row in reader)

    if delete_no_hdr:
        os.remove(trd_save_hdr + no_hdr_tag)
