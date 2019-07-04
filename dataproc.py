# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:09 2019

@author: Andy
"""

import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.signal import medfilt


def compute_D_exp(i, diam_cruc, df_meas, b):
    """
    """
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    v_samp = df_meas['sample volume [mL]'].values[i] # sample volume [mL]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute diffusivity using formula derived from Crank 1956 eqn 10.168 [cm^2/s]
    D_exp = -4*h_samp**2/np.pi**2*b

    return D_exp
    

def compute_D_sqrt(i, a, t_mp1, w_gas_act, n_pts_exp, maxfev, diam_cruc, 
                        df_meas):
    """
    """
    # Perform exponential fit to estimate the saturation mass
    w_gas_inf = extrapolate_equilibrium(t_mp1[:-n_pts_exp], w_gas_act[:-n_pts_exp], maxfev=maxfev)
    # estimate sample thickness based on estimated volume and cross-sectional area of crucible
    area_cruc = np.pi*(diam_cruc/2)**2 # area [cm^2]
    v_samp = df_meas['sample volume [mL]'].values[i] # sample volume [mL]
    h_samp = v_samp / area_cruc # height of sample [cm]
    # compute mean diffusivity using formula from Vrentas et al. 1977 (found in Pastore et al. 2011 as well) [cm^2/s]
    D_sqrt = np.pi*h_samp**2/4*(a/w_gas_inf)**2

    return D_sqrt

def compute_gas_mass(i, p_arr, p_set_arr, df_meas, bp_arr, br_arr, t_grav, 
                     p_thresh_frac, last_bound, v_ref_he):
    """
    """
    # get current set pressure
    p_set = p_set_arr[i]
    
    # get indices of corresponding to the current pressure
    i_p0, i_p1 = get_curr_p_interval(p_arr, p_set, p_thresh_frac, last_bound=last_bound)
    bp_select = bp_arr[i_p0:i_p1]
    br_select = br_arr[i_p0:i_p1]
    t_select = t_grav[i_p0:i_p1]
    
    # extract mp1 measurements and corresponding times for the current pressure set point
    is_mp1 = (bp_select == 2)
    mp1 = medfilt(br_select[is_mp1], kernel_size=5) # medfilt removes spikes from unstable measurements
    t_mp1 = t_select[is_mp1]
    
    # Is the sample adsorbing or desorbing gas?
    is_adsorbing = (p_set_arr[i] - p_set_arr[max(i-1,0)]) >= 0
    # Cut off data points at the beginning and end from the transition between pressure set points
    i_start, i_end = get_mp1_interval(mp1, is_adsorbing)
    mp1 = mp1[i_start:i_end]
    t_mp1 = t_mp1[i_start:i_end]
    
    # estimate the mass of adsorbed gas
    zero = df_meas['zero [g]'].values[i]
    br_eq = mp1 - zero # balance reading (not corrected for buoyancy) [g]
    br_eq_0 = br_eq[0] # vacuum balance reading [g]
    # subtract the balance reading under vacuum
    w_gas_app = br_eq - br_eq_0
    # compute the buoyancy correction (approximate volume of sample by equilibrium value) [g]
    buoyancy = rho_co2(p_set)*(df_meas['sample volume [mL]'].values[i] + v_ref_he)
    # correct for buoyancy to get the true mass of the sample
    w_gas_act = w_gas_app + buoyancy
    
    return w_gas_act, t_mp1, df_meas, i_p1, is_adsorbing


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
    # Extract the various units of time
    mo = np.array([int(d[:d.find('/')]) for d in date])
    dy = np.array([int(d[d.find('/')+1:d.find('/')+3]) for d in date])
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


def cut_ends(data, thresh):
    """
    ASSUMPTIONS:
        -User does not want to cut the data from the first to third quartiles.
    """
    # extract indices of quarter, halfway, and three-quarters through data
    i1 = int(len(data)/4)
    i2 = int(len(data)/2)
    i3 = int(len(data)*3/4)
    # check if data is generally increasing or decreasing
    is_increasing = (data[i3] - data[i1]) > 0
    # check that this is generally true
    if (is_increasing != (data[i2] - data[i1])) or (is_increasing != (data[i3] - data[i2])):
        print("Quartiles of data are not monotonic (dataproc.cut_ends()).")
    # during adsorption, remove sections of decreasing weight measurements
    if is_increasing:
        i_start = np.where(np.diff(data[:i1]) > thresh)[0][0]
        i_end = np.where(np.diff(data[i3:]) < -thresh)[0][0] + i3
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


def get_mp1_interval_2nd_deriv(mp1, is_adsorbing, w_thresh=0.00005):
    """
    """
    mp1 = medfilt(mp1, kernel_size=min(21, 2*(len(mp1)/2)+1))
    # estimate the halfway point in the data set
    i_halfway = int(len(mp1)/2)
    d_halfway = mp1[i_halfway+1] - mp1[i_halfway]
    # during adsorption, remove sections of decreasing weight measurements
    if is_adsorbing:
        i_start = np.where(np.diff(mp1[:i_halfway]) < 0.2*d_halfway)[0]
        i_end = np.where(-np.diff(mp1[i_halfway:]) > w_thresh)[0] + i_halfway
    # during desorption, remove sections of increasing weight measurements
    else:
        i_start = np.where(-np.diff(mp1[:i_halfway]) < -0.2*d_halfway)[0]
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
        
    print('i_start = {0}, i_end = {1}, i_halfway = {2}'.format(i_start, i_end,
          i_halfway))
        
    return i_start, i_end
        
  
def load_raw_data(adsa_folder, adsa_file_list, adsa_t0_list, grav_file_path, p_set_arr,
              hdr_adsa=1, hdr_grav=3, 
              columns=['p set [kPa]', 'p actual [kPa]', 'p std [kPa]',
                       'zero [g]', 'zero std [g]', 
                       'mp1 [g]', 'mp1 std [g]', 'mp2 [g]', 'mp2 std [g]', 
                       'M_0 (extrap) [g]', 'M_0 (prev) [g]', 
                       'M_infty (extrap) [g]', 'M_infty (final) [g]',
                       'if tension [mN/m]', 'if tension std [mN/m]', 
                       'drop volume [uL]', 'drop volume std [uL]', 
                       'sample volume [mL]', 'dissolved gas balance reading [g]',
                       'buoyancy correction [g]', 'actual weight of dissolved gas [g]',
                       'solubility [w/w]', 'specific volume [mL/g]', 
                       'specific volume error [mL/g]',
                       'diffusivity (sqrt) [cm^2/s]', 'diffusivity (exp) [cm^2/s]',
                       'diffusion time constant [s]']):
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
        if_tension = np.concatenate((if_tension, df_adsa['IFT'].values))
        v_drop = np.concatenate((v_drop, df_adsa['PndVol'].values))
        t_adsa = np.concatenate((t_adsa, df_adsa['Secs.1'].values + adsa_t0_list[i]))
    
    # TODO validate ADSA data
    # interfacial tension between 3 and 72 mN/m
    # drop volume between 1 and 10 uL
    # time between 0 and max(t_grav)
    
    # load rubotherm data and process
    df = pd.read_csv(grav_file_path, header=hdr_grav)
    # Extract time in terms of seconds after start
    date_raw = df['DATE'].values
    time_raw = df['TIME'].values
    t_grav = convert_time(date_raw, time_raw)
    # shift time so initial time is zero to match interfacial tension time
    t_grav -= t_grav[0]
    
    # load rubotherm data in sync with time
    br_arr = df['WEITGHT(g)'].values
    bp_arr = df['BALANCE POSITION'].values
    p_arr = df['Now Pressure(kPa)'].values
    
    # TODO validate rubotherm data
    # are there at least 120 data points per pressure?
    
    # initialize data frame to store data
    df_meas = pd.DataFrame(columns=columns)
    df_meas['p set [kPa]'] = p_set_arr

    return df_meas, br_arr, bp_arr, p_arr, t_grav, if_tension, v_drop, t_adsa
  
    
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


def rho_co2(p):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    webbook.nist.gov at 30.5 C.
    The density is returned in term of g/mL as a function of pressure in kPa.
    Will perform the interpolation if an input pressure p is given.
    PARAMETERS:
        p : int (or array of ints)
            pressure in kPa of CO2 @ 30.5 C
    
    RETURNS:
        rho : same as p
            density in g/mL of co2 @ 30.5 C 
    """

    p_co2_kpa = 1E2*np.arange(0,75,5)
    # density in g/mL (at 30.5 C)
    rho_co2 = np.array([0, 8.9277, 18.316, 28.229, 38.747, 49.971,
                        62.032, 75.103, 89.425, 105.34, 123.37,144.38,
                        169.97, 	203.77, 258.74])/1000
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")

    return f_rho(p)



def square_root_3param(t, a, b, t0):
    return a*(t-t0)**(0.5) + b


def square_root_1param(t, a):
    return a*t**(0.5)


def store_grav_adsa(df_meas, i, i_p0, i_p1, t_grav, t_adsa, br_arr, bp_arr, if_tension, 
                v_drop, n_adsa, w_resolution=1E-5):
    """
    """
    # extract data for current pressure
    t_select = t_grav[i_p0:i_p1]
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
    df_meas['zero [g]'].iloc[i] = np.mean(br_select[i_zero])
    std_zero = np.std(br_select[i_zero])
    df_meas['zero std [g]'].iloc[i] = max(std_zero, w_resolution)
    df_meas['mp1 [g]'].iloc[i] = np.mean(br_select[i_mp1_f])
    std_mp1 = np.std(reject_outliers(br_select[i_mp1_f]))
    df_meas['mp1 std [g]'].iloc[i] = max(std_mp1, w_resolution)
    df_meas['mp2 [g]'].iloc[i] = np.mean(br_select[i_mp2])
    std_mp2 = np.std(br_select[i_mp2])
    df_meas['mp2 std [g]'].iloc[i] = max(std_mp2, w_resolution)
    
    # identify the final time of gravimetry measurement
    t_f = t_select[-1]
    # get last indices of ADSA data points before final time of gravimetry measurement for synchronization (if possible)
    try:
        # indices of corresponding ADSA data points
        i_adsa = np.where(t_adsa <= t_f)[0][-n_adsa:]
        # interfacial tension [mN/m]
        if_mean = np.mean(if_tension[i_adsa])
        print('Interfacial tension = %f mN/m.' % if_mean)
        # drop volume [uL]
        v_drop_mean = np.mean(v_drop[i_adsa])
        print('Drop volume = %f uL.' % v_drop_mean)
        # store data
        df_meas['if tension [mN/m]'].iloc[i] = if_mean
        df_meas['if tension std [mN/m]'].iloc[i] = np.std(if_tension[i_adsa])
        df_meas['drop volume [uL]'].iloc[i] = v_drop_mean
        df_meas['drop volume std [uL]'].iloc[i] = np.std(v_drop[i_adsa])
    except:
        print('no adsa data for current pressure.')
        
    return df_meas