# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 18:36:12 2019

@author: Andy
"""

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

import dataproc

# Set parameters
adsa_folder = '../../EXPERIMENTS/Italy/data/adsa/'
adsa_file_list = ['20190612_0614_1k2f_adsa_data.csv', '20190614_0617_1k2f_adsa_data.csv']
adsa_t0_list = [3600*25 + 60*8 + 2, 24*3600*2 + 60*56 + 2] # belsorp: 6/12 6:05:58pm; v1 6/12 6:16pm, v2 6/14 7:02pm
w_samp_atm = 0.686 # mass of polyol sample in atmospheric pressure [g]
v_drop_0_manual = 3.332 # if not available in video, input initial drop volume [uL]; o/w put 0
rho_samp = 1.02 # density of polyol sample from TDS [g/cc]
# initial rubotherm at 6:16pm 6/12; video started 7:14pm 6/14
grav_file_path = '../../EXPERIMENTS/Italy/data/gravimetry/v2110b-TRD-061219-1804.csv'
save_file_path = '../../EXPERIMENTS/Italy/data/gravimetry/1k2f_1.csv'
# number of measurements to average for surface tension and volume readings
n_adsa = 3
# volume of hook and crucible as measured in helium [mL]
v_ref_he = 2.2675 # measured by Maria Rosaria Di Caprio @ 35 C [mL]
# pressure
p_step_arr = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
                       5000, 5500]) # user-defined set points [kPa]
p_thresh_frac_lo = 0.04 # threshold for acceptable difference between actual pressure and set pressure [fraction]
p_thresh_frac_hi = 0.01 # threshold for acceptable difference between actual pressure and set pressure [fraction]
dp_desorp = 700 # decrease in pressure during desorption [kPa]
di_desorp = 10 # duration of desorption step [# of measurements]

def reject_outliers(data, m=2):
    """from https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list"""
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def rho_gas(p):
    """
    Returns an interpolation function for the density of carbon dioxide
    according to the equation of state (data taken from
    http://www.peacesoftware.de/einigewerte/co2_e.html) at 30.5 C.
    The density is returned in term of g/mL as a function of pressure in Pascals.
    Will perform the interpolation if an input pressure p is given.
    """

    p_co2_kpa = 1E2*np.arange(0,80,5)
    # density in g/mL (at 30.5 C)
    rho_co2 = np.array([0, 8.9286, 18.316, 28.53525, 38.7545, 50.39925,
                        62.044, 75.752, 89.46, 106.455, 123.45,146.92,
                        170.39, 	216.7325, 779.4605627226, 	667.215])/1000
    f_rho = interp1d(p_co2_kpa, rho_co2, kind="cubic")

    return f_rho(p)

# initialize list to store interfacial tension measurements [mN/m]
if_tension = np.array([])
# also store drop volume measurements [uL]
drop_vol = np.array([])
# record time
t_adsa = np.array([])
# extract data from all data files for the pendant drop (ADSA)
for i in range(len(adsa_file_list)):
    adsa_file = adsa_file_list[i]
    df_adsa = pd.read_csv(adsa_folder + adsa_file, header=1)
    if_tension = np.concatenate((if_tension, df_adsa['IFT'].values))
    drop_vol = np.concatenate((drop_vol, df_adsa['PndVol'].values))
    t_adsa = np.concatenate((t_adsa, df_adsa['Secs.1'].values + adsa_t0_list[i]))

# load rubotherm data and process
df = pd.read_csv(grav_file_path, header=3)
# Extract time in terms of seconds after start
date_raw = df['DATE'].values
time_raw = df['TIME'].values
t_grav = dataproc.convert_time(date_raw, time_raw)
# shift time so initial time is zero to match interfacial tension time
t_grav -= t_grav[0]

# load rubotherm data in sync with time
br_arr = df['WEITGHT(g)'].values
bp_arr = df['BALANCE POSITION'].values
p_arr = df['Now Pressure(kPa)'].values

# initialize data frame to store data
df_meas = pd.DataFrame(columns=['p [kPa]', 'zero [g]', 'zero std [g]', 'mp1 [g]',
                        'mp1 std [g]', 'mp2 [g]', 'mp2 std [g]', 'if tension [mN/m]',
                        'if tension std [mN/m]', 'drop volume [uL]', 
                        'drop volume std [uL]'])
df_meas['p [kPa]'] = p_step_arr

# index where pressure = 0
i_p_zero = int(np.where(p_arr == 0)[0][0])
i_desorp_guess = np.where((p_arr[di_desorp:] - p_arr[:-di_desorp]) < -dp_desorp)[0][0]
i_desorp = i_desorp_guess + np.where(p_step_arr[-1] - p_arr[i_desorp_guess:] > 
                                     p_thresh_frac_lo*p_step_arr[-1])[0][0]
inds_sorp = np.arange(i_p_zero, i_desorp)
p_sorp = p_arr[inds_sorp]
t_sorp = t_grav[inds_sorp]
br_sorp = br_arr[inds_sorp]
bp_sorp = bp_arr[inds_sorp]
# extract interfacial tension, drop volume, and mass at MP1 for each pressure
for i in range(len(p_step_arr)):
    p_step = p_step_arr[i]
    print(p_step)
    # get indices of each measurement with pressure within thresholds 
    # one threshold for above set pressure, one for below
    i_p = np.logical_and(p_step - p_sorp <= p_thresh_frac_lo*p_step, 
                         p_step - p_sorp >= -p_thresh_frac_hi*p_step)
    # extract data for current pressure
    t_select = t_sorp[i_p]
    br_select = br_sorp[i_p]
    bp_select = bp_sorp[i_p]
    # indices for different measuring positions at end of measurement
    i_zero = np.where(bp_select==1)[0]
    i_mp1 = np.where(bp_select==2)[0] 
    i_mp2 = np.where(bp_select==3)[0]
    i_mp1 = i_mp1[np.logical_and(i_mp1 > np.max(i_zero), i_mp1 < np.min(i_mp2))]
    # get averages and stdev of each measurement
    df_meas['zero [g]'].iloc[i] = np.mean(br_select[i_zero])
    df_meas['zero std [g]'].iloc[i] = np.std(br_select[i_zero])
    df_meas['mp1 [g]'].iloc[i] = np.mean(reject_outliers(br_select[i_mp1]))
    df_meas['mp1 std [g]'].iloc[i] = np.std(reject_outliers(br_select[i_mp1]))
    df_meas['mp2 [g]'].iloc[i] = np.mean(reject_outliers(br_select[i_mp2]))
    df_meas['mp2 std [g]'].iloc[i] = np.std(reject_outliers(br_select[i_mp2]))
    # now sync with if tension and drop volume measurements
    t_f = t_select[-1]
    print(t_f)
    # get last index of adsa data points before next pressure step
    try:
        i_adsa = np.where(t_adsa <= t_f)[0][-n_adsa:]
        if_mean = np.mean(if_tension[i_adsa])
        print(i_adsa)
        print(if_mean)
        df_meas['if tension [mN/m]'].iloc[i] = if_mean
        df_meas['if tension std [mN/m]'].iloc[i] = np.std(if_tension[i_adsa])
        df_meas['drop volume [uL]'].iloc[i] = np.mean(drop_vol[i_adsa])
        df_meas['drop volume std [uL]'].iloc[i] = np.std(drop_vol[i_adsa])
    except:
        print('no adsa data for p = %d kPa' % p_step)

# balance reading at each pressure at equilibrium
br_e = df_meas['mp1 [g]'].values - df_meas['zero [g]'].values
br_e_0 = br_e[0] # balance reading at 0 pressure
# approximate mass of gas at atmospheric pressure as that at 150 kPa (p_step #2)
# assume volume of sample is same as at atmospheric pressure
w_gas_atm = br_e[2] - br_e_0 + rho_gas(p_step_arr[2])*(w_samp_atm/rho_samp + v_ref_he)
w_samp_0 = w_samp_atm - w_gas_atm
# sample mass at 0 atmosphere
#w_samp_0 = w_samp_atm - (df_meas['mp1 [g]'].values[2] - df_meas['zero [g]'].values[2]) + \
#                (df_meas['mp1 [g]'].values[0] - df_meas['zero [g]'].values[0])
print('w_samp_0 = %f g vs. w_samp_atm = %f g.' % (w_samp_0, w_samp_atm))
# calculate volume [mL]
v_samp_0 = w_samp_0 / rho_samp
v_drop = df_meas['drop volume [uL]'].values
# if initial drop volume is given, use it as the reference
if v_drop_0_manual > 0:
    v_drop_0 = v_drop_0_manual
# otherwise, use the first measurement of the drop volume as the reference
else:
    v_drop_0 = v_drop[0]
v_samp = v_drop / v_drop_0 * v_samp_0


# calculate "actual gas weight gain"; volumes in mL, density in g/mL
w_gas_act = br_e - br_e_0 + rho_gas(p_step_arr)*(v_samp + v_ref_he)
# calculate solubility w/w
df_meas['solubility'] = w_gas_act / (w_samp_0 + w_gas_act)

# process results to calculate 
# 1) mass of CO2 in sample
# 2) specific volume
# 3) interfacial tension (shouldn't need further processing--just average)
# 4) diffusivity?

df_meas.to_csv(save_file_path)

