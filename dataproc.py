# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:09 2019

@author: Andy
"""

import numpy as np

from scipy.optimize import curve_fit


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
    dy = np.array([int(d[2:4]) for d in date])
    mo = np.array([int(d[:1]) for d in date])
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


def extrapolate_equilibrium(t, m):
    """Extrapolate mass over time with exponential fit to estimate equilibrium."""
    def exponential_approach(x, a, b, c):
        """Exponential approach to asymptote. Negatives and /100 there because I can't figure out how to change the initial
        parameters for the curve_fit function from all 1's."""
        return -a*np.exp(-b/100*x) + c
    
    popt, pcov = curve_fit(exponential_approach, t, m)
    a, b, c = popt
    m_eq = c
    
    return m_eq