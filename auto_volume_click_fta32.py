# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:28:37 2019

@author: Andy
"""

import pyautogui
import time


# parameters
get_location = False
n_ims = 144
duration = 0.2
short_sleep = 0.2

    
if get_location:
    while True:
        print(pyautogui.position())
        time.sleep(1)
            
else:
    time.sleep(10)    
    for i in range(n_ims):
        # click volume
        pyautogui.click(x=555, y=882, duration=duration)
        time.sleep(short_sleep)
        # click next image
        pyautogui.click(x=1086, y=398, duration=duration)
        time.sleep(short_sleep)    

    