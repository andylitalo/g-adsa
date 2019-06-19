# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 21:42:31 2019

@author: Andy
"""

import pyautogui
import time

# NOTE: PLEASE DEFINE ROI BASED ON FIRST AND LAST IMAGES BEFORE RUNNING PROGRAM
# THEN CLICK 'IF TENSION' BUTTON AND OPEN WINDOWS AS SHOWN IN 
# automated_if_tension_clicking_screen.ppt

# parameters
get_location = False
n_ims = 376 # number of images to analyze
d = 0 # duration of moving mouse to click
sleep = 0.1
pyautogui.FAILSAFE = True # fail-safe by moving mouse to upper left

print('Press Ctrl+C to quit.')

# give time to get to FTA32 screen from Python
time.sleep(5)

# run clicking loop
try:
    if get_location:
        while True:
            print(pyautogui.position())
            time.sleep(1)
    else:        
        for i in range(n_ims):
            # click on 'IT Tension' button to compute interfacial tension
            pyautogui.click(1085, 371, button='left', duration=d)
            # wait 10 s for calculation
            time.sleep(10)
            # right-click on grid of data
            pyautogui.click(1380, 245, button='right', duration=d)
            time.sleep(sleep)
            # copy grid
            pyautogui.click(1508, 330, button='left', duration=d)
            time.sleep(sleep)
            # click 'ok' to confirm copying grid
            pyautogui.click(1036, 626, button='left', duration=d)
            time.sleep(sleep)
            # click icon for 'notepad' to open notepad window
            pyautogui.click(696, 1049, button='left', duration=d)
            time.sleep(1)
            # right-click on lower-right corner of notepad to prepare to paste
            pyautogui.click(1589, 942, button='right', duration=d)
            time.sleep(1)
            # click 'paste' to paste the data
            pyautogui.click(1384, 683, button='left', duration=d)
            time.sleep(1)
            # save data
            pyautogui.click(1197, 512, button='left', duration=d) #file
            time.sleep(sleep)
            pyautogui.click(1304, 603, button='left', duration=d) # save
            time.sleep(sleep)
            # click to go to next image
            pyautogui.click(552, 883, button='left', duration=d)
            # wait for 10 seconds to go to next image
            time.sleep(10)
            # save analysis of interfacial tension
            pyautogui.click(24, 51, button='left', duration=d) # file
            time.sleep(sleep)
            pyautogui.click(122, 192, button='left', duration=d) # save
            time.sleep(sleep)
            print(str(i+1) + 'th click completed.')
            
# Ctrl-C ends the loop
except KeyboardInterrupt:
    print('\nDone.')