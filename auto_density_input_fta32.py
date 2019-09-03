# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 13:28:37 2019

@author: Andy
"""

import pyautogui
import time


# Stage 4: Mapping the Liquid-liquid Coexistence
get_location = False
already_analyzed = False
n_ims = 350
duration = 0.2
long_sleep = 16
med_sleep = 3   
short_sleep = 0.2
save_period = 10

def double_alt_tab():
    """Moves to the third window in the queue, like alt+tab twice."""
    pyautogui.keyDown('alt')
    pyautogui.press('tab')
    time.sleep(short_sleep)
    pyautogui.press('tab')
    pyautogui.keyUp('alt')
    time.sleep(short_sleep)
    return
    
if get_location:
    while True:
        print(pyautogui.position())
        time.sleep(1)
            
else:
    time.sleep(10)

    # Start with excel open and first cell in the column "CO2 Density [g/mL]"
    # selected. The next window (accessible by alt + tab) should be the FTA32
    # software with the desired video loaded to the first image. The third
    # window should be a blank notepad/text document. These windows should be
    # positioned as in automated_density_input_clicking_screen.ppt
    # ALSO: be sure to add a new column in the Excel csv titled "Interfacial
    # Tension [mN/m]"
    
    for i in range(n_ims):
        # copy entry in "CO2 Density [g/mL]" column in Excel
        pyautogui.hotkey('ctrl', 'c', duration=duration)
        time.sleep(short_sleep)
        # switch to FTA32 program
        pyautogui.hotkey('alt', 'tab')
        time.sleep(short_sleep)
        # click "Calibration"
        pyautogui.click(x=828, y=119, duration=duration)
        time.sleep(short_sleep)
        # double-click entry for "Density of light phase (g/cc):" to highlight it
        pyautogui.doubleClick(x=1056, y=435, duration=duration)
        time.sleep(short_sleep)
        # paste CO2 density
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(short_sleep)
        # delete any straggling digits from the previous entry (esp. after E-#)
        pyautogui.press('delete')
        pyautogui.press('delete')
        time.sleep(short_sleep)
        # switch to Excel
        pyautogui.hotkey('alt', 'tab')
        time.sleep(short_sleep)
        # move to "Sample Density [g/mL]" column and copy entry
        pyautogui.press('right')
        time.sleep(short_sleep)
        pyautogui.hotkey('ctrl', 'c')
        time.sleep(short_sleep)
        # switch to FTA32 program
        pyautogui.hotkey('alt', 'tab')
        time.sleep(short_sleep)
        # double-click entry for "Density of heavy phase (g/cc):" to highlight it
        pyautogui.doubleClick(x=1055, y=544, duration=duration)
        time.sleep(short_sleep)
        # paste sample density
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(short_sleep)
        # click "Images" tab
        pyautogui.click(x=151, y=121, duration=duration)
        time.sleep(short_sleep)
        # click "IF Tension" button and wait at least 6 seconds
        pyautogui.click(x=1088, y=374, duration=duration)
        time.sleep(long_sleep)
        # right-click results window and copy grid
        pyautogui.click(x=1533, y=120, button='right', duration=duration)
        time.sleep(short_sleep)
        pyautogui.click(x=1641, y=208, duration=duration)
        time.sleep(short_sleep)
        pyautogui.press('enter')
        time.sleep(short_sleep)
        # switch to text document and paste grid
        double_alt_tab()
        time.sleep(short_sleep)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(short_sleep)
        # double-click to select interfacial tension measurement and copy
        pyautogui.doubleClick(x=1720, y=543, duration=duration)
        pyautogui.hotkey('ctrl', 'c')
        # select all and delete text in text document
        pyautogui.hotkey('ctrl', 'a')
        time.sleep(short_sleep)
        pyautogui.press('backspace')
        time.sleep(short_sleep)
        # switch to Excel spreadsheet, move to "Interfacial..." column and paste
        double_alt_tab()
        pyautogui.press('right')
        time.sleep(short_sleep)
        pyautogui.hotkey('ctrl', 'v')
        time.sleep(short_sleep)
        # move to next entry in "CO2 Density [g/mL]" column
        pyautogui.press('left')
        time.sleep(0.05)
        pyautogui.press('left')
        time.sleep(0.05)
        pyautogui.press('down')
        time.sleep(short_sleep)
        # switch to FTA32 & click to go to the next image and wait at least 6 seconds
        double_alt_tab()
        pyautogui.click(x=552, y=882, duration=duration)
        if already_analyzed:
            time.sleep(med_sleep)
        else:
            time.sleep(long_sleep)
        
        # save results
        if i%save_period == 0:
            # click "File" and "Save..."
            pyautogui.click(x=23, y=48, duration=duration)
            time.sleep(short_sleep)
            pyautogui.click(x=172, y=191, duration=duration)
        # switch to Excel
        pyautogui.hotkey('alt', 'tab')
        time.sleep(short_sleep)
        
        # save results
        if i%save_period==0:
            # ctrl + s and accept csv format
            pyautogui.hotkey('ctrl', 's')
            time.sleep(short_sleep)
            pyautogui.press('enter')
            time.sleep(short_sleep)
        

    