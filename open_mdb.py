# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:44:09 2019

@author: Andy
"""

# PARAMETERS

import csv, pyodbc

# set up some constants
MDB = 'C:/Users\Andy.DESKTOP-CFRG05F\OneDrive - California Institute of Technology\Documents\Research\Kornfield\EXPERIMENTS\Italy\DATA\adsa\20190612_v2110b_small_drop_high.mdb'
DRV = '{Microsoft Access Driver (*.mdb)}'
PWD = 'pw'

# connect to db
con = pyodbc.connect('DRIVER={};DBQ={};PWD={}'.format(DRV,MDB,PWD))
cur = con.cursor()

# run a query and get the results 
SQL = 'SELECT * FROM mytable;' # your query goes here
rows = cur.execute(SQL).fetchall()
cur.close()
con.close()

# you could change the mode from 'w' to 'a' (append) for any subsequent queries
with open('20190612_v2110b_small_drop_high.csv', 'wb') as fou:
    csv_writer = csv.writer(fou) # default field-delimiter is ","
    csv_writer.writerows(rows)