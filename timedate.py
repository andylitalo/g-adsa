# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 11:26:16 2019

@author: Andy
"""

class TimeDate:
    def __init__(self, second=0, minute=0, hour=0, day=1, month=1, year=2019,
                 date_str='', time_str=''):
        # set values
        self.second = second
        self.minute = minute
        self.hour = hour
        self.day = day
        self.month = month
        self.year = year
        # validate values
        self.validate(second, minute, hour, day, month, year)
        # Load time and date strings
        if len(date_str) > 0 and len(time_str) > 0:
            self.load_string(date_str, time_str)
        
    def is_second(self, second):
        return 0 <= second and second < 60 and isinstance(second, int)
    
    def is_minute(self, minute):
        return 0 <= minute and minute < 60 and isinstance(minute, int)
    
    def is_hour(self, hour):
        return 0 <= hour and hour < 24 and isinstance(hour, int)
    
    def is_day(self, day):
        return 1 <= day and day <= self.days_in_month() and isinstance(day, int)
    
    def is_month(self, month):
        return 1 <= month and month <= 12 and isinstance(month, int)
    
    def is_year(self, year):
        return 2019 <= year and year < 2100 and isinstance(year, int)
    
    def validate(self, second, minute, hour, day, month, year):
        assert self.is_second(second), 'Invalid format for "second" parameter.'
        assert self.is_minute(minute), 'Invalid format for "minute" parameter.'
        assert self.is_hour(hour), 'Invalid format for "hour" parameter.'
        assert self.is_day(day), 'Invalid format for "second" parameter.'
        assert self.is_month(month), 'Invalid format for "second" parameter.'
        assert self.is_year(year), 'Invalid format for "second" parameter.'
    
    def add_second(self, second):
        self.second += int(second)
        while not self.is_second(self.second):
            self.second -= 60
            self.minute += 1
            
    def add_minute(self, minute):
        self.minute += int(minute)
        while not self.is_minute(self.minute):
            self.minute -= 60
            self.hour += 1
            
    def add_hour(self, hour):
        self.hour += int(hour)
        while not self.is_hour(self.hour):
            self.hour -= 24
            self.day += 1
            
    def add_day(self, day):
        self.day += int(day)
        while not self.is_day(self.day):
            self.day -= self.days_in_month()
            self.month += 1
            
    def add_month(self, month):
        self.month += int(month)
        while not self.is_month(self.month):
            self.month -= 12
            self.year += 1
            
    def add_year(self, year):
        self.year += int(year)
            
    def add(self, second=0, minute=0, hour=0, day=0, month=0, year=0):
        self.add_second(second)
        self.add_minute(minute)
        self.add_hour(hour)
        self.add_day(day)
        self.add_month(month)
        self.add_year(year)
        
    def subtract_second(self, second):
        self.second -= int(second)
        while not self.is_second(self.second):
            self.second += 60
            self.minute -= 1
            
    def subtract_minute(self, minute):
        self.minute -= int(minute)
        while not self.is_minute(self.minute):
            self.minute += 60
            self.hour -= 1
            
    def subtract_hour(self, hour):
        self.hour -= int(hour)
        while not self.is_hour(self.hour):
            self.hour += 24
            self.day -= 1
            
    def subtract_day(self, day):
        self.day -= int(day)
        while not self.is_day(self.day):
            self.month -= 1
            self.day += self.days_in_month()
            
    def subtract_month(self, month):
        self.month -= int(month)
        while not self.is_month(self.month):
            self.month += 12
            self.year -= 1
            
    def subtract_year(self, year):
        self.year -= int(year)
        
    def subtract(self, second=0, minute=0, hour=0, day=0, month=0, year=0):
        self.subtract_second(second)
        self.subtract_minute(minute)
        self.subtract_hour(hour)
        self.subtract_day(day)
        self.subtract_month(month)
        self.subtract_year(year)
        
    def diff_min(td1, td2):
        diff = 0
        diff += td2.minute - td1.minute
        diff += 60*(td2.hour - td1.hour)
        diff += 24*60*(td2.day - td1.day)
        if td2.year != td1.year:
            print("year not the same. current method cannot subtract.")
        # if second date has a later month, add seconds of months in between
        if td2.month > td1.month:
            for m in range(td1.month, td2.month):
                diff += TimeDate.days_in_given_month(m, td1.year)*24*60
        # if second date has earlier month, subtract seconds of months in between
        elif td2.month < td1.month:
            for m in range(td2.month-1, td1.month-1):
                diff -= TimeDate.days_in_given_month(m, td1.year)*24*60
        
        return diff
    
    def days_in_given_month(month, year):
        if month in [1,3,5,7,8,10,12]:
            return 31
        elif month in [4,6,9,11]:
            return 30
        elif month == 2:
            if year % 4 != 0:
                return 28
            else:
                return 29
        else:
            print('invalid month format.')
            return -1 
        
    def days_in_month(self):
        if self.month in [1,3,5,7,8,10,12]:
            return 31
        elif self.month in [4,6,9,11]:
            return 30
        elif self.month == 2:
            if self.year % 4 != 0:
                return 28
            else:
                return 29
        else:
            self.is_month(month)
            print('invalid month format.')
            return -1
        
    def get_date_string(self):
        return str(self.month) + '/' + str(self.day) + '/' + str(self.year)

    def get_time_string(self):
        return str(self.hour) + ':' + '{0:02d}'.format(self.minute) + ':' + \
                '{0:02d}'.format(self.second)
                
    def load_string(self, date_string, time_string):
        # Determine the character used to separate the values in the date entries
        if '/' in date_string:
            d_sep = '/'
        elif '-' in date_string:
            d_sep = '-'
        else:
            print('date in unrecognized format by dataTimeDate class.')
        self.second = int(time_string[time_string.find(':')+4:])
        self.minute = int(time_string[time_string.find(':')+1: \
                                      time_string.find(':')+3])
        self.hour = int(time_string[:time_string.find(':')])
        self.day = int(date_string[date_string.find(d_sep)+1: \
                                   date_string.find(d_sep, date_string.find(d_sep)+1)])
        self.month = int(date_string[:date_string.find(d_sep)])
        self.year = int(date_string[-4:])
        
        
def __init__():
    pass