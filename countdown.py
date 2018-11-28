# -*- coding: utf-8 -*-
import time

def countdown(t):
    """
    Enter a positive number
    """
    try:
        t = int(t)
        if t > 0:
            while t:
                mins, secs = divmod(t, 60)
                if mins >= 60:
                    hours, mins = divmod(mins, 60)
                    timeformat = '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
                else:
                    hours = 0
                    timeformat = '{:02d}:{:02d}'.format(mins, secs)
                print(timeformat, end='\r')
                time.sleep(1)
                t -= 1
            print('Time Up!')
        else:
            print('Please enter a positive number')
    except ValueError:
        print('Input is not a number!')

