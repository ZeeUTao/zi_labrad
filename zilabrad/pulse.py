# -*- coding: utf-8 -*-
"""
waveform script

Created on 2021.04.23
@author: Huang Wenhui
"""

import numpy as np
from math import ceil, pi
from scipy.special import erf
import math
import functools
from numba import jit
from collections import Iterable

from zilabrad.pyle.envelopes import Envelope
from zilabrad.pyle.tools import convertUnits





def limit_trange(start,end,smooth=0e-9):
    """ limit timeFunc(t) range, 
        return zeros value out of [start,end]
        to optimize calculate;
    """
    def wrap(func):
        def wrap_func(t):
            if not isinstance(t,Iterable):
                return func(t)
            t_gap = t[1]-t[0]
            if t[0] > (start-smooth):
                start_idx = 0
            else:
                start_idx = int((start-smooth-t[0])/t_gap)
            if t[-1] < (end+smooth):
                end_idx = -1
            else:
                end_idx = -int((t[-1]-end-smooth)/t_gap)
            _array = np.zeros(len(t),dtype='complex64')
            _array[start_idx:end_idx] = func(t[start_idx:end_idx])
            return _array
        return wrap_func
    return wrap







#####################
###     tools     ###
#####################
@convertUnits(delay='s')
def time_shift(env,delay=0):
    """ delay --> time unit: second
        let timeFunc shifting 'delay', 

        Example:
            Square 0 ns ~ 15 ns after 
            shift with delay = 5 ns is 
            Square 5 n ~ 20 ns.
    """
    def timeFunc(t):
        return env.timeFunc(t-delay)
    return Envelope(timeFunc=timeFunc,start=env.start+delay,end=env.end+delay)

@convertUnits(freq='Hz')
def mix(env,freq=0,phase=0):
    @limit_trange(env.start-1e-15,env.end+1e-15)
    def timeFunc(t):
        return env(t)*(np.cos(2*pi*freq*t+phase)+1j*np.sin(2*pi*freq*t+phase)) # np.exp(2j*pi*freq*t+1j*phase)
    return Envelope(timeFunc=timeFunc,start=env.start,end=env.end)




##########################
###   Basic Envelope   ###
##########################

@convertUnits(start='s', end='s', amp=None, length='s')
def rect(start=50e-9,end=None,amp=1.0,length=100e-9):
    if end is None: end = start + length
    @limit_trange(start,end)
    def timeFunc(t):
        return amp*(t < end)*(t >= start)
    env = Envelope(timeFunc, start=start, end=end)
    return env


@convertUnits(amp=None,start='s',end='s',freq='Hz',length='s')
def sine(amp=1, phase=0, start=0, end=None, freq=10e6, length=100e-9):
    if end is None: end = start + length
    @limit_trange(start,end)
    def timeFunc(t): 
        return amp*np.sin(2*pi*freq*(t-start)+phase)
    env = Envelope(timeFunc, start=start, end=end)
    return env

@convertUnits(amp=None,start='s',end='s',length='s')
def gaussian(amp=1,start=0,end=None,length=40e-9,sigma_times=3,phase=0):
    if isinstance(end,type(None)):
        end = start + length
    two_sigma_square = 1/2*((end-start)/sigma_times)**2
    t0 = (start + end)/2
    @limit_trange(start,end,10e-9)
    def timeFunc(t):
        return amp * np.exp(-(t-t0)**2/two_sigma_square+1j*phase)
    return Envelope(timeFunc, start=start, end=end)


@convertUnits(amp=None,start='s',end='s',length='s',w='s')
def flattop(amp=1,start=0,end=None,length=100e-9,w=5e-9):
    if end is None: end = start + length
    a = 1.66511 / w  # 1.66511 = 2*np.sqrt(np.log(2)) ;
    @limit_trange(start,end,2*w)
    def timeFunc(t):
        print(t[0],t[-1],len(t))
        return amp*(erf(a*(t-start))+erf(a*(end-t)))/2
    return Envelope(timeFunc, start=start, end=end)


@convertUnits(amp=None,t0='s',length='s')
def edge(amp=1,t0=0,length=5e-9,rise=False,mode='cos'):
    if abs(length)<1e-15: return NOTHING
    if rise: 
        start,end = t0-length,t0
    else: 
        start,end = t0,t0+length
        
    @limit_trange(start,end)
    def timeFunc_cos(t):
        return amp*(np.cos((t-t0)/length*pi)+1)/2*(t>=start)*(t<end)
    
    @limit_trange(start,end)
    def timeFunc_line(t):
        return amp*(1-abs(t-t0)/length)*(t>=start)*(t<end)

    if mode is 'cos':
        return Envelope(timeFunc_cos, start=start, end=end)
    elif mode is 'line':
        return Envelope(timeFunc_line, start=start, end=end)


NOTHING = Envelope(lambda t: np.zeros(len(t)),start=None,end=None)

##########################
###  Special Envelope  ###
##########################

@convertUnits(amp=None, start='s', end='s', freq='Hz', length='s')
def readout(amp=0.1, phase=0.0, start=0, end=None, freq=10e6, length=100e-9):
    if end is None: end = start + length

    @limit_trange(start,end)
    def timeFunc(t): # two np.sin() faster than np.exp() ; 
        return amp*(np.cos(2*pi*freq*(t-start)+phase)
                    +1j*np.sin(2*pi*freq*(t-start)+phase))
    env = Envelope(timeFunc, start=start, end=end)
    return env


# @convertUnits(amp=None, start='s', end='s', freq='Hz', length='s')
# def spectroscopy(amp=0.1,start=0,end=None,length=1e-6,freq=0e-6):
#     if end is None: end = start + length

#     env = rect(start=start,end=end,amp=amp)
#     return mix(env,freq,phase=0) ## have some error in mix

@convertUnits(amp=None, start='s', end='s', freq='Hz', length='s')
def spectroscopy(amp=0.1,start=0,end=None,length=1e-6,freq=0e-6,phase=0):
    if end is None: end = start + length
    @limit_trange(start,end)
    def timeFunc(t): 
        return amp*(np.cos(2*pi*freq*t+phase)+1j*np.sin(2*pi*freq*t+phase))
    env = Envelope(timeFunc, start=start, end=end)
    return env

@convertUnits(amp=None, start='s', end='s', freq='Hz', length='s')
def gauss_gate(amp=0.1,start=0,end=None,length=1e-6,freq=0e-6,phase=0,alpha=0,sigma_times=3):
    if end is None: end = start + length
    # env = gaussian(amp=amp,start=start,end=end,sigma_times=sigma_times)
    two_sigma_square = 1/2*((end-start)/sigma_times)**2
    t0 = (start + end)/2
    @limit_trange(start,end,10e-9)
    def timeFunc(t):
        ## default f21-f10 = 250MHz for alpha range ~ 1
        new_amp = (1+1j*alpha/2/pi/250e6*(-2/two_sigma_square)*(start-t0)) * amp
        assert abs(new_amp) <= 1, '%.3f > 1 '%new_amp
        return new_amp * np.exp(-(t-t0)**2/two_sigma_square)
    env = Envelope(timeFunc, start=start, end=end)
    return mix(env,freq=freq,phase=phase)

