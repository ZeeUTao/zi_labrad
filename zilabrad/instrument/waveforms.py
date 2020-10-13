# -*- coding: utf-8 -*-
"""
waveform script

Created on 2020.10.06
@author: Tao Ziyu
"""

import numpy as np
from math import ceil,pi
from zilabrad.pyle.envelopes import Envelope,NOTHING
import math
import inspect
import functools

def convertUnits(**unitdict):
    """
    Decorator to create functions that automatically
    convert arguments into specified units.  If a unit
    is specified for an argument and the user passes
    an argument with incompatible units, an Exception
    will be raised.  Inside the decorated function, the
    arguments no longer have any units, they are just
    plain floats.  Not all arguments to the function need
    to be specified in the decorator.  Those that are not
    specified will be passed through unmodified.
    
    Usage:
    
    @convertUnits(t0='ns', amp=None)
    def func(t0, amp):
        <do stuff>
        
    This is essentially equivalent to:
    
    def func(t0, amp):
        t0 = convert(t0, 'ns')
        amp = convert(amp, None)
        <do stuff>
        
    The convert function is defined internally, and will
    convert any quantities with units into the specified
    units, or strip off any units if unit is None.
    """
    def convert(v, u):
        if hasattr(v, 'unit'): # prefer over subclass check: isinstance(v, Value)
            if u is None:
                return v[v.unit]
            else:
                return v[u]
        else:
            return v
    
    def wrap(f):
        args = inspect.getargspec(f)[0] # list of argument names
        for arg in unitdict:
            if arg not in args:
                    raise Exception('function %s does not take arg "%s"' % (f, arg))
        # unitdict maps argument names to units
        # posdict maps argument positions to units
        posdict = dict((i, unitdict[arg]) for i, arg in enumerate(args) if arg in unitdict)
        
        @functools.wraps(f)
        def wrapped(*a, **kw):
            # convert positional arguments if they have a unit
            a = [convert(val, posdict.get(i, None)) for i, val in enumerate(a)]
            # convert named arguments if they have a unit
            for arg, val in kw.items():
                if arg in unitdict:
                    kw[arg] = convert(val, unitdict[arg])
            # call the function with converted arguments
            return f(*a, **kw)
        return wrapped
    return wrap

class waveform(object):
    """ 
    Represents a control waveform as a function of time 

    """
    def __init__(self,all_length=1e-6,fs=1.8e9,origin=0,name='default'):
        self.name = name
        self.fs = fs
        self.sample_number = ceil(all_length*self.fs/16)*16 ## QA lenght最小16,最小单位间隔8; HD length最小32,最小单位间隔16;
        # self.len = self.sample_number/self.fs ## set as 16 sample integral multiple;
        # self.origin = origin ## mark real start in waveform; set QA trigger as 0  
        # self.tlist = np.asarray([k/self.fs+self.origin for k in range(self.sample_number)])
        # self.bias_sample = 0

    # We can use env from pyle.envelopes to define some complicated waveforms    
    # use: 
    # w_qa = waveform()
    # w_qa(env)
    # on the other hand, some frequently used waveforms are provided below


    def func2array_withoutNumpy(self,envelope,
        start: float,
        end: float,
        fs: None or float = None):
        """
        Try to use numpy, or it will be slow in python
        """
        if fs == None:
            fs = self.fs

        start = envelope.start
        end = envelope.end

        if end <= start:
            return []
        
        interval = 1./fs
        steps = ceil( (end-start)*fs)

        return [func(start + idx*interval) for idx in range(steps)]
        
 
    def func2array(self,func,
        start: float,
        end: float,
        fs: None or float = None):
        """
        Args:
            func: func(t) contains only Numpy function
        """
        if fs == None:
            fs = self.fs 
        if end <= start:
            return []
        interval = 1./fs
        return func(np.arange(start,end,interval))


@convertUnits(start='s',end='s',amp=None,length='s')
def square(amp=1.0,start=0.,end=None,length=100e-9,fs=1.8e9):
    if end is None:
        end = start + length

    steps = ceil( (end-start)*fs)
    return amp*np.ones(steps)


@convertUnits(freq='Hz',start='s',end='s',length='s')
def spectroscopyPulse(amp=0.0,freq=10e6,phase=0.,
    start=0.,end=None,length=1e-6,
    fs=1.8e9):
    """
    Args: fs (float): sampling rate
    Returns: numpy.array
    """
    if end is None:
        end = start + length
    return amp * np.cos( freq * np.arange(start,end,1./fs) + phase )


@convertUnits(freq='Hz',start='s',end='s',length='s')
def func_with_envelope(amp=0.0,freq=10e6,
    start=0.,end=None,length=1e-6,
    fs=1.8e9):
    """
    It's an example for using zilabrad.pyle.envelopes.Envelope to define timefunction, 
    which can be added, multiplied...
    """
    if end is None:
        end = start + length
    def timeFunc(t):
        return amp*np.cos(freq * t)
    return Envelope(timeFunc,None,start,end)




@convertUnits(freq='Hz',start='s',end='s',length='s')
def sine(amp=0.0,freq=10e6,
    start=0.,end=None,length=1e-6,
    fs=1.8e9):
    if end is None:
        end = start + length
    return amp * np.sin( freq * np.arange(start,end,1./fs) )

@convertUnits(freq='Hz',start='s',end='s',length='s')
def cosine(amp=0.0,freq=10e6,
    start=0.,end=None,length=1e-6,
    fs=1.8e9):
    if end is None:
        end = start + length
    return amp * np.cos( freq * np.arange(start,end,1./fs) )



@convertUnits(freq='Hz',start='s',end='s',length='s')
def readoutPulse(amp=0.0,freq=10e6,
    start=0.,end=None,length=1e-6,fs=1.8e9):
    """
    Args: fs (float): sampling rate
    Returns: numpy.array
    """
    if end is None:
        end = start + length
    return [
    cosine(amp=amp,freq=freq,phase=phase,start=start,end=end,length=length,
    fs=fs),
    sine(amp=amp,freq=freq,phase=phase,start=start,end=end,length=length,
    fs=fs)
    ]


def readoutPulseMany(amps: list = [0.],freqs: list = [10e6],
    start: float = 0., end: float = 1e-6,length=1e-6,
    fs: float = 1.8e9):
    """
    Args: fs (float): sampling rate
    Returns: numpy.array
    """
    if end is None:
        end = start + length

    _amps = np.asarray(amps)
    _freqs = np.asarray(freqs)
    
    para_length = len(_amps)
    if  len(_freqs) != para_length:
        raise ValueError("All of the list as parameters should has the same length")
    
    # pulse_I = 0.

    # Do not use np.sum([ for i in range()]) to generate pulse
    # Alough it is short, it cost more times
    for idx in range(para_length):
        if idx == 0:
            pulse_I = cosine(_amps[0],_freqs[0],start,end,fs)
        else:
            pulse_I += cosine(_amps[0],_freqs[0],start,end,fs)

    for idx in range(para_length):
        if idx == 0:
            pulse_Q = sine(_amps[0],_freqs[0],start,end,fs)
        else:
            pulse_Q += sine(_amps[0],_freqs[0],start,end,fs) 
    return [pulse_I,pulse_Q]




# @convertUnits(start='s',end='s',amp=None,length='s')
# def square(start=50e-9,end=None,amp=1.0,length=100e-9):
#     if end is None: end = start + length
#     timeFunc = lambda t: amp*(start<=t<end)
#     envelopes = Envelope(timeFunc,None,start,end)
#     return envelopes


# @convertUnits(start='s',end='s',freq='Hz',length='s')
# def readout(amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
#     if end is None: end = start + length
#     timeFunc1 = lambda t: amp*np.cos(2*pi*freq*(t-start)+phase)*(start<=t<end)
#     env1 = Envelope(timeFunc1,None,start,end)
#     timeFunc2 = lambda t: amp*np.sin(2*pi*freq*(t-start)+phase)*(start<=t<end)
#     env2 = Envelope(timeFunc2,None,start,end)
#     return env1,env2

