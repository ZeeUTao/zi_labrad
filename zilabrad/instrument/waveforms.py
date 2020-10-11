# -*- coding: utf-8 -*-
"""
waveform script

Created on 2020.10.06
@author: Tao Ziyu
"""

import numpy as np
from math import ceil,pi
from zilabrad.pyle.envelopes import Envelope,NOTHING

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
    def __init__(self,all_length=1e-6,fs=1.8e9,origin=0):
        self.fs = fs
        self.sample_number = ceil(all_length*self.fs/16)*16 ## QA lenght最小16,最小单位间隔8; HD length最小32,最小单位间隔16;
        self.tlist = []
        # self.len = self.sample_number/self.fs ## set as 16 sample integral multiple;
        # self.origin = origin ## mark real start in waveform; set QA trigger as 0  
        # self.tlist = np.asarray([k/self.fs+self.origin for k in range(self.sample_number)])
        # self.bias_sample = 0

    # We can use env from pyle.envelopes to define some complicated waveforms    
    # use: 
    # w_qa = waveform()
    # w_qa(env)
    # on the other hand, some frequently used waveforms are provided below
    def func2array(self,envelopes,fs=None):
        if len(self.tlist) == 0:
            start = envelopes.start
            end = envelopes.end
            if fs == None:
                fs = self.fs
            tlist = np.arange(start,end,1/fs)
        else:
            tlist = self.tlist

        pulse = [envelopes(t) for t in tlist]
        return pulse
    
    def __call__(self,envelopes):
        return self.func2array(envelopes)
        
    @convertUnits(origin='s',end='s')
    def set_tlist(self,origin,end,fs):
        self.tlist = np.arange(origin,end,1/fs)
        
    # @convertUnits(amp='V')
    # def bias(self,amp=0,length=None):
    #     if length != None:
    #         self.bias_sample = ceil(length*self.fs/16)*16
    #     # pulse = [amp,self.bias_len]
    #     return np.ones(self.sample_number)*amp


    # def readout(self,qubits):
    #     pulse = np.array([np.zeros(self.sample_number),np.zeros(self.sample_number)])
    #     for q in qubits:
    #         pulse[0] += np.asarray([q.power_r*np.cos(2*pi*q.demod_freq*t+q.demod_phase) for t in self.tlist])
    #         pulse[1] += np.asarray([q.power_r*np.sin(2*pi*q.demod_freq*t+q.demod_phase) for t in self.tlist])
    #     pulse = pulse/len(qubits)
    #     return pulse


    # @convertUnits(start='s',end='s',amp=None,length='s')
    # def square(self,start=50e-9,end=None,amp=1.0,length=100e-9):
    #     if end is None: end = start + length
    #     timeFunc = lambda t: amp*(start<=t<end)
    #     envelopes = Envelope(timeFunc,None,start,end)
    #     return self.func2array(envelopes)

    # @convertUnits(start='s',end='s',freq='Hz',length='s')
    # def sine(self,amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
    #     if end is None: end = start + length
    #     timeFunc = lambda t: amp*np.sin(2*pi*freq*(t-start)+phase)*(start<=t<end)
    #     envelopes = Envelope(timeFunc,None,start,end)
    #     return self.func2array(envelopes)
    
    # @convertUnits(start='s',end='s',freq='Hz',length='s')
    # def cosine(self,amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
    #     if end is None: end = start + length
    #     timeFunc = lambda t: amp*np.cos(2*pi*freq*(t-start)+phase)*(start<=t<end)
    #     envelopes = Envelope(timeFunc,None,start,end)
    #     return self.func2array(envelopes)



    # @convertUnits(start='s',end='s',amp=None,length='s')
    # def square(self,start=50e-9,end=None,amp=1.0,length=100e-9):
    #     if end is None: end = start + length
    #     timeFunc = lambda t: amp*(start<=t<end)
    #     envelopes = Envelope(timeFunc,None,start,end)
    #     return self.func2array(envelopes)



@convertUnits(start='s',end='s',freq='Hz',length='s')
def sine(amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
    if end is None: end = start + length
    timeFunc = lambda t: amp*np.sin(2*pi*freq*(t-start)+phase)*(start<=t<end)
    envelopes = Envelope(timeFunc,None,start,end)
    return envelopes

@convertUnits(start='s',end='s',freq='Hz',length='s')
def cosine(amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
    if end is None: end = start + length
    timeFunc = lambda t: amp*np.cos(2*pi*freq*(t-start)+phase)*(start<=t<end)
    envelopes = Envelope(timeFunc,None,start,end)
    return envelopes


@convertUnits(start='s',end='s',amp=None,length='s')
def square(start=50e-9,end=None,amp=1.0,length=100e-9):
    if end is None: end = start + length
    timeFunc = lambda t: amp*(start<=t<end)
    envelopes = Envelope(timeFunc,None,start,end)
    return envelopes


@convertUnits(start='s',end='s',freq='Hz',length='s')
def readout(amp=0.1,phase=0.0,start=0,end=None,freq=10e6,length=100e-9):
    if end is None: end = start + length
    timeFunc1 = lambda t: amp*np.cos(2*pi*freq*(t-start)+phase)*(start<=t<end)
    env1 = Envelope(timeFunc1,None,start,end)
    timeFunc2 = lambda t: amp*np.sin(2*pi*freq*(t-start)+phase)*(start<=t<end)
    env2 = Envelope(timeFunc2,None,start,end)
    return env1,env2

