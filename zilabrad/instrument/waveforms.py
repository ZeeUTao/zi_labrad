# -*- coding: utf-8 -*-
"""
waveform script

Created on 2020.10.06
@author: Tao Ziyu
"""

import numpy as np
from math import ceil, pi
import math
import inspect
import functools
from numba import jit

from zilabrad.pyle.envelopes import Envelope, NOTHING
from zilabrad.util import singleton


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
        # prefer over subclass check: isinstance(v, Value)
        if hasattr(v, 'unit'):
            if u is None:
                return v[v.unit]
            else:
                return v[u]
        else:
            return v

    def wrap(f):
        args = inspect.getfullargspec(f)[0]
        for arg in unitdict:
            if arg not in args:
                raise Exception(
                    'function %s does not take arg "%s"' % (f, arg)
                    )
        # unitdict maps argument names to units
        # posdict maps argument positions to units
        posdict = dict((i, unitdict[arg])
                       for i, arg in enumerate(args) if arg in unitdict)

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


@singleton
class waveServer(object):
    """
    Represents a control waveform as a function of time

    """

    def __init__(
            self, all_length=1e-6, fs=1.8e9,
            origin=0, name='default'):
        self.name = name
        self.fs = fs
        # QA lenght最小16,最小单位间隔8; HD length最小32,最小单位间隔16;
        self.sample_number = ceil(all_length*self.fs/16)*16

    # We can use pyle.envelopes to define some complicated waveforms.
    # Some frequently used waveforms are provided below

    def set_tlist(self, start, end, fs):
        """just set parameters, not really generate a list which is slow
        """
        self.start = start
        self.end = end
        self.fs = fs

    def func2array_withoutNumpy(self, func,
                                start: float = None,
                                end: float = None,
                                fs: None or float = None):
        """
        Try to use numpy, or it will be slow in python
        """
        if fs is None:
            fs = self.fs

        if hasattr(func, 'start') and start is None:
            start = func.start
        if hasattr(func, 'end') and end is None:
            end = func.end

        if end <= start:
            return []

        interval = 1./fs
        steps = ceil((end-start)*fs)

        return [func(start + idx*interval) for idx in range(steps)]

    def func2array_withNumpy(self, func,
                             start: float = None,
                             end: float = None,
                             fs: None or float = None):
        """
        Args:
            func: func(t) contains only Numpy function
            func can also be wrapped by zilabrad.pyle.envelopes.Envelope
        """
        if fs is None:
            fs = self.fs
        if hasattr(func, 'start') and start is None:
            start = func.start
        if hasattr(func, 'end') and end is None:
            end = func.end
        if end <= start:
            return []

        interval = 1./fs
        return func(np.arange(start, end, interval))

    def func2array(self, func,
                   start: float = None,
                   end: float = None,
                   fs: None or float = None):
        """
        Args:
            func: func(t) contains only Numpy function
            func can also be wrapped by zilabrad.pyle.envelopes.Envelope
        """
        try:
            result = self.func2array_withNumpy(func, start, end, fs)
        except Exception:
            result = self.func2array_withoutNumpy(func, start, end, fs)
            return result
        else:
            return result


# Collection of Envelope timeFunc
# Envelope to define timefunction, which can be added, multiplied...


@convertUnits(start='s', end='s', amp=None, length='s')
def square(start=50e-9, end=None, amp=1.0, length=100e-9):
    if end is None:
        end = start + length
    # @jit(nopython=True)

    def timeFunc(t):
        return amp*(t < end)*(t >= start)
    envelopes = Envelope(timeFunc, None, start, end)
    return envelopes


@convertUnits(start='s', end='s', freq='Hz', length='s')
def sine(amp=0.1, phase=0.0, start=0, end=None, freq=10e6, length=100e-9):
    if end is None:
        end = start + length

    def timeFunc(t): return amp*np.sin(2*pi*freq *
                                       (t-start)+phase)*(t < end)*(t >= start)
    envelopes = Envelope(timeFunc, None, start, end)
    return envelopes


@convertUnits(start='s', end='s', freq='Hz', length='s')
def cosine(amp=0.1, phase=0.0, start=0, end=None, freq=10e6, length=100e-9):
    if end is None:
        end = start + length

    def timeFunc(t): return amp*np.cos(2*pi*freq *
                                       (t-start)+phase)*(t < end)*(t >= start)
    envelopes = Envelope(timeFunc, None, start, end)
    return envelopes


@convertUnits(start='s', end='s', freq='Hz', length='s')
def readout(amp=0.1, phase=0.0, start=0, end=None, freq=10e6, length=100e-9):
    if end is None:
        end = start + length

    def timeFunc1(t): return amp*np.cos(2*pi*freq *
                                        (t-start)+phase)*(t < end)*(t >= start)
    env1 = Envelope(timeFunc1, None, start, end)
    def timeFunc2(t): return amp*np.sin(2*pi*freq *
                                        (t-start)+phase)*(t < end)*(t >= start)
    env2 = Envelope(timeFunc2, None, start, end)
    return env1, env2

# Collection of Array timeFunc, which returns an array


@convertUnits(start='s', end='s', amp=None, length='s')
def squareArray(amp=1.0, start=0., end=None, length=100e-9, fs=1.8e9):
    if end is None:
        end = start + length
    steps = ceil((end-start)*fs)
    return amp*np.ones(steps)


@convertUnits(freq='Hz', start='s', end='s', length='s')
def spectroscopyPulseArray(amp=0.0, freq=10e6, phase=0.,
                           start=0., end=None, length=1e-6,
                           fs=1.8e9):
    """
    Args: fs (float): sampling rate
    Returns: numpy.array
    """
    if end is None:
        end = start + length
    return amp * np.cos(freq * np.arange(start, end, 1./fs) + phase)


@convertUnits(freq='Hz', start='s', end='s', length='s')
def func_with_envelope(
        amp=0.0, freq=10e6, start=0., end=None, length=1e-6,
        fs=1.8e9):
    """
    It's an example for using zilabrad.pyle.envelopes.Envelope
    to define timefunction.
    You can also directly define it in your script like multiplied.py.
    """
    if end is None:
        end = start + length

    def timeFunc(t):
        return amp*np.cos(freq * t)
    return Envelope(timeFunc, None, start, end)


@convertUnits(freq='Hz', start='s', end='s', length='s')
def sineArray(amp=0.0, freq=10e6,
              start=0., end=None, length=1e-6,
              fs=1.8e9):
    if end is None:
        end = start + length
    return amp * np.sin(freq * np.arange(start, end, 1./fs))


@convertUnits(freq='Hz', start='s', end='s', length='s')
def cosineArray(amp=0.0, freq=10e6,
                start=0., end=None, length=1e-6,
                fs=1.8e9):
    if end is None:
        end = start + length
    return amp * np.cos(freq * np.arange(start, end, 1./fs))


@convertUnits(freq='Hz', start='s', end='s', length='s')
def readoutArray(amp=0.0, freq=10e6,
                 start=0., end=None, length=1e-6, fs=1.8e9):
    """
    Args: fs (float): sampling rate
    Returns: numpy.array
    """
    if end is None:
        end = start + length
    return [
        cosine(
            amp=amp, freq=freq, phase=phase, start=start, end=end,
            length=length, fs=fs),
        sine(
            amp=amp, freq=freq, phase=phase, start=start, end=end,
            length=length, fs=fs)
    ]


def readoutArrayMany(amps: list = [0.], freqs: list = [10e6],
                     start: float = 0., end: float = 1e-6, length=1e-6,
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
    if len(_freqs) != para_length:
        raise ValueError(
            "All of the list as parameters should has the same length")

    # pulse_I = 0.

    # Do not use np.sum([ for i in range()]) to generate pulse
    # Alough it is short, it cost more times
    for idx in range(para_length):
        if idx == 0:
            pulse_I = cosine(_amps[0], _freqs[0], start, end, fs)
        else:
            pulse_I += cosine(_amps[0], _freqs[0], start, end, fs)

    for idx in range(para_length):
        if idx == 0:
            pulse_Q = sine(_amps[0], _freqs[0], start, end, fs)
        else:
            pulse_Q += sine(_amps[0], _freqs[0], start, end, fs)
    return [pulse_I, pulse_Q]
