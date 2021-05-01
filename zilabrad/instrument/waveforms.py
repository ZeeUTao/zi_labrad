# -*- coding: utf-8 -*-
"""
waveform script

Created on 2020.10.06
@author: Tao Ziyu
"""

import numpy as np
from math import ceil
from zilabrad.pyle.tools import singleton


@singleton
class waveGenerator(object):
    """
    Represents a control waveform as a function of time

    """
    def __init__(self, name='default', address=None):
        self.name = name

    # We can use pyle.envelopes to define some complicated waveforms.
    # Some frequently used waveforms are provided below

    def func2array_withoutNumpy(self, func,
                                start: float = None,
                                end: float = None,
                                fs: float = 2.4e9,
                                mode: str = 'AWG'):
        """
        Try to use numpy, or it will be slow in python
        """

        if hasattr(func, 'start') and start is None:
            start = func.start
        if hasattr(func, 'end') and end is None:
            end = func.end

        if end <= start:
            return []

        interval = 1./fs
        steps = ceil((end-start)*fs)
        if mode.upper() == 'AWG':
            return [func(start + idx*interval) for idx in range(steps)]
        elif mode.upper() == 'IQ':
            _list = [func(start + idx*interval) for idx in range(steps)]
            return [np.real(_list), np.imag(_list)]

    def func2array_withNumpy(self, func,
                             start: float = None,
                             end: float = None,
                             fs: float = 2.4e9,
                             mode: str = 'AWG'):
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
        if mode.upper() == 'AWG':
            return func(np.arange(start, end, interval))
        elif mode.upper() == 'IQ':
            return [func(np.arange(start, end, interval)).real,
                    func(np.arange(start, end, interval)).imag]

    def func2array(self, func,
                   start: float = None,
                   end: float = None,
                   fs: None or float = None,
                   mode: str = 'AWG'):
        """
        Args:
            func: func(t) contains only Numpy function
            func can also be wrapped by zilabrad.pyle.envelopes.Envelope
        """
        try:
            result = self.func2array_withNumpy(func, start, end, fs, mode)
        except Exception:
            result = self.func2array_withoutNumpy(func, start, end, fs, mode)
            return result
        else:
            return result
