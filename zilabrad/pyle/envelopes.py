# Control envelopes in time domain and frequency domain
#
# For a pair of functions g(t) <-> h(f) we use the following
# convention for the Fourier Transform:
#
#          / +inf
#         |
# h(f) =  | g(t) * exp(-2j*pi*f*t) dt
#         |
#        / -inf
#
#          / +inf
#         |
# g(t) =  | h(f) * exp(2j*pi*f*t) df
#         |
#        / -inf
#
# Note that we are working with frequency in GHz, rather than
# angular frequency.  Also note that the sign convention is opposite
# to what is normally taken in physics.  But this is the convention
# used here and in the DAC deconvolution code, so you should use it.

 # have to do this so we get math std library

import math

import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

from pyle.util import convertUnits


class Envelope(object):
    """Represents a control envelope as a function of time or frequency.
    
    Envelopes can be added to each other or multiplied by constant values.
    Multiplication of two envelopes and addition of a constant value (other
    than zero) are not equivalent in time and fourier domains, so these
    operations are not supported.
    
    Envelopes keep track of their start and end time, and when added
    together the new envelope will use the earliest start and latest end,
    to cover the entire range of its constituent parts.
    
    Envelopes can be evaluated as functions of time or frequency using the
    fourier flag.  By default, they are evaluated as a function of time.
    """
    def __init__(self, timeFunc, freqFunc, start=None, end=None):
        self.timeFunc = timeFunc
        self.freqFunc = freqFunc
        self.start = start
        self.end = end

    def __call__(self, x, fourier=False):
        if fourier:
            return self.freqFunc(x)
        else:
            return self.timeFunc(x)

    def __add__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return self.timeFunc(t) + other.timeFunc(t)
            def freqFunc(f):
                return self.freqFunc(f) + other.freqFunc(f)
            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            raise Exception("Cannot add a constant to hybrid time/fourier envelopes")
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return self.timeFunc(t) - other.timeFunc(t)
            def freqFunc(f):
                return self.freqFunc(f) - other.freqFunc(f)
            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return -self
            raise Exception("Cannot subtract a constant from hybrid time/fourier envelopes")
        
    def __rsub__(self, other):
        if isinstance(other, Envelope):
            start, end = timeRange((self, other))
            def timeFunc(t):
                return other.timeFunc(t) - self.timeFunc(t)
            def freqFunc(f):
                return other.freqFunc(f) - self.freqFunc(f)
            return Envelope(timeFunc, freqFunc, start=start, end=end)
        else:
            # if we try to add envelopes with the built in sum() function,
            # the first envelope is added to 0 before adding the rest.  To support
            # this, we add a special case here since adding 0 in time or fourier
            # is equivalent
            if other == 0:
                return self
            raise Exception("Cannot subtract a constant from hybrid time/fourier envelopes")

    def __mul__(self, other):
        if isinstance(other, Envelope):
            raise Exception("Hybrid time/fourier envelopes can only be multiplied by constants")
        else:
            def timeFunc(t):
                return self.timeFunc(t) * other
            def freqFunc(f):
                return self.freqFunc(f) * other
            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)
    __rmul__ = __mul__

    def __div__(self, other):
        if isinstance(other, Envelope):
            raise Exception("Hybrid time/fourier envelopes can only be divided by constants")
        else:
            def timeFunc(t):
                return self.timeFunc(t) / other
            def freqFunc(f):
                return self.freqFunc(f) / other
            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)
        
    def __rdiv__(self, other):
        if isinstance(other, Envelope):
            raise Exception("Hybrid time/fourier envelopes can only be divided by constants")
        else:
            def timeFunc(t):
                return other / self.timeFunc(t)
            def freqFunc(f):
                return other / self.freqFunc(f)
            return Envelope(timeFunc, freqFunc, start=self.start, end=self.end)

    def __neg__(self):
        return -1 * self
    
    def __pos__(self):
        return self


_zero = lambda x: 0*x


# empty envelope
NOTHING = Envelope(_zero, _zero, start=None, end=None)


@convertUnits(t0='ns', w='ns', amp=None, phase=None, df='GHz')
def gaussian(t0, w, amp=1.0, phase=0.0, df=0.0):
    """A gaussian pulse with specified center and full-width at half max."""
    sigma = w / np.sqrt(8*np.log(2)) # convert fwhm to std. deviation
    def timeFunc(t):
        return amp * np.exp(-(t-t0)**2/(2*sigma**2) - 2j*np.pi*df*(t-t0) + 1j*phase)
    
    sigmaf = 1 / (2*np.pi*sigma) # width in frequency space
    ampf = amp * np.sqrt(2*np.pi*sigma**2) # amp in frequency space
    def freqFunc(f):
        return ampf * np.exp(-(f+df)**2/(2*sigmaf**2) - 2j*np.pi*f*t0 + 1j*phase)
    
    return Envelope(timeFunc, freqFunc, start=t0-w, end=t0+w)


@convertUnits(t0='ns', len='ns', amp=None)
def triangle(t0, len, amp, fall=True):
    """A triangular pulse, either rising or falling."""
    if not fall:
        return triangle(t0+len, -len, amp, fall=True)
    
    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)
    
    if len == 0 or amp == 0:
        return Envelope(_zero, _zero, start=tmin, end=tmax)
    
    def timeFunc(t):
        return amp * (t >= tmin) * (t < tmax) * (1 - (t-t0)/len)
    
    def freqFunc(f):
        # this is tricky because the fourier transform has a 1/f term, which blows up for f=0
        # the z array allows us to separate the zero-frequency part from the rest
        z = f == 0
        f = 2j*np.pi*(f + z)
        return amp * ((1-z)*np.exp(-f*t0)*(1.0/f - (1-np.exp(-f*len))/(f**2*len)) + z*len/2.0)
    
    return Envelope(timeFunc, freqFunc, start=tmin, end=tmax)


@convertUnits(t0='ns', len='ns', amp=None, overshoot=None)
def rect(t0, len, amp, overshoot=0.0, overshoot_w=1.0):
    """A rectangular pulse with sharp turn on and turn off.
    
    Note that the overshoot_w parameter, which defines the FWHM of the gaussian overshoot peaks
    is only used when evaluating the envelope in the time domain.  In the fourier domain, as is
    used in the dataking code which uploads sequences to the boards, the overshoots are delta
    functions.
    """
    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)
    tmid = (tmin + tmax) / 2.0
    
    overshoot *= np.sign(amp) # overshoot will be zero if amp is zero
    
    # to add overshoots in time, we create an envelope with two gaussians
    if overshoot:
        o_w = overshoot_w
        o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING
    def timeFunc(t):
        return (amp * (t >= tmin) * (t < tmax) +
                overshoot * o_env(t))
    
    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    def freqFunc(f):
        return (amp * abs(len) * np.sinc(len*f) * np.exp(-2j*np.pi*f*tmid) +
                overshoot * (np.exp(-2j*np.pi*f*tmin) + np.exp(-2j*np.pi*f*tmax)))
    
    return Envelope(timeFunc, freqFunc, start=tmin, end=tmax)


@convertUnits(t0='ns', len='ns', w='ns', amp=None)
def flattop(t0, len, w, amp=1.0, overshoot=0.0, overshoot_w=1.0):
    """A rectangular pulse convolved with a gaussian to have smooth rise and fall."""
    tmin = min(t0, t0+len)
    tmax = max(t0, t0+len)
    
    overshoot *= np.sign(amp) # overshoot will be zero if amp is zero
    
    # to add overshoots in time, we create an envelope with two gaussians
    a = 2*np.sqrt(np.log(2)) / w
    if overshoot:
        o_w = overshoot_w
        o_amp = 2*np.sqrt(np.log(2)/np.pi) / o_w # total area == 1
        o_env = gaussian(tmin, o_w, o_amp) + gaussian(tmax, o_w, o_amp)
    else:
        o_env = NOTHING
    def timeFunc(t):
        return (amp * (erf(a*(tmax - t)) - erf(a*(tmin - t)))/2.0 +
                overshoot * o_env(t))
    
    # to add overshoots in frequency, use delta funcs (smoothed by filters)
    rect_env = rect(t0, len, 1.0)
    kernel = gaussian(0, w, 2*np.sqrt(np.log(2)/np.pi) / w) # area = 1
    def freqFunc(f):
        return (amp * rect_env(f, fourier=True) * kernel(f, fourier=True) + # convolve with gaussian kernel
                overshoot * (np.exp(-2j*np.pi*f*tmin) + np.exp(-2j*np.pi*f*tmax)))
    
    return Envelope(timeFunc, freqFunc, start=tmin, end=tmax)


@convertUnits(t0='ns', rise='ns', hold='ns', fall='ns', amp=None)
def trapezoid(t0, rise, hold, fall, amp):
    """Create a trapezoidal pulse, built up from triangles and rectangles."""
    return (triangle(t0, rise, amp, fall=False) +
            rect(t0+rise, hold, amp) +
            triangle(t0+rise+hold, fall, amp))


@convertUnits(df='GHz')
def mix(env, df=0.0):
    """Apply sideband mixing at difference frequency df."""
    def timeFunc(t):
        return env(t) * np.exp(-2j*np.pi*df*t)
    def freqFunc(f):
        return env(f + df, fourier=True)
    return Envelope(timeFunc, freqFunc, start=env.start, end=env.end)


@convertUnits(dt='ns')
def deriv(env, dt=0.1):
    """Get the time derivative of a given envelope."""
    def timeFunc(t):
        return (env(t+dt) - env(t-dt)) / (2*dt)
    def freqFunc(f):
        return 2j*np.pi*f * env(f, fourier=True)
    return Envelope(timeFunc, freqFunc, start=env.start, end=env.end)


@convertUnits(dt='ns')
def shift(env, dt=0.0):
    """Shift an envelope in time."""
    def timeFunc(t):
        return env(t - dt)
    def freqFunc(f):
        return env(f, fourier=True) * np.exp(-2j*np.pi*f*dt)
    return Envelope(timeFunc, freqFunc, start=env.start, end=env.end)


# utility functions

def timeRange(envelopes):
    """Calculate the earliest start and latest end of a list of envelopes.
    
    Returns a tuple (start, end) giving the time range.  Note that one or
    both of start and end may be None if the envelopes do not specify limits.
    """
    starts = [env.start for env in envelopes if env.start is not None]
    start = min(starts) if len(starts) else None
    ends = [env.end for env in envelopes if env.end is not None]
    end = max(ends) if len(ends) else None
    return start, end


def fftFreqs(time=1024):
    """Get a list of frequencies for evaluating fourier envelopes.
    
    The time is rounded up to the nearest power of two, since powers
    of two are best for the fast fourier transform.  Returns a tuple
    of frequencies to be used for complex and for real signals.
    """
    nfft = 2**int(math.ceil(math.log(time, 2)))
    f_complex = np.fft.fftfreq(nfft)
    f_real = f_complex[:nfft/2+1]
    return f_complex, f_real


def ifft(envelope, t0=-200, n=1000):
    f = np.fft.fftfreq(n)
    return np.fft.ifft(envelope(f, fourier=True) * np.exp(2j*np.pi*t0*f))


def fft(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    return np.fft.fft(envelope(t))


def plotFT(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = ifft(envelope, t0, n)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))


def plotTD(envelope, t0=-200, n=1000):
    t = t0 + np.arange(n)
    y = envelope(t)
    plt.plot(t, np.real(y))
    plt.plot(t, np.imag(y))
    


