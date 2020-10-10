from contextlib import contextmanager
import functools
import inspect

import numpy as np


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
        if hasattr(v, 'value'): # prefer over subclass check: isinstance(v, Value)
            if u is None:
                return v.value
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


# processPriority
#
# This function allows us to dynamically reduce the priority of the currently-
# running process.  It can be useful to decrease the priority while running
# simulations, for example, so as not to lock up the machine while work
# is in progress.  If you have a multi-processor machine, this is probably
# not necessary.  For cross-platform compatibility, we catch the import error
# and make a dummy priority manager for platforms other than windows.

try:
    import win32api, win32process, win32con
    
    @contextmanager
    def processPriority(priority=1):
        """Context manager to execute a with clause as a different priority."""        
        priorityclasses = [win32process.IDLE_PRIORITY_CLASS,
                           win32process.BELOW_NORMAL_PRIORITY_CLASS,
                           win32process.NORMAL_PRIORITY_CLASS,
                           win32process.ABOVE_NORMAL_PRIORITY_CLASS,
                           win32process.HIGH_PRIORITY_CLASS,
                           win32process.REALTIME_PRIORITY_CLASS]
        pid = win32api.GetCurrentProcessId()
        handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
        original_priority = win32process.GetPriorityClass(handle)
        win32process.SetPriorityClass(handle, priorityclasses[priority])
        try:
            yield # execute with-statement here
        finally:
            win32process.SetPriorityClass(handle, original_priority)

except ImportError:
    @contextmanager
    def processPriority(priority=1):
        yield # a do-nothing context manager for non-windows compatibility


def memoize(f):
    """Wraps a one-argument function so that it caches computed results."""
    _cache = {}
    @functools.wraps(f)
    def wrapped(a):
        try:
            return _cache[a]
        except KeyError:
            val = _cache[a] = f(a)
            return val
    wrapped._cache = _cache
    return wrapped

@memoize
def sierp(n):
    if n == 0:
        return np.array([[1.0]])
    s = sierp(n-1)
    z = np.zeros_like(s)
    return np.vstack([np.hstack([s, z]),
                      np.hstack([s, s])])

@memoize
def sierpinv(n):
    return np.linalg.inv(sierp(n))



