# -*- coding: utf-8 -*-
"""
some basic tools
"""
from abc import ABC
from functools import wraps
import gc
import inspect



#####################
###   Unit Tools  ###
#####################

def Unit2SI(a):
    if not hasattr(a, 'unit'):
        return a
    elif a.unit in ['GHz', 'MHz']:
        return a['Hz']
    elif a.unit in ['ns', 'us']:
        return a['s']
    else:
        return a[a.unit]


def Unit2num(a):
    if not hasattr(a, 'unit'):
        return a
    else:
        return a[a.unit]



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

        @wraps(f)
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




#######################
### singleton class ###
#######################

def singletonMany(class_):
    """
    We do not want to initialize the same device after we created it.
    obj_name: required to identify whether the device object has been created
    Example:
        @singleton
        class MyDevice(object)
            def __init__(self,obj_name,*args, **kwargs):
        Therefore, we can use like:
        obj_names = ['1','2']
        for i,devId in enumerate(obj_names):
            if i == 0:
                devs = MyDevice(devId)
            else:
                MyDevice(devId)

        Then if you call 'devs', it gives a dict of those objects
        devs = {'1':MyDevice('1'),'2':MyDevice('2')}

        If you lose your parameters for some reason, and want to
        create it again, use: dev1 = MyDevice('1')
        The object will not be initialized again, but just give
        it to you, since it always stored in your cache even you forget it.
    """
    class SingletonFactory(ABC):
        instance = {}

        def __new__(cls, obj_name, *args, **kwargs):
            if obj_name not in cls.instance:
                cls.instance[obj_name] = class_(obj_name, *args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory


def singleton(class_):
    class SingletonFactory(ABC):
        instance = None

        def __new__(cls, *args, **kwargs):
            if not cls.instance:
                cls.instance = class_(*args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory


def clear_singletonMany(class_):
    _keys = list(class_.instance.keys())
    for _key in _keys:
        del class_(_key)[_key]
    del _keys
    gc.collect()
    return




