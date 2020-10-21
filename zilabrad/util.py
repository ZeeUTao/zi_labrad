
from abc import ABC
from functools import wraps

    
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
        
        If you lose your parameters for some reason, and want to create it again, 
        dev1 = MyDevice('1')
        The object will not be initialized again, but just give it to you, since it always 
        stored in your cache even you forget it.
    """

    class SingletonFactory(ABC):
        instance = {}
        def __new__(cls,obj_name,*args, **kwargs):
            if obj_name not in cls.instance:
                cls.instance[obj_name] = class_(obj_name,*args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory
    
def singleton(class_):

    class SingletonFactory(ABC):
        instance = None
        def __new__(cls,*args, **kwargs):
            if not cls.instance:
                cls.instance = class_(*args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory
    
    
    
    