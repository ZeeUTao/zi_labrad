
from abc import ABC



def singleton(class_):
    """
    We do not want to initialize the same device after we created it.
    device_id: required to identify whether the device object has been created
    
    Example: 
        
        @singleton
        class MyDevice(object)
            def __init__(self,device_id,*args, **kwargs):
        
        Therefore, we can use like: 
        
        device_ids = ['1','2']
        for i,devId in enumerate(device_ids):
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
        def __new__(cls,device_id,*args, **kwargs):
            if device_id not in cls.instance:
                cls.instance[device_id] = class_(device_id,*args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory