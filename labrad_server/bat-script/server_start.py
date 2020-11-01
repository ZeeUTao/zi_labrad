# -*- coding: utf-8 -*-
"""
Simple python3 interface for running bash commands
"""

import os
from abc import ABC
from functools import wraps
import sys


_glob_paras = {"script_dir":r"M:\labrad_server"}


def singletonMany(class_):
    class SingletonFactory(ABC):
        instance = {}
        def __new__(cls,idx,*args, **kwargs):
            if idx not in cls.instance:
                cls.instance[idx] = class_(idx,*args, **kwargs)
            return cls.instance
    SingletonFactory.register(class_)
    return SingletonFactory
    

class Servers(object):
    """
    servers factory, every server is singleton

        
    Example: 
    >>> server = Servers(idx=1,name='1',func='1')
    >>> Servers(idx=2,name='2',func='2')
    >>> Servers(idx=2,name='2',func='2')
    
    >>> server.all
    {name1 : obj1, name2 : obj2, name3 : obj3}
    >>> server().idx
    1
    >>> server.select(2).idx
    2
    >>> server().idx
    2
    >>> server[1].idx,server[2].idx,server[3].idx
    (1, 2, 3)
    """
    
    _instance = {}
    _default_idx = 1
    
    def __new__(cls, *args, **kwargs):
        """
        if no specified idx, idx = 0
        """
        _default_idx = cls._default_idx
        if "idx" in kwargs:
            idx = kwargs["idx"]
        elif len(args)>0:
            idx = args[0]
        else:
            idx = _default_idx
        
        # if instance with the same idx exist, do not create again
        if idx in cls._instance:
            return cls._instance[idx]
        else:
            obj = super().__new__(cls)
            cls._instance[idx] = obj
            return obj
            
    def __init__(self,idx,name,func):
        self.idx = idx
        self.name = name
        self.func = func
    
    @staticmethod
    def default_obj():
        return _instance[_default_idx]
        
    def __call__(self):
        cls = self.__class__
        return cls._instance[cls._default_idx]
        
    def __getitem__(self, key):
        return self.__class__._instance[key]


    def select(self,idx=None):
        if idx != None:
            self.__class__._default_idx = idx
            return self[idx]
        else:
            return self.__call__()
    
    @property
    def all(self):
        return self.__class__._instance
    
    def run(self,*args,**kwargs):
        if hasattr(self.func,"__call__"):
            self.func(*args,**kwargs)



def start_cmd_command(command):
    os.system(r"start cmd /k %s"%(command))

def start_cmd_ipython3(path,dir = None):
    if dir == None:
        path = os.path.join(_glob_paras['script_dir'],path) 
    def func():
        commands = r"start cmd /k ipython3 %s"%(path)
        os.system(commands)
    return func
    
def start_labrad(
    path = r"M:\scalabrad-0.8.3\bin\labrad",
    registryFile = r"file:///M:/Registry?format=delphi"):
    start_cmd_command("%s --registry %s"%(path,registryFile))

def add_servers():
    # add servers    
    server = Servers(idx=1,
        name='scalabrad',
        func=start_labrad)
    Servers(
        idx=2,
        name = "py3_data_vault",
        func=start_cmd_ipython3(r"py3_data_vault.py"))
    Servers(
        idx=3,
        name = "gpib_server",
        func=start_cmd_ipython3(r"gpib_server.py"))
    Servers(
        idx=4,
        name = "gpib_device_manager",
        func=start_cmd_ipython3(r"gpib_device_manager.py"))
    Servers(
        idx=5,
        name = "anritsu",
        func=start_cmd_ipython3(r"anritsu.py"))
    return server


def dict2prettyWords(serverDict):
    words = ""
    for key,ser in serverDict.items():
        words += f"\n{key} {ser.name}"
    return words
    
def choose_server():
    arg = input("Input the index and press 'Enter' to start server \n")
    try:
        choose = eval(arg)
    except: choose = arg
    
    print("You choose %s \n"%(choose))
    return choose




def helper():
    print("="*30)
    words = "This is server_start bash by Ziyu Tao"
    print(words)
    print("="*30)
    
    
    helpwords = "Present Directory for running scripts is %s "%_glob_paras["script_dir"]
    print(helpwords)
    
    
    newDir = input("Enter new directory to change or use the default to continue")
    if os.path.exists(newDir):
        _glob_paras["script_dir"] = newDir
        print("Present Directory for running scripts is %s "%_glob_paras["script_dir"])        
        return
    else:
        print(f"Empty input or Invalid Directory, %s"%(newDir))
        print("Present Directory for running scripts is %s "%_glob_paras["script_dir"])
        return 
    
def main():
    server = add_servers()
    serverDict = dict(server.all)
    
    helper()
    
    while True:
        words = dict2prettyWords(serverDict)
        print(words)
        print("a start all")
        print("q quit")
        print("c clear screen")
        print("h help")
        print("")
        
        choice = choose_server()
        if choice=='q':
            break
            
        elif choice == 'c':
            os.system('cls')
            
        elif choice == 'h':
            helper()
            
        elif choice == 'a':
            for key in serverDict.keys():
                server[key].run()
           
        elif choice in serverDict.keys():
            server[choice].run()
        else:
            print('Invalid input %s '%(choice))



# seefiles = sorted((fn for fn in os.listdir(r"M:/xxx") if fn.endswith('.py')))

if __name__ == '__main__':
    main()