# -*- coding: utf-8 -*-
"""
Simple python3 interface for running bash commands
"""

import os


_glob_paras = {"script_dir":r"M:\labrad_server"}
_glob_paras["old_getcwd"] = os.getcwd()

os.chdir(_glob_paras["script_dir"])

def cd_oldDir():
    os.chdir(_glob_paras["old_getcwd"])
    
    
class Servers(object):
    """
    Example
    >>> server = Servers(idx=1,name='1',func='1')
    >>> Servers(idx=2,name='2',func='2')
    >>> Servers(idx=2,name='2',func='2')
    
    >>> server[1].idx,server[2].idx,server[3].idx
    (1, 2, 3)
    """
    
    _instance = {}
    def __new__(cls, *args, **kwargs):
        if "idx" in kwargs:
            idx = kwargs["idx"]
        else:
            idx = args.pop(0)
            
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

    def __getitem__(self, key):
        return self.__class__._instance[key]

    @property
    def all(self):
        return self.__class__._instance
    
    def run(self,*args,**kwargs):
        if hasattr(self.func,"__call__"):
            self.func(*args,**kwargs)



def add_servers():
    def start_cmd_command(command):
        os.system(r"start cmd /k %s"%(command))

    def start_cmd_ipython3(path,dir = None):    
        if dir == None:
            path = os.path.join(_glob_paras['script_dir'],path)
        
        def func():
            commands = r"start cmd /k ipython3 %s"%(path)
            # if you have more than one versions of ipython3, and you want to specify ones, 
            # you need to modify {ipython3} to the located path that you want to specify
            os.system(commands)
            
        return func
        
    def start_labrad(
        path = r"M:\scalabrad-0.8.3\bin\labrad",
        registryFile = r"file:///M:/Registry?format=delphi"):
        start_cmd_command("%s --registry %s"%(path,registryFile))
        
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


def start_all(serverDict):
    for key in serverDict.keys():
        serverDict[key].run()
        if serverDict[key].name == 'scalabrad':
            os.system("TIMEOUT /T 7")

def start_choices(serverDict,choices):		
    for key in choices:
        serverDict[key].run()
        if serverDict[key].name == 'scalabrad':
            os.system("TIMEOUT /T 7")



def dict2prettyWords(serverDict):
    words = ""
    for key,ser in serverDict.items():
        words += f"\n{key} {ser.name}"
    return words
    
def helper():
    print("="*40)
    words = "This is server_start bash by Ziyu Tao"
    print(words)
    print("="*40)
    
    
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

def choose_server():
    arg = input("Input the index and press 'Enter' to start server \n")
    try:
        choose = eval(arg)
    except: choose = arg
    
    print("You choose %s \n"%(choose))
    return choose
    
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
            start_all(serverDict)
           
        elif choice in serverDict.keys():
            start_choices(serverDict,[choice])
        else:
            print('Invalid input %s '%(choice))



# seefiles = sorted((fn for fn in os.listdir(r"M:/xxx") if fn.endswith('.py')))

if __name__ == '__main__':
    main()
    cd_oldDir()