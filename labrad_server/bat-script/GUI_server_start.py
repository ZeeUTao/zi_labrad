#!/usr/bin/python
"""
Simple python GUI for starting servers
"""


from server_start import start_choices,add_servers,cd_oldDir
import numpy as np
import tkinter as tk
import tkinter.font as tkFont
import os


# width=50
# height=30
server = add_servers()
servers_GUI = {}
for key in server.all:
    servers_GUI[server[key].name] = server[key]
    
    
def lazy_property(func):
    attr_name = "_lazy_" + func.__name__
    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return _lazy_property

class App:
    def __init__(self, root):
        self.servers_GUI = servers_GUI
        self.start_choices = start_choices
        
        width=40
        height=10
        
        root.title("Run Servers (@Ziyu Tao)")
        root.update()
        
        screenwidth = root.winfo_screenwidth()
        screenheight = root.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height,screenwidth/2.-width/2., screenheight/4-height/2.)
        root.geometry(alignstr)
#         root.resizable(width=True, height=True)
        
        
        self.frame = tk.Frame(root)
        self.frame.pack()
        
        
        # initiate Gitems
        self.Gitems = {
            "GListBox_servers":self.GListBox_servers,
            "GButton_confirm":self.GButton_confirm,
            "GCheckBox_selectAll":self.GCheckBox_selectAll,
        }
        
        

    @lazy_property
    def GListBox_servers(self):
        GListBox=tk.Listbox(self.frame)
        GListBox["font"] = tkFont.Font(family='Times',size=18)
        GListBox["justify"] = "left"
        GListBox["selectmode"] = "multiple"
        GListBox["setgrid"] = "True"
        
        for serName in servers_GUI.keys():
            GListBox.insert(tk.END,serName)
        GListBox.pack(side='top', fill='both')
        return GListBox
            
    @lazy_property
    def GButton_confirm(self):
        GButton=tk.Button(self.frame)
        GButton["font"] = tkFont.Font(family='Times',size=18)
        GButton["justify"] = "center"
        GButton["text"] = "confirm"
        GButton["command"] = self.command_confirm
        GButton.pack(side='top',fill='both')
        return GButton

    def command_confirm(self):
        mylistbox = self.GListBox_servers
        choices = [mylistbox.get(idx) for idx in mylistbox.curselection()]
        self.start_choices(self.servers_GUI,choices)        
 
    
    @lazy_property
    def GCheckBox_selectAll(self):
        GCheckBox=tk.Checkbutton(self.frame)
        GCheckBox["cursor"] = "arrow"
        ft = tkFont.Font(family='Times',size=18)
        GCheckBox["font"] = ft
        GCheckBox["justify"] = "left"
        GCheckBox["text"] = "select all"
        GCheckBox["command"] = self.command_selectAll
        GCheckBox.pack(side='top',fill='both')
        return GCheckBox
    
    @lazy_property
    def selectAll_val(self):
        """select or deselect value
        usage: val.get(), val.get(int)
        """
        val = tk.IntVar()
        val.set(0)
        self.GCheckBox_selectAll["variable"] = val
        return val

        
    
    def command_selectAll(self):
        """select or deselect all command
        """
        num_lines = self.GListBox_servers.size()
        value = self.selectAll_val.get()
        if value == True:
            for idx in range(num_lines):
                self.GListBox_servers.select_set(idx)
        else:
            for idx in range(num_lines):
                self.GListBox_servers.select_clear(idx)
                

             
_glob_paras = {"script_dir":r"M:\zi_labrad\labrad_server"}
_glob_paras["old_getcwd"] = os.getcwd()

os.chdir(_glob_paras["script_dir"])

root = tk.Tk()
app = App(root)
root.mainloop()

# os.chdir(_glob_paras["old_getcwd"])
    


