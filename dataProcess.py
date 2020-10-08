# -*- coding: utf-8 -*-
"""
helpers for data process

Example:
In ipython console:
    run dataProcess
and use functions:
    fitT1(dh,idx,dv): use labrad.dataVault to get the idx-th filename 
    fitT1(dh,idx): get all the filenames and sort, choose filenames[idx]
    
    when idx = 1, '00001 - xxxx.csv' can match
"""


import os
import csv

import numpy as np
import scipy
from scipy import optimize
from scipy import fftpack
from scipy.fftpack import hilbert
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import leastsq, curve_fit
import sys
# sys.path.insert(0,'M:\\')

# labrad module
import labrad
from pyle.workflow import switchSession
# labrad module end


_default_dir = r'M:\Experimental Data'
_default_session = ['', 'hwh', 'sample1']

encodings = [
    ('%','%p'),
    ('/','%f'),
    ('\\','%b'),
    (':','%c'),
    ('*','%a'),
    ('?','%q'),
    ('"','%r'),
    ('<','%l'),
    ('>','%g'),
    ('|','%v')
]

def dsEncode(name):
    for char, code in encodings:
        name = name.replace(char, code)
    return name
            
def read_csvdata(file_name):
    print('reading ',file_name)
    
    with open(file_name,'r') as f:
        reader = csv.reader(f)
        rows = [row for row in reader]
        rows = np.asarray(rows,dtype=float)
    return rows
    
    
class datahelp(object):
    """ 
    helper to get data from specified path
    """
    def __init__(self,path = None,session=None):
        self.path = path
        self.session = session
        
        if self.session is None:
            self.session = _default_session
            
            
        if self.path is None:
            _path = _default_dir
            for sub in self.session:
                if sub is '': continue
                _path = os.path.join(_path,sub+r'.dir')
            self.path = _path
            
        
    @staticmethod
    def listnames(path,ext='.csv'):
        """
        return a list of file names in the self.path
        """
        names = sorted((fn for fn in os.listdir(path) if fn.endswith(ext)))
        return names
    
    
    def getDataset(self,idx,dv=None,name=None):
        """
        Args: 
            idx (int): indicate the name of dataset ('00001 - Test1028_bias2d.csv')
            name (str): directly give the name like '00001 - Test1028_bias2d.csv'
            dv (labrad.dataVault)
        Returns: 
            data in .csv (numpy.array)
        """
        if name:
            path_data = os.path.join(self.path,name)
            data = read_csvdata(path_data)
            return data
            
        if dv is None:
            names = self.listnames(self.path,ext='.csv')
            name = names[idx]
            path_data =  os.path.join(self.path,name)
            data = read_csvdata(path_data)
            return data
        
        # use dataVault
        # dv.cd(self.session)
        
        
        if idx is None:
            idx = 0
        name = dv.dir()[1][idx] + '.csv'
        name = dsEncode(name)

        path_data = os.path.join(self.path,name)
        data = read_csvdata(path_data)
        return data
        
def _connect_labrad():    
    cxn = labrad.connect()
    dv = cxn.data_vault
    _default_user = "hwh"
    user = input("Enter user (default:hwh)") or _default_user
    ss = switchSession(cxn,user=user,session=None)         
    return cxn,dv,ss
        
def fitT1(dh,idx,dv=None,trange=13,data=None,doPlot=True,fig=None):
    data = dh.getDataset(idx,dv)

    idx1 = 3
    idx2 = 6
    residue = np.mean(data[:,idx2])
    
    def fitfunc(p,t):
        return p[0]*np.exp(-p[1]*t)+residue
    
    t = data[data[data[:,0]<trange,0]>0.0,0]*1000
    prob = data[data[data[:,0]<trange,0]>0.0,idx1]
    
    def errfunc(p):
        return prob-fitfunc(p,t)
    out = leastsq(errfunc,np.array([max(prob)-min(prob),1/(t[-1]/3)]),full_output=1)
    p = out[0]
    print('probability: %g;   T1: %g ns;   Residue: %g' % (p[0], 1.0/p[1], residue) )
    deviation = np.sqrt(np.mean((fitfunc(p,t)-prob)**2))
    print('deviation: ', deviation)
    if doPlot:
        plt.figure(fig)
        plt.plot(t/1000.,prob,'bo')
        xs = np.linspace(t[0],20*1000,1000)
        plt.plot(xs/1000.,fitfunc(p,xs),'k',linewidth=2)
        plt.xlabel('Delay (us)',size=20)
        plt.ylabel('Probability |1>',size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.ylim(0,1)
        
    if data.shape[1]>=5:
        plt.plot(data[:,0],data[:,-1],'ro')
    plt.tight_layout()
    return p[0], 1.0/p[1]       
    


    
if __name__=="__main__":
    dh = datahelp()
    cxn,dv,ss = _connect_labrad()
    
    
    

