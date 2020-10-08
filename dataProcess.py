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
from importlib import reload

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
_default_session = ['', 'hwh', '20200930_12Q_sample_C_test']

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
            name = names[idx-1]
            path_data =  os.path.join(self.path,name)
            data = read_csvdata(path_data)
            return data
        
        # use dataVault
        # dv.cd(self.session)
        
        if isinstance(idx,int):
            raise VauleError('idx should be int')

        if idx >= 1:
            # to insure 1 represent '00001 - xxxx.csv'
            idx = idx-1
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


def plot_label(xlabel,ylabel,size=15,size2 =20):
    plt.xlabel(xlabel,size=size2)
    plt.ylabel(ylabel,size=size2)
    plt.xticks(size=size)
    plt.yticks(size=size) 

def fitT1(dh,idx,dv=None,trange=40,data=None,doPlot=True,fig=None,title=''):
    """
    trange (float/int): should be larger than time max of T1 measurement
    """
    data = dh.getDataset(idx,dv)

    idx1 = 3
    idx2 = 6
    residue = np.mean(data[:,idx2])
    
    def fitfunc(p,t):
        return p[0]*np.exp(-p[1]*t)+residue
    
    t = data[data[data[:,0]<trange,0]>=0.0,0]*1000
    prob = data[data[data[:,0]<trange,0]>=0.0,idx1]
    
    def errfunc(p):
        return prob-fitfunc(p,t)
    out = leastsq(errfunc,np.array([max(prob)-min(prob),1/(t[-1]/3)]),full_output=1)
    p = out[0]
    
    deviation = np.sqrt(np.mean((fitfunc(p,t)-prob)**2))
    size,size2 = 15,20
    if doPlot:
        plt.figure(fig)
        plt.plot(t/1000.,prob,'bo')
        xs = np.linspace(t[0],np.max(t),1000)
        plt.plot(xs/1000.,fitfunc(p,xs),'k',linewidth=2)
        plt.xlabel(r'delay $(\mu s)$',size=size2)
        plt.ylabel(r'$P(1)$',size=size2)
        plt.xticks(size=size)
        plt.yticks(size=size)
        plt.ylim(0,1)
        
    print('probability: %g;   T1: %g ns;   Residue: %g' % (p[0], 1.0/p[1], residue) )    
    print('deviation: ', deviation)
    
    plt.title(title+'T1: %.1f ns;' % (1.0/p[1]/1e3 ),size=size)
    
    if data.shape[1]>=5:
        plt.plot(data[:,0],data[:,-1],'ro')
    plt.tight_layout()
    return p[0], 1.0/p[1]       
    
def plotIQraw(dh,idx,dv=None,level=2):
    """level (int): IQ points with N-level system (default 2, qubit)
    """

    data = dh.getDataset(idx,dv)
    if level <= 3:
        colors = ['b','r','g']
    else:
        colors = [None]*level

    for i in range(level):
        Is,Qs = data[:,2*i],data[:,2*i+1]
        plt.scatter(Is,Qs,s=20,color=colors[i],label=str(i))

    plt.title(dh.path+'\n --#'+str(idx),size=10)
    plt.legend()
    plot_label('I','Q')

def fitRamsey(dh,idx,fingefreq=0.002,sign=0,T1=13306,trange=[0,2],debug=False, fig=None, pro_idx=-1):
    '''gaussian-fit of dephasing time; input T1 in nanoseconds'''
    data = dh.getDataset(idx,dv)
    
    if np.max(data[:,0])>50:
        data[:,0] = data[:,0]/1000.

    t = data[data[data[:,0]<trange[1],0]>=trange[0],0]*1000
    prob = data[data[data[:,0]<trange[1],0]>=trange[0],pro_idx]
    mid = (np.min(prob)+np.max(prob))/2.0
    prob = prob-mid
    hprob = hilbert(prob)
    probEnv = np.sqrt(prob**2+hprob**2)
    
    def fitfunc(p,t):
        return p[0]*exp(-t/T1/2.-t**2/p[1]**2)
        # return p[0]*exp(-t/p[1])
    def errfunc(p):
        return probEnv-fitfunc(p,t)
    out = leastsq(errfunc,array([max(probEnv)-min(probEnv),t[-1]/3.0]),full_output=1)
    p = out[0]

    t1 = data[:,0]*1000
    prob1 = data[:,pro_idx] - mid
    plt.figure(fig)
    plt.plot(t1/1000.,prob1+mid,'bo-')
    xs = np.linspace(t1[0],t1[-1],1000)
    plt.plot(xs/1000.,fitfunc(p,xs)+mid,'k',linewidth=2)
    plt.plot(xs/1000.,fitfunc(p,xs)*np.cos(2*np.pi*fingefreq*xs+sign*np.pi)+mid,'r',linewidth=2)

    plot_label(r'delay $(\mu s)$',r'$P_1$')

    print('T2_gaussian: %.2f ns'%(p[1]))
    print('T2: %.2f ns'%(0.5*p[1]**2*((0.25/T1**2+4/p[1]**2)**0.5-0.5/T1)))
    plt.ylim(0,1)
 
    return p[1]
    
if __name__=="__main__":
    dh = datahelp()
    cxn,dv,ss = _connect_labrad()
    
    
    

