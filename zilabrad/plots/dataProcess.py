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

# labrad module
import zilabrad.plots.adjuster as adjuster
import labrad
from zilabrad.pyle.workflow import switchSession
# labrad module end
import configparser



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
        self.dir = r'M:\Experimental Data'
        self.path = path

        if session is None:
            self.session = _default_session
        else:
            self.session = session
            
        if self.path is None:
            _path = self.dir
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

def rotateData(data, center0, center1, labels=['|0>','|1>'], doPlot=True):
    center = (center0+center1)/2.0
    theta = np.angle(center0-center1)
    I0s, Q0s, I1s, Q1s = data.T
    sig0s = I0s + 1j*Q0s
    sig1s = I1s + 1j*Q1s
    if doPlot:
        plt.figure(101)
        plt.plot(I0s, Q0s, 'b.', label=labels[0])
        plt.plot(I1s, Q1s, 'r.', label=labels[1])
        plt.plot((np.real(center0), np.real(center1)),(np.imag(center0), np.imag(center1)),'ko--',markersize=10,lw=3)
        plt.xlabel('I',size=20)
        plt.ylabel('Q',size=20)
        plt.xticks(size=20)
        plt.yticks(size=20)
        plt.legend()
        plt.subplots_adjust(bottom=0.125, left=0.15)
    sig0s = (sig0s-center)*np.exp(-1j*theta)
    sig1s = (sig1s-center)*np.exp(-1j*theta)
    return sig0s, sig1s

def fit_double_gaussian(count,xs,plot=True,output=True,color='black',method='bfgs'):
    def fitfunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2)) + np.abs(p[3])*np.exp(-(xs-p[4])**2/(2*p[5]**2))

    def errfunc(p):
        x = count - fitfunc(p,xs)
        return np.sum(x**2)
    
    xs_0 = xs[xs[:]>0]
    xs_1 = xs[xs[:]<0]
    count_0 = count[xs_1.shape[0]:]
    count_1 = count[0:xs_1.shape[0]]
    # out = minimize(errfunc,(np.max(count_0),np.mean(xs_0),(np.max(xs_0)-np.mean(xs_0))/2.0,np.max(count_1),np.mean(xs_1),(np.max(xs_1)-np.mean(xs_1))/2.0),
    # method = 'TNC', bounds = ((0.,6000.),(min(xs_1)/2.,max(xs_0)),(2.,max(xs_0)),(0.,6000.),(min(xs_1),max(xs_0)/2.),(2.,max(xs_0))))
    
    x0 = (np.max(count_0),np.mean(xs_0),(np.max(xs_0)-np.mean(xs_0))/2.0,np.max(count_1),np.mean(xs_1),(np.max(xs_1)-np.mean(xs_1))/2.0)
    bnds = ((0.0,6000.0),(0.0,max(xs_0)),(4.0,max(xs_0)),(0.0,6000.0),(min(xs_1),0.0),(4.0,max(xs_0)))
    # print x0
    # print bnds
    if method == 'bfgs':
        res = scipy.optimize.fmin_l_bfgs_b(errfunc,x0,bounds=bnds,approx_grad=True)[0]
    elif method == 'tnc':
        res = scipy.optimize.fmin_tnc(errfunc,x0,bounds=bnds,approx_grad=True)[0]
    else:
        res = scipy.optimize.fmin_slsqp(errfunc,x0,bounds=bnds)
        
    p = res
    if output:
        print('Amp|0>: ',np.abs(p[0]),'mean|0>: ',p[1],'deviation|0>: ',np.abs(p[2]))
        print('Amp|1>: ',np.abs(p[3]),'mean|1>: ',p[4],'deviation|1>: ',np.abs(p[5]))
    if plot:
        plt.plot(xs,fitfunc(p,xs),color=color,linewidth=2)
        
    return p

def anaSignal(sig0s, sig1s, stats, labels=['|0>','|1>'], fignum=[102, 103], debugFlag=False):
    stats0 = np.size(sig0s)
    stats1 = np.size(sig1s)

    if stats1 != stats:
        print('warning!!!!!!!!!!!!!!!!')
        print(stats0, stats1)

    plt.figure(fignum[0])
    plt.subplot(311)
    plt.plot(np.real(sig0s), np.imag(sig0s), 'bo', label=labels[0])
    plt.plot(np.real(sig1s), np.imag(sig1s), 'ro', label=labels[1],alpha=0.5)
    plt.legend()
    
    plt.subplot(312)
    sig_min = np.min([np.min(np.real(sig0s)), np.min(np.real(sig1s))])
    sig_max = np.max([np.max(np.real(sig0s)), np.max(np.real(sig1s))])
    counts0,positions0,patch0 = plt.hist(np.real(sig0s),range=(sig_min, sig_max),bins=50,alpha=0.5,label=labels[0])
    counts1,positions1,patch1 = plt.hist(np.real(sig1s),range=(sig_min, sig_max),bins=50,alpha=0.5,label=labels[1])
    positions0 = (positions0[0:-1] + positions0[1:])/2.0
    positions1 = (positions1[0:-1] + positions1[1:])/2.0
    
    result0 = fit_double_gaussian(counts0,positions0,color='blue')
    result1 = fit_double_gaussian(counts1,positions1,color='red')
    separation = result0[1]-result1[4]
    deviation0 = result0[2]
    deviation1 = result1[5]
    p00 = result0[:3]
    p01 = result0[3:]
    p11 = result1[3:]
    p10 = result1[:3]
    print('Separation: %.2f, SNR: %.2f'%(p00[1]-p11[1],(p00[1]-p11[1])/(p00[2]+p11[2])))
    def gaussianFunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2))
    def fitFunc(p,xs):
        return np.abs(p[0])*np.exp(-(xs-p[1])**2/(2*p[2]**2)) + np.abs(p[3])*np.exp(-(xs-p[4])**2/(2*p[5]**2))
    
    c0 = np.array([np.sum(counts0[:idx0]) for idx0 in range(len(positions0))])
    c1 = np.array([np.sum(counts1[:idx0]) for idx0 in range(len(positions0))])
    vis = np.array([-np.sum(counts0[:idx0])+np.sum(counts1[:idx0]) for idx0 in range(len(positions0))])
    idx = np.argmax(vis)
    vism = max(vis/float(stats))
    print('sepPoint: %.2f, Visibility: %.2f'%(positions0[idx], vism))
    
    plt.subplot(313)
    plt.plot(positions0, gaussianFunc(p00, positions0), 'b-', label=labels[0])
    plt.plot(positions0, gaussianFunc(p11, positions0), 'r-', label=labels[1])
    plt.fill_between(positions0, gaussianFunc(p00,positions0), where=(positions0<=positions0[idx-1]), color='b', alpha=0.5)
    plt.fill_between(positions0, gaussianFunc(p11,positions0), where=(positions0>=positions0[idx-1]), color='r', alpha=0.5)
    
    plt.figure(fignum[1])
    plt.plot(positions0, c0/float(stats), 'b-', lw=2,label='|0>')
    plt.plot(positions0, c1/float(stats), 'r-', lw=2,label='|1>')
    plt.plot(positions0, vis/float(stats), 'k-', lw=2)
    # plt.plot(positions0, vis/float(stats), 'r-', lw=1)
    plt.xlabel('Integration Axis', size=20)
    plt.ylabel('Visibility', size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.title('Maximum Visibility: %.2f'%np.max(vis/float(stats)))
    plt.subplots_adjust(bottom=0.125, left=0.18)

    separationError0 = np.sum(gaussianFunc(p00,positions0[:idx]))/np.sum(fitFunc(result0,positions0)) #calculate separation error
    separationError1 = np.sum(gaussianFunc(p11,positions0[idx:]))/np.sum(fitFunc(result1,positions0))
    Error0 = np.sum(counts0[1:idx])/np.float(stats)
    Error1 = np.sum(counts1[idx:])/np.float(stats)
    stateErrorp0 = Error0 - separationError0
    stateErrorp1 = Error1 - separationError1
    print('stateError|0>: %.1f%%, stateError|1>: %.1f%%'%(100*stateErrorp0, 100*stateErrorp1))
    print('Error|0>: %.1f%%, Error|1>: %.1f%%'%(100*Error0, 100*Error1))
    anaData = [separation, deviation0, deviation1, separationError0, separationError1, stateErrorp0, stateErrorp1, vism, positions0[idx], np.abs(positions0[idx]-result0[1])]
    
    if debugFlag:
        print(anaData)
        pdb.set_trace()
    return anaData 


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
    
    t = data[:,0]
    prob = data[:,idx1]
    
    def errfunc(p):
        return prob-fitfunc(p,t)
    out = leastsq(errfunc,np.array([max(prob)-min(prob),1/(np.max(t))]),full_output=1)
    p = out[0]
    
    deviation = np.sqrt(np.mean((fitfunc(p,t)-prob)**2))
    size,size2 = 15,20
    if doPlot:
        plt.figure(fig)
        plt.plot(t,prob,'ro')
        xs = np.linspace(np.min(t),np.max(t),1000)
        plt.plot(xs,fitfunc(p,xs),'k',linewidth=2)
        plt.xlabel(r'delay $(\mu s)$',size=size2)
        plt.ylabel(r'$P(1)$',size=size2)
        plt.xticks(size=size)
        plt.yticks(size=size)
        plt.ylim(0,1)
        
    print('probability: %g;   T1: %g us;   Residue: %g' % (p[0], 1.0/p[1], residue) )    
    print('deviation: ', deviation)
    
    plt.title(title+'T1: %.1f us;' % (1.0/p[1] ),size=size)
    
    if data.shape[1]>=5:
        plt.plot(data[:,0],data[:,-1],'bo')
    plt.tight_layout()
    return p[0], 1.0/p[1]       


def _updateIQraw2(data,Qb,dv=None,update=True,analyze=False):
    Is0,Qs0,Is1,Qs1 =data.T
    stats = len(Is0)
    if update:
        Qb['center|0>'] = [np.mean(Is0),np.mean(Qs0)]
        Qb['center|1>'] = [np.mean(Is1),np.mean(Qs1)]
        adjuster.IQ_center(qubit=Qb, data=data)

    center0 = Qb['center|0>'][0] + 1j*Qb['center|0>'][1]
    center1 = Qb['center|1>'][0] + 1j*Qb['center|1>'][1]

    if analyze:
        sig0s, sig1s = rotateData(data, center0, center1,doPlot=True)
        result = anaSignal(sig0s, sig1s, stats, labels=['|0>','|1>'], fignum=[102, 103])

        plt.figure(101)
        theta = np.linspace(0, 2*np.pi, 100)
        for idx in range(3):
            plt.plot(np.real(center0+(idx+1)*result[1]*np.exp(1j*theta)),np.imag(center0+(idx+1)*result[1]*np.exp(1j*theta)),'g-',lw=2)
            plt.plot(np.real(center1+(idx+1)*result[2]*np.exp(1j*theta)),np.imag(center1+(idx+1)*result[2]*np.exp(1j*theta)),'k-',lw=2)
        sepPoint = center0+np.cos(np.angle(center1-center0))*result[-1]+1j*np.sin(np.angle(center1-center0))*result[-1]
        sepPoint1 = sepPoint+np.cos(np.angle(center1-center0)+np.pi/2)*50.0+1j*np.sin(np.angle(center1-center0)+np.pi/2)*50.0
        sepPoint2 = sepPoint+np.cos(np.angle(center1-center0)-np.pi/2)*50.0+1j*np.sin(np.angle(center1-center0)-np.pi/2)*50.0
        plt.plot(np.real(np.array([sepPoint1,sepPoint,sepPoint2])), np.imag(np.array([sepPoint1,sepPoint,sepPoint2])), 'g-', lw=2)
    return

def UpdateIQraw2(dh,idx,Qb,dv=None,update=True,analyze=False):
    data = dh.getDataset(idx,dv)
    _updateIQraw2(data,Qb,dv,update,analyze)
    return

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

def fitRamsey(dh,idx,dv=None,fingefreq=0.002,sign=0,T1=13306,trange=[0,2],debug=False,pro_idx=-1):
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
        return p[0]*np.exp(-t/T1/2.-t**2/p[1]**2)
        # return p[0]*exp(-t/p[1])
    def errfunc(p):
        return probEnv-fitfunc(p,t)
    out = leastsq(errfunc,np.array([max(probEnv)-min(probEnv),t[-1]/3.0]),full_output=1)
    p = out[0]

    t1 = data[:,0]*1000
    prob1 = data[:,pro_idx] - mid

    plt.plot(t1/1000.,prob1+mid,'bo-')
    xs = np.linspace(t1[0],t1[-1],1000)
    plt.plot(xs/1000.,fitfunc(p,xs)+mid,'k',linewidth=2)
    plt.plot(xs/1000.,fitfunc(p,xs)*np.cos(2*np.pi*fingefreq*xs+sign*np.pi)+mid,'r',linewidth=2)

    plot_label(r'delay $(\mu s)$',r'$P_1$')

    print('T2_gaussian: %.2f ns'%(p[1]))
    print('T2: %.2f ns'%(0.5*p[1]**2*((0.25/T1**2+4/p[1]**2)**0.5-0.5/T1)))
    plt.ylim(0,1)
 
    return p[1]

def RamseyFFT(dh,idx, dv=None):
    '''FFT to ramsey data'''
    data = dh.getDataset(idx,dv)
    ts = data[:,0]
    ps = data[:,-1]
    ps = ps - np.mean(ps)
    freq = np.fft.fftfreq(len(ts), ts[1]-ts[0])
    fourier = abs(np.fft.fft(ps, len(ts)))

    fm = freq[np.where(fourier==np.max(fourier))][0]
    plt.plot(freq, fourier, '.-')
    plt.xlabel('freq (MHz)', size=20)
    plt.ylabel('fft',size=20)
    plt.xticks(size=20)
    plt.yticks(size=20)
    plt.subplots_adjust(bottom=0.15,left=0.15)
    return fm
        
# if __name__=="__main__":
#     dh = datahelp()
#     cxn,dv,ss = _connect_labrad()
    
    
    

