from queue import Empty
from multiprocessing import Process, Queue
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import widgets
from scipy import interpolate

from labrad.units import Unit
from labrad.types import Value
V, mV, us, ns, GHz, rad = [Unit(s) for s in ('V', 'mV', 'us', 'ns', 'GHz', 'rad')]


def IQ_center_test(data):# Chao Song 2015-02-06
    I0s, Q0s, I1s, Q1s = data.T
    traces = [{'x': I0s, 'y': Q0s, 'args': ('b.',)},
              {'x': I1s, 'y': Q1s, 'args': ('r.',)}]
    params = [{'name':'I0', 'val':I0, 'range':(min(I0s),max(I0s)), 'axis':'x', 'color': 'b'},
              {'name':'I1', 'val':I1, 'range':(min(I1s),max(I1s)), 'axis':'x', 'color': 'k'},
              {'name':'Q0', 'val':Q0, 'range':(min(Q0s),max(Q0s)), 'axis':'y', 'color': 'b'},
              {'name':'Q1', 'val':Q1, 'range':(min(Q1s),max(Q1s)), 'axis':'y', 'color': 'k'}]
    result = adjust(params, traces)
    
def IQ_center(qubit, data):# Ziyu 2020-10-06
    I0s, Q0s, I1s, Q1s = data.T
    traces = [{'x': I0s, 'y': Q0s, 'args': ('b.',)},
              {'x': I1s, 'y': Q1s, 'args': ('r.',)}]
    params = [{'name':'I0', 'val':qubit['center|0>'][0], 'range':(min(I0s),max(I0s)), 'axis':'x', 'color': 'b'},
              {'name':'I1', 'val':qubit['center|1>'][0], 'range':(min(I1s),max(I1s)), 'axis':'x', 'color': 'k'},
              {'name':'Q0', 'val':qubit['center|0>'][1], 'range':(min(Q0s),max(Q0s)), 'axis':'y', 'color': 'b'},
              {'name':'Q1', 'val':qubit['center|1>'][1], 'range':(min(Q1s),max(Q1s)), 'axis':'y', 'color': 'k'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['center|0>'] = [result['I0'], result['Q0']]
        qubit['center|1>'] = [result['I1'], result['Q1']]

def IQ_center_multilevel(qubit, data):# Chao Song 2015-04-29
    I0s, Q0s, I1s, Q1s = data.T
    traces = [{'x': I0s, 'y': Q0s, 'args': ('b.',)},
              {'x': I1s, 'y': Q1s, 'args': ('r.',)}]
    params = [{'name':'I0', 'val':qubit['center|0>'][0][''], 'range':(min(I0s),max(I0s)), 'axis':'x', 'color': 'b'},
              {'name':'I1', 'val':qubit['center|1>'][0][''], 'range':(min(I1s),max(I1s)), 'axis':'x', 'color': 'k'},
              {'name':'I2', 'val':qubit['center|2>'][0][''], 'range':(min(I1s),max(I1s)), 'axis':'x', 'color': 'y'},
              {'name':'Q0', 'val':qubit['center|0>'][1][''], 'range':(min(Q0s),max(Q0s)), 'axis':'y', 'color': 'b'},
              {'name':'Q1', 'val':qubit['center|1>'][1][''], 'range':(min(Q1s),max(Q1s)), 'axis':'y', 'color': 'k'},
              {'name':'Q2', 'val':qubit['center|2>'][1][''], 'range':(min(Q1s),max(Q1s)), 'axis':'y', 'color': 'y'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['center|0>'] = [Value(result['I0'],''), Value(result['Q0'],'')]
        qubit['center|1>'] = [Value(result['I1'],''), Value(result['Q1'],'')]
        qubit['center|2>'] = [Value(result['I2'],''), Value(result['Q2'],'')]

def findMinimum(data,fit=True):
    if fit:
        p = np.polyfit(data[:,0],data[:,1],2)
        xMin = -1.0*p[1]/(2.0*p[0])
        yMin = np.polyval(p,xMin)
        return (xMin,yMin)
    else:
        index = np.argmin(data[:,1])
        return data[index,0],data[index,1]
        
def adc_delay(qubit, data):
    t, I, Q = data.T
    traces = [{'x': t, 'y': I, 'args': ('b',)},
          {'x':t, 'y': Q, 'args': ('r',)}]
    params = [{'name':'adc delay', 'val': qubit['adc delay'][ns], 'range':(0, 500), 'axis':'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['adc delay'] = qubit['adc delay'] + result['adc delay']*ns
    
def adjust_s_scanning(qubit, data, qnd=False):
    f, phase = data.T
    traces = [{'x':f, 'y': phase, 'args':('b.',)}]
    if qnd:
        params = [{'name': 'qnd_readout frequency', 'val': qubit['qnd_readout frequency'][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    else:
        params = [{'name': 'readout frequency', 'val': qubit['readout frequency'][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        if qnd:
            qubit['qnd_readout frequency'] = result['qnd_readout frequency']*GHz
        else:    
            qubit['readout frequency'] = result['readout frequency']*GHz

        
def adjust_phase(qubit, data):
    fb, left, right = data.T
    traces = [{'x': fb, 'y': left, 'args': ('b.',)},
              {'x': fb, 'y': right, 'args': ('r.',)}]
    params = [{'name':'adc adjusted phase', 'val': qubit['adc adjusted phase'][rad], 'range':(-np.pi, np.pi), 'axis':'y', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['adc adjusted phase'] = (-2 - result['adc adjusted phase'])*rad + qubit['adc adjusted phase']
        if qubit['adc adjusted phase']>np.pi:
            qubit['adc adjusted phase'] = qubit['adc adjusted phase']-2*np.pi
        elif qubit['adc adjusted phase']<-np.pi:
            qubit['adc adjusted phase'] = qubit['adc adjusted phase']+2*np.pi

def adjust_phase_arc(qubit, data):
    fb, left, right = data.T
    traces = [{'x': fb, 'y': left, 'args': ('b.',)},
              {'x': fb, 'y': right, 'args': ('r.',)}]
    params = [{'name': 'operate', 'val': qubit['biasOperate'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'b'},
              {'name': 'readout', 'val': qubit['biasReadout'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'reset0', 'val': qubit['biasReset'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'},
              {'name': 'reset1', 'val': qubit['biasReset'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'm'},
              {'name': 'Phase', 'val': qubit['critical phase'][rad], 'range': (-np.pi,np.pi), 'axis': 'y', 'color': 'k'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['operate']*V
        qubit['biasReadout'] = result['readout']*V
        qubit['biasReset'] = [result['reset0']*V, result['reset1']*V] * 2
        qubit['critical phase'] = result['Phase']*rad
        
def adjust_squid_steps(qubit, data):
    fb, low, high = data.T
    traces = [{'x': fb, 'y': low, 'args': ('b.',)},
              {'x': fb, 'y': high, 'args': ('r.',)}]
    params = [{'name': 'operate', 'val': qubit['biasOperate'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'b'},
              {'name': 'readout', 'val': qubit['biasReadout'][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'reset0', 'val': qubit['biasReset'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'},
              {'name': 'reset1', 'val': qubit['biasReset'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'm'},
              {'name': 'timing0', 'val': qubit['squidSwitchIntervals'][0][0][us], 'range': (0,40), 'axis': 'y', 'color': 'k'},
              {'name': 'timing1', 'val': qubit['squidSwitchIntervals'][0][1][us], 'range': (0,40), 'axis': 'y', 'color': 'gray'},
              {'name': 'Edge_left', 'val': qubit['squidEdges'][0][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'g'},
              {'name': 'Edge_right', 'val': qubit['squidEdges'][1][V], 'range': (-2.5,2.5), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['operate']*V
        qubit['biasReadout'] = result['readout']*V
        qubit['biasReset'] = [result['reset0']*V, result['reset1']*V] * 2
        qubit['squidSwitchIntervals'] = [(result['timing0']*us, result['timing1']*us)]
        qubit['squidEdges'] = [result['Edge_left']*V,result['Edge_right']*V] #mark the edge of two branches of the same color. Converts voltage-to-Phi_not


def adjust_time(data):
    t, probs = data[:,0], data[:,1:].T
    traces = [{'x': t, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 't', 'val': (min(t)+max(t))/2, 'range': (min(t), max(t)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        return result['t']


def adjust_operate_bias(qubit, data):
    fb, prob = data.T
    traces = [{'x': fb, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': 'fb', 'val': qubit['biasOperate'][mV], 'range': (min(fb),max(fb)), 'axis': 'x', 'color': 'b'},
              {'name': 'step', 'val': qubit['biasStepEdge'][mV], 'range': (min(fb),max(fb)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['biasOperate'] = result['fb']*mV
        qubit['biasStepEdge'] = result['step']*mV


def adjust_scurve(qubit, data):
    mpa, prob = data.T
    traces = [{'x': mpa, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': 'mpa', 'val': float(qubit['measureAmp']), 'range': (min(mpa),max(mpa)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['measureAmp'] = result['mpa']


def adjust_visibility(qubit, data):
    mpas, probs = data.T[0], data.T[1:]
    #We have to make sure that the mpa axis is monotonically increasing for scipy.interpolation.interp1d to work properly
    if mpas[0]>mpas[-1]:        #If mpas runs negatively
        mpas = mpas[::-1]       #Reverse it's order
        probs = probs[:,::-1]   #and also reverse the order of the dependent variables.
    traces = [{'x':mpas, 'y':prob, 'args': ('.-',)} for prob in probs]
    params = [{'name':'mpa', 'val': float(qubit['measureAmp']), 'range': (min(mpas),max(mpas)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        f0 = interpolate.interp1d(mpas,probs[0],'cubic')
        f1 = interpolate.interp1d(mpas,probs[1],'cubic')
        mpa = result['mpa']
        qubit['measureAmp']=mpa
        qubit['calScurve']=[float(f0(mpa)),float(f1(mpa))]


def adjust_scurve2(qubit, data, states):
    colors = ['b','g','r','c','m','y','k']
    keynames=['measureAmp']+['measureAmp'+str(i) for i in list(np.arange(2,max(max(states)+2,2)))]
    mpa, probs = data.T[0], data.T[1:]
    traces = [{'x': mpa, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'mpa'+str(state+1), 'val': float(qubit[keynames[state]]), 'range': (min(mpa),max(mpa)), 'axis': 'x', 'color': colors[state]} for state in states]
    result = adjust(params, traces)
    if result is not None:
        for state in states:
            qubit[keynames[state]] = result['mpa'+str(state+1)]


def adjust_visibility2(qubit, data, states):
    numstates=len(states)
    mpas, probs, visis = data.T[0], data.T[1:numstates], data.T[numstates:]
    colors = ['b','g','r','c','m','y','k']
    keynames=['measureAmp']+['measureAmp'+str(i) for i in list(np.arange(2,max(max(states)+1,2)))]
    #We have to make sure that the mpa axis is monotonically increasing for scipy.interpolation.interp1d to work properly
    if mpas[0]>mpas[-1]:        #If mpas runs negatively
        mpas = mpas[::-1]       #Reverse it's order
        probs = probs[:,::-1]   #and also reverse the order of the probabilities.
        visis = visis[:,::-1]   #and also reverse the order of the visibilities.
    traces = [{'x':mpas, 'y':vis, 'args': ('.-',)} for vis in visis]+[{'x':mpas, 'y':prob, 'args': ('.-',)} for prob in probs]
    params = [{'name':'mpa'+str(state), 'val': float(qubit[keynames[state-1]]), 'range': (min(mpas),max(mpas)), 'axis': 'x', 'color': colors[state-1]} for state in states[1:]]
    result = adjust(params, traces)
    if result is not None:
        for state in states[1:]:
            qubit[keynames[state-1]] = result['mpa'+str(state)]


def adjust_frequency(qubit, data, paramName=None):
    if paramName is None:
        paramName = 'f10'
    f, prob = data.T
    traces = [{'x': f, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': paramName, 'val': qubit[paramName][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit[paramName] = result[paramName]*GHz

def adjust_frequency_new(qubit, data, paramName=None):#-Chao, 15-05-09
    if paramName is None:
        paramName = 'f10'
    f = data[:,0]
    prob = data[:,3]
    traces = [{'x': f, 'y': prob, 'args': ('b.-',)}]
    params = [{'name': paramName, 'val': qubit[paramName][GHz], 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        qubit[paramName] = result[paramName]*GHz
        
def adjust_frequency_02(qubit, data):
    f10 = qubit['f10'][GHz]
    f21 = qubit['f21'][GHz]
    f20_2ph = (f21 + f10) / 2
    f, probs = data.T[0], data.T[1:]
    traces = [{'x': f, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'f10', 'val': f10, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'},
              {'name': 'f20_2ph', 'val': f20_2ph, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        f10 = result['f10']
        f20_2ph = result['f20_2ph']
        f21 = 2*f20_2ph - f10
        qubit['f10'] = f10*GHz
        qubit['f21'] = f21*GHz

def adjust_frequency_02_new(qubit, data):
    f10 = qubit['f10'][GHz]
    f21 = qubit['f21'][GHz]
    f20_2ph = (f21 + f10) / 2
    f, probs = data[:,0], data[:,3]
    traces = [{'x': f, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'f10', 'val': f10, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'b'},
              {'name': 'f20_2ph', 'val': f20_2ph, 'range': (min(f),max(f)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        f10 = result['f10']
        f20_2ph = result['f20_2ph']
        f21 = 2*f20_2ph - f10
        qubit['f10'] = f10*GHz
        qubit['f21'] = f21*GHz

def adjust_fc(qubit, data):
    f10 = qubit['f10'][GHz]
    fc = qubit['fc'][GHz]
    f, probs = data.T[0], data.T[1:]
    traces = [{'x': f, 'y': prob, 'args': ('.-',)} for prob in probs]
    params = [{'name': 'f10', 'val': qubit['f10'][GHz], 'range': (min(f)-.2,max(f)+.2), 'axis': 'x', 'color': 'b'},
              {'name': 'fc', 'val': qubit['fc'][GHz], 'range': (min(f)-.2,max(f)+.2), 'axis': 'x', 'color': 'g'}]
    result = adjust(params, traces)
    if result is not None:
        qubit['f10'] = result['f10']*GHz
        qubit['fc'] = result['fc']*GHz

        
def saveKeyNumber(key,state):
    """Create key name for higher state pulses for saving into registry.
    
    Inputs the registry key name and the state. Outputs the corresponding
    registry key referring to that state. Not valid for frequencies.
    
    Examples: setKeyNumber('piAmp',1) returns 'piAmp'
              setKeyNumber('piAmp',3) returns 'piAmp3'
    """
    statenum = str(state) if state>1 else ''
    newkey = key + statenum
    return newkey
    
def adjust_rabihigh(qubit, data, state=1, diff=0.4):
    rabiheight, probs = data.T
    traces = [{'x': rabiheight, 'y': probs, 'args': ('.-',)}]
    params = [{'name': 'maxRabi', 'val': float(qubit[saveKeyNumber('piAmp',state)]), 'range': (min(rabiheight),max(rabiheight)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        # Fit parabola to part of rabihigh centered about chosen point.
        rabiselect = result['maxRabi']
        rabikeep = np.logical_and(rabiheight>(rabiselect*(1-diff)),rabiheight<(rabiselect*(1+diff)))
        rabichoose = rabiheight[rabikeep]
        probchoose = probs[rabikeep]
        datachoose = np.array([rabichoose,probchoose]).T
        amp,maxrabiprob = findMinimum(datachoose)
        qubit[saveKeyNumber('piAmp',state)] = amp
        

def adjust_cZControlPhaseCorrAmp(qubit, data):
    cZCorrAmpMax = qubit['cZControlPhaseCorrAmpMax']
    cZCorrAmpMin = qubit['cZControlPhaseCorrAmpMin']
    controlAmp, probs = data.T
    traces = [{'x': controlAmp, 'y': probs, 'args': ('b.-',)}]
    params = [{'name': 'MAX', 'val': float(qubit['cZControlPhaseCorrAmpMax']), 'range': (min(controlAmp),max(controlAmp)), 'axis': 'x', 'color': 'b'},
              {'name': 'MIN', 'val': float(qubit['cZControlPhaseCorrAmpMin']), 'range': (min(controlAmp),max(controlAmp)), 'axis': 'x', 'color': 'b'}]
    result = adjust(params, traces)
    if result is not None:
        cZCorrAmpMax = result['MAX']
        cZCorrAmpMin = result['MIN']
        qubit['cZControlPhaseCorrAmpMax'] = cZCorrAmpMax
        qubit['cZControlPhaseCorrAmpMin'] = cZCorrAmpMin
        
def adjust_cZTargetPhaseCorrAmp(qubit, data):
    cZCorrAmpMax = qubit['cZTargetPhaseCorrAmpMax']
    cZCorrAmpMin = qubit['cZTargetPhaseCorrAmpMin']
    targetAmp, probs = data.T
    traces = [{'x': targetAmp, 'y': probs, 'args': ('b.-',)}]
    params = [{'name': 'MAX', 'val': float(qubit['cZTargetPhaseCorrAmpMax']), 'range': (min(targetAmp),max(targetAmp)), 'axis': 'x', 'color': 'b'},
              {'name': 'MIN', 'val': float(qubit['cZTargetPhaseCorrAmpMin']), 'range': (min(targetAmp),max(targetAmp)), 'axis': 'x', 'color': 'r'}]
    result = adjust(params, traces)
    if result is not None:
        cZCorrAmpMax = result['MAX']
        cZCorrAmpMin = result['MIN']
        qubit['cZTargetPhaseCorrAmpMax'] = cZCorrAmpMax
        qubit['cZTargetPhaseCorrAmpMin'] = cZCorrAmpMin

def adjust(*a, **kw):
    return runInSubprocess(_adjustGeneric, *a, **kw)

def _adjustGeneric(params, traces):
    xlines = []
    ylines = []
    xsliders = []
    ysliders = []
    
    xparams = [p for p in params if p['axis'] == 'x']
    yparams = [p for p in params if p['axis'] == 'y']
    
    result = [None]

    sh = 0.03
    dy = 0.04
    
    sw = 0.025
    dx = 0.05#0.03

    top = 0.95
    left = 0.08
    bottom = sh + dy*(len(xparams)+2)
    right = 1 - sw - dx*(len(yparams)+1)
    
    bgap = 0.1
    bw = (right - left - bgap)/2.0
    
    # plot the original data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(top=top, left=left, bottom=bottom, right=right)
    for trace in traces:
        args = trace.get('args', ())
        kw = trace.get('kw', {})
        ax.plot(trace['x'], trace['y'], *args, **kw)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
        
    # draw lines and create sliders for adjustable params
    for i, p in enumerate(xparams):
        val = p['val']
        x = [val, val] if p['axis'] == 'x' else xlim
        y = [val, val] if p['axis'] == 'y' else ylim
        line, = ax.plot(x, y, c=p['color'])
        xlines.append(line)
        min, max = p['range']
        slider_ax = fig.add_axes([left, sh+dy*(len(xparams)-i), right-left, sh])
        s = widgets.Slider(slider_ax, p['name'], min, max, p['val'], valfmt='%0.3f', color=p['color'])
        xsliders.append(s)
    
    for i, p in enumerate(yparams):
        val = p['val']
        x = [val, val] if p['axis'] == 'x' else xlim
        y = [val, val] if p['axis'] == 'y' else ylim
        line, = ax.plot(x, y, c=p['color'])
        ylines.append(line)
        min, max = p['range']
        slider_ax = fig.add_axes([1-sw-dx*(len(yparams)-i), bottom, sw, top-bottom])
        s = VerticalSlider(slider_ax, p['name'], min, max, p['val'], valfmt='%0.3f', color=p['color'])
        ysliders.append(s)
    
    # create save and cancel buttons
    btn_ax = fig.add_axes([left, sh, bw, sh])
    save_btn = widgets.Button(btn_ax, 'Save')
    
    btn_ax = fig.add_axes([right-bw, sh, bw, sh])
    cancel_btn = widgets.Button(btn_ax, 'Cancel')
    
    # event callbacks
    def update(val):
        for p, line, slider in zip(xparams, xlines, xsliders):
            val = p['val'] = slider.val
            x = [val, val] if p['axis'] == 'x' else xlim
            y = [val, val] if p['axis'] == 'y' else ylim
            line.set_xdata(x)
            line.set_ydata(y)
        for p, line, slider in zip(yparams, ylines, ysliders):
            val = p['val'] = slider.val
            x = [val, val] if p['axis'] == 'x' else xlim
            y = [val, val] if p['axis'] == 'y' else ylim
            line.set_xdata(x)
            line.set_ydata(y)
        plt.draw() # redraw the figure
    
    def save(e):
        result[0] = dict((p['name'], p['val']) for p in params)
        plt.close(fig)
    
    def cancel(e):
        plt.close(fig)
    
    # hook up events    
    for slider in xsliders + ysliders:
        slider.on_changed(update)
    cancel_btn.on_clicked(cancel)
    save_btn.on_clicked(save)
    
    # initial update
    update(None)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.show()
    return result[0]


class VerticalSlider(widgets.Widget):
    """
    A vertical slider representing a floating point range

    The following attributes are defined
      ax     : the slider axes.Axes instance
      val    : the current slider value
      vline  : a Line2D instance representing the initial value
      poly   : A patch.Polygon instance which is the slider
      valfmt : the format string for formatting the slider text
      label  : a text.Text instance, the slider label
      closedmin : whether the slider is closed on the minimum
      closedmax : whether the slider is closed on the maximum
      slidermin : another slider - if not None, this slider must be > slidermin
      slidermax : another slider - if not None, this slider must be < slidermax
      dragging : allow for mouse dragging on slider

    Call on_changed to connect to the slider event
    """
    def __init__(self, ax, label, valmin, valmax, valinit=0.5, valfmt='%1.2f',
                 closedmin=True, closedmax=True, slidermin=None, slidermax=None,
                 dragging=True, **kwargs):
        """
        Create a slider from valmin to valmax in axes ax;

        valinit -  the slider initial position

        label - the slider label

        valfmt - used to format the slider value

        closedmin and closedmax - indicate whether the slider interval is closed

        slidermin and slidermax - be used to contrain the value of
          this slider to the values of other sliders.

        additional kwargs are passed on to self.poly which is the
        matplotlib.patches.Rectangle which draws the slider.  See the
        matplotlib.patches.Rectangle documentation for legal property
        names (eg facecolor, edgecolor, alpha, ...)
          """
        self.ax = ax

        self.valmin = valmin
        self.valmax = valmax
        self.val = valinit
        self.valinit = valinit
        self.poly = ax.axhspan(valmin,valinit,0,1, **kwargs)

        self.hline = ax.axhline(valinit,0,1, color='r', lw=1)


        self.valfmt = valfmt
        ax.set_yticks([])
        ax.set_ylim((valmin, valmax))
        ax.set_xticks([])
        ax.set_navigate(False)

        ax.figure.canvas.mpl_connect('button_press_event', self._update)
        if dragging:
            ax.figure.canvas.mpl_connect('motion_notify_event', self._update)
        # TODO fix text
        # self.label = ax.text(-0.02, 0.5, label, transform=ax.transAxes,
                             # verticalalignment='center',
                             # horizontalalignment='right')

        # self.valtext = ax.text(1.02, 0.5, valfmt%valinit,
                               # transform=ax.transAxes,
                               # verticalalignment='center',
                               # horizontalalignment='left')
                               
        self.label = ax.text(0.8, -0.03, label, transform=ax.transAxes,rotation='vertical',
                             verticalalignment='center',
                             horizontalalignment='right')

        self.valtext = ax.text(-0.6, 0.5, valfmt%valinit,rotation='vertical',
                               transform=ax.transAxes,
                               verticalalignment='center',
                               horizontalalignment='left')

        self.cnt = 0
        self.observers = {}

        self.closedmin = closedmin
        self.closedmax = closedmax
        self.slidermin = slidermin
        self.slidermax = slidermax

    def _update(self, event):
        'update the slider position'
        if event.button !=1: return
        if event.inaxes != self.ax: return
        val = event.ydata
        if not self.closedmin and val <= self.valmin: return
        if not self.closedmax and val >= self.valmax: return

        if self.slidermin is not None:
            if val <= self.slidermin.val: return

        if self.slidermax is not None:
            if val >= self.slidermax.val: return

        self.set_val(val)

    def set_val(self, val):
        xy = self.poly.xy
        xy[1] = 0, val
        xy[2] = 1, val
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt%val)
        if self.drawon: self.ax.figure.canvas.draw()
        self.val = val
        if not self.eventson: return
        for _cid, func in self.observers.items():
            func(val)

    def on_changed(self, func):
        """
        When the slider valud is changed, call this func with the new
        slider position

        A connection id is returned which can be used to disconnect
        """
        cid = self.cnt
        self.observers[cid] = func
        self.cnt += 1
        return cid

    def disconnect(self, cid):
        'remove the observer with connection id cid'
        try: del self.observers[cid]
        except KeyError: pass

    def reset(self):
        "reset the slider to the initial value if needed"
        if (self.val != self.valinit):
            self.set_val(self.valinit)


def runInSubprocess(f, *a, **kw):
    q = Queue()
    p = Process(target=_run, args=(q, f, a, kw))
    p.start()
    while True:
        try:
            result = q.get(timeout=0.3)
            break
        except Empty:
            if not p.is_alive():
                raise Exception('Child process died!')
    p.join()
    return result


def _run(q, f, a, kw):
    q.put(f(*a, **kw))

