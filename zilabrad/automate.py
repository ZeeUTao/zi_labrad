"""
Automatic calibration

dh: zilabrad.plots.dataProcess.datahelp
"""

import time
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import numpy as np
from zilabrad import multiplex as mp
from zilabrad.instrument.QubitContext import loadQubits
from labrad.units import Unit, Value

ar = mp.ar
_unitSpace = ('V', 'mV', 'us', 'ns', 's', 'GHz',
              'MHz', 'kHz', 'Hz', 'dBm', 'rad', 'None')
V, mV, us, ns, s, GHz, MHz, kHz, Hz, dBm, rad, _l = [
    Unit(s) for s in _unitSpace]

try:
    from sklearn.cluster import KMeans
except Exception:
    _scikit_import_error = "sklearn not installed, use, \
pip install -U scikit-learn"
    raise ImportError(_scikit_import_error)


def color_generator(level):
    """
    example: colors = color_generator(level=3)
    for i in range(3):
        plt.scatter(1,1,color=next(colors)['color'])
    """
    if level >= 5:
        colors = plt.rcParams["axes.prop_cycle"]()
    else:
        colors = ({'color': i} for i in ['b', 'r', 'g', 'purple'])
    return colors


def _tune_piamp(
    sample, dh, Qubit, idx=-1, idx_pro=5, plot=True, update=True,
    amp_key='piAmp', _error=0.2
):
    data = dh.getDataset(idx, None)
    xdata = data[:, 0]
    ydata = data[:, idx_pro]

    def func(x, a, b, c):
        return a*(np.sin(np.pi/2. * x/b))**2+c
    _piamp0 = Qubit[amp_key]

    popt, pcov = curve_fit(
        func, xdata, ydata,
        p0=[np.max(ydata)-np.min(ydata), _piamp0, np.min(ydata)]
    )

    piamp = np.round(popt[1], 4)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(xdata, ydata, 'ko')
        plt.plot(xdata, func(xdata, *popt), 'r-')
        plt.plot(piamp+xdata*0, ydata, 'b', linewidth=5)
        plt.grid()
        title = Qubit._dir[-1] + f': {amp_key}->{piamp}'
        plt.title(title)
        plt.show()
    Qubit[amp_key] = piamp
    return


def tune_piamp(
    sample, dh, idx=-1, idx_pro=5, measure=0, amp_key='piAmp', steps=30
):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Qubit = Qubits[measure]
    piamp0 = Qubit[amp_key]
    mp.rabihigh(
        sample, piamp=ar[0.:2.*piamp0:2.*piamp0/steps],
        measure=measure, name='rabihigh '+amp_key)
    time.sleep(0.5)
    _tune_piamp(
        sample, dh, Qubit, idx=idx, idx_pro=idx_pro, plot=True, update=True,
        amp_key=amp_key)


def tune_piamp21(
    sample, dh, idx=-1, idx_pro=7, measure=0, amp_key='piAmp21', steps=30
):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Qubit = Qubits[measure]
    piamp0 = Qubit[amp_key]
    mp.rabihigh21(
        sample, piamp21=ar[0.:2.*piamp0:2.*piamp0/steps],
        measure=measure, name='rabihigh '+amp_key)
    time.sleep(0.5)
    _tune_piamp(
        sample, dh, Qubit, idx=idx, idx_pro=idx_pro, plot=True, update=True,
        amp_key=amp_key)


def _tune_freq(
    sample, dh, Qubit, idx=0, idx_pro=5, plot=True, update=True,
    _key='f10'
):
    """
    Note: xdata is in MHz, e.g., ar[-40:40:2,MHz]
    """
    data = dh.getDataset(idx, None)
    xdata = data[:, 0]
    ydata = data[:, idx_pro]

    n_data = len(xdata)
    mean0 = 0.
    sigma0 = sum(ydata*(xdata-mean0)**2)/n_data

    def gauss(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    freq0 = Qubit[_key]

    popt, pcov = curve_fit(
        gauss, xdata, ydata,
        p0=[1, mean0, sigma0]
    )

    freq_shift = np.round(popt[1], 2)
    freq_new = freq0 + Value(freq_shift, 'MHz')
    freq_new = Value(freq_new['GHz'], 'GHz')
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        plt.plot(xdata, ydata, 'ko:', label='data')
        plt.plot(xdata, gauss(xdata, *popt), 'r-', linewidth=2)
        plt.plot(freq_shift+xdata*0, ydata, 'b', linewidth=5)
        plt.grid()
        title = Qubit._dir[-1] + f': {_key}->{freq_new}'
        plt.title(title)
        plt.show()
    if update:
        Qubit[_key] = freq_new
    return


def tune_f10(
    sample, dh, df=ar[-40:40:2, MHz], idx_pro=5, measure=0, _key='f10'
):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Qubit = Qubits[measure]
    mp.rabihigh(
        sample, piamp=None, df=df, measure=measure,
        name='rabihigh '+_key)
    time.sleep(0.5)
    _tune_freq(sample, dh, Qubit, idx=-1, idx_pro=idx_pro, _key=_key)


def tune_f21(
    sample, dh, df=ar[-40:40:2, MHz], idx_pro=7, measure=0, _key='f21'
):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Qubit = Qubits[measure]
    mp.rabihigh21(
        sample, piamp21=None, df=df, measure=measure,
        name='rabihigh '+_key)
    time.sleep(0.5)
    _tune_freq(sample, dh, Qubit, idx=-1, idx_pro=idx_pro, _key=_key)


def IQ_cali(
        sample, dh, idx=-1, measure=0, n_cluster=None, plot=True, update=True,
        plot_scatter=True, cmap='Greys', do_return=False
):
    """
    Args:
        dh: zilabrad.plots.dataProcess.datahelp
        idx: iq raw data index
        level: number of clusters, default is len(data) //2,
        for example, data is [I0,Q0,I1,Q1], then level = 4//2 = 2
        measure: index for qubit, whose IQ center will be updated
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Q = Qubits[measure]

    data = dh.getDataset(idx)
    _centers_average = np.mean(data, 0)
    level = len(_centers_average)//2
    if n_cluster is None:
        n_cluster = level
    centers_average = _centers_average.reshape(level, 2)

    def get_IQ_data(i):
        return data[:, [2*i, 2*i+1]]

    def IQ_center_assign(center_state, centers):
        # center_state (array): the averaged IQ data
        # [I,Q] for a given state
        dis = []
        for i, center in enumerate(centers):
            _dis = np.linalg.norm(center-center_state, ord=2)
            dis.extend([_dis])
        dis = np.asarray(dis)
        idx = np.where(dis == np.min(dis))
        return np.int(idx[0])

    data_cluster = np.vstack(list(map(get_IQ_data, range(level))))
    state_pred = KMeans(n_clusters=n_cluster).fit_predict(data_cluster)

    _return = {}

    centers = np.zeros((level, 2))
    for i in range(level):
        xs = data_cluster[state_pred == i, 0]
        ys = data_cluster[state_pred == i, 1]
        center = np.mean(xs), np.mean(ys)
        centers[i] = center

    idx_assign = []
    if update:
        for i in range(level):
            idx = IQ_center_assign(centers_average[i], centers)
            idx_assign += [idx]
            Q[f'center|{i}>'] = centers[idx]
            print(f'update: center|{i}> = {np.round(centers[idx],3)}')

    centers_I, centers_Q = centers[idx_assign, 0], centers[idx_assign, 1]

    def plot_cali_center():
        facecolor = 'w'
        colors = color_generator(level)
        for i in range(level):
            color_i = next(colors)['color']
            ax.scatter(
                centers_I[i],
                centers_Q[i], linewidth=2, facecolor=facecolor,
                edgecolor=color_i, s=200, marker="*", label=i)
        plt.legend()
        return

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        data_hist2d = ax.hist2d(
            *data_cluster.T, bins=50, cmap=cmap,
            shading='auto')

        plot_cali_center()
        plt.show()

    if plot_scatter:
        colors = color_generator(level)
        for i in range(level):
            color_i = next(colors)['color']
            plt.scatter(
                *data[:1000, [2*i, 2*i+1]].T, alpha=0.2,
                label=i, color=color_i)
        plot_cali_center()
        plt.legend()
    # for test
    if do_return:
        _return['centers'] = centers
        _return['data_hist2d'] = data_hist2d
        return _return
