"""
Automatic calibration

dh: zilabrad.plots.dataProcess.datahelp
"""

from zilabrad.instrument.QubitContext import loadQubits
from zilabrad import multiplex as mp
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import time

ar = mp.ar


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
    sample, dh, Qubit, idx=0, idx_pro=5, plot=True, update=True,
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
    sample, dh, idx=0, idx_pro=5, measure=0, amp_key='piAmp', steps=20
):
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Qubit = Qubits[measure]
    piamp0 = Qubit[amp_key]
    mp.rabihigh(
        sample, piamp=ar[0.:2.*piamp0:2.*piamp0/steps],
        measure=measure)
    time.sleep(0.5)
    _tune_piamp(
        sample, dh, Qubit, idx=idx, idx_pro=idx_pro, plot=True, update=True,
        amp_key=amp_key)


def IQ_cali(
        sample, dh, idx=0, measure=0, n_cluster=None, plot=True, update=True,
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
        for i in range(level):
            color_i = next(colors)['color']
            ax.scatter(
                centers_I[i],
                centers_Q[i], linewidth=2, facecolor=facecolor,
                edgecolor=color_i, s=200, marker="*", label=i)
        plt.legend()
        return

    if plot:
        colors = color_generator(level)
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
