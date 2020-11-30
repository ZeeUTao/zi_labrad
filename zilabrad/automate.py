"""
Automatic calibration

dh: zilabrad.plots.dataProcess.datahelp
"""

from zilabrad.instrument.QubitContext import loadQubits
import numpy as np
from matplotlib import pyplot as plt

try:
    from sklearn.cluster import KMeans
except Exception:
    _scikit_import_error = "sklearn not installed, use, \
pip install -U scikit-learn"
    raise ImportError(_scikit_import_error)


def IQ_cali(
        sample, dh, idx=0, idx_q=0, level=2, plot=True, update=True,
        plot_scatter=False
        ):
    """
    Args:
        dh: zilabrad.plots.dataProcess.datahelp
        idx: iq raw data index
        level: number of clusters
        idx_q: index for qubit, whose IQ center will be updated
    """
    sample, qubits, Qubits = loadQubits(sample, write_access=True)
    Q = Qubits[idx_q]

    data = dh.getDataset(idx)

    centers_average = np.mean(data, 0).reshape(level, 2)

    def get_IQ_data(i): return data[:, [2*i, 2*i+1]]

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
    state_pred = KMeans(n_clusters=level).fit_predict(data_cluster)

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

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        _ = ax.hist2d(*data_cluster.T, bins=50, cmap='pink')
        for i in range(level):
            ax.scatter(
                centers_I[i],
                centers_Q[i],
                s=200, marker='X', label=i)
        plt.legend()
        plt.show()
    if plot_scatter:
        for i in range(3):
            plt.scatter(*data[:1000, [2*i, 2*i+1]].T, alpha=0.2, label=i)
        plt.legend()
    _return['centers'] = centers
    return _return
