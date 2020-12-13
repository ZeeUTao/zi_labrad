import numpy as np
from twisted.internet.defer import returnValue

from zilabrad.pyle.util import getch
from zilabrad.pyle.datasaver import Dataset


def prepDataset(sample, name, axes=None, dependents=None, measure=None, kw=None):
    """Prepare dataset for a sweep.

    This function builds a dictionary of keyword arguments to be used to create
    a Dataset object for a sweep.  Sample should be a dict-like object
    (usually a copy of the sample as returned by loadQubits) that contains current
    parameter settings.  Name is the name of the dataset, which will get prepended
    with a list indicating which qubits are in the sample config, and which of them
    are to be measured.  kw is a dictionary of additional parameters that should
    be added to the dataset.

    Axes can be specified explicitly as a tuple of (<name>, <unit>), or else
    by value for use with grid sweeps.  In the latter case (grid sweep), you should
    specify axes as (<value>, <name>).  If value is iterable, the axis will be
    added to the list of dependent variables so the value can be swept (we look
    at element [0] to get the units); if value is not iterable it will be added
    to the dictionary of static parameters.

    Dependents is either a list of (<name>, <label>, <unit>) designations for the
    dependent variables, or None.  If no list is provided, then the dependents are
    assumed to be probabilities.  In this case, the measure variable is used to
    determine the appropriate set of probabilities: for one qubit, we assume only
    P1 will be measured, while for N qubits all 2^N probabilities are assumed to
    be measured, in the order P00...00, P00...01, P00...10,..., P11...11.  If this
    is not what you want, you must specify the independent variables explicitly.

    Note that measure can be None (all qubits assumed to be measured), an integer
    (just one qubit measured, identified by index in sample['config']) or a list
    of integers (multiple qubits measured).
    """
    conf = list(sample['config'])

    # copy parameters
    kw = {} if kw is None else dict(kw)
    kw.update(sample)  # copy all sample data

    if measure is None:
        measure = list(range(len(conf)))
    elif isinstance(measure, int):
        measure = [measure]

    if hasattr(measure, 'params'):
        # this is a Measurer
        kw.update(measure.params())
    else:
        kw['measure'] = measure

    # update dataset name to reflect which qubits are measured
    for i, q in enumerate(conf):
        if i in kw['measure']:
            conf[i] = '|%s>' % q
    name = '%s: %s' % (', '.join(conf), name)

    # create list of independent vars
    independents = []
    for param, label in axes:
        if isinstance(param, str):
            # param specified as string name
            independents.append((param, label))
        elif np.iterable(param):
            # param value will be swept
            try:
                units = param[0].units
            except Exception:
                units = ''
            independents.append((label, units))
        else:
            # param value is static
            kw[label] = param

    # create list of dependent vars
    if dependents is None:
        if hasattr(measure, 'dependents'):
            # this is a Measurer
            dependents = measure.dependents()
        else:
            n = len(measure)
            if n == 1:
                labels = ['|1>']
            else:
                labels = ['|%s>' % bin(i)[2:].rjust(n, '0')
                          for i in range(2**n)]
            dependents = [('Probability', s, '') for s in labels]

    return Dataset(
        path=list(sample._dir),
        name=name,
        independents=independents,
        dependents=dependents,
        params=kw
    )


def gridSweep(axes):
    if not len(axes):
        yield (), ()
    else:
        (param, _label), rest = axes[0], axes[1:]
        if np.iterable(param):  # TODO: different way to detect if something should be swept
            for val in param:
                for all, swept in gridSweep(rest):
                    yield (val,) + all, (val,) + swept
        else:
            for all, swept in gridSweep(rest):
                yield (param,) + all, swept


def grid(func, axes, **kw):
    """Run a pipelined sweep on a grid over the given list of axes.

    The axes should be specified as a list of (value, label) tuples.
    We iterate over each axis that is iterable, leaving others constant.
    Func should be written to return only the dependent variable data
    (e.g. probabilities), and the independent variables that are being
    swept will be prepended automatically before the data is passed along.

    All other keyword arguments to this function are passed directly to run.
    """
    def gridSweep(axes):
        if not len(axes):
            yield (), ()
        else:
            (param, _label), rest = axes[0], axes[1:]
            if np.iterable(param):  # TODO: different way to detect if something should be swept
                for val in param:
                    for all, swept in gridSweep(rest):
                        yield (val,) + all, (val,) + swept
            else:
                for all, swept in gridSweep(rest):
                    yield (param,) + all, swept

    # pass in all params to the function, but only prepend swept params to data
    def wrapped(server, args):
        all, swept = args
        ans = yield func(server, *all)
        ans = np.asarray(ans)
        pre = np.asarray(swept)
        if len(ans.shape) != 1:
            pre = np.tile([pre], (ans.shape[0], 1))
        returnValue(np.hstack((pre, ans)))

    return run(wrapped, gridSweep(axes), abortPrefix=[1], **kw)


def checkAbort(iterable, labels=[], prefix=[], func=None):
    """Wrap an iterator to allow it to be aborted during iteration.

    Pressing ESC will cause the iterable to abort immediately.
    Alternately, pressing a number key (1, 2, 3, etc.) will abort
    the next time there is a change at a specific index in the value
    produced by the iterable.  This assumes that the source iterable
    returns values at each step that are indexable (e.g. tuples) so
    that we can grab a particular element and check if it has changed.

    In addition, the optional prefix parameter allows to specify a part
    of the value at each step to be monitored for changes.  For example,
    grid sweeps produce two tuples, a tuple of all current values,
    and a second tuple of the current values of only the swept parameters
    (the tuple of all values is what gets passed to the sweep function,
    while the second tuple of just swept parameters is what gets passed
    to the data vault).  In this case, the prefix would be set to [1]
    so that we only check the second tuple for changes.
    """
    idx = -1
    last = None
    for val in iterable:
        curr = val
        for i in prefix:
            curr = curr[i]
        key = getch.getch()
        key2 = getch.getch()
        if key is not None:
            if key == b'\x1b' and key2 == b'\x1b':
                print('Abort scan')
                if func:
                    try:
                        func()
                    except Exception:
                        pass
                break
            elif hasattr(curr, '__len__') and key in [str(i+1) for i in range(len(curr))]:
                idx = int(key) - 1
                if labels:
                    print('Abort scan on next change of %s' % labels[idx])
                else:
                    print('Abort scan on next change at index %d' % idx)
            elif key == '\r':
                if idx >= 0:
                    idx = -1
                    print('Abort canceled')
        if (idx >= 0) and (last is not None):
            if curr[idx] != last[idx]:
                break
        yield val
        last = curr
