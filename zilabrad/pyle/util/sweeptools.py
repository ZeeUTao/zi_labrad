import math
import random

import numpy as np

from zilabrad.pyle.units import Value, Unit


def iterable(x):
    """Check whether x is iterable, but make an exception for Value's, which override getitem."""
    return np.iterable(x) and not hasattr(x, 'value')


def shuffle(iterable):
    L = list(iterable)
    random.shuffle(L)
    return L


def unshuffle(data):
    """Unshuffles data taken with the sweep server in shuffle mode.
    
    For a 2D array of data, np.lexsort sorts columns using values in the
    last row, then the next to last row, etc. (don't ask me why it works
    that way, but it does).  We want to sort rows using the first
    column, then the second column, etc.  Hence, we transform the data
    by first reversing the order of columns and transposing, to put it
    in the form appropriate for lexsort.  Lexsort returns an array of
    indices giving the order, so we return a new array in which we take
    the rows of the input matrix in the order returned by lexsort.
    """
    data = np.asarray(data)
    indices = np.lexsort(data[:,::-1].T)
    return data.take(indices, axis=0)


def nearest(val, incr):
    """Round a value to the nearest multiple of increment."""
    return round(val / incr) * incr


def inUnits(v, units):
    """Convert a value into the given units if possible."""
    if hasattr(v, 'value'):
        return v[units]
    else:
        return v


def iterUnits(iterable, unit=None):
    """Add units to each element of an iterable."""
    if unit is None:
        return iterable
    else:
        return (val*unit for val in iterable)


class Range(object):
    """Simple object that encapsulates a range of values with units.
    
    Rather than generating a list of Values, this object is iterable and
    generates Values on demand, as needed.
    """
    def __init__(self, range, unit):
        self.range = range
        self.unit = unit
        
    def fill(self, default):
        """Create a new range with missing values filled in from a default range."""
        ra, rb = self.range, default.range
        choose = lambda a, b: a if a is not None else b
        start = choose(ra.start, rb.start)
        stop = choose(ra.stop, rb.stop)
        step = choose(ra.step, rb.step)
        unit = choose(self.unit, default.unit)
        return r[start:stop:step, unit]
    
    def _count(self):
        r, u = self.range, self.unit
        start, stop, step = inUnits(r.start, u), inUnits(r.stop, u), inUnits(r.step, u)
        d = stop - start
        s = abs(step) if d >= 0 else -abs(step)
        start = start * u if u is not None else start
        step = s * u if u is not None else s
        count = int(math.floor(d / s + 0.000000001)) + 1
        return start, step, count
    
    def __iter__(self):
        start, step, count = self._count()
        for i in range(count):
            yield start + step * i
    
    def __getitem__(self, i):
        start, step, count = self._count()
        if i >= 0:
            return start + step * i
        elif i < 0:
            return start + step * (count + i)
    
    def __len__(self):
        return self._count()[2]
    
    def __repr__(self):
        r, u = self.range, self.unit
        return 'r[%s:%s:%s,%s]' % (r.start, r.stop, r.step, u)

class Range2(object):
    """Simple object that encapsulates a range of values with units.
    
    Rather than generating a list of Values, this object is iterable and
    generates Values on demand, as needed.
    """
    def __init__(self, range, unit):
        if isinstance(range, list):
            self.rangeList = range
        else:
            self.rangeList = [range]
        self.unit = unit
        
    def fill(self, default):
        """Create a new range with missing values filled in from a default range."""
        raise Exception('fill is not yet implemented for object Range2')
##        ra, rb = self.range, default.range
##        choose = lambda a, b: a if a is not None else b
##        start = choose(ra.start, rb.start)
##        stop = choose(ra.stop, rb.stop)
##        step = choose(ra.step, rb.step)
##        unit = choose(self.unit, default.unit)
##        return r[start:stop:step, unit]

    def _count(self):
        u = self.unit
        info = []
        for r in self.rangeList:
            start, stop, step = inUnits(r.start, u), inUnits(r.stop, u), inUnits(r.step, u)
            d = stop - start
            s = abs(step) if d >= 0 else -abs(step)
            start = start * u if u is not None else start
            step = s * u if u is not None else s
            count = int(math.floor(d / s + 0.000000001)) + 1
            info.append((start,step,count))
        return info
    
    def __iter__(self):
        info = self._count()
        for start, step, count in info:
            for i in range(count):
                yield start + step * i
    
    def __getitem__(self, i):
        info = self._count()
        if i>=0:
            for start, step, count in info:
                if i>count-1:
                    i -= count
                else:
                    return start + step * i
        elif i < 0:
            for start, step, count in reversed(info):
                if abs(i) >= count:
                    i += count
                else:
                    return start + step * (count+i)

    def __add__(self, other):
        if not self.unit.isCompatible(other.unit):
            raise Exception('Incompatible units in range objects')
        range = self.rangeList[:] #Use slice to get a copy!!! Don't mutate the object!!!
        range.extend(other.rangeList[:]) 
        unit = self.unit
        return Range2(range, unit)

    def __len__(self):
        info = self._count()
        return sum([info[i][2] for i in range(len(info))])
    
    def __repr__(self):
        reprs = ''
        ranges, u = self.rangeList, self.unit
        for r in ranges:
            reprs = reprs + 'r[%s,%s,%s,%s], ' %(r.start, r.stop, r.step, u)
        reprs = reprs[0:-2] #Chop off tailing comma and space characters
        return reprs


class LogRange(object):
    """Simple object that encapsulates a log-spaced range of values with units.
    
    Rather than generating a list of Values, this object is iterable and
    generates Values on demand, as needed.
    """
    def __init__(self, range, unit):
        self.range = range
        self.unit = unit
        
    def fill(self, default):
        """Create a new range with missing values filled in from a default range."""
        ra, rb = self.range, default.range
        choose = lambda a, b: a if a is not None else b
        start = choose(ra.start, rb.start)
        stop = choose(ra.stop, rb.stop)
        step = choose(ra.step, rb.step)
        unit = choose(self.unit, default.unit)
        return r[start:stop:step, unit]
    
    def _range(self):
        r, u = self.range, self.unit
        start, stop, N = inUnits(r.start, u), inUnits(r.stop, u), inUnits(r.step, u)
        return np.logspace(np.log10(start), np.log10(stop), N)
    
    def __iter__(self):
        r = self._range()
        for i in r:
            yield i * self.unit
    
    def __len__(self):
        r = self._range()
        return len(r)
    
    def __repr__(self):
        r, u = self.range, self.unit
        return 'lr[%s:%s:%s,%s]' % (r.start, r.stop, r.step, u)        

class RangeCreator(object):
    """Create ranges with units using slice notation."""
    def __getitem__(self, range):
        if isinstance(range, tuple):
            range, unit = range
        else:
            unit = None
        if isinstance(unit, str):
            unit = Unit(unit)
        return Range(range, unit)

class RangeCreator2(object):
    """Create ranges with units using slice notation."""
    def __getitem__(self, range):
        if isinstance(range, tuple):
            range, unit = range
        else:
            unit = None
        if isinstance(unit, str):
            unit = Unit(unit)
        return Range2(range, unit)


r = RangeCreator() # convenient alias for creating ranges with slice notation
r2 = RangeCreator2()

class LogRangeCreator(object):
    """Create ranges with units using slice notation."""
    def __getitem__(self, range):
        if isinstance(range, tuple):
            range, unit = range
        else:
            unit = None
        if isinstance(unit, str):
            unit = Unit(unit)
        return LogRange(range, unit)

lr = LogRangeCreator() # convenient alias for creating ranges with slice notation



def PQ(val, unit):
    return Value(float(val), unit)

def PQrange(start, stop, step, unit):
    a = np.arange(start, stop, step)
    return [PQ(n, unit) for n in a]

arangePQ = PQrange

def PQlinspace(start, stop, number, unit):
    a = np.linspace(start, stop, number)
    return [PQ(n, unit) for n in a]

def PQscan(start, step, unit):
    while True:
        yield PQ(start, unit)
        start += step

def PQcenter(start, step, unit):
    n = 1
    yield PQ(start, unit)
    while True:
        yield PQ(start + step * n, unit)
        yield PQ(start - step * n, unit)
        n += 1
    
def centerscanPQ(center, radius, step, unit):
    return [PQ(center + x * sign, unit)
            for x in np.arange(0.0, radius, step)
            for sign in [-1.0, 1.0]][1:]


