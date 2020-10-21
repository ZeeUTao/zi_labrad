"""
Pipelining

The goal of pipelining is to run several operations in parallel
when each operation can take multiple steps.  For datataking, the
steps are typically something like (1) run sequence on the DACs,
(2) process the returned data, (3) save to the data vault.  In the
past, most of our pipelining tools have taken the approach of
assuming (and hence enforcing) this basic structure, which limits
what one can do.  Here, we take a more generic approach, by requiring
only that each process is provided as a python generator.

To enable parallelism with pipelining, functions should return a Future,
which is basically an object that knows how to wait for the result of
some computation that may take a long time, a sort of promise to get a
result in the future.  When you make a Labrad request with the option
wait=False, for example, the result is a Future.  The call returns
immediately and you can go do other things.  When you want the result,
you call .wait() and only then does the system wait for the answer.
So the simplest way to enable parallelism is to write functions that
make Labrad requests with wait=False and return.

In addition, you can use generators that yield Future's for more
complicated work, where for example you need to looping, conditionals,
or postprocessing, and you want this to continue to work with pipelining.
"""

import functools
import itertools


# try to import returnValue and _DefGen_Return from twisted
# if not found, we provide our own replacements, to avoid a
# direct dependency on twisted in this module
try:
    from twisted.internet.defer import returnValue, _DefGen_Return
except ImportError:
    class _DefGen_Return(BaseException):
        def __init__(self, value):
            self.value = value
    def returnValue(val):
        raise _DefGen_Return(val)


def pmap(function, iterable, size=10):
    """Apply function to every item from iterable, up to size calls in parallel.
    
    function - the function to be called for each element in iterable.
        to enable parallelism, this function should return a Future,
        or should be a generator that yields Future's.  Whenever a
        future comes out, the iterable will be advanced and more function
        calls will be started before waiting for this future to complete.
    
    iterable - iterated over to produce values that are fed to function.
    
    size - the maximum number of calls to make in parallel.
    
    The interface to this function is similar to the builtin map function,
    however the name has been changed so as not to overwrite the builtin name,
    and to highlight an important difference with the builtin function.
    Unlike the builtin python map function, pmap returns not a list of results,
    but rather an iterator that produces the results, somewhat like itertools.imap.
    So, in order to ensure that the mapping actually happens, you must 'exhaust'
    the resulting iterator, for example by collecting it into a list or
    iterating over it in a for loop. 
    """
    runners = []
    
    def advance():
        # loop over runners, advancing until one completes
        for runner in itertools.cycle(runners):
            runner.advance()
            if runner.done:
                break
        # if the first runner is done, pop it and return its output
        # Otherwise say we aren't done and return no information
        if len(runners) and runners[0].done:
            return (True, runners.pop(0).result)
        return (False, None)
    
    for val in iterable:
        result = function(val)
        if is_future(result):
            runner = _FutureRunner(result)
        elif is_generator(result):
            runner = _GeneratorRunner(result)
        else:
            runner = _ValueRunner(result)
        runners.append(runner)
        if len(runners) >= size:
            done, result = advance()
            if done:
                yield result
    
    # finish all runners
    while len(runners):
        done, result = advance()
        if done:
            yield result


# runners are helper classes used by pmap to manage execution

class _GeneratorRunner(object):
    def __init__(self, gen):
        self.gen = _run(gen)
        self.done = False
        self.advance()
    
    def advance(self):
        if not self.done:
            try:
                next(self.gen)
            except _DefGen_Return as e:
                self.done = True
                self.result = e.value
            except StopIteration:
                self.done = True
                self.result = None


class _FutureRunner(object):
    def __init__(self, future):
        self.future = future
        self.done = False
    
    def advance(self):
        if not self.done:
            self.result = self.future.wait()
            self.done = True


class _ValueRunner(object):
    def __init__(self, result):
        self.result = result
        self.done = True
        
    def advance(self):
        pass


def _run(gen):
    """Run a pipelineable generator.
    
    The input generator gen is run step by step, keeping track of each
    result it yields.  If this result is a future, we yield, then wait
    for the future the next time we are invoked.
    
    If the result is itself a generator, we call _run recursively to
    handle it in the same manner.
    """
    result = None
    while True:
        try:
            if is_future(result):
                yield
                try:
                    result = result.wait()
                except Exception as e:
                    result = gen.throw(e)
            elif is_generator(result):
                try:
                    for _ in _run(result):
                        yield
                    result = None
                except _DefGen_Return as e:
                    result = e.value
            else:
                result = gen.send(result)
        except _DefGen_Return as e:
            returnValue(e.value)
        except StopIteration:
            returnValue(None)


def call(function, *a, **kw):
    """Call a function written to work with pipelining.
    
    If the function returns a Future, we wait for it; if it returns
    a generator, we run it using the standard pipeline mechanisms.
    Anything else gets returned unchanged.
    """
    result = function(*a, **kw)
    if is_future(result):
        return result.wait()
    elif is_generator(result):
        try:
            for _ in _run(result):
                pass
        except _DefGen_Return as e:
            return e.value
    else:
        return result


def wrap(func, out=None):
    """Wrap a function written to work with pipelining.
    
    Wraps func into a new function that calls out(value) on every
    value coming out of func.  This works for any function that can
    be used in pipelining, even if it is a generator.
    """
    out = out or (lambda x: x)
    def wrap_gen(gen):
        result = None
        while True:
            result = gen.send(result)
            if is_generator(result):
                # recursively wrap subgenerators
                result = wrap_gen(result)
            else:
                result = out(result)
            result = yield result
                    
    @functools.wraps(func)
    def wrapped(*a, **kw):
        result = func(*a, **kw)
        if is_generator(result):
            return wrap_gen(result)
        else:
            return out(result)
    return wrapped


class FutureList(object):
    """Takes a list of futures and returns one future that waits for all of them."""
    def __init__(self, futures):
        self.futures = futures
        self.callbacks = []
        self.result = None
        self.done = False
        
    def addCallback(self, f, *args, **kw):
        if self.done:
            self.result = f(self.result, *args, **kw)
        else:
            self.callbacks.append((f, args, kw))
        return self
        
    def wait(self):
        if self.done:
            return self.result
        self.result = [(f.wait() if is_future(f) else f) for f in self.futures]
        self.done = True
        for f, args, kw in self.callbacks:
            self.result = f(self.result, *args, **kw)
        return self.result
    
    def __repr__(self):
        if self.done:
            return '<FutureList: result=%r>' % (self.result,)
        else:
            return '<FutureList: pending...>'


_gen_type = type(i for i in [])

def is_generator(obj):
    """Determine whether obj is a generator object."""
    return isinstance(obj, _gen_type)


def is_future(obj):
    """Determine whether obj is a future that can be waited for."""
    return hasattr(obj, 'wait') and callable(obj.wait)


def test_pipeline():
    def simple_pipe(i):
        for j in range(4):
            print('pipe %d, stage %d' % (i, j))
            yield
        returnValue(i)
    for result in pmap(simple_pipe, range(10), 5):
        print('got result:', result)
    
    import random
    def weird_pipe(val):
        for _ in range(random.randint(1,10)):
            yield
        returnValue(val)
    ans = [v for v in pmap(weird_pipe, range(20), 5)]
    print(ans)
    
    # test subgenerators
    def subgen(val):
        for _ in range(2):
            yield
        returnValue(val*2)
    def subgen_pipe(val):
        for _ in range(2):
            val = yield subgen(val)
        returnValue(val)
    ans = [v for v in pmap(subgen_pipe, range(10), 5)]
    print(ans)
    
    # test subgenerators with Futures
#    import labrad
#    with labrad.connect() as cxn:
#        mgr = cxn.manager
#        def get_setting_info(server):
#            settings = yield mgr.lr_settings(server, wait=False)
#            settings = [s for w, s in settings]
#            helps = {}
#            for setting in settings:
#                help = yield mgr.help(server, setting, wait=False)
#                helps[setting] = help
#            returnValue((server, helps))
#        servers = [s for w, s in mgr.servers()]
#        ans = dict(pipe(get_setting_info, servers, 5))
#        print ans

    
if __name__ == '__main__':
    test_pipeline()
    
    