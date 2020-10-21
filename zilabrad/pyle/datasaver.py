import labrad
from zilabrad.pyle import types


class Dataset(object):
    """Encapsulates a dataset that is created and written to via the data vault.
    
    If no connection is provided, a new one will be created.  Whether a new
    connection is established or not, all communication happens in our own
    private context, so that we will not interfere with other communication
    happening with datasets in other contexts.
    
    As data are added to the dataset, requests are fired and buffered, but not
    waited for, so as not to block program execution.  Once a certain number
    of requests have been fired (as set by the delay parameter), adding more
    data will cause the oldest request to be popped out of the buffer and waited
    for.  When the disconnect method is called, all remaining requests will be
    popped and waited for, ensuring that the data are fully saved.
    
    This object is compatible with the context management protocol used by the
    'with' statement in python, which will call connect at the beginning and
    disconnect at the end, ensuring that all data added during the body of the
    with statement will get saved before program execution continues.
    
    The lazy parameter controls when the dataset is actually created.  If lazy
    is set to True (the default), then the dataset will not be created until
    the first time data are added to the dataset.  Thus if an error occurs
    and no data are ever added, we do not even create a dataset, which should
    reduce the number of junk empty datasets in the data vault.  Alternately,
    if lazy is set to False, then a dataset will be created immediately when
    connect is called.  Note that in either case, we do not actually wait for
    the result of the creation request until later when the request buffer is
    filled or disconnect is called.
    """
    def __init__(self, path, name, independents, dependents,
                 params={}, mkdir=True, delay=10, lazy=True, cxn=None):
        self.cxn = cxn
        self.path = path
        self.name = name
        self.independents = [(n, str(u)) for n, u in independents]
        self.dependents = [(n, l, str(u)) for n, l, u in dependents]
        self.params = extractParams(params)
        self.mkdir = mkdir
        self.delay = delay
        self.lazy = lazy
        self.created = False
        self.first_request = True
        self.requests = []
        self.connected = False
    
    def connect(self):
        """Connect to the data vault and (possibly) create the dataset."""
        if self.connected:
            raise Exception('Dataset already connected to data vault')
        self.connected = True
        if self.cxn is None:
            self.cxn = labrad.connect()
            self.closeCxn = True
        else:
            self.closeCxn = False
        self.server = self.cxn.data_vault
        self.context = self.cxn.context()
        if not self.lazy:
            self._create()
    
    def disconnect(self):
        """Wait for pending requests and disconnect from the data vault.""" 
        try:
            for req in self.requests:
                req.running()
        except: pass
        finally:
            if self.closeCxn:
                cxn = self.cxn
                self.cxn = None
                cxn.disconnect()
    
    def __enter__(self):
        """Call connect when entering a with-statement."""
        self.connect()
        return self
    
    def __exit__(self, t, val, tb):
        """Call disconnect when exiting a with-statement."""
        try:
            self.disconnect()
        except:
            # if an error happens in disconnect, we reraise it
            # only in the case that no error was passed in to
            # the exit method.
            if t is None:
                raise
    
    def _create(self):
        """Create the dataset."""
        p = self.server.packet(context=self.context)
        p.cd(self.path, self.mkdir)
        p.new(self.name, self.independents, self.dependents)
        if len(self.params):
            p.add_parameters(tuple(self.params))
        self.requests.append(p.send_future())
        self.created = True
    
    # see labrad.client
    # and concurrent.futures.Future
    def add(self, data):
        """Add data to this dataset.
        
        If the buffer of pending requests is full, the oldest
        request will be popped out of the buffer and waited for
        before the new data are added.
        """
        if self.lazy and not self.created:
            self._create() # make sure the dataset has been created
        if len(self.requests) >= self.delay:
            result = self.requests.pop(0)
            if result: 
                result = result.result()
            if self.first_request:
                # the first request is to create the dataset, so the
                # response contains our path and the name assigned by the
                # data vault.  This name has a number prefix to make it
                # unique.  We pull out this number so it can be used
                # later if we need to retrieve the dataset.
                self.path, self.fullName = result['new']
                self.num = int(self.fullName.split(' - ')[0])
                self.first_request = False
        self.requests.append(self.server.add(data, context=self.context))
        return data
    
    def capture(self, iterable):
        """Capture all data from iterable and add it to this dataset.
        
        This method returns a generator that internally uses a
        with statement to connect to LabRAD and disconnect when
        done.  Once the generator is done the with-statement is
        exited and no more data can be added to the dataset.
        In other words, the iterable will be the _only_ data
        that can get added to this dataset.
        """
        with self as dataset:
            for data in iterable:
                dataset.add(data)
                yield data


def extractParams(params, ignore='_'):
    """Extract parameters from a dictionary into a list of pairs.
    
    Parameters in nested dictionaries are pulled out into one flat list
    with dotted names for the keys.  The optional 'ignore' parameter
    can be used to prevent certain parameters from being included in the
    dataset.  Any parameters (at any level of nesting) that begin with
    the ignore string will be skipped.
    """
    def extract(d, prefix=''):
        parameters = []
        for key, val in sorted(d.items()):
            if ignore and key.startswith(ignore):
                continue
            elif val is None:
                continue
            elif isinstance(val, dict):
                parameters.extend(extract(val, prefix + key + '.'))
            else:
                try:
                    types.flatten(val)
                    parameters.append((prefix + key, val))
                except Exception:
                    pass # skip anything that cannot be flattened by LabRAD
        return parameters
    return extract(params)

