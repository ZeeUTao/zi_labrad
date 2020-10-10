import itertools

import labrad


class ContextCycler(object):
    """Connect to a Labrad server and send requests in different contexts.
    
    This class helps to do pipelining by sending requests to a labrad server
    in different contexts, a list of which is managed internally so the user
    doesn't have to worry about them.  It conforms to python's with-statement
    protocol, allowing you to create a connection to labrad and then have
    it automatically closed when the with-statement is exited.  In addition,
    we wait for all pending requests to finish before disconnecting.  To make
    requests, call the 'send' method.
    """
    
    def __init__(self, name, contexts=10, cxn=None):
        self.cxn = cxn
        self.num_contexts = contexts
        self.request_nums = itertools.count()
        self.requests = {}
        self.server_name = name
    
    def connect(self):
        """Connect to labrad and set up contexts."""
        if self.cxn is None:
            self.cxn = labrad.connect()
            self.close_cxn = True
        else:
            self.close_cxn = False
        self.server = self.cxn[self.server_name]
        self.contexts = itertools.cycle(self.server.context() for _ in range(self.num_contexts))

    def disconnect(self):
        """Wait for pending requests and disconnect from labrad."""
        try:
            # wait for any requests that are still pending
            reqs = list(self.requests.values())
            if len(reqs):
                print('Waiting for %d requests to complete...' % len(reqs))
            for req in reqs:
                req.wait()
        finally:
            if self.close_cxn:
                cxn = self.cxn
                self.cxn = None
                cxn.disconnect()

    def __enter__(self):
        """Call connect when entering a 'with' statement."""
        self.connect()
        return self
    
    def __exit__(self, t, val, tb):
        """Call disconnect when exiting a 'with' statement."""
        try:
            self.disconnect()
        except:
            # if an error happens in disconnect, we reraise it
            # only in the case that no error was passed in to
            # the exit method.
            if t is None:
                raise
    
    def packet(self, **kwargs):
        kwargs['context'] = next(self.contexts)    
        p = self.server.packet(**kwargs)
        return ManagedPacket(self, p)
        
    def _start_request(self, req):
        i = next(self.request_nums)
        self.requests[i] = req
        req.addCallback(self._finish_request, i)
    
    def _finish_request(self, result, i):
        """Remove a request from the list of pending requests."""
        del self.requests[i]
        return result


class ManagedPacket(object):
    """A packet that notifies a manager when it gets sent.
    
    Allows the manager to keep track of all pending futures that
    have been sent to it, so that they can be properly waited for
    before the manager's connection is closed.
    """
    def __init__(self, manager, packet):
        self._manager = manager
        self._packet = packet
        
    def __getattr__(self, name):
        return getattr(self._packet, name)
    
    def __getitem__(self, name):
        return self._packet[name]
    
    def __iter__(self):
        raise TypeError("'ManagedPacket' object is not iterable")
    
    def send(self, wait=False, **kwargs):
        req = self._packet.send(wait=wait, **kwargs)
        if not wait:
            self._manager._start_request(req)
        return req

