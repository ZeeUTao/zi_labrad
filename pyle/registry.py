# Copyright (C) 2008  Max Hofheinz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


class AttrDict(dict):
    """A dict whose entries can also be accessed as attributes.
    
    The copy method returns a copy, including recursive copies of
    any sub directories as AttrDict's.  Note that mutable values of
    other types, such as lists, are not copied.
    
    The where method returns a copy with updates passed as a dictionary
    or keyword parameters.  This also allows updates to be made to
    keys in subdirectories by passing dotted names (or, when using
    keyword parameters, by using '__' in place of '.').
    """
    def __getattr__(self, name):
        try:
            return dict.__getitem__(self, name)
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        dict.__setitem__(self, name, value)
    
    def __delattr__(self, name):
        try:
            dict.__delitem__(self, name)
        except KeyError:
            raise AttributeError(name)
    
    def copy(self):
        d = AttrDict(self)
        object.__setattr__(d, '_dir', self._dir)
        object.__setattr__(d, '__name__', self.__name__)
        for k, v in d.items():
            if isinstance(v, AttrDict):
                d[k] = v.copy()
        return d
    
    def where(self, dict={}, **kwds):
        d = self.copy()
        def add_all(updates):
            for k, v in updates.items():
                k = k.replace('__', '.')
                if '.' in k:
                    path = k.split('.')
                    path, k = path[:-1], path[-1]
                    subdir = d
                    for dir in path:
                        subdir = subdir[dir]
                    subdir[k] = v
                else: # just set the key
                    d[k] = v
        add_all(dict)
        add_all(kwds)
        return d


class RegistryWrapper(object):
    """Accesses the labrad registry with a dict-like interface.
    
    Keys or directories in the registry can be accesed using either
    dict-like notation d["key"] or attribute access d.key.  Obviously
    the former is required if you want to compute the string name of
    an element to access and then pass the key as a variable.
    
    For every packet that gets sent to the registry, we first cd
    into the directory wrapped by this wrapper, then at the end
    cd back into the root directory.  That way this directory
    (and any subdirectories that we visit which get wrapped) can
    always be deleted because there is not a stray context hanging
    out in any particular directory.
    
    The copy method returns a local copy of this registry directory,
    including recursive copies of all subdirectories. 
    """
    def __init__(self, cxn, dir='', ctx=None):
        if isinstance(dir, str):
            dir = [dir]
        else:
            dir = list(dir)
        if '' in dir[1:]:
            raise Exception('Empty string is invalid subdirectory name')
        if dir[0] != '':
            dir = [''] + dir
        if ctx is None:
            ctx = cxn.context()
        srv = cxn.registry
            
        object.__setattr__(self, '_dir', dir)
        object.__setattr__(self, '_cxn', cxn)
        object.__setattr__(self, '_srv', srv)
        object.__setattr__(self, '_ctx', ctx)
        
        # make sure the directory gets created
        self._send(self._packet())
        
    def _subdir(self, name):
        """Create a wrapper for a subdirectory, reusing our connection."""
        if name == '':
            raise Exception('Empty string is invalid subdirectory name')
        return RegistryWrapper(self._cxn, self._dir + [name], self._ctx)
        
    def _packet(self):
        """Create a packet with the correct context and directory."""
        return self._srv.packet(context=self._ctx).cd(self._dir, True)
    
    def _send(self, pkt):
        """Change back into the root directory after each request."""
        return pkt.cd(['']).send()
    
    def _get_list(self):
        """Get current directory listing (dirs and keys)."""
        return self._send(self._packet().dir()).dir
        
    ## dict interface
    def __getitem__(self, name):
        dirs, keys = self._get_list()
        if name in dirs:
            return self._subdir(name)
        elif name in keys:
            return self._send(self._packet().get(name)).get
        else:
            raise KeyError(name)

    def __setitem__(self, name, value):
        if isinstance(value, (dict, RegistryWrapper)):
            subdir = self._subdir(name)
            for element in value:
                subdir[element] = value[element]
        else:
            self._send(self._packet().set(name, value))

    def __delitem__(self, name):
        dirs, keys = self._get_list()
        if name in dirs:
            subdir = self._subdir(name)
            for k in subdir:
                del subdir[k]
            self._send(self._packet().rmdir(name))
        elif name in keys:
            self._send(self._packet()['del'](name))
        else:
            raise KeyError(name)

    ## attribute interface
    def __getattr__(self, name):
        try:
            return self.__getitem__(name)
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self.__setitem__(name, value)

    def __delattr__(self, name):
        try:
            return self.__delitem__(name)
        except KeyError:
            raise AttributeError(name)

    def copy(self):
        """Make a local copy, recursively copying subdirs as well."""
        dirs, keys = self._get_list()
        d = AttrDict()
        object.__setattr__(d, '_dir', self._dir)
        object.__setattr__(d, '__name__', self._dir[-1])
        for name in dirs:
            d[name] = self._subdir(name).copy()
        if len(keys):
            p = self._packet()
            for name in keys:
                p.get(name, key=name)
            ans = self._send(p)
            for name in keys:
                d[name] = ans[name]
        return d

    def __contains__(self, name):
        dirs, keys = self._get_list()
        return name in keys or name in dirs
    
    def keys(self):
        dirs, keys = self._get_list()
        return list(dirs) + list(keys)
    
    def __iter__(self):
        return iter(list(self.keys()))
    
    def __repr__(self):
        return '<RegistryWrapper: %r>' % (self._dir,)

