# cython_weakdict.pyx
from cpython.weakref cimport WeakValueDictionary

cdef class CyWeakDict:
    cdef WeakValueDictionary _weak_refs

    def __init__(self):
        self._weak_refs = WeakValueDictionary()

    def __getitem__(self, key):
        val = self._weak_refs.get(key)
        if val is None:
            raise KeyError(key)
        return val

    def __setitem__(self, key, value):
        self._weak_refs[key] = value

    def __delitem__(self, key):
        del self._weak_refs[key]

    def __contains__(self, key):
        return key in self._weak_refs

    def get(self, key, default=None):
        return self._weak_refs.get(key, default)

    def clear(self):
        self._weak_refs.clear()
