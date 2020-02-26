# cython: embedsignature=True
# cython: profile=True
# cython: language_version=3


cdef class DataDescription:

    cdef public:
        long index
        list transforms

