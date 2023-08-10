#include "numpy/ndarraytypes.h"

// int NumPyArray_NDIM(PyObject *obj);
// npy_intp NumPyArray_DIM(PyObject *obj, int i);
// void *NumPyArray_DATA(PyObject *obj);

// Bridge::Bridge();
// {
//     Py_Initialize();
//     _import_array();
// }

// /// Wraps PyArray_NDIM
// int NumPyArray_NDIM(PyObject *obj)
// {
//     return PyArray_NDIM((PyArrayObject *)obj);
// }

// /// Wraps PyArray_DIM
// npy_intp NumPyArray_DIM(PyObject *obj, int i)
// {
//     return PyArray_DIM((PyArrayObject *)obj, i);
// }

// /// Wraps PyArray_DATA
// void *NumPyArray_DATA(PyObject *obj)
// {
//     return PyArray_DATA((PyArrayObject *)obj);
// }
