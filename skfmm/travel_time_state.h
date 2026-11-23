#pragma once
#include "Python.h"

PyObject *state_create(PyObject *, PyObject *);
PyObject *state_advance(PyObject *, PyObject *);
PyObject *state_result(PyObject *, PyObject *);
PyObject *state_flags(PyObject *, PyObject *);
PyObject *state_checkpoint(PyObject *, PyObject *);
PyObject *state_restore(PyObject *, PyObject *);
PyObject *state_stats(PyObject *, PyObject *);
