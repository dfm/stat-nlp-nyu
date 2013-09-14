#include <Python.h>
#include <numpy/arrayobject.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

double logsumexp(double a, double b)
{
    if (a >= b) return a + log(1 + exp(b - a));
    return b + log(1 + exp(a - b));
}

static PyObject
*maxent_objective (PyObject *self, PyObject *args)
{
    double sigma;
    PyObject *w_obj, *inds_obj, *f_obj;
    if (!PyArg_ParseTuple(args, "OOOd", &w_obj, &inds_obj, &f_obj, &sigma))
        return NULL;

    PyArrayObject *w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_IN_ARRAY),
                  *inds_array = (PyArrayObject*)PyArray_FROM_OTF(inds_obj, NPY_LONG, NPY_IN_ARRAY),
                  *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    int i, j, k,
        nweights = PyArray_DIM(w_array, 0),
        nfeatures = PyArray_DIM(f_array, 1),
        nsamples = PyArray_DIM(inds_array, 0),
        nclasses = nweights / nfeatures;

    // Allocate the memory for the Jacobian.
    npy_intp dim[1] = {nweights};
    PyArrayObject *grad_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                  NPY_DOUBLE);

    double nlp = 0.0, value, norm, factor,
           *probs = malloc(nclasses*sizeof(double)),
           *w = PyArray_DATA(w_array),
           *f = PyArray_DATA(f_array),
           *grad = PyArray_DATA(grad_array);
    long *inds = PyArray_DATA(inds_array);

    for (i = 0; i < nweights; ++i) grad[i] = 0.0;

    for (i = 0; i < nsamples; ++i) {
        norm = -INFINITY;
        for (j = 0; j < nclasses; ++j) {
            value = 0.0;
            for (k = 0; k < nfeatures; ++k)
                value += w[j*nfeatures+k] * f[i*nfeatures+k];
            probs[j] = value;
            norm = logsumexp(norm, value);
        }
        nlp -= probs[inds[i]] - norm;

        for (j = 0; j < nclasses; ++j) {
            factor = exp(probs[j] - norm);
            for (k = 0; k < nfeatures; ++k)
                grad[j*nfeatures+k] += factor * f[i*nfeatures+k];
        }

        for (k = 0; k < nfeatures; ++k)
            grad[inds[i]*nfeatures+k] -= f[i*nfeatures+k];
    }
    free(probs);

    // L2 norm.
    double l2 = 0.0;
    sigma = 1.0 / sigma / sigma;
    for (i = 0; i < nweights; ++i) {
        l2 += w[i] * w[i];
        grad[i] += sigma * w[i];
    }
    nlp += 0.5 * l2 * sigma;

    // Clean up.
    Py_DECREF(w_array);
    Py_DECREF(inds_array);
    Py_DECREF(f_array);

    printf("%f\n", -nlp);

    PyObject *ret = Py_BuildValue("dO", nlp, grad_array);
    Py_DECREF(grad_array);

    return ret;
}

static PyMethodDef maxent_methods[] = {
    {"objective",
     (PyCFunction) maxent_objective,
     METH_VARARGS,
     ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int maxent_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int maxent_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_maxent",
    NULL,
    sizeof(struct module_state),
    maxent_methods,
    NULL,
    maxent_traverse,
    maxent_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__maxent(void)
#else
#define INITERROR return

void init_maxent(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_maxent", maxent_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_maxent.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

    import_array();

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
