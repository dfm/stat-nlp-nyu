#include <Python.h>
#include <numpy/arrayobject.h>
#include <lbfgs.h>

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

typedef struct dataset {

    int nweights, nsamples, nclasses, nfeatures;
    long *inds;
    double *f, sigma;

} Dataset;

static
double evaluate_one (Dataset *data, double *w, double *grad, int n, int i)
{
    int j, k,
        nclasses = data->nclasses,
        nfeatures = data->nfeatures;
    long *inds = data->inds;
    double norm = -INFINITY, nlp = 0.0, factor, *f = data->f,
           *probs = malloc(nclasses*sizeof(double));
    for (j = 0; j < nclasses; ++j) {
        double value = 0.0;
        for (k = 0; k < nfeatures; ++k)
            value += w[j*nfeatures+k] * f[i*nfeatures+k];
        probs[j] = value;
        norm = logsumexp(norm, value);
    }
    nlp -= probs[inds[i]] - norm;
    free(probs);

    for (j = 0; j < nclasses; ++j) {
        factor = exp(probs[j] - norm);
        for (k = 0; k < nfeatures; ++k)
            grad[j*nfeatures+k] += factor * f[i*nfeatures+k];
    }

    for (k = 0; k < nfeatures; ++k)
        grad[inds[i]*nfeatures+k] -= f[i*nfeatures+k];

    return nlp;
}

static
lbfgsfloatval_t evaluate(void *instance, const lbfgsfloatval_t *w,
                         lbfgsfloatval_t *grad, const int n,
                         const lbfgsfloatval_t step)
{
    int i, j, k,
        nweights = ((Dataset*)instance)->nweights,
        nsamples = ((Dataset*)instance)->nsamples,
        nclasses = ((Dataset*)instance)->nclasses,
        nfeatures = ((Dataset*)instance)->nfeatures;
    long *inds = ((Dataset*)instance)->inds;
    double *f = ((Dataset*)instance)->f,
           norm, factor, nlp = 0.0;

    for (i = 0; i < nweights; ++i) grad[i] = 0.0;

    for (i = 0; i < nsamples; ++i)
        nlp += evaluate_one();

    // L2 norm.
    double l2 = 0.0, sigma = ((Dataset*)instance)->sigma;
    sigma = 1.0 / sigma / sigma;
    for (i = 0; i < nweights; ++i) {
        l2 += w[i] * w[i];
        grad[i] += sigma * w[i];
    }
    nlp += 0.5 * l2 * sigma;

    return nlp;
}

static
int progress(void *instance, const lbfgsfloatval_t *x,
             const lbfgsfloatval_t *g, const lbfgsfloatval_t fx,
             const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm,
             const lbfgsfloatval_t step, int n, int k, int ls)
{
    printf("Iteration %d: ", k);
    printf("fx = %f\n", fx);
    printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
    printf("\n");
    return 0;
}

static PyObject
*maxent_optimize (PyObject *self, PyObject *args)
{
    int maxiter;
    double sigma;
    PyObject *w_obj, *inds_obj, *f_obj;
    if (!PyArg_ParseTuple(args, "OOOdi", &w_obj, &inds_obj, &f_obj, &sigma,
                          &maxiter))
        return NULL;

    PyArrayObject *w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_IN_ARRAY),
                  *inds_array = (PyArrayObject*)PyArray_FROM_OTF(inds_obj, NPY_LONG, NPY_IN_ARRAY),
                  *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    int i,
        nweights = PyArray_DIM(w_array, 0),
        nfeatures = PyArray_DIM(f_array, 1),
        nsamples = PyArray_DIM(inds_array, 0),
        nclasses = nweights / nfeatures;

    // Allocate the memory for the Jacobian.
    npy_intp dim[1] = {nweights};
    PyArrayObject *out_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                 NPY_DOUBLE);

    double nlp = 0.0,
           *w = PyArray_DATA(w_array),
           *f = PyArray_DATA(f_array),
           *out_data = PyArray_DATA(out_array);
    long *inds = PyArray_DATA(inds_array);

    // Set up the dataset.
    Dataset *dataset = malloc(sizeof(Dataset));
    dataset->nweights = nweights;
    dataset->nsamples = nsamples;
    dataset->nclasses = nclasses;
    dataset->nfeatures = nfeatures;
    dataset->inds = inds;
    dataset->f = f;
    dataset->sigma = sigma;

    lbfgsfloatval_t fx;
    lbfgsfloatval_t *x = lbfgs_malloc(nweights);
    lbfgs_parameter_t param;

    for (i = 0; i < nweights; ++i) x[i] = w[i];
    lbfgs_parameter_init(&param);
    param.max_iterations = maxiter;
    int r = lbfgs(nweights, x, &fx, evaluate, progress, dataset, &param);
    printf("L-BFGS optimization terminated with status code = %d\n", r);
    printf("  fx = %f\n", fx);

    for (i = 0; i < nweights; ++i) out_data[i] = x[i];
    nlp = fx;

    lbfgs_free(x);
    free(dataset);

    // Clean up.
    Py_DECREF(w_array);
    Py_DECREF(inds_array);
    Py_DECREF(f_array);

    PyObject *ret = Py_BuildValue("dO", nlp, out_array);
    Py_DECREF(out_array);

    return ret;
}

static PyObject
*maxent_online (PyObject *self, PyObject *args)
{
    int maxiter;
    double sigma;
    PyObject *w_obj, *inds_obj, *f_obj;
    if (!PyArg_ParseTuple(args, "OOOdi", &w_obj, &inds_obj, &f_obj, &sigma,
                          &maxiter))
        return NULL;

    PyArrayObject *w_array = (PyArrayObject*)PyArray_FROM_OTF(w_obj, NPY_DOUBLE, NPY_IN_ARRAY),
                  *inds_array = (PyArrayObject*)PyArray_FROM_OTF(inds_obj, NPY_LONG, NPY_IN_ARRAY),
                  *f_array = (PyArrayObject*)PyArray_FROM_OTF(f_obj, NPY_DOUBLE, NPY_IN_ARRAY);
    int i,
        nweights = PyArray_DIM(w_array, 0),
        nfeatures = PyArray_DIM(f_array, 1),
        nsamples = PyArray_DIM(inds_array, 0),
        nclasses = nweights / nfeatures;

    npy_intp dim[1] = {nweights};
    PyArrayObject *out_array = (PyArrayObject*)PyArray_SimpleNew(1, dim,
                                                                 NPY_DOUBLE);

    double nlp = 0.0,
           *w = PyArray_DATA(w_array),
           *f = PyArray_DATA(f_array),
           *out_data = PyArray_DATA(out_array);
    long *inds = PyArray_DATA(inds_array);

    for (iteration = 0; iterations < maxiter; ++iterations) {

    }

    for (i = 0; i < nweights; ++i) out_data[i] = x[i];
    nlp = fx;

    lbfgs_free(x);
    free(dataset);

    // Clean up.
    Py_DECREF(w_array);
    Py_DECREF(inds_array);
    Py_DECREF(f_array);

    PyObject *ret = Py_BuildValue("dO", nlp, out_array);
    Py_DECREF(out_array);

    return ret;
}

static PyMethodDef maxent_methods[] = {
    {"optimize",
     (PyCFunction) maxent_optimize,
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
