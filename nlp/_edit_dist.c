#include <Python.h>

struct module_state {
    PyObject *error;
};

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

#define INSERT_COST     1.0
#define DELETE_COST     1.0
#define SUBSTITUTE_COST 1.0

double min(double a, double b)
{
    if (a <= b) return a;
    return b;
}

double get_distance(PyObject *list1, PyObject *list2, int l1, int l2,
                    int p1, int p2, double **best)
{
    double distance = INFINITY;
    if (p1 > l1 || p2 > l2) return INFINITY;
    if (p1 == l1 && p2 == l2) return 0.0;
    if (best[p1][p2] < 0.0) {
        distance = min(distance, INSERT_COST
                       + get_distance(list1, list2, l1, l2, p1+1, p2, best));
        distance = min(distance, DELETE_COST
                       + get_distance(list1, list2, l1, l2, p1, p2+1, best));
        distance = min(distance, SUBSTITUTE_COST
                       + get_distance(list1, list2, l1, l2, p1+1, p2+1, best));
        if (p1 < l1 && p2 < l2) {
            if (PyObject_RichCompareBool(PyList_GetItem(list1, p1),
                                         PyList_GetItem(list2, p2),
                                         Py_EQ) == 1)
                distance = min(distance, get_distance(list1, list2, l1, l2,
                                                      p1+1, p2+1, best));

        }
        best[p1][p2] = distance;
    }
    return best[p1][p2];
}

static PyObject
*edit_distance (PyObject *self, PyObject *args)
{
    PyObject *list1, *list2;
    if (!PyArg_ParseTuple(args, "OO", &list1, &list2)) return NULL;

    int i, j,
        l1 = (int) PyList_Size(list1),
        l2 = (int) PyList_Size(list2);

    double **best = malloc((l1+1)*sizeof(double*));
    for (i = 0; i <= l1; ++i) {
        best[i] = malloc((l2+1)*sizeof(double));
        for (j = 0; j <= l2; ++j) best[i][j] = -1.0;
    }

    double distance = get_distance(list1, list2, l1, l2, 0, 0, best);

    for (i = 0; i <= l1; ++i) free(best[i]);
    free(best);

    return Py_BuildValue("d", distance);
}

static PyMethodDef edit_methods[] = {
    {"distance",
     (PyCFunction) edit_distance,
     METH_VARARGS,
     ""},
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3

static int edit_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int edit_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_edit",
    NULL,
    sizeof(struct module_state),
    edit_methods,
    NULL,
    edit_traverse,
    edit_clear,
    NULL
};

#define INITERROR return NULL

PyObject *PyInit__edit(void)
#else
#define INITERROR return

void init_edit(void)
#endif
{
#if PY_MAJOR_VERSION >= 3
    PyObject *module = PyModule_Create(&moduledef);
#else
    PyObject *module = Py_InitModule("_edit", edit_methods);
#endif

    if (module == NULL)
        INITERROR;
    struct module_state *st = GETSTATE(module);

    st->error = PyErr_NewException("_edit.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        INITERROR;
    }

#if PY_MAJOR_VERSION >= 3
    return module;
#endif
}
