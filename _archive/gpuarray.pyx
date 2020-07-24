# -->
# -->
# -->
# -->
# -->
# -->

cimport numpy as np
import numpy as np

import sys

from cpython cimport Py_INCREF, PyNumber_Index
from cpython.object cimport Py_EQ, Py_NE

def api_version():
    """api_version()
    """
    # (library version, module version)
    return (GPUARRAY_API_VERSION, 0)

def abi_version():
    """abi_version()
    """
    major_version = GPUARRAY_ABI_VERSION / 1000
    minor_version = GPUARRAY_ABI_VERSION % 1000
    return (major_version, minor_version)

np.import_array()

# to export the numeric value
SIZE = GA_SIZE
SSIZE = GA_SSIZE

# Numpy API steals dtype references and this breaks cython
cdef object PyArray_Empty(int a, np.npy_intp *b, np.dtype c, int d):
    Py_INCREF(c)
    return _PyArray_Empty(a, b, c, d)

cdef bytes _s(s):
    if isinstance(s, unicode):
        return (<unicode>s).encode('ascii')
    if isinstance(s, bytes):
        return s
    raise TypeError("Expected a string")

cdef size_t countis(l, object val):
    cdef size_t count
    cdef size_t i
    count = 0
    for i in range(len(l)):
        if l[i] is val:
            count += 1
    return count

def cl_wrap_ctx(size_t ptr):
    """
    cl_wrap_ctx(ptr)   ||XX"""
# -->
# -->
# -->
# -->
    cdef gpucontext *(*cl_make_ctx)(void *, int)
    cdef GpuContext res
    cl_make_ctx = <gpucontext *(*)(void *, int)>gpuarray_get_extension("cl_make_ctx")
    if cl_make_ctx == NULL:
        raise RuntimeError, "cl_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
    res.ctx = cl_make_ctx(<void *>ptr, 0)
    if res.ctx == NULL:
        raise RuntimeError, "cl_make_ctx call failed"
    return res

def cuda_wrap_ctx(size_t ptr, bint own):
    """
    cuda_wrap_ctx(ptr) ||XX"""
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
    cdef int flags
    cdef GpuContext res
    cuda_make_ctx = <gpucontext *(*)(void *, int)>gpuarray_get_extension("cuda_make_ctx")
    if cuda_make_ctx == NULL:
        raise RuntimeError, "cuda_make_ctx extension is absent"
    res = GpuContext.__new__(GpuContext)
    flags = 0
    if not own:
        flags |= GPUARRAY_CUDA_CTX_NOFREE
    res.ctx = cuda_make_ctx(<void *>ptr, flags)
    if res.ctx == NULL:
        raise RuntimeError, "cuda_make_ctx call failed"
    return res

import numpy

cdef dict NP_TO_TYPE = {
    np.dtype('bool'): GA_BOOL,
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
    np.dtype('float16'): GA_HALF,
}

cdef dict TYPE_TO_NP = dict((v, k) for k, v in NP_TO_TYPE.iteritems())

def register_dtype(np.dtype dtype, cname):
    """
    register_dtype(dtype, cname)   ||XX"""
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
    cdef char *tmp

    t = <gpuarray_type *>malloc(sizeof(gpuarray_type))
    if t == NULL:
        raise MemoryError("Can't allocate new type")
    tmp = <char *>malloc(len(cname)+1)
    if tmp == NULL:
        free(t)
        raise MemoryError
    memcpy(tmp, <char *>cname, len(cname)+1)
    t.size = dtype.itemsize
    t.align = dtype.alignment
    t.cluda_name = tmp
    typecode = gpuarray_register_type(t, NULL)
    if typecode == -1:
        free(tmp)
        free(t)
        raise RuntimeError("Could not register type")
    NP_TO_TYPE[dtype] = typecode
    TYPE_TO_NP[typecode] = dtype

cdef np.dtype typecode_to_dtype(int typecode):
    res = TYPE_TO_NP.get(typecode, None)
    if res is not None:
        return res
    else:
        raise NotImplementedError("TODO")

# This function takes a flexible dtype as accepted by the functions of
# this module and ensures it becomes a numpy dtype.
cdef np.dtype dtype_to_npdtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, int):
        return typecode_to_dtype(dtype)
    try:
        return np.dtype(dtype)
    except TypeError:
        pass
    if isinstance(dtype, np.dtype):
        return dtype
    raise ValueError("data type not understood", dtype)

# This is a stupid wrapper to avoid the extra argument introduced by having
# dtype_to_typecode declared 'cpdef'.
cdef int get_typecode(dtype) except -1:
    return dtype_to_typecode(dtype)

cpdef int dtype_to_typecode(dtype) except -1:
    """
    dtype_to_typecode(dtype)
# -->
# -->
# -->
# -->
# -->
# -->
# -->
    """
    if isinstance(dtype, int):
        return dtype
    try:
        dtype = np.dtype(dtype)
    except TypeError:
        pass
    if isinstance(dtype, np.dtype):
        res = NP_TO_TYPE.get(dtype, None)
        if res is not None:
            return res
    raise ValueError("don't know how to convert to dtype: %s"%(dtype,))

def dtype_to_ctype(dtype):
    """
    dtype_to_ctype(dtype)  ||XX"""
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# 
#    """
    cdef int typecode = dtype_to_typecode(dtype)
    cdef const gpuarray_type *t = gpuarray_get_type(typecode)
    cdef bytes res
    if t.cluda_name == NULL:
        raise ValueError, "No mapping for %s"%(dtype,)
    res = t.cluda_name
    return res.decode('ascii')

cdef ga_order to_ga_order(ord) except <ga_order>-2:
    if ord == "C" or ord == "c":
        return GA_C_ORDER
    elif ord == "A" or ord == "a" or ord is None:
        return GA_ANY_ORDER
    elif ord == "F" or ord == "f":
        return GA_F_ORDER
    else:
        raise ValueError("Valid orders are: 'A' (any), 'C' (C), 'F' (Fortran)")

cdef int strides_ok(GpuArray a, strides):
    # Check that the passed in strides will not go outside of the
    # memory of the array.  It is assumed that the strides are of the
    # proper length.
    cdef ssize_t max_axis_offset
    cdef size_t lower = a.ga.offset
    cdef size_t upper = a.ga.offset
    cdef size_t itemsize = gpuarray_get_elsize(a.ga.typecode)
    cdef size_t size
    cdef unsigned int i

    gpudata_property(a.ga.data, GA_BUFFER_PROP_SIZE, &size)

    for i in range(a.ga.nd):
        if a.ga.dimensions[i] == 0:
            return 1

        max_axis_offset = <ssize_t>(strides[i])<ssize_t>(a.ga.dimensions[i] - 1)
        if max_axis_offset > 0:
            if upper + max_axis_offset > size:
                return 0
            upper += max_axis_offset
        else:
            if lower < <size_t>(-max_axis_offset):
                return 0
            lower += max_axis_offset
    return (upper + itemsize) <= size

# -->
# -->
# -->
# -->
# -->
# -->
# -->
    # pass

cdef type get_exc(int errcode):
    if errcode == GA_VALUE_ERROR:
        return ValueError
    if errcode == GA_DEVSUP_ERROR:
        return UnsupportedException
    else:
        return GpuArrayException

cdef bint py_CHKFLAGS(GpuArray a, int flags):
    return GpuArray_CHKFLAGS(&a.ga, flags)

cdef bint py_ISONESEGMENT(GpuArray a):
    return GpuArray_ISONESEGMENT(&a.ga)

cdef void array_fix_flags(GpuArray a):
    GpuArray_fix_flags(&a.ga)

cdef int array_empty(GpuArray a, gpucontext *ctx,
        int typecode, unsigned int nd, const size_t *dims,
        ga_order ord) except -1:
    cdef int err
    err = GpuArray_empty(&a.ga, ctx, typecode, nd, dims, ord)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int array_fromdata(GpuArray a,
           gpudata *data, size_t offset, int typecode,
           unsigned int nd, const size_t *dims,
           const ssize_t *strides, int writeable) except -1:
    cdef int err
    err = GpuArray_fromdata(&a.ga, data, offset, typecode, nd, dims,
  strides, writeable)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(gpudata_context(data), err)

cdef int array_view(GpuArray v, GpuArray a) except -1:
    cdef int err
    err = GpuArray_view(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_sync(GpuArray a) except -1:
    cdef int err
    with nogil:
        err = GpuArray_sync(&a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_index(GpuArray r, GpuArray a, const ssize_t *starts,
        const ssize_t *stops, const ssize_t *steps) except -1:
    cdef int err
    err = GpuArray_index(&r.ga, &a.ga, starts, stops, steps)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_take1(GpuArray r, GpuArray a, GpuArray i,
        int check_err) except -1:
    cdef int err
    err = GpuArray_take1(&r.ga, &a.ga, &i.ga, check_err)
    if err != GA_NO_ERROR:
        if err == GA_VALUE_ERROR:
            raise IndexError, GpuArray_error(&r.ga, err)
        raise get_exc(err), GpuArray_error(&r.ga, err)

cdef int array_setarray(GpuArray v, GpuArray a) except -1:
    cdef int err
    err = GpuArray_setarray(&v.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&v.ga, err)

cdef int array_reshape(GpuArray res, GpuArray a, unsigned int nd,
          const size_t *newdims, ga_order ord,
          bint nocopy) except -1:
    cdef int err
    err = GpuArray_reshape(&res.ga, &a.ga, nd, newdims, ord, nocopy)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_transpose(GpuArray res, GpuArray a,
            const unsigned int *new_axes) except -1:
    cdef int err
    err = GpuArray_transpose(&res.ga, &a.ga, new_axes)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_clear(GpuArray a) except -1:
    GpuArray_clear(&a.ga)

cdef bint array_share(GpuArray a, GpuArray b):
    return GpuArray_share(&a.ga, &b.ga)

cdef gpucontext *array_context(GpuArray a) except NULL:
    cdef gpucontext *res
    res = GpuArray_context(&a.ga)
    if res is NULL:
        raise GpuArrayException, "Invalid array or destroyed context"
    return res

cdef int array_move(GpuArray a, GpuArray src) except -1:
    cdef int err
    err = GpuArray_move(&a.ga, &src.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_write(GpuArray a, void *src, size_t sz) except -1:
    cdef int err
    with nogil:
        err = GpuArray_write(&a.ga, src, sz)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_read(void *dst, size_t sz, GpuArray src) except -1:
    cdef int err
    with nogil:
        err = GpuArray_read(dst, sz, &src.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&src.ga, err)

cdef int array_memset(GpuArray a, int data) except -1:
    cdef int err
    err = GpuArray_memset(&a.ga, data)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_copy(GpuArray res, GpuArray a, ga_order order) except -1:
    cdef int err
    err = GpuArray_copy(&res.ga, &a.ga, order)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_transfer(GpuArray res, GpuArray a) except -1:
    cdef int err
    with nogil:
        err = GpuArray_transfer(&res.ga, &a.ga)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_split(_GpuArray **res, GpuArray a, size_t n, size_t *p,
        unsigned int axis) except -1:
    cdef int err
    err = GpuArray_split(res, &a.ga, n, p, axis)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(&a.ga, err)

cdef int array_concatenate(GpuArray r, const _GpuArray **a, size_t n,
 unsigned int axis, int restype) except -1:
    cdef int err
    err = GpuArray_concatenate(&r.ga, a, n, axis, restype)
    if err != GA_NO_ERROR:
        raise get_exc(err), GpuArray_error(a[0], err)

cdef const char *kernel_error(GpuKernel k, int err) except NULL:
    return gpucontext_error(gpukernel_context(k.k.k), err)

cdef int kernel_init(GpuKernel k, gpucontext *ctx,
        unsigned int count, const char **strs, const size_t *len,
        const char *name, unsigned int argcount, const int *types,
        int flags) except -1:
    cdef int err
    cdef char *err_str = NULL
    err = GpuKernel_init(&k.k, ctx, count, strs, len, name, argcount,
            types, flags, &err_str)
    if err != GA_NO_ERROR:
        if err_str != NULL:
            try:
                py_err_str = err_str.decode('UTF-8')
            finally:
                free(err_str)
            raise get_exc(err), py_err_str
        raise get_exc(err), gpucontext_error(ctx, err)

cdef int kernel_clear(GpuKernel k) except -1:
    GpuKernel_clear(&k.k)

cdef gpucontext *kernel_context(GpuKernel k) except NULL:
    cdef gpucontext *res
    res = GpuKernel_context(&k.k)
    if res is NULL:
        raise GpuArrayException, "Invalid kernel or destroyed context"
    return res

cdef int kernel_sched(GpuKernel k, size_t n, size_t *gs, size_t *ls) except -1:
    cdef int err
    err = GpuKernel_sched(&k.k, n, gs, ls)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_call(GpuKernel k, unsigned int n, const size_t *gs,
        const size_t *ls, size_t shared, void **args) except -1:
    cdef int err
    err = GpuKernel_call(&k.k, n, gs, ls, shared, args)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef int kernel_property(GpuKernel k, int prop_id, void *res) except -1:
    cdef int err
    err = gpukernel_property(k.k.k, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), kernel_error(k, err)

cdef GpuContext pygpu_default_context():
    return default_context

cdef GpuContext default_context = None

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1:
    cdef int err
    err = gpucontext_property(c.ctx, prop_id, res)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(c.ctx, err)

def set_default_context(GpuContext ctx):
    """
    set_default_context(ctx) ||XX"""
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
    # """
    global default_context
    default_context = ctx

def get_default_context():
    """
    get_default_context()
    # -->
    Return the currently defined default context (or `None`).
    """
    return default_context

cdef GpuContext ensure_context(GpuContext c):
    global default_context
    if c is None:
        if default_context is None:
            raise TypeError, "No context specified."
        return default_context
    return c

cdef bint pygpu_GpuArray_Check(object o):
    return isinstance(o, GpuArray)

def count_platforms(kind):
    """
    count_platforms(kind) ||XX """
#-->
#-->
#-->
    cdef unsigned int platcount
    cdef int err
    err = gpu_get_platform_count(_s(kind), &platcount)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(NULL, err)
    return platcount

def count_devices(kind, unsigned int platform):
    """
    count_devices(kind, platform)  ||XX """
 */
#-->
#-->
    cdef unsigned int devcount
    cdef int err
    err = gpu_get_device_count(_s(kind), platform, &devcount)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(NULL, err)
    return devcount

cdef GpuContext pygpu_init(dev, gpucontext_props *p):
    cdef int err
    cdef GpuContext res

    if dev.startswith('cuda'):
        kind = b"cuda"
        if dev[4:] == '':
            devnum = -1
        else:
            devnum = int(dev[4:])
        gpucontext_props_cuda_dev(p, devnum)
    elif dev.startswith('opencl'):
        kind = b"opencl"
        devspec = dev[6:].split(':')
        if len(devspec) < 2:
            raise ValueError("OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev)
        if not devspec[0].isdigit() or not devspec[1].isdigit():
            raise ValueError("OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev)
            gpucontext_props_opencl_dev(p, int(devspec[0]), int(devspec[1]))
        else:
            gpucontext_props_opencl_dev(p, int(devspec[0]), int(devspec[1]))
    else:
        raise ValueError("Unknown device format:" + dev)

    res = GpuContext.__new__(GpuContext)
    res.kind = kind
    err = gpucontext_init(&res.ctx, <char *>res.kind, p)
    if err != GA_NO_ERROR:
        raise get_exc(err), gpucontext_error(NULL, err)
    return res

def init(dev, sched='default', single_stream=False, kernel_cache_path=None,
         max_cache_size=sys.maxsize, initial_cache_size=0):
    """
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
# -->
    """
    cdef gpucontext_props *p = NULL
    cdef int err
    cdef bytes kernel_cache_path_b
    err = gpucontext_props_new(&p)
    if err != GA_NO_ERROR:
        raise MemoryError
    try:
        if sched == 'single':
            err = gpucontext_props_sched(p, GA_CTX_SCHED_SINGLE)
        elif sched == 'multi':
            err = gpucontext_props_sched(p, GA_CTX_SCHED_MULTI)
        elif sched != 'default':
            raise TypeError('unexpected value for parameter sched: %s' % (sched,))
        if err != GA_NO_ERROR:
            raise get_exc(err), gpucontext_error(NULL, err)

        if kernel_cache_path:
            kernel_cache_path_b = _s(kernel_cache_path)
            gpucontext_props_kernel_cache(p, <const char *>kernel_cache_path_b)

        err = gpucontext_props_alloc_cache(p, initial_cache_size,
                max_cache_size)
        if err != GA_NO_ERROR:
            raise get_exc(err), gpucontext_error(NULL, err)
        if single_stream:
            gpucontext_props_set_single_stream(p)
    except:
        gpucontext_props_del(p)
        raise
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
 */
  "pygpu/gpuarray.pyx":590
    return res

def init(dev, sched='default', single_stream=False, kernel_cache_path=None,
         max_cache_size=sys.maxsize, initial_cache_size=0):
    """
 */
"pygpu/gpuarray.pyx":660
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
    "pygpu/gpuarray.pyx":661

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
    zeros(shape, dtype='float64', order='C', context=None, cls=None)
 */
  "pygpu/gpuarray.pyx":660
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":682

    """
    res = empty(shape, dtype=dtype, order=order, context=context, cls=cls)
    array_memset(res, 0)
    return res
 */
  "pygpu/gpuarray.pyx":683
    """
    res = empty(shape, dtype=dtype, order=order, context=context, cls=cls)
    array_memset(res, 0)
    return res

 */
  "pygpu/gpuarray.pyx":684
    res = empty(shape, dtype=dtype, order=order, context=context, cls=cls)
    array_memset(res, 0)
    return res

cdef GpuArray pygpu_zeros(unsigned int nd, const size_t *dims, int typecode,
 */
  "pygpu/gpuarray.pyx":660
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
"pygpu/gpuarray.pyx":686
    return res

cdef GpuArray pygpu_zeros(unsigned int nd, const size_t *dims, int typecode,
ga_order order, GpuContext context, object cls):
    cdef GpuArray res
 */
  "pygpu/gpuarray.pyx":689
ga_order order, GpuContext context, object cls):
    cdef GpuArray res
    res = pygpu_empty(nd, dims, typecode, order, context, cls)
    array_memset(res, 0)
    return res
 */
  "pygpu/gpuarray.pyx":690
    cdef GpuArray res
    res = pygpu_empty(nd, dims, typecode, order, context, cls)
    array_memset(res, 0)
    return res

 */
  "pygpu/gpuarray.pyx":691
    res = pygpu_empty(nd, dims, typecode, order, context, cls)
    array_memset(res, 0)
    return res

cdef GpuArray pygpu_empty(unsigned int nd, const size_t *dims, int typecode,
 */
  "pygpu/gpuarray.pyx":686
    return res

cdef GpuArray pygpu_zeros(unsigned int nd, const size_t *dims, int typecode,
ga_order order, GpuContext context, object cls):
    cdef GpuArray res
 */
"pygpu/gpuarray.pyx":693
    return res

cdef GpuArray pygpu_empty(unsigned int nd, const size_t *dims, int typecode,
ga_order order, GpuContext context, object cls):
    cdef GpuArray res
 */
  "pygpu/gpuarray.pyx":697
    cdef GpuArray res

    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
 */
  "pygpu/gpuarray.pyx":699
    context = ensure_context(context)

    res = new_GpuArray(cls, context, None)
    array_empty(res, context.ctx, typecode, nd, dims, order)
    return res
 */
  "pygpu/gpuarray.pyx":700

    res = new_GpuArray(cls, context, None)
    array_empty(res, context.ctx, typecode, nd, dims, order)
    return res

 */
  "pygpu/gpuarray.pyx":701
    res = new_GpuArray(cls, context, None)
    array_empty(res, context.ctx, typecode, nd, dims, order)
    return res

cdef GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
 */
  "pygpu/gpuarray.pyx":693
    return res

cdef GpuArray pygpu_empty(unsigned int nd, const size_t *dims, int typecode,
ga_order order, GpuContext context, object cls):
    cdef GpuArray res
 */
"pygpu/gpuarray.pyx":703
    return res

cdef GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
      unsigned int nd, const size_t *dims,
      const ssize_t *strides, GpuContext context,
 */
  "pygpu/gpuarray.pyx":709
    cdef GpuArray res

    res = new_GpuArray(cls, context, base)
    array_fromdata(res, buf, offset, typecode, nd, dims,
      strides, writable)
 */
  "pygpu/gpuarray.pyx":710

    res = new_GpuArray(cls, context, base)
    array_fromdata(res, buf, offset, typecode, nd, dims,
      strides, writable)
    return res
 */
  "pygpu/gpuarray.pyx":712
    array_fromdata(res, buf, offset, typecode, nd, dims,
      strides, writable)
    return res


 */
  "pygpu/gpuarray.pyx":703
    return res

cdef GpuArray pygpu_fromgpudata(gpudata *buf, size_t offset, int typecode,
      unsigned int nd, const size_t *dims,
      const ssize_t *strides, GpuContext context,
 */
"pygpu/gpuarray.pyx":715


cdef GpuArray pygpu_copy(GpuArray a, ga_order ord):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, None)
 */
  "pygpu/gpuarray.pyx":717
cdef GpuArray pygpu_copy(GpuArray a, ga_order ord):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, None)
    array_copy(res, a, ord)
    return res
 */
  "pygpu/gpuarray.pyx":718
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, None)
    array_copy(res, a, ord)
    return res

 */
  "pygpu/gpuarray.pyx":719
    res = new_GpuArray(type(a), a.context, None)
    array_copy(res, a, ord)
    return res

cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
 */
  "pygpu/gpuarray.pyx":715


cdef GpuArray pygpu_copy(GpuArray a, ga_order ord):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, None)
 */
"pygpu/gpuarray.pyx":721
    return res

cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
    array_move(a, src)
    return 0
 */
  "pygpu/gpuarray.pyx":722

cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
    array_move(a, src)
    return 0

 */
  "pygpu/gpuarray.pyx":723
cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
    array_move(a, src)
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
 */
  "pygpu/gpuarray.pyx":721
    return res

cdef int pygpu_move(GpuArray a, GpuArray src) except -1:
    array_move(a, src)
    return 0
 */
"pygpu/gpuarray.pyx":725
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
    "pygpu/gpuarray.pyx":726

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
    empty(shape, dtype='float64', order='C', context=None, cls=None)
 */
  "pygpu/gpuarray.pyx":725
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":750
    cdef unsigned int nd

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
      "pygpu/gpuarray.pyx":751

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
 */
      "pygpu/gpuarray.pyx":750
    cdef unsigned int nd

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
    "pygpu/gpuarray.pyx":752
    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]
 */
      "pygpu/gpuarray.pyx":753
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]

 */
      "pygpu/gpuarray.pyx":754
    except TypeError:
        nd = 1
        shape = [shape]

    cdims = <size_t *>calloc(nd, sizeof(size_t))
 */
    "pygpu/gpuarray.pyx":750
    cdef unsigned int nd

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
  "pygpu/gpuarray.pyx":756
        shape = [shape]

    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
 */
  "pygpu/gpuarray.pyx":757

    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
 */
    "pygpu/gpuarray.pyx":758
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
 */
    "pygpu/gpuarray.pyx":757

    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
 */
  "pygpu/gpuarray.pyx":759
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
 */
    "pygpu/gpuarray.pyx":760
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 */
      "pygpu/gpuarray.pyx":761
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 dtype_to_typecode(dtype), to_ga_order(order),
 */
      "pygpu/gpuarray.pyx":760
        raise MemoryError, "could not allocate cdims"
    try:
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 */
    "pygpu/gpuarray.pyx":762
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 dtype_to_typecode(dtype), to_ga_order(order),
 context, cls)
 */
    "pygpu/gpuarray.pyx":763
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 dtype_to_typecode(dtype), to_ga_order(order),
 context, cls)
    finally:
 */
    "pygpu/gpuarray.pyx":762
        for i, d in enumerate(shape):
            cdims[i] = d
        return pygpu_empty(nd, cdims,
 dtype_to_typecode(dtype), to_ga_order(order),
 context, cls)
 */
  "pygpu/gpuarray.pyx":766
 context, cls)
    finally:
        free(cdims)

def asarray(a, dtype=None, order='A', GpuContext context=None):
 */
  "pygpu/gpuarray.pyx":725
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
"pygpu/gpuarray.pyx":768
        free(cdims)

def asarray(a, dtype=None, order='A', GpuContext context=None):
    """
    asarray(a, dtype=None, order='A', context=None)
 */
  "pygpu/gpuarray.pyx":794

    """
    return array(a, dtype=dtype, order=order, copy=False, context=context,
    cls=GpuArray)

 */
  "pygpu/gpuarray.pyx":795
    """
    return array(a, dtype=dtype, order=order, copy=False, context=context,
    cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
 */
  "pygpu/gpuarray.pyx":794

    """
    return array(a, dtype=dtype, order=order, copy=False, context=context,
    cls=GpuArray)

 */
  "pygpu/gpuarray.pyx":768
        free(cdims)

def asarray(a, dtype=None, order='A', GpuContext context=None):
    """
    asarray(a, dtype=None, order='A', context=None)
 */
"pygpu/gpuarray.pyx":797
    cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
    """
    ascontiguousarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":816

    """
    return array(a, order='C', dtype=dtype, ndmin=1, copy=False,
    context=context)

 */
  "pygpu/gpuarray.pyx":817
    """
    return array(a, order='C', dtype=dtype, ndmin=1, copy=False,
    context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
 */
  "pygpu/gpuarray.pyx":816

    """
    return array(a, order='C', dtype=dtype, ndmin=1, copy=False,
    context=context)

 */
  "pygpu/gpuarray.pyx":797
    cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
    """
    ascontiguousarray(a, dtype=None, context=None)
 */
"pygpu/gpuarray.pyx":819
    context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
    """
    asfortranarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":838

    """
    return array(a, order='F', dtype=dtype, ndmin=1, copy=False,
    context=context)

 */
  "pygpu/gpuarray.pyx":839
    """
    return array(a, order='F', dtype=dtype, ndmin=1, copy=False,
    context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
 */
  "pygpu/gpuarray.pyx":838

    """
    return array(a, order='F', dtype=dtype, ndmin=1, copy=False,
    context=context)

 */
  "pygpu/gpuarray.pyx":819
    context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
    """
    asfortranarray(a, dtype=None, context=None)
 */
"pygpu/gpuarray.pyx":841
    context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    """
    may_share_memory(a, b)
 */
  "pygpu/gpuarray.pyx":847
    Returns True if `a` and `b` may share memory, False otherwise.
    """
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
 */
  "pygpu/gpuarray.pyx":841
    context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    """
    may_share_memory(a, b)
 */
"pygpu/gpuarray.pyx":849
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
 */
    "pygpu/gpuarray.pyx":850

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
    from_gpudata(data, offset, dtype, shape, context=None, strides=None, writable=True, base=None, cls=None)
 */
  "pygpu/gpuarray.pyx":849
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":890

    """
    cdef size_t *cdims = NULL
    cdef ssize_t *cstrides = NULL
    cdef unsigned int nd
 */
  "pygpu/gpuarray.pyx":891
    """
    cdef size_t *cdims = NULL
    cdef ssize_t *cstrides = NULL
    cdef unsigned int nd
    cdef size_t size
 */
  "pygpu/gpuarray.pyx":896
    cdef int typecode

    context = ensure_context(context)

    try:
 */
  "pygpu/gpuarray.pyx":898
    context = ensure_context(context)

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
      "pygpu/gpuarray.pyx":899

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
 */
      "pygpu/gpuarray.pyx":898
    context = ensure_context(context)

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
    "pygpu/gpuarray.pyx":900
    try:
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]
 */
      "pygpu/gpuarray.pyx":901
        nd = <unsigned int>len(shape)
    except TypeError:
        nd = 1
        shape = [shape]

 */
      "pygpu/gpuarray.pyx":902
    except TypeError:
        nd = 1
        shape = [shape]

    if strides is not None and len(strides) != nd:
 */
    "pygpu/gpuarray.pyx":898
    context = ensure_context(context)

    try:
        nd = <unsigned int>len(shape)
    except TypeError:
 */
  "pygpu/gpuarray.pyx":904
        shape = [shape]

    if strides is not None and len(strides) != nd:
        raise ValueError, "strides must be the same length as shape"

 */
    "pygpu/gpuarray.pyx":905

    if strides is not None and len(strides) != nd:
        raise ValueError, "strides must be the same length as shape"

    typecode = dtype_to_typecode(dtype)
 */
    "pygpu/gpuarray.pyx":904
        shape = [shape]

    if strides is not None and len(strides) != nd:
        raise ValueError, "strides must be the same length as shape"

 */
  "pygpu/gpuarray.pyx":907
        raise ValueError, "strides must be the same length as shape"

    typecode = dtype_to_typecode(dtype)

    try:
 */
  "pygpu/gpuarray.pyx":909
    typecode = dtype_to_typecode(dtype)

    try:
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
 */
    "pygpu/gpuarray.pyx":910

    try:
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
 */
    "pygpu/gpuarray.pyx":911
    try:
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
 */
    "pygpu/gpuarray.pyx":912
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
 */
      "pygpu/gpuarray.pyx":913
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
            cdims[i] = d
 */
      "pygpu/gpuarray.pyx":912
        cdims = <size_t *>calloc(nd, sizeof(size_t))
        cstrides = <ssize_t *>calloc(nd, sizeof(ssize_t))
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
 */
    "pygpu/gpuarray.pyx":914
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
 */
      "pygpu/gpuarray.pyx":915
            raise MemoryError
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
 */
      "pygpu/gpuarray.pyx":914
        if cdims == NULL or cstrides == NULL:
            raise MemoryError
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
 */
    "pygpu/gpuarray.pyx":916
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
   cstrides[i] = s
 */
      "pygpu/gpuarray.pyx":917
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
   cstrides[i] = s
        else:
 */
        "pygpu/gpuarray.pyx":918
        if strides:
            for i, s in enumerate(strides):
   cstrides[i] = s
        else:
            size = gpuarray_get_elsize(typecode)
 */
        "pygpu/gpuarray.pyx":917
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
   cstrides[i] = s
        else:
 */
      "pygpu/gpuarray.pyx":916
        for i, d in enumerate(shape):
            cdims[i] = d
        if strides:
            for i, s in enumerate(strides):
   cstrides[i] = s
 */
    "pygpu/gpuarray.pyx":920
   cstrides[i] = s
        else:
            size = gpuarray_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
   cstrides[i] = size
 */
      "pygpu/gpuarray.pyx":921
        else:
            size = gpuarray_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
   cstrides[i] = size
   size *= cdims[i]
 */
        "pygpu/gpuarray.pyx":922
            size = gpuarray_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
   cstrides[i] = size
   size *= cdims[i]

 */
        "pygpu/gpuarray.pyx":923
            for i in range(nd-1, -1, -1):
   cstrides[i] = size
   size *= cdims[i]

        return pygpu_fromgpudata(<gpudata *>data, offset, typecode, nd, cdims,
 */
        "pygpu/gpuarray.pyx":921
        else:
            size = gpuarray_get_elsize(typecode)
            for i in range(nd-1, -1, -1):
   cstrides[i] = size
   size *= cdims[i]
 */
    "pygpu/gpuarray.pyx":925
   size *= cdims[i]

        return pygpu_fromgpudata(<gpudata *>data, offset, typecode, nd, cdims,
       cstrides, context, writable, base, cls)
    finally:
 */
    "pygpu/gpuarray.pyx":926

        return pygpu_fromgpudata(<gpudata *>data, offset, typecode, nd, cdims,
       cstrides, context, writable, base, cls)
    finally:
        free(cdims)
 */
    "pygpu/gpuarray.pyx":925
   size *= cdims[i]

        return pygpu_fromgpudata(<gpudata *>data, offset, typecode, nd, cdims,
       cstrides, context, writable, base, cls)
    finally:
 */
  "pygpu/gpuarray.pyx":928
       cstrides, context, writable, base, cls)
    finally:
        free(cdims)
        free(cstrides)

 */
        "pygpu/gpuarray.pyx":929
    finally:
        free(cdims)
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
 */
      "pygpu/gpuarray.pyx":928
       cstrides, context, writable, base, cls)
    finally:
        free(cdims)
        free(cstrides)

 */
      "pygpu/gpuarray.pyx":929
    finally:
        free(cdims)
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
 */
  "pygpu/gpuarray.pyx":849
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
 */
"pygpu/gpuarray.pyx":931
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
 */
    "pygpu/gpuarray.pyx":932

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
    array(obj, dtype='float64', copy=True, order=None, ndmin=0, context=None, cls=None)
 */
  "pygpu/gpuarray.pyx":931
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":966

    """
    return carray(proto, dtype, copy, order, ndmin, context, cls)

cdef carray(proto, dtype, copy, order, unsigned int ndmin,
 */
  "pygpu/gpuarray.pyx":931
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
 */
"pygpu/gpuarray.pyx":968
    return carray(proto, dtype, copy, order, ndmin, context, cls)

cdef carray(proto, dtype, copy, order, unsigned int ndmin,
            GpuContext context, cls):
    cdef GpuArray res
 */
  "pygpu/gpuarray.pyx":975
    cdef np.ndarray a

    if isinstance(proto, GpuArray):
        arg = proto

 */
    "pygpu/gpuarray.pyx":976

    if isinstance(proto, GpuArray):
        arg = proto

        if context is not None and  context.ctx != array_context(arg):
 */
    "pygpu/gpuarray.pyx":978
        arg = proto

        if context is not None and  context.ctx != array_context(arg):
            raise ValueError, "cannot copy an array to a different context"

 */
      "pygpu/gpuarray.pyx":979

        if context is not None and  context.ctx != array_context(arg):
            raise ValueError, "cannot copy an array to a different context"

        if (not copy
 */
      "pygpu/gpuarray.pyx":978
        arg = proto

        if context is not None and  context.ctx != array_context(arg):
            raise ValueError, "cannot copy an array to a different context"

 */
    "pygpu/gpuarray.pyx":981
            raise ValueError, "cannot copy an array to a different context"

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
 */
    "pygpu/gpuarray.pyx":982

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
 */
    "pygpu/gpuarray.pyx":983
        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
 */
    "pygpu/gpuarray.pyx":984
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
 */
    "pygpu/gpuarray.pyx":985
            and (order is None or order == 'A' or
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
   shp = arg.shape
 */
    "pygpu/gpuarray.pyx":981
            raise ValueError, "cannot copy an array to a different context"

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
 */
      "pygpu/gpuarray.pyx":986
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
 */
        "pygpu/gpuarray.pyx":987
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
   shp = idx + shp
 */
        "pygpu/gpuarray.pyx":988
            if arg.ga.nd < ndmin:
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
   shp = idx + shp
   arg = arg.reshape(shp)
 */
        "pygpu/gpuarray.pyx":989
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
   shp = idx + shp
   arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
 */
        "pygpu/gpuarray.pyx":990
   idx = (1,)*(ndmin-len(shp))
   shp = idx + shp
   arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
   arg = arg.view(cls)
 */
        "pygpu/gpuarray.pyx":986
    (order == 'C' and py_CHKFLAGS(arg, GA_C_CONTIGUOUS)) or
    (order == 'F' and py_CHKFLAGS(arg, GA_F_CONTIGUOUS)))):
            if arg.ga.nd < ndmin:
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
 */
      "pygpu/gpuarray.pyx":991
   shp = idx + shp
   arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
   arg = arg.view(cls)
            return arg
 */
        "pygpu/gpuarray.pyx":992
   arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
   arg = arg.view(cls)
            return arg
        shp = arg.shape
 */
        "pygpu/gpuarray.pyx":991
   shp = idx + shp
   arg = arg.reshape(shp)
            if not (cls is None or arg.__class__ is cls):
   arg = arg.view(cls)
            return arg
 */
      "pygpu/gpuarray.pyx":993
            if not (cls is None or arg.__class__ is cls):
   arg = arg.view(cls)
            return arg
        shp = arg.shape
        if len(shp) < ndmin:
 */
      "pygpu/gpuarray.pyx":981
            raise ValueError, "cannot copy an array to a different context"

        if (not copy
            and (dtype is None or dtype_to_typecode(dtype) == arg.typecode)
            and (order is None or order == 'A' or
 */
    "pygpu/gpuarray.pyx":994
   arg = arg.view(cls)
            return arg
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
 */
    "pygpu/gpuarray.pyx":995
            return arg
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
 */
      "pygpu/gpuarray.pyx":996
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
 */
      "pygpu/gpuarray.pyx":997
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
 */
      "pygpu/gpuarray.pyx":995
            return arg
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
 */
    "pygpu/gpuarray.pyx":998
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
 */
      "pygpu/gpuarray.pyx":999
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
 */
        "pygpu/gpuarray.pyx":1000
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
 */
        "pygpu/gpuarray.pyx":999
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
 */
      "pygpu/gpuarray.pyx":1001
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
        if cls is None:
 */
        "pygpu/gpuarray.pyx":1002
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
        if cls is None:
            cls = type(proto)
 */
        "pygpu/gpuarray.pyx":1001
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
        if cls is None:
 */
      "pygpu/gpuarray.pyx":998
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
            if py_CHKFLAGS(arg, GA_C_CONTIGUOUS):
   order = 'C'
 */
    "pygpu/gpuarray.pyx":1003
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
        if cls is None:
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
 */
      "pygpu/gpuarray.pyx":1004
   order = 'F'
        if cls is None:
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
       context=arg.context)
 */
      "pygpu/gpuarray.pyx":1003
            elif py_CHKFLAGS(arg, GA_F_CONTIGUOUS):
   order = 'F'
        if cls is None:
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
 */
    "pygpu/gpuarray.pyx":1005
        if cls is None:
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
       context=arg.context)
        res.base = arg.base
 */
    "pygpu/gpuarray.pyx":1006
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
       context=arg.context)
        res.base = arg.base
        if len(shp) < ndmin:
 */
    "pygpu/gpuarray.pyx":1005
        if cls is None:
            cls = type(proto)
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
       context=arg.context)
        res.base = arg.base
 */
    "pygpu/gpuarray.pyx":1007
        res = empty(shp, dtype=(dtype or arg.dtype), order=order, cls=cls,
       context=arg.context)
        res.base = arg.base
        if len(shp) < ndmin:
            tmp = res[idx]
 */
    "pygpu/gpuarray.pyx":1008
       context=arg.context)
        res.base = arg.base
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
 */
      "pygpu/gpuarray.pyx":1009
        res.base = arg.base
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
            tmp = res
 */
      "pygpu/gpuarray.pyx":1008
       context=arg.context)
        res.base = arg.base
        if len(shp) < ndmin:
            tmp = res[idx]
        else:
 */
    "pygpu/gpuarray.pyx":1011
            tmp = res[idx]
        else:
            tmp = res
        array_move(tmp, arg)
        return res
 */
    "pygpu/gpuarray.pyx":1012
        else:
            tmp = res
        array_move(tmp, arg)
        return res

 */
    "pygpu/gpuarray.pyx":1013
            tmp = res
        array_move(tmp, arg)
        return res

    context = ensure_context(context)
 */
    "pygpu/gpuarray.pyx":975
    cdef np.ndarray a

    if isinstance(proto, GpuArray):
        arg = proto

 */
  "pygpu/gpuarray.pyx":1015
        return res

    context = ensure_context(context)

    # We need a contiguous array for the copy
 */
  "pygpu/gpuarray.pyx":1018

    # We need a contiguous array for the copy
    if order != 'C' and order != 'F':
        order = 'C'

 */
    "pygpu/gpuarray.pyx":1019
    # We need a contiguous array for the copy
    if order != 'C' and order != 'F':
        order = 'C'

    a = numpy.array(proto, dtype=dtype_to_npdtype(dtype), order=order,
 */
    "pygpu/gpuarray.pyx":1018

    # We need a contiguous array for the copy
    if order != 'C' and order != 'F':
        order = 'C'

 */
  "pygpu/gpuarray.pyx":1021
        order = 'C'

    a = numpy.array(proto, dtype=dtype_to_npdtype(dtype), order=order,
       ndmin=ndmin, copy=False)

 */
  "pygpu/gpuarray.pyx":1022

    a = numpy.array(proto, dtype=dtype_to_npdtype(dtype), order=order,
       ndmin=ndmin, copy=False)

    res = pygpu_empty(np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
 */
  "pygpu/gpuarray.pyx":1021
        order = 'C'

    a = numpy.array(proto, dtype=dtype_to_npdtype(dtype), order=order,
       ndmin=ndmin, copy=False)

 */











"pygpu/gpuarray.pyx":1190


cdef class flags(object):
    cdef int fl

 */

"pygpu/gpuarray.pyx":1268
        raise KeyError, "Unknown flag"

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
 */
"pygpu/gpuarray.pyx":1556
    if d is NULL:
        raise GpuArrayException, gpucontext_error(c.ctx, 0)
    return <size_t>d

cdef class GpuArray:
    """
    Device array  |X-X-X-X-X-X-X-X-X-X-X-: """
 */

"pygpu/gpuarray.pyx":1928
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
        cdef unsigned int i

 */
"pygpu/gpuarray.pyx":1939
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
            else:
 */
"pygpu/gpuarray.pyx":1949
            key = (key,)
        else:
            if all(isinstance(k, list) for k in key):
   raise NotImplementedError, "fancy indexing not supported"

 */
"pygpu/gpuarray.pyx":1975

        # Remove the None entries for indexing
        getitem_idcs = tuple(k for k in key if k is not None)

        # For less than 1 index, fill up with slice(None) to the right.
 */
"pygpu/gpuarray.pyx":2071
            free(steps)

    def __setitem__(self, idx, v):
        cdef GpuArray tmp, gv

        if isinstance(idx, list):
            if any(isinstance(i, slice) or i is Ellipsis for i in idx):
   self.__setitem__(tuple(idx), v)
            else:
   raise NotImplementedError, "fancy indexing not supported"
 */

"pygpu/gpuarray.pyx":2084
            idx = (idx,)
        else:
            if all(isinstance(i, list) for i in idx):
   raise NotImplementedError, "fancy indexing not supported"

 */
"pygpu/gpuarray.pyx":2093

        # Remove None entries, they should be ignored (as in Numpy)
        idx = tuple(i for i in idx if i is not None)
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
 */

"pygpu/gpuarray.pyx":2269


cdef class GpuKernel:
    """
    GpuKernel(source, name, types, context=None, have_double=False, have_small=False, have_complex=False, have_half=False, cuda=False, opencl=False) XX||"""
 */





  
  


  "pygpu/gpuarray.pyx":1025

    res = pygpu_empty(np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
         dtype_to_typecode(a.dtype), to_ga_order(order),
         context, cls)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
 */
  "pygpu/gpuarray.pyx":1024
       ndmin=ndmin, copy=False)

    res = pygpu_empty(np.PyArray_NDIM(a), <size_t *>np.PyArray_DIMS(a),
         dtype_to_typecode(a.dtype), to_ga_order(order),
         context, cls)
 */
  "pygpu/gpuarray.pyx":1027
         dtype_to_typecode(a.dtype), to_ga_order(order),
         context, cls)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
    return res

 */
  "pygpu/gpuarray.pyx":1028
         context, cls)
    array_write(res, np.PyArray_DATA(a), np.PyArray_NBYTES(a))
    return res

cdef void (*cuda_enter)(gpucontext *)
 */
  "pygpu/gpuarray.pyx":968
    return carray(proto, dtype, copy, order, ndmin, context, cls)

cdef carray(proto, dtype, copy, order, unsigned int ndmin,
            GpuContext context, cls):
    cdef GpuArray res
 */
"pygpu/gpuarray.pyx":1060

    """
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)
 */
  "pygpu/gpuarray.pyx":1061
    """
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)

 */
    "pygpu/gpuarray.pyx":1062
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)

    def __reduce__(self):
 */
    "pygpu/gpuarray.pyx":1061
    """
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)

 */
  "pygpu/gpuarray.pyx":1060

    """
    def __dealloc__(self):
        if self.ctx != NULL:
            gpucontext_deref(self.ctx)
 */
"pygpu/gpuarray.pyx":1064
            gpucontext_deref(self.ctx)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuContext object"

 */
  "pygpu/gpuarray.pyx":1065

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuContext object"

    def __init__(self):
 */
  "pygpu/gpuarray.pyx":1064
            gpucontext_deref(self.ctx)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuContext object"

 */
"pygpu/gpuarray.pyx":1067
        raise RuntimeError, "Cannot pickle GpuContext object"

    def __init__(self):
        if type(self) is GpuContext:
            raise RuntimeError, "Called raw GpuContext.__init__"
 */
  "pygpu/gpuarray.pyx":1068

    def __init__(self):
        if type(self) is GpuContext:
            raise RuntimeError, "Called raw GpuContext.__init__"

 */
    "pygpu/gpuarray.pyx":1069
    def __init__(self):
        if type(self) is GpuContext:
            raise RuntimeError, "Called raw GpuContext.__init__"

    def __enter__(self):
 */
    "pygpu/gpuarray.pyx":1068

    def __init__(self):
        if type(self) is GpuContext:
            raise RuntimeError, "Called raw GpuContext.__init__"

 */
  "pygpu/gpuarray.pyx":1067
        raise RuntimeError, "Cannot pickle GpuContext object"

    def __init__(self):
        if type(self) is GpuContext:
            raise RuntimeError, "Called raw GpuContext.__init__"
 */
"pygpu/gpuarray.pyx":1071
            raise RuntimeError, "Called raw GpuContext.__init__"

    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
 */
  "pygpu/gpuarray.pyx":1072

    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
 */
    "pygpu/gpuarray.pyx":1073
    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
 */
    "pygpu/gpuarray.pyx":1072

    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
 */
  "pygpu/gpuarray.pyx":1074
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
 */
    "pygpu/gpuarray.pyx":1075
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
 */
    "pygpu/gpuarray.pyx":1074
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
 */
  "pygpu/gpuarray.pyx":1076
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
 */
    "pygpu/gpuarray.pyx":1077
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
        return self
 */
    "pygpu/gpuarray.pyx":1076
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
 */
  "pygpu/gpuarray.pyx":1078
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
        return self

 */
  "pygpu/gpuarray.pyx":1079
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
        return self

    def __exit__(self, t, v, tb):
 */
  "pygpu/gpuarray.pyx":1071
            raise RuntimeError, "Called raw GpuContext.__init__"

    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
 */
"pygpu/gpuarray.pyx":1081
        return self

    def __exit__(self, t, v, tb):
        cuda_exit(self.ctx)

 */
  "pygpu/gpuarray.pyx":1082

    def __exit__(self, t, v, tb):
        cuda_exit(self.ctx)

    property ptr:
 */
  "pygpu/gpuarray.pyx":1081
        return self

    def __exit__(self, t, v, tb):
        cuda_exit(self.ctx)

 */
"pygpu/gpuarray.pyx":1086
    property ptr:
        "Raw pointer value for the context object"
        def __get__(self):
            return <size_t>self.ctx

 */
  "pygpu/gpuarray.pyx":1087
        "Raw pointer value for the context object"
        def __get__(self):
            return <size_t>self.ctx

    property devname:
 */
  "pygpu/gpuarray.pyx":1086
    property ptr:
        "Raw pointer value for the context object"
        def __get__(self):
            return <size_t>self.ctx

 */
"pygpu/gpuarray.pyx":1091
    property devname:
        "Device name for this context"
        def __get__(self):
            cdef char tmp[256]

 */
  "pygpu/gpuarray.pyx":1094
            cdef char tmp[256]

            ctx_property(self, GA_CTX_PROP_DEVNAME, tmp)
            return tmp.decode('ascii')

 */
  "pygpu/gpuarray.pyx":1095

            ctx_property(self, GA_CTX_PROP_DEVNAME, tmp)
            return tmp.decode('ascii')

    property unique_id:
 */
  "pygpu/gpuarray.pyx":1091
    property devname:
        "Device name for this context"
        def __get__(self):
            cdef char tmp[256]

 */
"pygpu/gpuarray.pyx":1099
    property unique_id:
        "Device PCI Bus ID for this context"
        def __get__(self):
            cdef char tmp[16]

 */
  "pygpu/gpuarray.pyx":1102
            cdef char tmp[16]

            ctx_property(self, GA_CTX_PROP_UNIQUE_ID, tmp)
            return tmp.decode('ascii')

 */
  "pygpu/gpuarray.pyx":1103

            ctx_property(self, GA_CTX_PROP_UNIQUE_ID, tmp)
            return tmp.decode('ascii')

    property lmemsize:
 */
  "pygpu/gpuarray.pyx":1099
    property unique_id:
        "Device PCI Bus ID for this context"
        def __get__(self):
            cdef char tmp[16]

 */
"pygpu/gpuarray.pyx":1107
    property lmemsize:
        "Size of the local (shared) memory, in bytes, for this context"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LMEMSIZE, &res)
 */
  "pygpu/gpuarray.pyx":1109
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LMEMSIZE, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1110
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LMEMSIZE, &res)
            return res

    property numprocs:
 */
  "pygpu/gpuarray.pyx":1107
    property lmemsize:
        "Size of the local (shared) memory, in bytes, for this context"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LMEMSIZE, &res)
 */
"pygpu/gpuarray.pyx":1114
    property numprocs:
        "Number of compute units for this context"
        def __get__(self):
            cdef unsigned int res
            ctx_property(self, GA_CTX_PROP_NUMPROCS, &res)
 */
  "pygpu/gpuarray.pyx":1116
        def __get__(self):
            cdef unsigned int res
            ctx_property(self, GA_CTX_PROP_NUMPROCS, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1117
            cdef unsigned int res
            ctx_property(self, GA_CTX_PROP_NUMPROCS, &res)
            return res

    property bin_id:
 */
  "pygpu/gpuarray.pyx":1114
    property numprocs:
        "Number of compute units for this context"
        def __get__(self):
            cdef unsigned int res
            ctx_property(self, GA_CTX_PROP_NUMPROCS, &res)
 */
"pygpu/gpuarray.pyx":1121
    property bin_id:
        "Binary compatibility id"
        def __get__(self):
            cdef const char *res
            ctx_property(self, GA_CTX_PROP_BIN_ID, &res)
 */
  "pygpu/gpuarray.pyx":1123
        def __get__(self):
            cdef const char *res
            ctx_property(self, GA_CTX_PROP_BIN_ID, &res)
            return res;

 */
  "pygpu/gpuarray.pyx":1124
            cdef const char *res
            ctx_property(self, GA_CTX_PROP_BIN_ID, &res)
            return res;

    property total_gmem:
 */
  "pygpu/gpuarray.pyx":1121
    property bin_id:
        "Binary compatibility id"
        def __get__(self):
            cdef const char *res
            ctx_property(self, GA_CTX_PROP_BIN_ID, &res)
 */
"pygpu/gpuarray.pyx":1128
    property total_gmem:
        "Total size of global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_TOTAL_GMEM, &res)
 */
  "pygpu/gpuarray.pyx":1130
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_TOTAL_GMEM, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1131
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_TOTAL_GMEM, &res)
            return res

    property free_gmem:
 */
  "pygpu/gpuarray.pyx":1128
    property total_gmem:
        "Total size of global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_TOTAL_GMEM, &res)
 */
"pygpu/gpuarray.pyx":1135
    property free_gmem:
        "Size of free global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_FREE_GMEM, &res)
 */
  "pygpu/gpuarray.pyx":1137
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_FREE_GMEM, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1138
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_FREE_GMEM, &res)
            return res

    property maxlsize0:
 */
  "pygpu/gpuarray.pyx":1135
    property free_gmem:
        "Size of free global memory on the device"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_FREE_GMEM, &res)
 */
"pygpu/gpuarray.pyx":1142
    property maxlsize0:
        "Maximum local size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE0, &res)
 */
  "pygpu/gpuarray.pyx":1144
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE0, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1145
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE0, &res)
            return res

    property maxlsize1:
 */
  "pygpu/gpuarray.pyx":1142
    property maxlsize0:
        "Maximum local size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE0, &res)
 */
"pygpu/gpuarray.pyx":1149
    property maxlsize1:
        "Maximum local size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE1, &res)
 */
  "pygpu/gpuarray.pyx":1151
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE1, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1152
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE1, &res)
            return res

    property maxlsize2:
 */
  "pygpu/gpuarray.pyx":1149
    property maxlsize1:
        "Maximum local size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE1, &res)
 */
"pygpu/gpuarray.pyx":1156
    property maxlsize2:
        "Maximum local size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE2, &res)
 */
  "pygpu/gpuarray.pyx":1158
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE2, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1159
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE2, &res)
            return res

    property maxgsize0:
 */
  "pygpu/gpuarray.pyx":1156
    property maxlsize2:
        "Maximum local size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXLSIZE2, &res)
 */
"pygpu/gpuarray.pyx":1163
    property maxgsize0:
        "Maximum global size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE0, &res)
 */
  "pygpu/gpuarray.pyx":1165
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE0, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1166
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE0, &res)
            return res

    property maxgsize1:
 */
  "pygpu/gpuarray.pyx":1163
    property maxgsize0:
        "Maximum global size for dimension 0"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE0, &res)
 */
"pygpu/gpuarray.pyx":1170
    property maxgsize1:
        "Maximum global size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE1, &res)
 */
  "pygpu/gpuarray.pyx":1172
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE1, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1173
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE1, &res)
            return res

    property maxgsize2:
 */
  "pygpu/gpuarray.pyx":1170
    property maxgsize1:
        "Maximum global size for dimension 1"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE1, &res)
 */
"pygpu/gpuarray.pyx":1177
    property maxgsize2:
        "Maximum global size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE2, &res)
 */
  "pygpu/gpuarray.pyx":1179
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE2, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1180
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE2, &res)
            return res

    property largest_memblock:
 */
  "pygpu/gpuarray.pyx":1177
    property maxgsize2:
        "Maximum global size for dimension 2"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_MAXGSIZE2, &res)
 */
"pygpu/gpuarray.pyx":1184
    property largest_memblock:
        "Size of the largest memory block you can allocate"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LARGEST_MEMBLOCK, &res)
 */
  "pygpu/gpuarray.pyx":1186
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LARGEST_MEMBLOCK, &res)
            return res

 */
  "pygpu/gpuarray.pyx":1187
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LARGEST_MEMBLOCK, &res)
            return res


 */
  "pygpu/gpuarray.pyx":1184
    property largest_memblock:
        "Size of the largest memory block you can allocate"
        def __get__(self):
            cdef size_t res
            ctx_property(self, GA_CTX_PROP_LARGEST_MEMBLOCK, &res)
 */
"pygpu/gpuarray.pyx":1193
    cdef int fl

    def __cinit__(self, fl):
        self.fl = fl

 */
  "pygpu/gpuarray.pyx":1193
    cdef int fl

    def __cinit__(self, fl):
        self.fl = fl

    def __reduce__(self):
        return (flags, (self.fl,))

    def __getitem__(self, idx):
 */
"pygpu/gpuarray.pyx":1199
        return (flags, (self.fl,))

    def __getitem__(self, idx):
        cdef const char *key
        cdef size_t n
 */
    "pygpu/gpuarray.pyx":1204
        cdef char c

        if isinstance(idx, unicode):
            idx = idx.encode('UTF-8')
        if isinstance(idx, bytes):
            key = idx
            n = len(idx)
        else:
            raise KeyError, "Unknown flag"
        if n == 1:
            c = key[0]
            if c == 'C':
   return self.c_contiguous
            elif c == 'F':
   return self.f_contiguous
            elif c == 'W':
   return self.writeable
            elif c == 'B':
   return self.behaved
            elif c == 'O':
   return self.owndata
            elif c == 'A':
   return self.aligned
            elif c == 'U':
   return self.updateifcopy
        elif n == 2:
            if strncmp(key, "CA", n) == 0:
   return self.carray
            if strncmp(key, "FA", n) == 0:
   return self.farray
        elif n == 3:
            if strncmp(key, "FNC", n) == 0:
   return self.fnc
        elif n == 4:
            if strncmp(key, "FORC", n) == 0:
   return self.forc
        elif n == 6:
            if strncmp(key, "CARRAY", n) == 0:
   return self.carray
            if strncmp(key, "FARRAY", n) == 0:
   return self.farray
        elif n == 7:
            if strncmp(key, "FORTRAN", n) == 0:
   return self.fortran
            if strncmp(key, "BEHAVED", n) == 0:
   return self.behaved
            if strncmp(key, "OWNDATA", n) == 0:
   return self.owndata
            if strncmp(key, "ALIGNED", n) == 0:
   return self.aligned
        elif n == 9:
            if strncmp(key, "WRITEABLE", n) == 0:
   return self.writeable
        elif n == 10:
            if strncmp(key, "CONTIGUOUS", n) == 0:
   return self.c_contiguous
        elif n == 12:
            if strncmp(key, "UPDATEIFCOPY", n) == 0:
   return self.updateifcopy
            if strncmp(key, "C_CONTIGUOUS", n) == 0:
   return self.c_contiguous
            if strncmp(key, "F_CONTIGUOUS", n) == 0:
   return self.f_contiguous

        raise KeyError, "Unknown flag"

    def __repr__(self):
 */
  "pygpu/gpuarray.pyx":1199
        return (flags, (self.fl,))

    def __getitem__(self, idx):
        cdef const char *key
        cdef size_t n
 */
"pygpu/gpuarray.pyx":1268
        raise KeyError, "Unknown flag"

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
 */
"pygpu/gpuarray.pyx":1269

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
 */
  "pygpu/gpuarray.pyx":1270
    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
            "updateifcopy"])
 */
    "pygpu/gpuarray.pyx":1269

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
 */
    "pygpu/gpuarray.pyx":1270
    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
            "updateifcopy"])
 */
  "pygpu/gpuarray.pyx":1269

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
 */
"pygpu/gpuarray.pyx":1268
        raise KeyError, "Unknown flag"

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
 */
  "pygpu/gpuarray.pyx":1269

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
 */
  "pygpu/gpuarray.pyx":1268
        raise KeyError, "Unknown flag"

    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
 */
"pygpu/gpuarray.pyx":1274
            "updateifcopy"])

    def __richcmp__(self, other, int op):
        cdef flags a
        cdef flags b
 */
  "pygpu/gpuarray.pyx":1277
        cdef flags a
        cdef flags b
        if not isinstance(self, flags) or not isinstance(other, flags):
            return NotImplemented
        a = self
 */
    "pygpu/gpuarray.pyx":1278
        cdef flags b
        if not isinstance(self, flags) or not isinstance(other, flags):
            return NotImplemented
        a = self
        b = other
 */
    "pygpu/gpuarray.pyx":1277
        cdef flags a
        cdef flags b
        if not isinstance(self, flags) or not isinstance(other, flags):
            return NotImplemented
        a = self
 */
  "pygpu/gpuarray.pyx":1279
        if not isinstance(self, flags) or not isinstance(other, flags):
            return NotImplemented
        a = self
        b = other
        if op == Py_EQ:
 */
  "pygpu/gpuarray.pyx":1280
            return NotImplemented
        a = self
        b = other
        if op == Py_EQ:
            return a.fl == b.fl
 */
  "pygpu/gpuarray.pyx":1281
        a = self
        b = other
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
 */
    "pygpu/gpuarray.pyx":1282
        b = other
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
            return a.fl != b.fl
 */
    "pygpu/gpuarray.pyx":1281
        a = self
        b = other
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
 */
  "pygpu/gpuarray.pyx":1283
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
            return a.fl != b.fl
        raise TypeError, "undefined comparison for flag object"
 */
    "pygpu/gpuarray.pyx":1284
            return a.fl == b.fl
        elif op == Py_NE:
            return a.fl != b.fl
        raise TypeError, "undefined comparison for flag object"

 */
    "pygpu/gpuarray.pyx":1283
        if op == Py_EQ:
            return a.fl == b.fl
        elif op == Py_NE:
            return a.fl != b.fl
        raise TypeError, "undefined comparison for flag object"
 */
  "pygpu/gpuarray.pyx":1285
        elif op == Py_NE:
            return a.fl != b.fl
        raise TypeError, "undefined comparison for flag object"

    property c_contiguous:
 */
  "pygpu/gpuarray.pyx":1274
            "updateifcopy"])

    def __richcmp__(self, other, int op):
        cdef flags a
        cdef flags b
 */
"pygpu/gpuarray.pyx":1288

    property c_contiguous:
        def __get__(self):
            return bool(self.fl & GA_C_CONTIGUOUS)

 */
  "pygpu/gpuarray.pyx":1289
    property c_contiguous:
        def __get__(self):
            return bool(self.fl & GA_C_CONTIGUOUS)

    property contiguous:
 */
  "pygpu/gpuarray.pyx":1288

    property c_contiguous:
        def __get__(self):
            return bool(self.fl & GA_C_CONTIGUOUS)

 */
"pygpu/gpuarray.pyx":1292

    property contiguous:
        def __get__(self):
            return self.c_contiguous

 */
  "pygpu/gpuarray.pyx":1293
    property contiguous:
        def __get__(self):
            return self.c_contiguous

    property f_contiguous:
 */
  "pygpu/gpuarray.pyx":1292

    property contiguous:
        def __get__(self):
            return self.c_contiguous

 */
"pygpu/gpuarray.pyx":1296

    property f_contiguous:
        def __get__(self):
            return bool(self.fl & GA_F_CONTIGUOUS)

 */
  "pygpu/gpuarray.pyx":1297
    property f_contiguous:
        def __get__(self):
            return bool(self.fl & GA_F_CONTIGUOUS)

    property fortran:
 */
  "pygpu/gpuarray.pyx":1296

    property f_contiguous:
        def __get__(self):
            return bool(self.fl & GA_F_CONTIGUOUS)

 */
"pygpu/gpuarray.pyx":1300

    property fortran:
        def __get__(self):
            return self.f_contiguous

 */
  "pygpu/gpuarray.pyx":1301
    property fortran:
        def __get__(self):
            return self.f_contiguous

    property updateifcopy:
 */
  "pygpu/gpuarray.pyx":1300

    property fortran:
        def __get__(self):
            return self.f_contiguous

 */
"pygpu/gpuarray.pyx":1305
    property updateifcopy:
        # Not supported.
        def __get__(self):
            return False

 */
  "pygpu/gpuarray.pyx":1306
        # Not supported.
        def __get__(self):
            return False

    property owndata:
 */
  "pygpu/gpuarray.pyx":1305
    property updateifcopy:
        # Not supported.
        def __get__(self):
            return False

 */
"pygpu/gpuarray.pyx":1310
    property owndata:
        # There is no equivalent for GpuArrays and it is always "True".
        def __get__(self):
            return True

 */
  "pygpu/gpuarray.pyx":1311
        # There is no equivalent for GpuArrays and it is always "True".
        def __get__(self):
            return True

    property aligned:
 */
  "pygpu/gpuarray.pyx":1310
    property owndata:
        # There is no equivalent for GpuArrays and it is always "True".
        def __get__(self):
            return True

 */
"pygpu/gpuarray.pyx":1314

    property aligned:
        def __get__(self):
            return bool(self.fl & GA_ALIGNED)

 */
  "pygpu/gpuarray.pyx":1315
    property aligned:
        def __get__(self):
            return bool(self.fl & GA_ALIGNED)

    property writeable:
 */
  "pygpu/gpuarray.pyx":1314

    property aligned:
        def __get__(self):
            return bool(self.fl & GA_ALIGNED)

 */
"pygpu/gpuarray.pyx":1318

    property writeable:
        def __get__(self):
            return bool(self.fl & GA_WRITEABLE)

 */
  "pygpu/gpuarray.pyx":1319
    property writeable:
        def __get__(self):
            return bool(self.fl & GA_WRITEABLE)

    property behaved:
 */
  "pygpu/gpuarray.pyx":1318

    property writeable:
        def __get__(self):
            return bool(self.fl & GA_WRITEABLE)

 */
"pygpu/gpuarray.pyx":1322

    property behaved:
        def __get__(self):
            return (self.fl & GA_BEHAVED) == GA_BEHAVED

 */
  "pygpu/gpuarray.pyx":1323
    property behaved:
        def __get__(self):
            return (self.fl & GA_BEHAVED) == GA_BEHAVED

    property carray:
 */
  "pygpu/gpuarray.pyx":1322

    property behaved:
        def __get__(self):
            return (self.fl & GA_BEHAVED) == GA_BEHAVED

 */
"pygpu/gpuarray.pyx":1326

    property carray:
        def __get__(self):
            return (self.fl & GA_CARRAY) == GA_CARRAY

 */
  "pygpu/gpuarray.pyx":1327
    property carray:
        def __get__(self):
            return (self.fl & GA_CARRAY) == GA_CARRAY

    # Yes these are really defined like that according to numpy sources.
 */
  "pygpu/gpuarray.pyx":1326

    property carray:
        def __get__(self):
            return (self.fl & GA_CARRAY) == GA_CARRAY

 */
"pygpu/gpuarray.pyx":1332
    # I don't know why.
    property forc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS or
       (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)
 */
  "pygpu/gpuarray.pyx":1333
    property forc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS or
       (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

 */
  "pygpu/gpuarray.pyx":1334
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS or
       (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

    property fnc:
 */
  "pygpu/gpuarray.pyx":1332
    # I don't know why.
    property forc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS or
       (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)
 */
"pygpu/gpuarray.pyx":1337

    property fnc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS and
       not (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)
 */
  "pygpu/gpuarray.pyx":1338
    property fnc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS and
       not (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

 */
  "pygpu/gpuarray.pyx":1339
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS and
       not (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)

    property farray:
 */
  "pygpu/gpuarray.pyx":1337

    property fnc:
        def __get__(self):
            return ((self.fl & GA_F_CONTIGUOUS) == GA_F_CONTIGUOUS and
       not (self.fl & GA_C_CONTIGUOUS) == GA_C_CONTIGUOUS)
 */
"pygpu/gpuarray.pyx":1342

    property farray:
        def __get__(self):
            return ((self.fl & GA_FARRAY) != 0 and
       not ((self.fl & GA_C_CONTIGUOUS) != 0))
 */
  "pygpu/gpuarray.pyx":1343
    property farray:
        def __get__(self):
            return ((self.fl & GA_FARRAY) != 0 and
       not ((self.fl & GA_C_CONTIGUOUS) != 0))

 */
  "pygpu/gpuarray.pyx":1344
        def __get__(self):
            return ((self.fl & GA_FARRAY) != 0 and
       not ((self.fl & GA_C_CONTIGUOUS) != 0))

    property num:
 */
  "pygpu/gpuarray.pyx":1342

    property farray:
        def __get__(self):
            return ((self.fl & GA_FARRAY) != 0 and
       not ((self.fl & GA_C_CONTIGUOUS) != 0))
 */
"pygpu/gpuarray.pyx":1347

    property num:
        def __get__(self):
            return self.fl

 */
  "pygpu/gpuarray.pyx":1348
    property num:
        def __get__(self):
            return self.fl

cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
 */
  "pygpu/gpuarray.pyx":1347

    property num:
        def __get__(self):
            return self.fl

 */
"pygpu/gpuarray.pyx":1350
            return self.fl

cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
    cdef GpuArray res
    if ctx is None:
 */
  "pygpu/gpuarray.pyx":1352
cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
    cdef GpuArray res
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
 */
    "pygpu/gpuarray.pyx":1353
    cdef GpuArray res
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
 */
    "pygpu/gpuarray.pyx":1352
cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
    cdef GpuArray res
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
 */
  "pygpu/gpuarray.pyx":1354
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
 */
    "pygpu/gpuarray.pyx":1355
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
        res = GpuArray.__new__(cls)
 */
    "pygpu/gpuarray.pyx":1354
    if ctx is None:
        raise RuntimeError, "ctx is None in new_GpuArray"
    if cls is None or cls is GpuArray:
        res = GpuArray.__new__(GpuArray)
    else:
 */
  "pygpu/gpuarray.pyx":1357
        res = GpuArray.__new__(GpuArray)
    else:
        res = GpuArray.__new__(cls)
    res.base = base
    res.context = ctx
 */
  "pygpu/gpuarray.pyx":1358
    else:
        res = GpuArray.__new__(cls)
    res.base = base
    res.context = ctx
    return res
 */
  "pygpu/gpuarray.pyx":1359
        res = GpuArray.__new__(cls)
    res.base = base
    res.context = ctx
    return res

 */
  "pygpu/gpuarray.pyx":1360
    res.base = base
    res.context = ctx
    return res

cdef GpuArray pygpu_view(GpuArray a, object cls):
 */
  "pygpu/gpuarray.pyx":1350
            return self.fl

cdef GpuArray new_GpuArray(object cls, GpuContext ctx, object base):
    cdef GpuArray res
    if ctx is None:
 */
"pygpu/gpuarray.pyx":1362
    return res

cdef GpuArray pygpu_view(GpuArray a, object cls):
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
 */
  "pygpu/gpuarray.pyx":1363

cdef GpuArray pygpu_view(GpuArray a, object cls):
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
    return res
 */
  "pygpu/gpuarray.pyx":1364
cdef GpuArray pygpu_view(GpuArray a, object cls):
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
    return res

 */
  "pygpu/gpuarray.pyx":1365
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
    return res

cdef int pygpu_sync(GpuArray a) except -1:
 */
  "pygpu/gpuarray.pyx":1362
    return res

cdef GpuArray pygpu_view(GpuArray a, object cls):
    cdef GpuArray res = new_GpuArray(cls, a.context, a.base)
    array_view(res, a)
 */
"pygpu/gpuarray.pyx":1367
    return res

cdef int pygpu_sync(GpuArray a) except -1:
    array_sync(a)
    return 0
 */
  "pygpu/gpuarray.pyx":1368

cdef int pygpu_sync(GpuArray a) except -1:
    array_sync(a)
    return 0

 */
  "pygpu/gpuarray.pyx":1369
cdef int pygpu_sync(GpuArray a) except -1:
    array_sync(a)
    return 0

cdef GpuArray pygpu_empty_like(GpuArray a, ga_order ord, int typecode):
 */
  "pygpu/gpuarray.pyx":1367
    return res

cdef int pygpu_sync(GpuArray a) except -1:
    array_sync(a)
    return 0
 */
"pygpu/gpuarray.pyx":1371
    return 0

cdef GpuArray pygpu_empty_like(GpuArray a, ga_order ord, int typecode):
    cdef GpuArray res

 */
  "pygpu/gpuarray.pyx":1374
    cdef GpuArray res

    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
 */
    "pygpu/gpuarray.pyx":1375

    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
            ord = GA_F_ORDER
 */
    "pygpu/gpuarray.pyx":1376
    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
            ord = GA_F_ORDER
        else:
 */
    "pygpu/gpuarray.pyx":1375

    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
            ord = GA_F_ORDER
 */
      "pygpu/gpuarray.pyx":1377
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
            ord = GA_F_ORDER
        else:
            ord = GA_C_ORDER
 */
      "pygpu/gpuarray.pyx":1375

    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
            ord = GA_F_ORDER
 */
    "pygpu/gpuarray.pyx":1379
            ord = GA_F_ORDER
        else:
            ord = GA_C_ORDER

    if typecode == -1:
 */
    "pygpu/gpuarray.pyx":1374
    cdef GpuArray res

    if ord == GA_ANY_ORDER:
        if (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
   not py_CHKFLAGS(a, GA_C_CONTIGUOUS)):
 */
  "pygpu/gpuarray.pyx":1381
            ord = GA_C_ORDER

    if typecode == -1:
        typecode = a.ga.typecode

 */
    "pygpu/gpuarray.pyx":1382

    if typecode == -1:
        typecode = a.ga.typecode

    res = new_GpuArray(type(a), a.context, None)
 */
    "pygpu/gpuarray.pyx":1381
            ord = GA_C_ORDER

    if typecode == -1:
        typecode = a.ga.typecode

 */
  "pygpu/gpuarray.pyx":1384
        typecode = a.ga.typecode

    res = new_GpuArray(type(a), a.context, None)
    array_empty(res, a.context.ctx, typecode,
   a.ga.nd, a.ga.dimensions, ord)
 */
  "pygpu/gpuarray.pyx":1385

    res = new_GpuArray(type(a), a.context, None)
    array_empty(res, a.context.ctx, typecode,
   a.ga.nd, a.ga.dimensions, ord)
    return res
 */
  "pygpu/gpuarray.pyx":1387
    array_empty(res, a.context.ctx, typecode,
   a.ga.nd, a.ga.dimensions, ord)
    return res

cdef np.ndarray pygpu_as_ndarray(GpuArray a):
 */
  "pygpu/gpuarray.pyx":1371
    return 0

cdef GpuArray pygpu_empty_like(GpuArray a, ga_order ord, int typecode):
    cdef GpuArray res

 */
"pygpu/gpuarray.pyx":1389
    return res

cdef np.ndarray pygpu_as_ndarray(GpuArray a):
    return _pygpu_as_ndarray(a, None)

 */
  "pygpu/gpuarray.pyx":1390

cdef np.ndarray pygpu_as_ndarray(GpuArray a):
    return _pygpu_as_ndarray(a, None)

cdef np.ndarray _pygpu_as_ndarray(GpuArray a, np.dtype ldtype):
 */
  "pygpu/gpuarray.pyx":1389
    return res

cdef np.ndarray pygpu_as_ndarray(GpuArray a):
    return _pygpu_as_ndarray(a, None)

 */
"pygpu/gpuarray.pyx":1392
    return _pygpu_as_ndarray(a, None)

cdef np.ndarray _pygpu_as_ndarray(GpuArray a, np.dtype ldtype):
    cdef np.ndarray res

 */
  "pygpu/gpuarray.pyx":1395
    cdef np.ndarray res

    if not py_ISONESEGMENT(a):
        a = pygpu_copy(a, GA_ANY_ORDER)

 */
    "pygpu/gpuarray.pyx":1396

    if not py_ISONESEGMENT(a):
        a = pygpu_copy(a, GA_ANY_ORDER)

    if ldtype is None:
 */
    "pygpu/gpuarray.pyx":1395
    cdef np.ndarray res

    if not py_ISONESEGMENT(a):
        a = pygpu_copy(a, GA_ANY_ORDER)

 */
  "pygpu/gpuarray.pyx":1398
        a = pygpu_copy(a, GA_ANY_ORDER)

    if ldtype is None:
        ldtype = a.dtype

 */
    "pygpu/gpuarray.pyx":1399

    if ldtype is None:
        ldtype = a.dtype

    res = PyArray_Empty(a.ga.nd, <np.npy_intp *>a.ga.dimensions,
 */
    "pygpu/gpuarray.pyx":1398
        a = pygpu_copy(a, GA_ANY_ORDER)

    if ldtype is None:
        ldtype = a.dtype

 */
  "pygpu/gpuarray.pyx":1402

    res = PyArray_Empty(a.ga.nd, <np.npy_intp *>a.ga.dimensions,
           ldtype, (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
       not py_CHKFLAGS(a, GA_C_CONTIGUOUS)))

 */
  "pygpu/gpuarray.pyx":1403
    res = PyArray_Empty(a.ga.nd, <np.npy_intp *>a.ga.dimensions,
           ldtype, (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
       not py_CHKFLAGS(a, GA_C_CONTIGUOUS)))

    array_read(np.PyArray_DATA(res), np.PyArray_NBYTES(res), a)
 */
  "pygpu/gpuarray.pyx":1401
        ldtype = a.dtype

    res = PyArray_Empty(a.ga.nd, <np.npy_intp *>a.ga.dimensions,
           ldtype, (py_CHKFLAGS(a, GA_F_CONTIGUOUS) and
       not py_CHKFLAGS(a, GA_C_CONTIGUOUS)))
 */
  "pygpu/gpuarray.pyx":1405
       not py_CHKFLAGS(a, GA_C_CONTIGUOUS)))

    array_read(np.PyArray_DATA(res), np.PyArray_NBYTES(res), a)

    return res
 */
  "pygpu/gpuarray.pyx":1407
    array_read(np.PyArray_DATA(res), np.PyArray_NBYTES(res), a)

    return res

cdef GpuArray pygpu_index(GpuArray a, const ssize_t *starts,
 */
  "pygpu/gpuarray.pyx":1392
    return _pygpu_as_ndarray(a, None)

cdef np.ndarray _pygpu_as_ndarray(GpuArray a, np.dtype ldtype):
    cdef np.ndarray res

 */
"pygpu/gpuarray.pyx":1409
    return res

cdef GpuArray pygpu_index(GpuArray a, const ssize_t *starts,
const ssize_t *stops, const ssize_t *steps):
    cdef GpuArray res
 */
  "pygpu/gpuarray.pyx":1412
const ssize_t *stops, const ssize_t *steps):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
 */
  "pygpu/gpuarray.pyx":1413
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
 */
      "pygpu/gpuarray.pyx":1414
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
        raise IndexError, "index out of bounds"
 */
      "pygpu/gpuarray.pyx":1413
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
 */
    "pygpu/gpuarray.pyx":1415
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
        raise IndexError, "index out of bounds"
    return res
 */
      "pygpu/gpuarray.pyx":1416
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
        raise IndexError, "index out of bounds"
    return res

 */
    "pygpu/gpuarray.pyx":1413
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    try:
        array_index(res, a, starts, stops, steps)
    except ValueError, e:
 */
  "pygpu/gpuarray.pyx":1417
    except ValueError, e:
        raise IndexError, "index out of bounds"
    return res

cdef GpuArray pygpu_reshape(GpuArray a, unsigned int nd, const size_t *newdims,
 */
  "pygpu/gpuarray.pyx":1409
    return res

cdef GpuArray pygpu_index(GpuArray a, const ssize_t *starts,
const ssize_t *stops, const ssize_t *steps):
    cdef GpuArray res
 */
"pygpu/gpuarray.pyx":1419
    return res

cdef GpuArray pygpu_reshape(GpuArray a, unsigned int nd, const size_t *newdims,
  ga_order ord, bint nocopy, int compute_axis):
    cdef GpuArray res
 */
  "pygpu/gpuarray.pyx":1422
  ga_order ord, bint nocopy, int compute_axis):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
 */
  "pygpu/gpuarray.pyx":1423
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
 */
    "pygpu/gpuarray.pyx":1424
    res = new_GpuArray(type(a), a.context, a.base)
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
    cdef unsigned int caxis = <unsigned int>compute_axis
 */
    "pygpu/gpuarray.pyx":1425
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
 */
    "pygpu/gpuarray.pyx":1423
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    if compute_axis < 0:
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
 */
  "pygpu/gpuarray.pyx":1426
        array_reshape(res, a, nd, newdims, ord, nocopy)
        return res
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
        raise ValueError("compute_axis is out of bounds")
 */
  "pygpu/gpuarray.pyx":1427
        return res
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
        raise ValueError("compute_axis is out of bounds")

 */
    "pygpu/gpuarray.pyx":1428
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
        raise ValueError("compute_axis is out of bounds")

    cdef size_t *cdims
 */
    "pygpu/gpuarray.pyx":1427
        return res
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
        raise ValueError("compute_axis is out of bounds")

 */
  "pygpu/gpuarray.pyx":1431

    cdef size_t *cdims
    cdef size_t tot = 1
    cdef unsigned int i
    for i in range(nd):
 */
  "pygpu/gpuarray.pyx":1433
    cdef size_t tot = 1
    cdef unsigned int i
    for i in range(nd):
        if i != caxis:
            tot *= newdims[i]
 */
    "pygpu/gpuarray.pyx":1434
    cdef unsigned int i
    for i in range(nd):
        if i != caxis:
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
 */
      "pygpu/gpuarray.pyx":1435
    for i in range(nd):
        if i != caxis:
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
 */
      "pygpu/gpuarray.pyx":1434
    cdef unsigned int i
    for i in range(nd):
        if i != caxis:
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
 */
  "pygpu/gpuarray.pyx":1436
        if i != caxis:
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"
 */
  "pygpu/gpuarray.pyx":1437
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"

 */
    "pygpu/gpuarray.pyx":1438
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"

    cdef size_t d
 */
    "pygpu/gpuarray.pyx":1437
            tot *= newdims[i]
    cdims = <size_t *>calloc(nd, sizeof(size_t))
    if cdims == NULL:
        raise MemoryError, "could not allocate cdims"

 */
  "pygpu/gpuarray.pyx":1441

    cdef size_t d
    try:
        for i in range(nd):
            d = newdims[i]
 */
    "pygpu/gpuarray.pyx":1442
    cdef size_t d
    try:
        for i in range(nd):
            d = newdims[i]
            if i == caxis:
 */
      "pygpu/gpuarray.pyx":1443
    try:
        for i in range(nd):
            d = newdims[i]
            if i == caxis:
   d = a.size // tot
 */
      "pygpu/gpuarray.pyx":1444
        for i in range(nd):
            d = newdims[i]
            if i == caxis:
   d = a.size // tot

 */
        "pygpu/gpuarray.pyx":1445
            d = newdims[i]
            if i == caxis:
   d = a.size // tot

   if dtot != a.size:
 */
        "pygpu/gpuarray.pyx":1447
   d = a.size // tot

   if dtot != a.size:
       raise GpuArrayException, "..."
            cdims[i] = d
 */
          "pygpu/gpuarray.pyx":1448

   if dtot != a.size:
       raise GpuArrayException, "..."
            cdims[i] = d

 */
          "pygpu/gpuarray.pyx":1447
   d = a.size // tot

   if dtot != a.size:
       raise GpuArrayException, "..."
            cdims[i] = d
 */
        "pygpu/gpuarray.pyx":1444
        for i in range(nd):
            d = newdims[i]
            if i == caxis:
   d = a.size // tot

 */
      "pygpu/gpuarray.pyx":1449
   if dtot != a.size:
       raise GpuArrayException, "..."
            cdims[i] = d

        array_reshape(res, a, nd, cdims, ord, nocopy)
 */
    "pygpu/gpuarray.pyx":1451
            cdims[i] = d

        array_reshape(res, a, nd, cdims, ord, nocopy)
        return res
    finally:
 */
    "pygpu/gpuarray.pyx":1452

        array_reshape(res, a, nd, cdims, ord, nocopy)
        return res
    finally:
        free(cdims)
 */
  "pygpu/gpuarray.pyx":1454
        return res
    finally:
        free(cdims)


 */
  "pygpu/gpuarray.pyx":1419
    return res

cdef GpuArray pygpu_reshape(GpuArray a, unsigned int nd, const size_t *newdims,
  ga_order ord, bint nocopy, int compute_axis):
    cdef GpuArray res
 */
"pygpu/gpuarray.pyx":1457


cdef GpuArray pygpu_transpose(GpuArray a, const unsigned int *newaxes):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
 */
  "pygpu/gpuarray.pyx":1459
cdef GpuArray pygpu_transpose(GpuArray a, const unsigned int *newaxes):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    array_transpose(res, a, newaxes)
    return res
 */
  "pygpu/gpuarray.pyx":1460
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
    array_transpose(res, a, newaxes)
    return res

 */
  "pygpu/gpuarray.pyx":1461
    res = new_GpuArray(type(a), a.context, a.base)
    array_transpose(res, a, newaxes)
    return res

cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
 */
  "pygpu/gpuarray.pyx":1457


cdef GpuArray pygpu_transpose(GpuArray a, const unsigned int *newaxes):
    cdef GpuArray res
    res = new_GpuArray(type(a), a.context, a.base)
 */
"pygpu/gpuarray.pyx":1463
    return res

cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
    array_transfer(res, a)
    return 0
 */
  "pygpu/gpuarray.pyx":1464

cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
    array_transfer(res, a)
    return 0

 */
  "pygpu/gpuarray.pyx":1465
cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
    array_transfer(res, a)
    return 0

def _split(GpuArray a, ind, unsigned int axis):
 */
  "pygpu/gpuarray.pyx":1463
    return res

cdef int pygpu_transfer(GpuArray res, GpuArray a) except -1:
    array_transfer(res, a)
    return 0
 */
"pygpu/gpuarray.pyx":1467
    return 0

def _split(GpuArray a, ind, unsigned int axis):
    """
    _split(a, ind, axis)
 */
  "pygpu/gpuarray.pyx":1471
    _split(a, ind, axis)
    """
    cdef list r = [None](len(ind) + 1)
    cdef Py_ssize_t i
    if not axis < a.ga.nd:
 */
  "pygpu/gpuarray.pyx":1473
    cdef list r = [None](len(ind) + 1)
    cdef Py_ssize_t i
    if not axis < a.ga.nd:
        raise ValueError, "split on non-existant axis"
    cdef size_t m = a.ga.dimensions[axis]
 */
    "pygpu/gpuarray.pyx":1474
    cdef Py_ssize_t i
    if not axis < a.ga.nd:
        raise ValueError, "split on non-existant axis"
    cdef size_t m = a.ga.dimensions[axis]
    cdef size_t v
 */
    "pygpu/gpuarray.pyx":1473
    cdef list r = [None](len(ind) + 1)
    cdef Py_ssize_t i
    if not axis < a.ga.nd:
        raise ValueError, "split on non-existant axis"
    cdef size_t m = a.ga.dimensions[axis]
 */
  "pygpu/gpuarray.pyx":1475
    if not axis < a.ga.nd:
        raise ValueError, "split on non-existant axis"
    cdef size_t m = a.ga.dimensions[axis]
    cdef size_t v
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t)len(ind))
 */
  "pygpu/gpuarray.pyx":1477
    cdef size_t m = a.ga.dimensions[axis]
    cdef size_t v
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t)len(ind))
    if p == NULL:
        raise MemoryError()
 */
  "pygpu/gpuarray.pyx":1478
    cdef size_t v
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t)len(ind))
    if p == NULL:
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
 */
    "pygpu/gpuarray.pyx":1479
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t)len(ind))
    if p == NULL:
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
    if rs == NULL:
 */
    "pygpu/gpuarray.pyx":1478
    cdef size_t v
    cdef size_t *p = <size_t *>PyMem_Malloc(sizeof(size_t)len(ind))
    if p == NULL:
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
 */
  "pygpu/gpuarray.pyx":1480
    if p == NULL:
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
    if rs == NULL:
        PyMem_Free(p)
 */
  "pygpu/gpuarray.pyx":1481
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
    if rs == NULL:
        PyMem_Free(p)
        raise MemoryError()
 */
    "pygpu/gpuarray.pyx":1482
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
    if rs == NULL:
        PyMem_Free(p)
        raise MemoryError()
    try:
 */
    "pygpu/gpuarray.pyx":1483
    if rs == NULL:
        PyMem_Free(p)
        raise MemoryError()
    try:
        for i in range(len(r)):
 */
    "pygpu/gpuarray.pyx":1481
        raise MemoryError()
    cdef _GpuArray **rs = <_GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(r))
    if rs == NULL:
        PyMem_Free(p)
        raise MemoryError()
 */
  "pygpu/gpuarray.pyx":1484
        PyMem_Free(p)
        raise MemoryError()
    try:
        for i in range(len(r)):
            r[i] = new_GpuArray(type(a), a.context, a.base)
 */
    "pygpu/gpuarray.pyx":1485
        raise MemoryError()
    try:
        for i in range(len(r)):
            r[i] = new_GpuArray(type(a), a.context, a.base)
            rs[i] = &(<GpuArray>r[i]).ga
 */
      "pygpu/gpuarray.pyx":1486
    try:
        for i in range(len(r)):
            r[i] = new_GpuArray(type(a), a.context, a.base)
            rs[i] = &(<GpuArray>r[i]).ga
        for i in range(len(ind)):
 */
      "pygpu/gpuarray.pyx":1487
        for i in range(len(r)):
            r[i] = new_GpuArray(type(a), a.context, a.base)
            rs[i] = &(<GpuArray>r[i]).ga
        for i in range(len(ind)):
            v = ind[i]
 */
    "pygpu/gpuarray.pyx":1488
            r[i] = new_GpuArray(type(a), a.context, a.base)
            rs[i] = &(<GpuArray>r[i]).ga
        for i in range(len(ind)):
            v = ind[i]
            # cap the values to the end of the array
 */
      "pygpu/gpuarray.pyx":1489
            rs[i] = &(<GpuArray>r[i]).ga
        for i in range(len(ind)):
            v = ind[i]
            # cap the values to the end of the array
            p[i] = v if v < m else m
 */
      "pygpu/gpuarray.pyx":1491
            v = ind[i]
            # cap the values to the end of the array
            p[i] = v if v < m else m
        array_split(rs, a, len(ind), p, axis)
        return r
 */
    "pygpu/gpuarray.pyx":1492
            # cap the values to the end of the array
            p[i] = v if v < m else m
        array_split(rs, a, len(ind), p, axis)
        return r
    finally:
 */
    "pygpu/gpuarray.pyx":1493
            p[i] = v if v < m else m
        array_split(rs, a, len(ind), p, axis)
        return r
    finally:
        PyMem_Free(p)
 */
  "pygpu/gpuarray.pyx":1495
        return r
    finally:
        PyMem_Free(p)
        PyMem_Free(rs)

 */
        "pygpu/gpuarray.pyx":1496
    finally:
        PyMem_Free(p)
        PyMem_Free(rs)

cdef GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
 */
      "pygpu/gpuarray.pyx":1495
        return r
    finally:
        PyMem_Free(p)
        PyMem_Free(rs)

 */
      "pygpu/gpuarray.pyx":1496
    finally:
        PyMem_Free(p)
        PyMem_Free(rs)

cdef GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
 */
  "pygpu/gpuarray.pyx":1467
    return 0

def _split(GpuArray a, ind, unsigned int axis):
    """
    _split(a, ind, axis)
 */
"pygpu/gpuarray.pyx":1498
        PyMem_Free(rs)

cdef GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
      unsigned int axis, int restype,
      object cls, GpuContext context):
 */
  "pygpu/gpuarray.pyx":1501
      unsigned int axis, int restype,
      object cls, GpuContext context):
    cdef res = new_GpuArray(cls, context, None)
    array_concatenate(res, a, n, axis, restype)
    return res
 */
  "pygpu/gpuarray.pyx":1502
      object cls, GpuContext context):
    cdef res = new_GpuArray(cls, context, None)
    array_concatenate(res, a, n, axis, restype)
    return res

 */
  "pygpu/gpuarray.pyx":1503
    cdef res = new_GpuArray(cls, context, None)
    array_concatenate(res, a, n, axis, restype)
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
 */
  "pygpu/gpuarray.pyx":1498
        PyMem_Free(rs)

cdef GpuArray pygpu_concatenate(const _GpuArray **a, size_t n,
      unsigned int axis, int restype,
      object cls, GpuContext context):
 */
"pygpu/gpuarray.pyx":1505
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
    GpuContext context):
    """
 */
  "pygpu/gpuarray.pyx":1511
    """
    cdef Py_ssize_t i
    context = ensure_context(context)
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(al))
    if als == NULL:
 */
  "pygpu/gpuarray.pyx":1512
    cdef Py_ssize_t i
    context = ensure_context(context)
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(al))
    if als == NULL:
        raise MemoryError()
 */
  "pygpu/gpuarray.pyx":1513
    context = ensure_context(context)
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(al))
    if als == NULL:
        raise MemoryError()
    try:
 */
    "pygpu/gpuarray.pyx":1514
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(al))
    if als == NULL:
        raise MemoryError()
    try:
        for i in range(len(al)):
 */
    "pygpu/gpuarray.pyx":1513
    context = ensure_context(context)
    cdef const _GpuArray **als = <const _GpuArray **>PyMem_Malloc(sizeof(_GpuArray *)len(al))
    if als == NULL:
        raise MemoryError()
    try:
 */
  "pygpu/gpuarray.pyx":1515
    if als == NULL:
        raise MemoryError()
    try:
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
 */
    "pygpu/gpuarray.pyx":1516
        raise MemoryError()
    try:
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
   raise TypeError, "expected GpuArrays to concatenate"
 */
      "pygpu/gpuarray.pyx":1517
    try:
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
   raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
 */
        "pygpu/gpuarray.pyx":1518
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
   raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
        return pygpu_concatenate(als, len(al), axis, restype, cls, context)
 */
        "pygpu/gpuarray.pyx":1517
    try:
        for i in range(len(al)):
            if not isinstance(al[i], GpuArray):
   raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
 */
      "pygpu/gpuarray.pyx":1519
            if not isinstance(al[i], GpuArray):
   raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
        return pygpu_concatenate(als, len(al), axis, restype, cls, context)
    finally:
 */
    "pygpu/gpuarray.pyx":1520
   raise TypeError, "expected GpuArrays to concatenate"
            als[i] = &(<GpuArray>al[i]).ga
        return pygpu_concatenate(als, len(al), axis, restype, cls, context)
    finally:
        PyMem_Free(als)
 */
  "pygpu/gpuarray.pyx":1522
        return pygpu_concatenate(als, len(al), axis, restype, cls, context)
    finally:
        PyMem_Free(als)

cdef int (*cuda_get_ipc_handle)(gpudata *, GpuArrayIpcMemHandle *)
 */
  "pygpu/gpuarray.pyx":1505
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
    GpuContext context):
    """
 */
"pygpu/gpuarray.pyx":1530
cuda_open_ipc_handle = <gpudata *(*)(gpucontext *, GpuArrayIpcMemHandle *, size_t)>gpuarray_get_extension("cuda_open_ipc_handle")

def open_ipc_handle(GpuContext c, bytes hpy, size_t l):
    """
    open_ipc_handle(c, hpy, l)
 */
  "pygpu/gpuarray.pyx":1550
    cdef gpudata *d

    b = hpy
    memcpy(&h, b, sizeof(h))

 */
  "pygpu/gpuarray.pyx":1551

    b = hpy
    memcpy(&h, b, sizeof(h))

    d = cuda_open_ipc_handle(c.ctx, &h, l)
 */
  "pygpu/gpuarray.pyx":1553
    memcpy(&h, b, sizeof(h))

    d = cuda_open_ipc_handle(c.ctx, &h, l)
    if d is NULL:
        raise GpuArrayException, gpucontext_error(c.ctx, 0)
 */
  "pygpu/gpuarray.pyx":1554

    d = cuda_open_ipc_handle(c.ctx, &h, l)
    if d is NULL:
        raise GpuArrayException, gpucontext_error(c.ctx, 0)
    return <size_t>d
 */
    "pygpu/gpuarray.pyx":1555
    d = cuda_open_ipc_handle(c.ctx, &h, l)
    if d is NULL:
        raise GpuArrayException, gpucontext_error(c.ctx, 0)
    return <size_t>d

 */
    "pygpu/gpuarray.pyx":1554

    d = cuda_open_ipc_handle(c.ctx, &h, l)
    if d is NULL:
        raise GpuArrayException, gpucontext_error(c.ctx, 0)
    return <size_t>d
 */

  "pygpu/gpuarray.pyx":1530
cuda_open_ipc_handle = <gpudata *(*)(gpucontext *, GpuArrayIpcMemHandle *, size_t)>gpuarray_get_extension("cuda_open_ipc_handle")

def open_ipc_handle(GpuContext c, bytes hpy, size_t l):
    """
    open_ipc_handle(c, hpy, l)
 */
"pygpu/gpuarray.pyx":1577
    interpreter.
    """
    def __dealloc__(self):
        array_clear(self)

 */
  "pygpu/gpuarray.pyx":1578
    """
    def __dealloc__(self):
        array_clear(self)

    def __cinit__(self):
 */
  "pygpu/gpuarray.pyx":1577
    interpreter.
    """
    def __dealloc__(self):
        array_clear(self)

 */
"pygpu/gpuarray.pyx":1580
        array_clear(self)

    def __cinit__(self):
        memset(&self.ga, 0, sizeof(_GpuArray))

 */
  "pygpu/gpuarray.pyx":1581

    def __cinit__(self):
        memset(&self.ga, 0, sizeof(_GpuArray))

    def __init__(self):
 */
  "pygpu/gpuarray.pyx":1580
        array_clear(self)

    def __cinit__(self):
        memset(&self.ga, 0, sizeof(_GpuArray))

 */
"pygpu/gpuarray.pyx":1583
        memset(&self.ga, 0, sizeof(_GpuArray))

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"
 */
  "pygpu/gpuarray.pyx":1584

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"

 */
    "pygpu/gpuarray.pyx":1585
    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"

    def __reduce__(self):
 */
    "pygpu/gpuarray.pyx":1584

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"

 */
  "pygpu/gpuarray.pyx":1583
        memset(&self.ga, 0, sizeof(_GpuArray))

    def __init__(self):
        if type(self) is GpuArray:
            raise RuntimeError, "Called raw GpuArray.__init__"
 */
"pygpu/gpuarray.pyx":1587
            raise RuntimeError, "Called raw GpuArray.__init__"

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuArray object"

 */
  "pygpu/gpuarray.pyx":1588

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuArray object"

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
 */
  "pygpu/gpuarray.pyx":1587
            raise RuntimeError, "Called raw GpuArray.__init__"

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuArray object"

 */
"pygpu/gpuarray.pyx":1590
        raise RuntimeError, "Cannot pickle GpuArray object"

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
           ssize_t *stop, ssize_t *step):
        cdef Py_ssize_t dummy
 */
  "pygpu/gpuarray.pyx":1594
        cdef Py_ssize_t dummy
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
 */
      "pygpu/gpuarray.pyx":1595
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
   k += self.ga.dimensions[i]
 */
      "pygpu/gpuarray.pyx":1596
        try:
            k = PyNumber_Index(key)
            if k < 0:
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
 */
        "pygpu/gpuarray.pyx":1597
            k = PyNumber_Index(key)
            if k < 0:
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
   raise IndexError, "index %d out of bounds" % (i,)
 */
        "pygpu/gpuarray.pyx":1596
        try:
            k = PyNumber_Index(key)
            if k < 0:
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
 */
      "pygpu/gpuarray.pyx":1598
            if k < 0:
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
   raise IndexError, "index %d out of bounds" % (i,)
            start[0] = k
 */
        "pygpu/gpuarray.pyx":1599
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
   raise IndexError, "index %d out of bounds" % (i,)
            start[0] = k
            step[0] = 0
 */
        "pygpu/gpuarray.pyx":1598
            if k < 0:
   k += self.ga.dimensions[i]
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
   raise IndexError, "index %d out of bounds" % (i,)
            start[0] = k
 */
      "pygpu/gpuarray.pyx":1600
            if k < 0 or (<size_t>k) >= self.ga.dimensions[i]:
   raise IndexError, "index %d out of bounds" % (i,)
            start[0] = k
            step[0] = 0
            return
 */
      "pygpu/gpuarray.pyx":1601
   raise IndexError, "index %d out of bounds" % (i,)
            start[0] = k
            step[0] = 0
            return
        except TypeError:
 */
      "pygpu/gpuarray.pyx":1602
            start[0] = k
            step[0] = 0
            return
        except TypeError:
            pass
 */
      "pygpu/gpuarray.pyx":1594
        cdef Py_ssize_t dummy
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
 */
    "pygpu/gpuarray.pyx":1603
            step[0] = 0
            return
        except TypeError:
            pass

 */
    "pygpu/gpuarray.pyx":1594
        cdef Py_ssize_t dummy
        cdef Py_ssize_t k
        try:
            k = PyNumber_Index(key)
            if k < 0:
 */
  "pygpu/gpuarray.pyx":1606
            pass

        if isinstance(key, slice):
            PySlice_GetIndicesEx(key, self.ga.dimensions[i],
       start, stop, step, &dummy)
 */
    "pygpu/gpuarray.pyx":1607

        if isinstance(key, slice):
            PySlice_GetIndicesEx(key, self.ga.dimensions[i],
       start, stop, step, &dummy)
            if stop[0] < start[0] and step[0] > 0:
 */
    "pygpu/gpuarray.pyx":1609
            PySlice_GetIndicesEx(key, self.ga.dimensions[i],
       start, stop, step, &dummy)
            if stop[0] < start[0] and step[0] > 0:
   stop[0] = start[0]
        elif key is Ellipsis:
 */
      "pygpu/gpuarray.pyx":1610
       start, stop, step, &dummy)
            if stop[0] < start[0] and step[0] > 0:
   stop[0] = start[0]
        elif key is Ellipsis:
            start[0] = 0
 */
      "pygpu/gpuarray.pyx":1609
            PySlice_GetIndicesEx(key, self.ga.dimensions[i],
       start, stop, step, &dummy)
            if stop[0] < start[0] and step[0] > 0:
   stop[0] = start[0]
        elif key is Ellipsis:
 */
    "pygpu/gpuarray.pyx":1606
            pass

        if isinstance(key, slice):
            PySlice_GetIndicesEx(key, self.ga.dimensions[i],
       start, stop, step, &dummy)
 */
  "pygpu/gpuarray.pyx":1611
            if stop[0] < start[0] and step[0] > 0:
   stop[0] = start[0]
        elif key is Ellipsis:
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
 */
    "pygpu/gpuarray.pyx":1612
   stop[0] = start[0]
        elif key is Ellipsis:
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
            step[0] = 1
 */
    "pygpu/gpuarray.pyx":1613
        elif key is Ellipsis:
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
            step[0] = 1
        else:
 */
    "pygpu/gpuarray.pyx":1614
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
            step[0] = 1
        else:
            raise IndexError, "cannot index with: %s" % (key,)
 */
    "pygpu/gpuarray.pyx":1611
            if stop[0] < start[0] and step[0] > 0:
   stop[0] = start[0]
        elif key is Ellipsis:
            start[0] = 0
            stop[0] = self.ga.dimensions[i]
 */
  "pygpu/gpuarray.pyx":1616
            step[0] = 1
        else:
            raise IndexError, "cannot index with: %s" % (key,)

    def write(self, np.ndarray src not None):
 */
  "pygpu/gpuarray.pyx":1590
        raise RuntimeError, "Cannot pickle GpuArray object"

    cdef __index_helper(self, key, unsigned int i, ssize_t *start,
           ssize_t *stop, ssize_t *step):
        cdef Py_ssize_t dummy
 */
"pygpu/gpuarray.pyx":1618
            raise IndexError, "cannot index with: %s" % (key,)

    def write(self, np.ndarray src not None):
        """
        write(src)
 */
  "pygpu/gpuarray.pyx":1646

        """
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
 */
    "pygpu/gpuarray.pyx":1647
        """
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
 */
    "pygpu/gpuarray.pyx":1646

        """
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
 */
  "pygpu/gpuarray.pyx":1648
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
 */
    "pygpu/gpuarray.pyx":1649
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
            src = np.asarray(src, order='F')
 */
    "pygpu/gpuarray.pyx":1648
        if not self.flags.behaved:
            raise ValueError, "Destination GpuArray is not well behaved: aligned and writeable"
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
 */
  "pygpu/gpuarray.pyx":1650
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
            src = np.asarray(src, order='F')
        else:
 */
    "pygpu/gpuarray.pyx":1651
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
            src = np.asarray(src, order='F')
        else:
            raise ValueError, "Destination GpuArray is not contiguous"
 */
    "pygpu/gpuarray.pyx":1650
        if self.flags.c_contiguous:
            src = np.asarray(src, order='C')
        elif self.flags.f_contiguous:
            src = np.asarray(src, order='F')
        else:
 */
  "pygpu/gpuarray.pyx":1653
            src = np.asarray(src, order='F')
        else:
            raise ValueError, "Destination GpuArray is not contiguous"
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
 */
  "pygpu/gpuarray.pyx":1654
        else:
            raise ValueError, "Destination GpuArray is not contiguous"
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
 */
    "pygpu/gpuarray.pyx":1655
            raise ValueError, "Destination GpuArray is not contiguous"
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
 */
    "pygpu/gpuarray.pyx":1654
        else:
            raise ValueError, "Destination GpuArray is not contiguous"
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
 */
  "pygpu/gpuarray.pyx":1656
        if self.dtype != src.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
 */
  "pygpu/gpuarray.pyx":1657
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(src)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
 */
  "pygpu/gpuarray.pyx":1659
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
 */
    "pygpu/gpuarray.pyx":1660
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
 */
  "pygpu/gpuarray.pyx":1661
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_write(self, np.PyArray_DATA(src), sz)
 */
    "pygpu/gpuarray.pyx":1662
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_write(self, np.PyArray_DATA(src), sz)

 */
    "pygpu/gpuarray.pyx":1661
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_write(self, np.PyArray_DATA(src), sz)
 */
  "pygpu/gpuarray.pyx":1663
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_write(self, np.PyArray_DATA(src), sz)

    def read(self, np.ndarray dst not None):
 */
  "pygpu/gpuarray.pyx":1618
            raise IndexError, "cannot index with: %s" % (key,)

    def write(self, np.ndarray src not None):
        """
        write(src)
 */
"pygpu/gpuarray.pyx":1665
        array_write(self, np.PyArray_DATA(src), sz)

    def read(self, np.ndarray dst not None):
        """
        read(dst)
 */
  "pygpu/gpuarray.pyx":1693

        """
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
 */
    "pygpu/gpuarray.pyx":1694
        """
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
 */
    "pygpu/gpuarray.pyx":1693

        """
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
 */
  "pygpu/gpuarray.pyx":1695
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
 */
  "pygpu/gpuarray.pyx":1696
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
    dst.flags['F_CONTIGUOUS']))):
 */
  "pygpu/gpuarray.pyx":1697
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
    dst.flags['F_CONTIGUOUS']))):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
 */
  "pygpu/gpuarray.pyx":1698
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
    dst.flags['F_CONTIGUOUS']))):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
 */
  "pygpu/gpuarray.pyx":1695
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
 */
    "pygpu/gpuarray.pyx":1699
   (self.flags.f_contiguous and self.flags.aligned and
    dst.flags['F_CONTIGUOUS']))):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
 */
    "pygpu/gpuarray.pyx":1695
        if not np.PyArray_ISBEHAVED(dst):
            raise ValueError, "Destination Numpy array is not well behaved: aligned and writeable"
        if (not ((self.flags.c_contiguous and self.flags.aligned and
     dst.flags['C_CONTIGUOUS']) or
   (self.flags.f_contiguous and self.flags.aligned and
 */
  "pygpu/gpuarray.pyx":1700
    dst.flags['F_CONTIGUOUS']))):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
 */
    "pygpu/gpuarray.pyx":1701
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
 */
    "pygpu/gpuarray.pyx":1700
    dst.flags['F_CONTIGUOUS']))):
            raise ValueError, "GpuArray and Numpy array do not match in contiguity or GpuArray is not aligned"
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
 */
  "pygpu/gpuarray.pyx":1702
        if self.dtype != dst.dtype:
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
 */
  "pygpu/gpuarray.pyx":1703
            raise ValueError, "GpuArray and Numpy array do not have matching data types"
        cdef size_t npsz = np.PyArray_NBYTES(dst)
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
 */
  "pygpu/gpuarray.pyx":1705
        cdef size_t sz = gpuarray_get_elsize(self.ga.typecode)
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
 */
    "pygpu/gpuarray.pyx":1706
        cdef unsigned i
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
 */
  "pygpu/gpuarray.pyx":1707
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_read(np.PyArray_DATA(dst), sz, self)
 */
    "pygpu/gpuarray.pyx":1708
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_read(np.PyArray_DATA(dst), sz, self)

 */
    "pygpu/gpuarray.pyx":1707
        for i in range(self.ga.nd):
            sz *= self.ga.dimensions[i]
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_read(np.PyArray_DATA(dst), sz, self)
 */
  "pygpu/gpuarray.pyx":1709
        if sz != npsz:
            raise ValueError, "GpuArray and Numpy array do not have the same size in bytes"
        array_read(np.PyArray_DATA(dst), sz, self)

    def get_ipc_handle(self):
 */
  "pygpu/gpuarray.pyx":1665
        array_write(self, np.PyArray_DATA(src), sz)

    def read(self, np.ndarray dst not None):
        """
        read(dst)
 */
"pygpu/gpuarray.pyx":1711
        array_read(np.PyArray_DATA(dst), sz, self)

    def get_ipc_handle(self):
        """
        get_ipc_handle()
 */
  "pygpu/gpuarray.pyx":1717
        cdef GpuArrayIpcMemHandle h
        cdef int err
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
 */
    "pygpu/gpuarray.pyx":1718
        cdef int err
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
 */
    "pygpu/gpuarray.pyx":1717
        cdef GpuArrayIpcMemHandle h
        cdef int err
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
 */
  "pygpu/gpuarray.pyx":1719
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
 */
    "pygpu/gpuarray.pyx":1720
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
 */
    "pygpu/gpuarray.pyx":1719
        if cuda_get_ipc_handle is NULL:
            raise SystemError, "Could not get necessary extension"
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
 */
  "pygpu/gpuarray.pyx":1721
        if self.context.kind != b'cuda':
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
 */
  "pygpu/gpuarray.pyx":1722
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
 */
    "pygpu/gpuarray.pyx":1723
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
        return res
 */
    "pygpu/gpuarray.pyx":1722
            raise ValueError, "Only works for cuda contexts"
        err = cuda_get_ipc_handle(self.ga.data, &h)
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
 */
  "pygpu/gpuarray.pyx":1724
        if err != GA_NO_ERROR:
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
        return res

 */
  "pygpu/gpuarray.pyx":1725
            raise get_exc(err), GpuArray_error(&self.ga, err)
        res = <bytes>(<char *>&h)[:sizeof(h)]
        return res

    def __array__(self, ldtype=None):
 */
  "pygpu/gpuarray.pyx":1711
        array_read(np.PyArray_DATA(dst), sz, self)

    def get_ipc_handle(self):
        """
        get_ipc_handle()
 */
"pygpu/gpuarray.pyx":1727
        return res

    def __array__(self, ldtype=None):
        """
        __array__(ldtype=None)
 */
  "pygpu/gpuarray.pyx":1735
        Automatically used by :meth:`numpy.asarray`.
        """
        return _pygpu_as_ndarray(self, ldtype)

    def __bool__(self):
 */
  "pygpu/gpuarray.pyx":1727
        return res

    def __array__(self, ldtype=None):
        """
        __array__(ldtype=None)
 */
"pygpu/gpuarray.pyx":1737
        return _pygpu_as_ndarray(self, ldtype)

    def __bool__(self):
        """
        __bool__()
 */
  "pygpu/gpuarray.pyx":1741
        __bool__()
        """
        if self.size == 0:
            return False
        elif self.size == 1:
 */
    "pygpu/gpuarray.pyx":1742
        """
        if self.size == 0:
            return False
        elif self.size == 1:
            return bool(numpy.asarray(self))
 */
    "pygpu/gpuarray.pyx":1741
        __bool__()
        """
        if self.size == 0:
            return False
        elif self.size == 1:
 */
  "pygpu/gpuarray.pyx":1743
        if self.size == 0:
            return False
        elif self.size == 1:
            return bool(numpy.asarray(self))
        else:
 */
    "pygpu/gpuarray.pyx":1744
            return False
        elif self.size == 1:
            return bool(numpy.asarray(self))
        else:
            raise ValueError('The truth value of a multi-element array is ambiguous')
 */
    "pygpu/gpuarray.pyx":1743
        if self.size == 0:
            return False
        elif self.size == 1:
            return bool(numpy.asarray(self))
        else:
 */
  "pygpu/gpuarray.pyx":1746
            return bool(numpy.asarray(self))
        else:
            raise ValueError('The truth value of a multi-element array is ambiguous')

    def _empty_like_me(self, dtype=None, order='C'):
 */
  "pygpu/gpuarray.pyx":1737
        return _pygpu_as_ndarray(self, ldtype)

    def __bool__(self):
        """
        __bool__()
 */
"pygpu/gpuarray.pyx":1748
            raise ValueError('The truth value of a multi-element array is ambiguous')

    def _empty_like_me(self, dtype=None, order='C'):
        """
        _empty_like_me(dtype=None, order='C')
 */
  "pygpu/gpuarray.pyx":1758
        cdef GpuArray res

        if dtype is None:
            typecode = -1
        else:
 */
    "pygpu/gpuarray.pyx":1759

        if dtype is None:
            typecode = -1
        else:
            typecode = dtype_to_typecode(dtype)
 */
    "pygpu/gpuarray.pyx":1758
        cdef GpuArray res

        if dtype is None:
            typecode = -1
        else:
 */
  "pygpu/gpuarray.pyx":1761
            typecode = -1
        else:
            typecode = dtype_to_typecode(dtype)

        return pygpu_empty_like(self, to_ga_order(order), typecode)
 */
  "pygpu/gpuarray.pyx":1763
            typecode = dtype_to_typecode(dtype)

        return pygpu_empty_like(self, to_ga_order(order), typecode)

    def copy(self, order='C'):
 */
  "pygpu/gpuarray.pyx":1748
            raise ValueError('The truth value of a multi-element array is ambiguous')

    def _empty_like_me(self, dtype=None, order='C'):
        """
        _empty_like_me(dtype=None, order='C')
 */
"pygpu/gpuarray.pyx":1765
        return pygpu_empty_like(self, to_ga_order(order), typecode)

    def copy(self, order='C'):
        """
        copy(order='C')
 */
  "pygpu/gpuarray.pyx":1777

        """
        return pygpu_copy(self, to_ga_order(order))

    def transfer(self, GpuContext new_ctx):
 */
  "pygpu/gpuarray.pyx":1765
        return pygpu_empty_like(self, to_ga_order(order), typecode)

    def copy(self, order='C'):
        """
        copy(order='C')
 */
"pygpu/gpuarray.pyx":1779
        return pygpu_copy(self, to_ga_order(order))

    def transfer(self, GpuContext new_ctx):
        """
        transfer(new_ctx)
 */
  "pygpu/gpuarray.pyx":1784
        """
        cdef GpuArray r
        if not GpuArray_ISONESEGMENT(&self.ga):
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
 */
    "pygpu/gpuarray.pyx":1786
        if not GpuArray_ISONESEGMENT(&self.ga):
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
        r = pygpu_empty(self.ga.nd, self.ga.dimensions, self.ga.typecode,
           GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
 */
    "pygpu/gpuarray.pyx":1784
        """
        cdef GpuArray r
        if not GpuArray_ISONESEGMENT(&self.ga):
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
 */
  "pygpu/gpuarray.pyx":1788
            raise ValueError("transfer() only works for contigous source")
        r = pygpu_empty(self.ga.nd, self.ga.dimensions, self.ga.typecode,
           GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
           new_ctx, None)
        pygpu_transfer(r, self)  # Will raise an error if needed
 */
  "pygpu/gpuarray.pyx":1787
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
        r = pygpu_empty(self.ga.nd, self.ga.dimensions, self.ga.typecode,
           GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
           new_ctx, None)
 */
  "pygpu/gpuarray.pyx":1790
           GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
           new_ctx, None)
        pygpu_transfer(r, self)  # Will raise an error if needed
        return r

 */
  "pygpu/gpuarray.pyx":1791
           new_ctx, None)
        pygpu_transfer(r, self)  # Will raise an error if needed
        return r

    def __copy__(self):
 */
  "pygpu/gpuarray.pyx":1779
        return pygpu_copy(self, to_ga_order(order))

    def transfer(self, GpuContext new_ctx):
        """
        transfer(new_ctx)
 */
"pygpu/gpuarray.pyx":1793
        return r

    def __copy__(self):
        return pygpu_copy(self, GA_C_ORDER)

 */
  "pygpu/gpuarray.pyx":1794

    def __copy__(self):
        return pygpu_copy(self, GA_C_ORDER)

    def __deepcopy__(self, memo):
 */
  "pygpu/gpuarray.pyx":1793
        return r

    def __copy__(self):
        return pygpu_copy(self, GA_C_ORDER)

 */
"pygpu/gpuarray.pyx":1796
        return pygpu_copy(self, GA_C_ORDER)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
 */
  "pygpu/gpuarray.pyx":1797

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
 */
    "pygpu/gpuarray.pyx":1798
    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
            return pygpu_copy(self, GA_C_ORDER)
 */
    "pygpu/gpuarray.pyx":1797

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
        else:
 */
  "pygpu/gpuarray.pyx":1800
            return memo[id(self)]
        else:
            return pygpu_copy(self, GA_C_ORDER)

    def sync(self):
 */
  "pygpu/gpuarray.pyx":1796
        return pygpu_copy(self, GA_C_ORDER)

    def __deepcopy__(self, memo):
        if id(self) in memo:
            return memo[id(self)]
 */
"pygpu/gpuarray.pyx":1802
            return pygpu_copy(self, GA_C_ORDER)

    def sync(self):
        """
        sync()
 */
  "pygpu/gpuarray.pyx":1811
        but can be useful as a separate operation for timings.
        """
        pygpu_sync(self)

    def view(self, object cls=GpuArray):
 */
  "pygpu/gpuarray.pyx":1802
            return pygpu_copy(self, GA_C_ORDER)

    def sync(self):
        """
        sync()
 */
"pygpu/gpuarray.pyx":1813
        pygpu_sync(self)

    def view(self, object cls=GpuArray):
        """
        view(cls=GpuArray)
 */
  "pygpu/gpuarray.pyx":1828

        """
        return pygpu_view(self, cls)

    def astype(self, dtype, order='A', copy=True):
 */
  "pygpu/gpuarray.pyx":1813
        pygpu_sync(self)

    def view(self, object cls=GpuArray):
        """
        view(cls=GpuArray)
 */
"pygpu/gpuarray.pyx":1830
        return pygpu_view(self, cls)

    def astype(self, dtype, order='A', copy=True):
        """
        astype(dtype, order='A', copy=True)
 */
  "pygpu/gpuarray.pyx":1853
        """
        cdef GpuArray res
        cdef int typecode = dtype_to_typecode(dtype)
        cdef ga_order ord = to_ga_order(order)

 */
  "pygpu/gpuarray.pyx":1854
        cdef GpuArray res
        cdef int typecode = dtype_to_typecode(dtype)
        cdef ga_order ord = to_ga_order(order)

        if (not copy and typecode == self.ga.typecode and
 */
  "pygpu/gpuarray.pyx":1856
        cdef ga_order ord = to_ga_order(order)

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
 */
  "pygpu/gpuarray.pyx":1857

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
            return self
 */
  "pygpu/gpuarray.pyx":1858
        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
            return self

 */
  "pygpu/gpuarray.pyx":1856
        cdef ga_order ord = to_ga_order(order)

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
 */
    "pygpu/gpuarray.pyx":1859
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
            return self

        res = self._empty_like_me(dtype=typecode, order=order)
 */
    "pygpu/gpuarray.pyx":1856
        cdef ga_order ord = to_ga_order(order)

        if (not copy and typecode == self.ga.typecode and
            ((py_CHKFLAGS(self, GA_F_CONTIGUOUS) and ord == GA_F_ORDER) or
(py_CHKFLAGS(self, GA_C_CONTIGUOUS) and ord == GA_C_ORDER))):
 */
  "pygpu/gpuarray.pyx":1861
            return self

        res = self._empty_like_me(dtype=typecode, order=order)
        array_move(res, self)
        return res
 */
  "pygpu/gpuarray.pyx":1862

        res = self._empty_like_me(dtype=typecode, order=order)
        array_move(res, self)
        return res

 */
  "pygpu/gpuarray.pyx":1863
        res = self._empty_like_me(dtype=typecode, order=order)
        array_move(res, self)
        return res

    def reshape(self, shape, order='C'):
 */
  "pygpu/gpuarray.pyx":1830
        return pygpu_view(self, cls)

    def astype(self, dtype, order='A', copy=True):
        """
        astype(dtype, order='A', copy=True)
 */
"pygpu/gpuarray.pyx":1865
        return res

    def reshape(self, shape, order='C'):
        """
        reshape(shape, order='C')
 */
  "pygpu/gpuarray.pyx":1879
        cdef int compute_axis

        try:
            nd = <unsigned int>len(shape)
        except TypeError:
 */
      "pygpu/gpuarray.pyx":1880

        try:
            nd = <unsigned int>len(shape)
        except TypeError:
            nd = 1
 */
      "pygpu/gpuarray.pyx":1879
        cdef int compute_axis

        try:
            nd = <unsigned int>len(shape)
        except TypeError:
 */
    "pygpu/gpuarray.pyx":1881
        try:
            nd = <unsigned int>len(shape)
        except TypeError:
            nd = 1
            shape = [shape]
 */
      "pygpu/gpuarray.pyx":1882
            nd = <unsigned int>len(shape)
        except TypeError:
            nd = 1
            shape = [shape]

 */
      "pygpu/gpuarray.pyx":1883
        except TypeError:
            nd = 1
            shape = [shape]

        newdims = <size_t *>calloc(nd, sizeof(size_t))
 */
    "pygpu/gpuarray.pyx":1879
        cdef int compute_axis

        try:
            nd = <unsigned int>len(shape)
        except TypeError:
 */
  "pygpu/gpuarray.pyx":1885
            shape = [shape]

        newdims = <size_t *>calloc(nd, sizeof(size_t))
        if newdims == NULL:
            raise MemoryError, "calloc"
 */
  "pygpu/gpuarray.pyx":1886

        newdims = <size_t *>calloc(nd, sizeof(size_t))
        if newdims == NULL:
            raise MemoryError, "calloc"
        compute_axis = -1
 */
    "pygpu/gpuarray.pyx":1887
        newdims = <size_t *>calloc(nd, sizeof(size_t))
        if newdims == NULL:
            raise MemoryError, "calloc"
        compute_axis = -1
        try:
 */
    "pygpu/gpuarray.pyx":1886

        newdims = <size_t *>calloc(nd, sizeof(size_t))
        if newdims == NULL:
            raise MemoryError, "calloc"
        compute_axis = -1
 */
  "pygpu/gpuarray.pyx":1888
        if newdims == NULL:
            raise MemoryError, "calloc"
        compute_axis = -1
        try:
            for i in range(nd):
 */
  "pygpu/gpuarray.pyx":1889
            raise MemoryError, "calloc"
        compute_axis = -1
        try:
            for i in range(nd):
   if shape[i] == -1:
 */
    "pygpu/gpuarray.pyx":1890
        compute_axis = -1
        try:
            for i in range(nd):
   if shape[i] == -1:
       assert compute_axis == -1
 */
      "pygpu/gpuarray.pyx":1891
        try:
            for i in range(nd):
   if shape[i] == -1:
       assert compute_axis == -1
       compute_axis = i
 */
        "pygpu/gpuarray.pyx":1892
            for i in range(nd):
   if shape[i] == -1:
       assert compute_axis == -1
       compute_axis = i
       newdims[i] = 1
 */
        "pygpu/gpuarray.pyx":1893
   if shape[i] == -1:
       assert compute_axis == -1
       compute_axis = i
       newdims[i] = 1
   else:
 */
        "pygpu/gpuarray.pyx":1894
       assert compute_axis == -1
       compute_axis = i
       newdims[i] = 1
   else:
       newdims[i] = shape[i]
 */
        "pygpu/gpuarray.pyx":1891
        try:
            for i in range(nd):
   if shape[i] == -1:
       assert compute_axis == -1
       compute_axis = i
 */
      "pygpu/gpuarray.pyx":1896
       newdims[i] = 1
   else:
       newdims[i] = shape[i]
            return pygpu_reshape(self, nd, newdims, to_ga_order(order), 0, compute_axis)
        finally:
 */
    "pygpu/gpuarray.pyx":1897
   else:
       newdims[i] = shape[i]
            return pygpu_reshape(self, nd, newdims, to_ga_order(order), 0, compute_axis)
        finally:
            free(newdims)
 */
  "pygpu/gpuarray.pyx":1899
            return pygpu_reshape(self, nd, newdims, to_ga_order(order), 0, compute_axis)
        finally:
            free(newdims)

    def transpose(self, *params):
 */
  "pygpu/gpuarray.pyx":1865
        return res

    def reshape(self, shape, order='C'):
        """
        reshape(shape, order='C')
 */
"pygpu/gpuarray.pyx":1901
            free(newdims)

    def transpose(self, *params):
        """
        transpose(*params)
 */
  "pygpu/gpuarray.pyx":1907
        cdef unsigned int *new_axes
        cdef unsigned int i
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
 */
    "pygpu/gpuarray.pyx":1908
        cdef unsigned int i
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
 */
    "pygpu/gpuarray.pyx":1907
        cdef unsigned int *new_axes
        cdef unsigned int i
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
 */
  "pygpu/gpuarray.pyx":1909
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
        else:
 */
    "pygpu/gpuarray.pyx":1910
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
        else:
            if len(params) != self.ga.nd:
 */
    "pygpu/gpuarray.pyx":1909
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
        else:
 */
  "pygpu/gpuarray.pyx":1912
            return pygpu_transpose(self, NULL)
        else:
            if len(params) != self.ga.nd:
   raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
 */
      "pygpu/gpuarray.pyx":1913
        else:
            if len(params) != self.ga.nd:
   raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
            try:
 */
      "pygpu/gpuarray.pyx":1912
            return pygpu_transpose(self, NULL)
        else:
            if len(params) != self.ga.nd:
   raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
 */
    "pygpu/gpuarray.pyx":1914
            if len(params) != self.ga.nd:
   raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
            try:
   for i in range(self.ga.nd):
 */
    "pygpu/gpuarray.pyx":1915
   raise ValueError("axes don't match: " + str(params))
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
            try:
   for i in range(self.ga.nd):
       new_axes[i] = params[i]
 */
      "pygpu/gpuarray.pyx":1916
            new_axes = <unsigned int *>calloc(self.ga.nd, sizeof(unsigned int))
            try:
   for i in range(self.ga.nd):
       new_axes[i] = params[i]
   return pygpu_transpose(self, new_axes)
 */
        "pygpu/gpuarray.pyx":1917
            try:
   for i in range(self.ga.nd):
       new_axes[i] = params[i]
   return pygpu_transpose(self, new_axes)
            finally:
 */
      "pygpu/gpuarray.pyx":1918
   for i in range(self.ga.nd):
       new_axes[i] = params[i]
   return pygpu_transpose(self, new_axes)
            finally:
   free(new_axes)
 */
    "pygpu/gpuarray.pyx":1920
   return pygpu_transpose(self, new_axes)
            finally:
   free(new_axes)

    def __len__(self):
 */
  "pygpu/gpuarray.pyx":1901
            free(newdims)

    def transpose(self, *params):
        """
        transpose(*params)
 */
"pygpu/gpuarray.pyx":1922
   free(new_axes)

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
 */
  "pygpu/gpuarray.pyx":1923

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
        else:
 */
    "pygpu/gpuarray.pyx":1924
    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
        else:
            raise TypeError, "len() of unsized object"
 */
    "pygpu/gpuarray.pyx":1923

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
        else:
 */
  "pygpu/gpuarray.pyx":1926
            return self.ga.dimensions[0]
        else:
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
 */
  "pygpu/gpuarray.pyx":1922
   free(new_axes)

    def __len__(self):
        if self.ga.nd > 0:
            return self.ga.dimensions[0]
 */
"pygpu/gpuarray.pyx":1928
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
        cdef unsigned int i

 */
"pygpu/gpuarray.pyx":1939
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
            else:
 */
"pygpu/gpuarray.pyx":1949
            key = (key,)
        else:
            if all(isinstance(k, list) for k in key):
   raise NotImplementedError, "fancy indexing not supported"

 */
"pygpu/gpuarray.pyx":1975

        # Remove the None entries for indexing
        getitem_idcs = tuple(k for k in key if k is not None)

        # For less than 1 index, fill up with slice(None) to the right.
 */
"pygpu/gpuarray.pyx":1928
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
        cdef unsigned int i

 */
  "pygpu/gpuarray.pyx":1931
        cdef unsigned int i

        if key is Ellipsis:
            return self.__cgetitem__(key)

 */
    "pygpu/gpuarray.pyx":1932

        if key is Ellipsis:
            return self.__cgetitem__(key)

        # A list or a sequence of list should trigger "fancy" indexing.
 */
    "pygpu/gpuarray.pyx":1931
        cdef unsigned int i

        if key is Ellipsis:
            return self.__cgetitem__(key)

 */
  "pygpu/gpuarray.pyx":1938
        # Conversely, if a list contains slice or Ellipsis objects, it behaves
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
 */
    "pygpu/gpuarray.pyx":1939
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
            else:
 */
      "pygpu/gpuarray.pyx":1940
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
            else:
   raise NotImplementedError, "fancy indexing not supported"
 */
      "pygpu/gpuarray.pyx":1939
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
            else:
 */
    "pygpu/gpuarray.pyx":1942
   return self.__getitem__(tuple(key))
            else:
   raise NotImplementedError, "fancy indexing not supported"

        try:
 */
    "pygpu/gpuarray.pyx":1938
        # Conversely, if a list contains slice or Ellipsis objects, it behaves
        # the same as a tuple.
        if isinstance(key, list):
            if any(isinstance(k, slice) or k is Ellipsis for k in key):
   return self.__getitem__(tuple(key))
 */
  "pygpu/gpuarray.pyx":1944
   raise NotImplementedError, "fancy indexing not supported"

        try:
            iter(key)
        except TypeError:
 */
      "pygpu/gpuarray.pyx":1945

        try:
            iter(key)
        except TypeError:
            key = (key,)
 */
      "pygpu/gpuarray.pyx":1944
   raise NotImplementedError, "fancy indexing not supported"

        try:
            iter(key)
        except TypeError:
 */
    "pygpu/gpuarray.pyx":1949
            key = (key,)
        else:
            if all(isinstance(k, list) for k in key):
   raise NotImplementedError, "fancy indexing not supported"

 */
        "pygpu/gpuarray.pyx":1950
        else:
            if all(isinstance(k, list) for k in key):
   raise NotImplementedError, "fancy indexing not supported"

            key = tuple(key)
 */
        "pygpu/gpuarray.pyx":1949
            key = (key,)
        else:
            if all(isinstance(k, list) for k in key):
   raise NotImplementedError, "fancy indexing not supported"

 */
      "pygpu/gpuarray.pyx":1952
   raise NotImplementedError, "fancy indexing not supported"

            key = tuple(key)

        # Need to massage Ellipsis here, to avoid packing it into a tuple.
 */
    "pygpu/gpuarray.pyx":1946
        try:
            iter(key)
        except TypeError:
            key = (key,)
        else:
 */
      "pygpu/gpuarray.pyx":1947
            iter(key)
        except TypeError:
            key = (key,)
        else:
            if all(isinstance(k, list) for k in key):
 */
    "pygpu/gpuarray.pyx":1944
   raise NotImplementedError, "fancy indexing not supported"

        try:
            iter(key)
        except TypeError:
 */
  "pygpu/gpuarray.pyx":1955

        # Need to massage Ellipsis here, to avoid packing it into a tuple.
        if countis(key, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

 */
    "pygpu/gpuarray.pyx":1956
        # Need to massage Ellipsis here, to avoid packing it into a tuple.
        if countis(key, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

        # The following code replaces an Ellipsis found in the key by
 */
    "pygpu/gpuarray.pyx":1955

        # Need to massage Ellipsis here, to avoid packing it into a tuple.
        if countis(key, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

 */
  "pygpu/gpuarray.pyx":1963
        # dimension with a[..., 1:] on any array (including 1-dim).  This
        # is also required for numpy compat.
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
 */
      "pygpu/gpuarray.pyx":1964
        # is also required for numpy compat.
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
            pass
 */
      "pygpu/gpuarray.pyx":1963
        # dimension with a[..., 1:] on any array (including 1-dim).  This
        # is also required for numpy compat.
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
 */
    "pygpu/gpuarray.pyx":1970
            # Need number of axes minus missing dimensions extra slice(None)
            # objects, not counting None entries and the Ellipsis itself
            num_slcs = self.ga.nd - (len(key) - countis(key, None) - 1)
            fill_slices = (slice(None),)num_slcs
            key = key[:ell_idx] + fill_slices + key[ell_idx + 1:]
 */
      "pygpu/gpuarray.pyx":1971
            # objects, not counting None entries and the Ellipsis itself
            num_slcs = self.ga.nd - (len(key) - countis(key, None) - 1)
            fill_slices = (slice(None),)num_slcs
            key = key[:ell_idx] + fill_slices + key[ell_idx + 1:]

 */
      "pygpu/gpuarray.pyx":1972
            num_slcs = self.ga.nd - (len(key) - countis(key, None) - 1)
            fill_slices = (slice(None),)num_slcs
            key = key[:ell_idx] + fill_slices + key[ell_idx + 1:]

        # Remove the None entries for indexing
 */
    "pygpu/gpuarray.pyx":1965
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
            pass
        else:
 */
    "pygpu/gpuarray.pyx":1963
        # dimension with a[..., 1:] on any array (including 1-dim).  This
        # is also required for numpy compat.
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
 */
  "pygpu/gpuarray.pyx":1975

        # Remove the None entries for indexing
        getitem_idcs = tuple(k for k in key if k is not None)

        # For less than 1 index, fill up with slice(None) to the right.
 */
  "pygpu/gpuarray.pyx":1981
        # slice is applied along the first axis only. It also allows
        # a[()], which simply is a view in Numpy.
        if len(getitem_idcs) <= 1:
            getitem_idcs = (getitem_idcs +
  (slice(None),)(self.ga.nd - len(getitem_idcs)))
 */
    "pygpu/gpuarray.pyx":1983
        if len(getitem_idcs) <= 1:
            getitem_idcs = (getitem_idcs +
  (slice(None),)(self.ga.nd - len(getitem_idcs)))

        # Slice into array, then reshape, accommodating for None entries in key
 */
    "pygpu/gpuarray.pyx":1982
        # a[()], which simply is a view in Numpy.
        if len(getitem_idcs) <= 1:
            getitem_idcs = (getitem_idcs +
  (slice(None),)(self.ga.nd - len(getitem_idcs)))

 */
    "pygpu/gpuarray.pyx":1981
        # slice is applied along the first axis only. It also allows
        # a[()], which simply is a view in Numpy.
        if len(getitem_idcs) <= 1:
            getitem_idcs = (getitem_idcs +
  (slice(None),)(self.ga.nd - len(getitem_idcs)))
 */
  "pygpu/gpuarray.pyx":1986

        # Slice into array, then reshape, accommodating for None entries in key
        sliced = self.__cgetitem__(getitem_idcs)
        if countis(key, None) == 0:
            # Avoid unnecessary reshaping if there was no None
 */
  "pygpu/gpuarray.pyx":1987
        # Slice into array, then reshape, accommodating for None entries in key
        sliced = self.__cgetitem__(getitem_idcs)
        if countis(key, None) == 0:
            # Avoid unnecessary reshaping if there was no None
            return sliced
 */
    "pygpu/gpuarray.pyx":1989
        if countis(key, None) == 0:
            # Avoid unnecessary reshaping if there was no None
            return sliced
        else:
            new_shape = []
 */
    "pygpu/gpuarray.pyx":1987
        # Slice into array, then reshape, accommodating for None entries in key
        sliced = self.__cgetitem__(getitem_idcs)
        if countis(key, None) == 0:
            # Avoid unnecessary reshaping if there was no None
            return sliced
 */
  "pygpu/gpuarray.pyx":1991
            return sliced
        else:
            new_shape = []
            i = 0
            if sliced.shape:
 */
    "pygpu/gpuarray.pyx":1992
        else:
            new_shape = []
            i = 0
            if sliced.shape:
   for k in key:
 */
    "pygpu/gpuarray.pyx":1993
            new_shape = []
            i = 0
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
 */
      "pygpu/gpuarray.pyx":1994
            i = 0
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
           continue
 */
        "pygpu/gpuarray.pyx":1995
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
           continue
       elif k is None:
 */
          "pygpu/gpuarray.pyx":1996
   for k in key:
       if isinstance(k, int):
           continue
       elif k is None:
           new_shape.append(1)
 */
          "pygpu/gpuarray.pyx":1995
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
           continue
       elif k is None:
 */
        "pygpu/gpuarray.pyx":1997
       if isinstance(k, int):
           continue
       elif k is None:
           new_shape.append(1)
       else:
 */
          "pygpu/gpuarray.pyx":1998
           continue
       elif k is None:
           new_shape.append(1)
       else:
           new_shape.append(sliced.shape[i])
 */
          "pygpu/gpuarray.pyx":1997
       if isinstance(k, int):
           continue
       elif k is None:
           new_shape.append(1)
       else:
 */
        "pygpu/gpuarray.pyx":2000
           new_shape.append(1)
       else:
           new_shape.append(sliced.shape[i])
           i += 1
            # Add remaining entries from sliced.shape if existing (happens
 */
          "pygpu/gpuarray.pyx":2001
       else:
           new_shape.append(sliced.shape[i])
           i += 1
            # Add remaining entries from sliced.shape if existing (happens
            # for 1 index or less if ndim >= 2).
 */
        "pygpu/gpuarray.pyx":1994
            i = 0
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
           continue
 */
      "pygpu/gpuarray.pyx":1993
            new_shape = []
            i = 0
            if sliced.shape:
   for k in key:
       if isinstance(k, int):
 */
    "pygpu/gpuarray.pyx":2004
            # Add remaining entries from sliced.shape if existing (happens
            # for 1 index or less if ndim >= 2).
            new_shape.extend(sliced.shape[i:])
            return sliced.reshape(new_shape)

 */
    "pygpu/gpuarray.pyx":2005
            # for 1 index or less if ndim >= 2).
            new_shape.extend(sliced.shape[i:])
            return sliced.reshape(new_shape)

    cdef __cgetitem__(self, key):
 */
  "pygpu/gpuarray.pyx":1928
            raise TypeError, "len() of unsized object"

    def __getitem__(self, key):
        cdef unsigned int i

 */
"pygpu/gpuarray.pyx":2007
            return sliced.reshape(new_shape)

    cdef __cgetitem__(self, key):
        cdef ssize_t *starts
        cdef ssize_t *stops
 */
  "pygpu/gpuarray.pyx":2015
        cdef unsigned int el

        if key is Ellipsis:
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
 */
    "pygpu/gpuarray.pyx":2016

        if key is Ellipsis:
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
 */
    "pygpu/gpuarray.pyx":2015
        cdef unsigned int el

        if key is Ellipsis:
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
 */
  "pygpu/gpuarray.pyx":2017
        if key is Ellipsis:
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
   return self
 */
    "pygpu/gpuarray.pyx":2018
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
   return self
            else:
 */
      "pygpu/gpuarray.pyx":2019
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
   return self
            else:
   raise IndexError, "0-d arrays can't be indexed"
 */
      "pygpu/gpuarray.pyx":2018
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
   return self
            else:
 */
    "pygpu/gpuarray.pyx":2021
   return self
            else:
   raise IndexError, "0-d arrays can't be indexed"

        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
 */
    "pygpu/gpuarray.pyx":2017
        if key is Ellipsis:
            return pygpu_view(self, None)
        elif self.ga.nd == 0:
            if isinstance(key, tuple) and len(key) == 0:
   return self
 */
  "pygpu/gpuarray.pyx":2023
   raise IndexError, "0-d arrays can't be indexed"

        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
 */
  "pygpu/gpuarray.pyx":2024

        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
 */
  "pygpu/gpuarray.pyx":2025
        starts = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
 */
  "pygpu/gpuarray.pyx":2026
        stops = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
   raise MemoryError
 */
    "pygpu/gpuarray.pyx":2027
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
   raise MemoryError

 */
      "pygpu/gpuarray.pyx":2028
        try:
            if starts == NULL or stops == NULL or steps == NULL:
   raise MemoryError

            d = 0
 */
      "pygpu/gpuarray.pyx":2027
        steps = <ssize_t *>calloc(self.ga.nd, sizeof(ssize_t))
        try:
            if starts == NULL or stops == NULL or steps == NULL:
   raise MemoryError

 */
    "pygpu/gpuarray.pyx":2030
   raise MemoryError

            d = 0

            if isinstance(key, (tuple, list)):
 */
    "pygpu/gpuarray.pyx":2032
            d = 0

            if isinstance(key, (tuple, list)):
   if Ellipsis in key:
       # The following code replaces the first Ellipsis
 */
      "pygpu/gpuarray.pyx":2033

            if isinstance(key, (tuple, list)):
   if Ellipsis in key:
       # The following code replaces the first Ellipsis
       # found in the key by a bunch of them depending on
 */
        "pygpu/gpuarray.pyx":2040
       # a[..., 1:] on any array (including 1-dim).  This
       # is also required for numpy compat.
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
 */
        "pygpu/gpuarray.pyx":2041
       # is also required for numpy compat.
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
 */
          "pygpu/gpuarray.pyx":2042
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
 */
          "pygpu/gpuarray.pyx":2043
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
       else:
 */
          "pygpu/gpuarray.pyx":2042
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
 */
          "pygpu/gpuarray.pyx":2044
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
       else:
           key = (key[:el] +
 */
          "pygpu/gpuarray.pyx":2043
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
       else:
 */
          "pygpu/gpuarray.pyx":2041
       # is also required for numpy compat.
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
 */
        "pygpu/gpuarray.pyx":2047
       else:
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
 */
          "pygpu/gpuarray.pyx":2046
     key[el+1:])
       else:
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
 */
          "pygpu/gpuarray.pyx":2047
       else:
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
 */
          "pygpu/gpuarray.pyx":2046
     key[el+1:])
       else:
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
 */
          "pygpu/gpuarray.pyx":2048
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
       raise IndexError, "too many indices"
 */
          "pygpu/gpuarray.pyx":2047
       else:
           key = (key[:el] +
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
 */
        "pygpu/gpuarray.pyx":2033

            if isinstance(key, (tuple, list)):
   if Ellipsis in key:
       # The following code replaces the first Ellipsis
       # found in the key by a bunch of them depending on
 */
      "pygpu/gpuarray.pyx":2049
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
       raise IndexError, "too many indices"
   for i in range(0, len(key)):
 */
        "pygpu/gpuarray.pyx":2050
     key[el+1:])
   if len(key) > self.ga.nd:
       raise IndexError, "too many indices"
   for i in range(0, len(key)):
       self.__index_helper(key[i], i, &starts[i], &stops[i],
 */
        "pygpu/gpuarray.pyx":2049
     [Ellipsis,]*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
   if len(key) > self.ga.nd:
       raise IndexError, "too many indices"
   for i in range(0, len(key)):
 */
      "pygpu/gpuarray.pyx":2051
   if len(key) > self.ga.nd:
       raise IndexError, "too many indices"
   for i in range(0, len(key)):
       self.__index_helper(key[i], i, &starts[i], &stops[i],
 &steps[i])
 */
        "pygpu/gpuarray.pyx":2052
       raise IndexError, "too many indices"
   for i in range(0, len(key)):
       self.__index_helper(key[i], i, &starts[i], &stops[i],
 &steps[i])
   d += <unsigned int>len(key)
 */
        "pygpu/gpuarray.pyx":2053
   for i in range(0, len(key)):
       self.__index_helper(key[i], i, &starts[i], &stops[i],
 &steps[i])
   d += <unsigned int>len(key)
            else:
 */
      "pygpu/gpuarray.pyx":2054
       self.__index_helper(key[i], i, &starts[i], &stops[i],
 &steps[i])
   d += <unsigned int>len(key)
            else:
   self.__index_helper(key, 0, starts, stops, steps)
 */
      "pygpu/gpuarray.pyx":2032
            d = 0

            if isinstance(key, (tuple, list)):
   if Ellipsis in key:
       # The following code replaces the first Ellipsis
 */
    "pygpu/gpuarray.pyx":2056
   d += <unsigned int>len(key)
            else:
   self.__index_helper(key, 0, starts, stops, steps)
   d += 1

 */
      "pygpu/gpuarray.pyx":2057
            else:
   self.__index_helper(key, 0, starts, stops, steps)
   d += 1

            for i in range(d, self.ga.nd):
 */
    "pygpu/gpuarray.pyx":2059
   d += 1

            for i in range(d, self.ga.nd):
   starts[i] = 0
   stops[i] = self.ga.dimensions[i]
 */
      "pygpu/gpuarray.pyx":2060

            for i in range(d, self.ga.nd):
   starts[i] = 0
   stops[i] = self.ga.dimensions[i]
   steps[i] = 1
 */
      "pygpu/gpuarray.pyx":2061
            for i in range(d, self.ga.nd):
   starts[i] = 0
   stops[i] = self.ga.dimensions[i]
   steps[i] = 1

 */
      "pygpu/gpuarray.pyx":2062
   starts[i] = 0
   stops[i] = self.ga.dimensions[i]
   steps[i] = 1

            return pygpu_index(self, starts, stops, steps)
 */
    "pygpu/gpuarray.pyx":2064
   steps[i] = 1

            return pygpu_index(self, starts, stops, steps)

        finally:
 */
  "pygpu/gpuarray.pyx":2067

        finally:
            free(starts)
            free(stops)
            free(steps)
 */
        "pygpu/gpuarray.pyx":2068
        finally:
            free(starts)
            free(stops)
            free(steps)

 */
        "pygpu/gpuarray.pyx":2069
            free(starts)
            free(stops)
            free(steps)

    def __setitem__(self, idx, v):
 */
      "pygpu/gpuarray.pyx":2067

        finally:
            free(starts)
            free(stops)
            free(steps)
 */
      "pygpu/gpuarray.pyx":2068
        finally:
            free(starts)
            free(stops)
            free(steps)

 */
      "pygpu/gpuarray.pyx":2069
            free(starts)
            free(stops)
            free(steps)

    def __setitem__(self, idx, v):
 */
  "pygpu/gpuarray.pyx":2007
            return sliced.reshape(new_shape)

    cdef __cgetitem__(self, key):
        cdef ssize_t *starts
        cdef ssize_t *stops
 */
"pygpu/gpuarray.pyx":2084
            idx = (idx,)
        else:
            if all(isinstance(i, list) for i in idx):
   raise NotImplementedError, "fancy indexing not supported"

 */
"pygpu/gpuarray.pyx":2093

        # Remove None entries, they should be ignored (as in Numpy)
        idx = tuple(i for i in idx if i is not None)
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
 */


      "pygpu/gpuarray.pyx":2075

        if isinstance(idx, list):
            if any(isinstance(i, slice) or i is Ellipsis for i in idx):
   self.__setitem__(tuple(idx), v)
            else:
 */
    "pygpu/gpuarray.pyx":2078
   self.__setitem__(tuple(idx), v)
            else:
   raise NotImplementedError, "fancy indexing not supported"
        try:
            iter(idx)
 */
  "pygpu/gpuarray.pyx":2079
            else:
   raise NotImplementedError, "fancy indexing not supported"
        try:
            iter(idx)
        except TypeError:
 */
      "pygpu/gpuarray.pyx":2080
   raise NotImplementedError, "fancy indexing not supported"
        try:
            iter(idx)
        except TypeError:
            idx = (idx,)
 */
      "pygpu/gpuarray.pyx":2079
            else:
   raise NotImplementedError, "fancy indexing not supported"
        try:
            iter(idx)
        except TypeError:
 */
    "pygpu/gpuarray.pyx":2084
            idx = (idx,)
        else:
            if all(isinstance(i, list) for i in idx):
   raise NotImplementedError, "fancy indexing not supported"

 */
        "pygpu/gpuarray.pyx":2085
        else:
            if all(isinstance(i, list) for i in idx):
   raise NotImplementedError, "fancy indexing not supported"

            idx = tuple(idx)
 */
        "pygpu/gpuarray.pyx":2084
            idx = (idx,)
        else:
            if all(isinstance(i, list) for i in idx):
   raise NotImplementedError, "fancy indexing not supported"

 */
      "pygpu/gpuarray.pyx":2087
   raise NotImplementedError, "fancy indexing not supported"

            idx = tuple(idx)

        if countis(idx, Ellipsis) > 1:
 */
    "pygpu/gpuarray.pyx":2081
        try:
            iter(idx)
        except TypeError:
            idx = (idx,)
        else:
 */
      "pygpu/gpuarray.pyx":2082
            iter(idx)
        except TypeError:
            idx = (idx,)
        else:
            if all(isinstance(i, list) for i in idx):
 */
    "pygpu/gpuarray.pyx":2079
            else:
   raise NotImplementedError, "fancy indexing not supported"
        try:
            iter(idx)
        except TypeError:
 */
  "pygpu/gpuarray.pyx":2089
            idx = tuple(idx)

        if countis(idx, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

 */
    "pygpu/gpuarray.pyx":2090

        if countis(idx, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

        # Remove None entries, they should be ignored (as in Numpy)
 */
    "pygpu/gpuarray.pyx":2089
            idx = tuple(idx)

        if countis(idx, Ellipsis) > 1:
            raise IndexError, "cannot use more than one Ellipsis"

 */
  "pygpu/gpuarray.pyx":2093

        # Remove None entries, they should be ignored (as in Numpy)
        idx = tuple(i for i in idx if i is not None)
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
 */
  "pygpu/gpuarray.pyx":2094
        # Remove None entries, they should be ignored (as in Numpy)
        idx = tuple(i for i in idx if i is not None)
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
        array_setarray(tmp, gv)
 */
  "pygpu/gpuarray.pyx":2095
        idx = tuple(i for i in idx if i is not None)
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
        array_setarray(tmp, gv)

 */
  "pygpu/gpuarray.pyx":2096
        tmp = self.__cgetitem__(idx)
        gv = carray(v, self.ga.typecode, False, 'A', 0, self.context, GpuArray)
        array_setarray(tmp, gv)

    def take1(self, GpuArray idx):
 */
"pygpu/gpuarray.pyx":2098
        array_setarray(tmp, gv)

    def take1(self, GpuArray idx):
        """
        take1(idx)
 */
  "pygpu/gpuarray.pyx":2104
        cdef GpuArray res
        cdef size_t odim
        if idx.ga.nd != 1:
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
 */
    "pygpu/gpuarray.pyx":2105
        cdef size_t odim
        if idx.ga.nd != 1:
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
        try:
 */
    "pygpu/gpuarray.pyx":2104
        cdef GpuArray res
        cdef size_t odim
        if idx.ga.nd != 1:
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
 */
  "pygpu/gpuarray.pyx":2106
        if idx.ga.nd != 1:
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
        try:
            self.ga.dimensions[0] = idx.ga.dimensions[0]
 */
  "pygpu/gpuarray.pyx":2107
            raise ValueError, "Expected index with nd=1"
        odim = self.ga.dimensions[0]
        try:
            self.ga.dimensions[0] = idx.ga.dimensions[0]
            res = pygpu_empty_like(self, GA_C_ORDER, -1)
 */
    "pygpu/gpuarray.pyx":2108
        odim = self.ga.dimensions[0]
        try:
            self.ga.dimensions[0] = idx.ga.dimensions[0]
            res = pygpu_empty_like(self, GA_C_ORDER, -1)
        finally:
 */
    "pygpu/gpuarray.pyx":2109
        try:
            self.ga.dimensions[0] = idx.ga.dimensions[0]
            res = pygpu_empty_like(self, GA_C_ORDER, -1)
        finally:
            self.ga.dimensions[0] = odim
 */
  "pygpu/gpuarray.pyx":2111
            res = pygpu_empty_like(self, GA_C_ORDER, -1)
        finally:
            self.ga.dimensions[0] = odim
        array_take1(res, self, idx, 1)
        return res
 */
  "pygpu/gpuarray.pyx":2112
        finally:
            self.ga.dimensions[0] = odim
        array_take1(res, self, idx, 1)
        return res

 */
  "pygpu/gpuarray.pyx":2113
            self.ga.dimensions[0] = odim
        array_take1(res, self, idx, 1)
        return res

    def __hash__(self):
 */
  "pygpu/gpuarray.pyx":2098
        array_setarray(tmp, gv)

    def take1(self, GpuArray idx):
        """
        take1(idx)
 */
"pygpu/gpuarray.pyx":2115
        return res

    def __hash__(self):
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

 */
  "pygpu/gpuarray.pyx":2116

    def __hash__(self):
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

    def __nonzero__(self):
 */
  "pygpu/gpuarray.pyx":2115
        return res

    def __hash__(self):
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

 */
"pygpu/gpuarray.pyx":2118
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
 */
  "pygpu/gpuarray.pyx":2119

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
 */
  "pygpu/gpuarray.pyx":2120
    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
 */
    "pygpu/gpuarray.pyx":2121
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
 */
    "pygpu/gpuarray.pyx":2120
    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
            return False
        if sz == 1:
 */
  "pygpu/gpuarray.pyx":2122
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
 */
    "pygpu/gpuarray.pyx":2123
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
            raise ValueError, "Truth value of array with more than one element is ambiguous"
 */
    "pygpu/gpuarray.pyx":2122
        if sz == 0:
            return False
        if sz == 1:
            return bool(numpy.asarray(self))
        else:
 */
  "pygpu/gpuarray.pyx":2125
            return bool(numpy.asarray(self))
        else:
            raise ValueError, "Truth value of array with more than one element is ambiguous"

    property shape:
 */
  "pygpu/gpuarray.pyx":2118
        raise TypeError, "unhashable type '%s'" % (self.__class__,)

    def __nonzero__(self):
        cdef int sz = self.size
        if sz == 0:
 */
"pygpu/gpuarray.pyx":2129
    property shape:
        "shape of this ndarray (tuple)"
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
 */
  "pygpu/gpuarray.pyx":2131
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.dimensions[i]
 */
  "pygpu/gpuarray.pyx":2132
            cdef unsigned int i
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.dimensions[i]
            return tuple(res)
 */
    "pygpu/gpuarray.pyx":2133
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.dimensions[i]
            return tuple(res)

 */
  "pygpu/gpuarray.pyx":2134
            for i in range(self.ga.nd):
   res[i] = self.ga.dimensions[i]
            return tuple(res)

        def __set__(self, newshape):
 */
  "pygpu/gpuarray.pyx":2129
    property shape:
        "shape of this ndarray (tuple)"
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
 */
"pygpu/gpuarray.pyx":2136
            return tuple(res)

        def __set__(self, newshape):
            # We support -1 only in a call to reshape
            cdef size_t *newdims
 */
  "pygpu/gpuarray.pyx":2142
            cdef unsigned int i
            cdef int err
            nd = <unsigned int>len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
 */
  "pygpu/gpuarray.pyx":2143
            cdef int err
            nd = <unsigned int>len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
   raise MemoryError, "calloc"
 */
  "pygpu/gpuarray.pyx":2144
            nd = <unsigned int>len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
   raise MemoryError, "calloc"
            try:
 */
    "pygpu/gpuarray.pyx":2145
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
   raise MemoryError, "calloc"
            try:
   for i in range(nd):
 */
    "pygpu/gpuarray.pyx":2144
            nd = <unsigned int>len(newshape)
            newdims = <size_t *>calloc(nd, sizeof(size_t))
            if newdims == NULL:
   raise MemoryError, "calloc"
            try:
 */
  "pygpu/gpuarray.pyx":2146
            if newdims == NULL:
   raise MemoryError, "calloc"
            try:
   for i in range(nd):
       newdims[i] = newshape[i]
 */
    "pygpu/gpuarray.pyx":2147
   raise MemoryError, "calloc"
            try:
   for i in range(nd):
       newdims[i] = newshape[i]
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
 */
      "pygpu/gpuarray.pyx":2148
            try:
   for i in range(nd):
       newdims[i] = newshape[i]
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
   if err != GA_NO_ERROR:
 */
    "pygpu/gpuarray.pyx":2149
   for i in range(nd):
       newdims[i] = newshape[i]
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
   if err != GA_NO_ERROR:
       raise get_exc(err), GpuArray_error(&self.ga, err)
 */
    "pygpu/gpuarray.pyx":2150
       newdims[i] = newshape[i]
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
   if err != GA_NO_ERROR:
       raise get_exc(err), GpuArray_error(&self.ga, err)
            finally:
 */
      "pygpu/gpuarray.pyx":2151
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
   if err != GA_NO_ERROR:
       raise get_exc(err), GpuArray_error(&self.ga, err)
            finally:
   free(newdims)
 */
      "pygpu/gpuarray.pyx":2150
       newdims[i] = newshape[i]
   err = GpuArray_reshape_inplace(&self.ga, nd, newdims, GA_C_ORDER)
   if err != GA_NO_ERROR:
       raise get_exc(err), GpuArray_error(&self.ga, err)
            finally:
 */
  "pygpu/gpuarray.pyx":2153
       raise get_exc(err), GpuArray_error(&self.ga, err)
            finally:
   free(newdims)

    property T:
 */
  "pygpu/gpuarray.pyx":2136
            return tuple(res)

        def __set__(self, newshape):
            # We support -1 only in a call to reshape
            cdef size_t *newdims
 */
"pygpu/gpuarray.pyx":2156

    property T:
        def __get__(self):
            return pygpu_transpose(self, NULL)

 */
  "pygpu/gpuarray.pyx":2157
    property T:
        def __get__(self):
            return pygpu_transpose(self, NULL)

    property size:
 */
  "pygpu/gpuarray.pyx":2156

    property T:
        def __get__(self):
            return pygpu_transpose(self, NULL)

 */
"pygpu/gpuarray.pyx":2161
    property size:
        "The number of elements in this object."
        def __get__(self):
            cdef size_t res = 1
            cdef unsigned int i
 */
  "pygpu/gpuarray.pyx":2162
        "The number of elements in this object."
        def __get__(self):
            cdef size_t res = 1
            cdef unsigned int i
            for i in range(self.ga.nd):
 */
  "pygpu/gpuarray.pyx":2164
            cdef size_t res = 1
            cdef unsigned int i
            for i in range(self.ga.nd):
   res *= self.ga.dimensions[i]
            return res
 */
    "pygpu/gpuarray.pyx":2165
            cdef unsigned int i
            for i in range(self.ga.nd):
   res *= self.ga.dimensions[i]
            return res

 */
  "pygpu/gpuarray.pyx":2166
            for i in range(self.ga.nd):
   res *= self.ga.dimensions[i]
            return res

    property strides:
 */
  "pygpu/gpuarray.pyx":2161
    property size:
        "The number of elements in this object."
        def __get__(self):
            cdef size_t res = 1
            cdef unsigned int i
 */
"pygpu/gpuarray.pyx":2170
    property strides:
        "data pointer strides (in bytes)"
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
 */
  "pygpu/gpuarray.pyx":2172
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.strides[i]
 */
  "pygpu/gpuarray.pyx":2173
            cdef unsigned int i
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.strides[i]
            return tuple(res)
 */
    "pygpu/gpuarray.pyx":2174
            res = [None]self.ga.nd
            for i in range(self.ga.nd):
   res[i] = self.ga.strides[i]
            return tuple(res)

 */
  "pygpu/gpuarray.pyx":2175
            for i in range(self.ga.nd):
   res[i] = self.ga.strides[i]
            return tuple(res)

        def __set__(self, newstrides):
 */
  "pygpu/gpuarray.pyx":2170
    property strides:
        "data pointer strides (in bytes)"
        def __get__(self):
            cdef unsigned int i
            res = [None]self.ga.nd
 */
"pygpu/gpuarray.pyx":2177
            return tuple(res)

        def __set__(self, newstrides):
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
 */
  "pygpu/gpuarray.pyx":2179
        def __set__(self, newstrides):
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
 */
    "pygpu/gpuarray.pyx":2180
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
 */
    "pygpu/gpuarray.pyx":2179
        def __set__(self, newstrides):
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
 */
  "pygpu/gpuarray.pyx":2181
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
 */
    "pygpu/gpuarray.pyx":2182
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
   self.ga.strides[i] = newstrides[i]
 */
    "pygpu/gpuarray.pyx":2181
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
 */
  "pygpu/gpuarray.pyx":2183
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
   self.ga.strides[i] = newstrides[i]
            array_fix_flags(self)
 */
    "pygpu/gpuarray.pyx":2184
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
   self.ga.strides[i] = newstrides[i]
            array_fix_flags(self)

 */
  "pygpu/gpuarray.pyx":2185
            for i in range(self.ga.nd):
   self.ga.strides[i] = newstrides[i]
            array_fix_flags(self)

    property ndim:
 */
  "pygpu/gpuarray.pyx":2177
            return tuple(res)

        def __set__(self, newstrides):
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
 */
"pygpu/gpuarray.pyx":2189
    property ndim:
        "The number of dimensions in this object"
        def __get__(self):
            return self.ga.nd

 */
  "pygpu/gpuarray.pyx":2190
        "The number of dimensions in this object"
        def __get__(self):
            return self.ga.nd

    property dtype:
 */
  "pygpu/gpuarray.pyx":2189
    property ndim:
        "The number of dimensions in this object"
        def __get__(self):
            return self.ga.nd

 */
"pygpu/gpuarray.pyx":2194
    property dtype:
        "The dtype of the element"
        def __get__(self):
            return typecode_to_dtype(self.ga.typecode)

 */
  "pygpu/gpuarray.pyx":2195
        "The dtype of the element"
        def __get__(self):
            return typecode_to_dtype(self.ga.typecode)

    property typecode:
 */
  "pygpu/gpuarray.pyx":2194
    property dtype:
        "The dtype of the element"
        def __get__(self):
            return typecode_to_dtype(self.ga.typecode)

 */
"pygpu/gpuarray.pyx":2199
    property typecode:
        "The gpuarray typecode for the data type of the array"
        def __get__(self):
            return self.ga.typecode

 */
  "pygpu/gpuarray.pyx":2200
        "The gpuarray typecode for the data type of the array"
        def __get__(self):
            return self.ga.typecode

    property itemsize:
 */
  "pygpu/gpuarray.pyx":2199
    property typecode:
        "The gpuarray typecode for the data type of the array"
        def __get__(self):
            return self.ga.typecode

 */
"pygpu/gpuarray.pyx":2204
    property itemsize:
        "The size of the base element."
        def __get__(self):
            return gpuarray_get_elsize(self.ga.typecode)

 */
  "pygpu/gpuarray.pyx":2205
        "The size of the base element."
        def __get__(self):
            return gpuarray_get_elsize(self.ga.typecode)

    property flags:
 */
  "pygpu/gpuarray.pyx":2204
    property itemsize:
        "The size of the base element."
        def __get__(self):
            return gpuarray_get_elsize(self.ga.typecode)

 */
"pygpu/gpuarray.pyx":2215
         UPDATEIFCOPY is not supported, therefore always False.
        """
        def __get__(self):
            return flags(self.ga.flags)

 */
  "pygpu/gpuarray.pyx":2216
        """
        def __get__(self):
            return flags(self.ga.flags)

    property offset:
 */
  "pygpu/gpuarray.pyx":2215
         UPDATEIFCOPY is not supported, therefore always False.
        """
        def __get__(self):
            return flags(self.ga.flags)

 */
"pygpu/gpuarray.pyx":2220
    property offset:
        "Return the offset into the gpudata pointer for this array."
        def __get__(self):
            return self.ga.offset

 */
  "pygpu/gpuarray.pyx":2221
        "Return the offset into the gpudata pointer for this array."
        def __get__(self):
            return self.ga.offset

    property data:
 */
  "pygpu/gpuarray.pyx":2220
    property offset:
        "Return the offset into the gpudata pointer for this array."
        def __get__(self):
            return self.ga.offset

 */
"pygpu/gpuarray.pyx":2228
        This will fail for arrays that have an offset.
        """
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
 */
  "pygpu/gpuarray.pyx":2229
        """
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
 */
    "pygpu/gpuarray.pyx":2230
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
 */
    "pygpu/gpuarray.pyx":2229
        """
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
 */
  "pygpu/gpuarray.pyx":2231
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
            # This wizadry grabs the actual backend pointer since it's
 */
    "pygpu/gpuarray.pyx":2232
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
    "pygpu/gpuarray.pyx":2231
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
            # This wizadry grabs the actual backend pointer since it's
 */
  "pygpu/gpuarray.pyx":2236
            # guarenteed to be the first element of the gpudata
            # structure.
            return <size_t>((<void **>self.ga.data)[0])

    property base_data:
 */
  "pygpu/gpuarray.pyx":2228
        This will fail for arrays that have an offset.
        """
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
 */
"pygpu/gpuarray.pyx":2240
    property base_data:
        "Return a pointer to the backing OpenCL object."
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
 */
  "pygpu/gpuarray.pyx":2241
        "Return a pointer to the backing OpenCL object."
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            # This wizadry grabs the actual backend pointer since it's
 */
    "pygpu/gpuarray.pyx":2242
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
    "pygpu/gpuarray.pyx":2241
        "Return a pointer to the backing OpenCL object."
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            # This wizadry grabs the actual backend pointer since it's
 */
  "pygpu/gpuarray.pyx":2246
            # guarenteed to be the first element of the gpudata
            # structure.
            return <size_t>((<void **>self.ga.data)[0])

    property gpudata:
 */
  "pygpu/gpuarray.pyx":2240
    property base_data:
        "Return a pointer to the backing OpenCL object."
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
 */
"pygpu/gpuarray.pyx":2250
    property gpudata:
        "Return a pointer to the raw backend object."
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
 */
  "pygpu/gpuarray.pyx":2251
        "Return a pointer to the raw backend object."
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
            # This wizadry grabs the actual backend pointer since it's
 */
    "pygpu/gpuarray.pyx":2252
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
    "pygpu/gpuarray.pyx":2251
        "Return a pointer to the raw backend object."
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
            # This wizadry grabs the actual backend pointer since it's
 */
  "pygpu/gpuarray.pyx":2256
            # guarenteed to be the first element of the gpudata
            # structure.
            return <size_t>((<void **>self.ga.data)[0]) + self.offset

    def __str__(self):
 */
  "pygpu/gpuarray.pyx":2250
    property gpudata:
        "Return a pointer to the raw backend object."
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
 */
"pygpu/gpuarray.pyx":2258
            return <size_t>((<void **>self.ga.data)[0]) + self.offset

    def __str__(self):
        return str(numpy.asarray(self))

 */
  "pygpu/gpuarray.pyx":2259

    def __str__(self):
        return str(numpy.asarray(self))

    def __repr__(self):
 */
  "pygpu/gpuarray.pyx":2258
            return <size_t>((<void **>self.ga.data)[0]) + self.offset

    def __str__(self):
        return str(numpy.asarray(self))

 */
"pygpu/gpuarray.pyx":2261
        return str(numpy.asarray(self))

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
 */
  "pygpu/gpuarray.pyx":2262

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
 */
      "pygpu/gpuarray.pyx":2263
    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
            return 'gpuarray.array(<content not available>)'
 */
      "pygpu/gpuarray.pyx":2262

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
 */
    "pygpu/gpuarray.pyx":2264
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
            return 'gpuarray.array(<content not available>)'

 */
      "pygpu/gpuarray.pyx":2265
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
            return 'gpuarray.array(<content not available>)'


 */
    "pygpu/gpuarray.pyx":2262

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
        except Exception:
 */
  "pygpu/gpuarray.pyx":2261
        return str(numpy.asarray(self))

    def __repr__(self):
        try:
            return 'gpuarray.' + repr(numpy.asarray(self))
 */
"pygpu/gpuarray.pyx":2342

    """
    def __dealloc__(self):
        cdef unsigned int numargs
        cdef int *types
 */
  "pygpu/gpuarray.pyx":2349
        # We need to do all of this at the C level to avoid touching
        # python stuff that could be gone and to avoid exceptions
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
 */
    "pygpu/gpuarray.pyx":2350
        # python stuff that could be gone and to avoid exceptions
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
   return
 */
    "pygpu/gpuarray.pyx":2351
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
 */
      "pygpu/gpuarray.pyx":2352
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
 */
      "pygpu/gpuarray.pyx":2351
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
 */
    "pygpu/gpuarray.pyx":2353
            if res != GA_NO_ERROR:
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
   return
 */
    "pygpu/gpuarray.pyx":2354
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
   return
            for i in range(numargs):
 */
      "pygpu/gpuarray.pyx":2355
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
   return
            for i in range(numargs):
   if types[i] != GA_BUFFER:
 */
      "pygpu/gpuarray.pyx":2354
   return
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_TYPES, &types)
            if res != GA_NO_ERROR:
   return
            for i in range(numargs):
 */
    "pygpu/gpuarray.pyx":2356
            if res != GA_NO_ERROR:
   return
            for i in range(numargs):
   if types[i] != GA_BUFFER:
       free(self.callbuf[i])
 */
      "pygpu/gpuarray.pyx":2357
   return
            for i in range(numargs):
   if types[i] != GA_BUFFER:
       free(self.callbuf[i])
            kernel_clear(self)
 */
        "pygpu/gpuarray.pyx":2358
            for i in range(numargs):
   if types[i] != GA_BUFFER:
       free(self.callbuf[i])
            kernel_clear(self)
        free(self.callbuf)
 */
        "pygpu/gpuarray.pyx":2357
   return
            for i in range(numargs):
   if types[i] != GA_BUFFER:
       free(self.callbuf[i])
            kernel_clear(self)
 */
    "pygpu/gpuarray.pyx":2359
   if types[i] != GA_BUFFER:
       free(self.callbuf[i])
            kernel_clear(self)
        free(self.callbuf)

 */
    "pygpu/gpuarray.pyx":2349
        # We need to do all of this at the C level to avoid touching
        # python stuff that could be gone and to avoid exceptions
        if self.k.k is not NULL:
            res = gpukernel_property(self.k.k, GA_KERNEL_PROP_NUMARGS, &numargs)
            if res != GA_NO_ERROR:
 */
  "pygpu/gpuarray.pyx":2360
       free(self.callbuf[i])
            kernel_clear(self)
        free(self.callbuf)

    def __reduce__(self):
 */
  "pygpu/gpuarray.pyx":2342

    """
    def __dealloc__(self):
        cdef unsigned int numargs
        cdef int *types
 */
"pygpu/gpuarray.pyx":2362
        free(self.callbuf)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuKernel object"

 */
  "pygpu/gpuarray.pyx":2363

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuKernel object"

    def __cinit__(self, source, name, types, GpuContext context=None,
 */
  "pygpu/gpuarray.pyx":2362
        free(self.callbuf)

    def __reduce__(self):
        raise RuntimeError, "Cannot pickle GpuKernel object"

 */
"pygpu/gpuarray.pyx":2365
        raise RuntimeError, "Cannot pickle GpuKernel object"

    def __cinit__(self, source, name, types, GpuContext context=None,
     have_double=False, have_small=False, have_complex=False,
     have_half=False, cuda=False, opencl=False, *a, **kwa):
 */
    "pygpu/gpuarray.pyx":2366

    def __cinit__(self, source, name, types, GpuContext context=None,
     have_double=False, have_small=False, have_complex=False,
     have_half=False, cuda=False, opencl=False, *a, **kwa):
        cdef const char *s[1]
 */
    "pygpu/gpuarray.pyx":2367
    def __cinit__(self, source, name, types, GpuContext context=None,
     have_double=False, have_small=False, have_complex=False,
     have_half=False, cuda=False, opencl=False, *a, **kwa):
        cdef const char *s[1]
        cdef size_t l
 */
  "pygpu/gpuarray.pyx":2365
        raise RuntimeError, "Cannot pickle GpuKernel object"

    def __cinit__(self, source, name, types, GpuContext context=None,
     have_double=False, have_small=False, have_complex=False,
     have_half=False, cuda=False, opencl=False, *a, **kwa):
 */
  "pygpu/gpuarray.pyx":2373
        cdef unsigned int i
        cdef int *_types
        cdef int flags = 0

        source = _s(source)
 */
  "pygpu/gpuarray.pyx":2375
        cdef int flags = 0

        source = _s(source)
        name = _s(name)

 */
  "pygpu/gpuarray.pyx":2376

        source = _s(source)
        name = _s(name)

        self.context = ensure_context(context)
 */
  "pygpu/gpuarray.pyx":2378
        name = _s(name)

        self.context = ensure_context(context)

        if have_double:
 */
  "pygpu/gpuarray.pyx":2380
        self.context = ensure_context(context)

        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
 */
    "pygpu/gpuarray.pyx":2381

        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
            flags |= GA_USE_SMALL
 */
    "pygpu/gpuarray.pyx":2380
        self.context = ensure_context(context)

        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
 */
  "pygpu/gpuarray.pyx":2382
        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
 */
    "pygpu/gpuarray.pyx":2383
            flags |= GA_USE_DOUBLE
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
            flags |= GA_USE_COMPLEX
 */
    "pygpu/gpuarray.pyx":2382
        if have_double:
            flags |= GA_USE_DOUBLE
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
 */
  "pygpu/gpuarray.pyx":2384
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
 */
    "pygpu/gpuarray.pyx":2385
            flags |= GA_USE_SMALL
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
            flags |= GA_USE_HALF
 */
    "pygpu/gpuarray.pyx":2384
        if have_small:
            flags |= GA_USE_SMALL
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
 */
  "pygpu/gpuarray.pyx":2386
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
            flags |= GA_USE_HALF
        if cuda:
 */
    "pygpu/gpuarray.pyx":2387
            flags |= GA_USE_COMPLEX
        if have_half:
            flags |= GA_USE_HALF
        if cuda:
            flags |= GA_USE_CUDA
 */
    "pygpu/gpuarray.pyx":2386
        if have_complex:
            flags |= GA_USE_COMPLEX
        if have_half:
            flags |= GA_USE_HALF
        if cuda:
 */
  "pygpu/gpuarray.pyx":2388
        if have_half:
            flags |= GA_USE_HALF
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
 */
    "pygpu/gpuarray.pyx":2389
            flags |= GA_USE_HALF
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL
 */
    "pygpu/gpuarray.pyx":2388
        if have_half:
            flags |= GA_USE_HALF
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
 */
  "pygpu/gpuarray.pyx":2390
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL

 */
    "pygpu/gpuarray.pyx":2391
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL

        s[0] = source
 */
    "pygpu/gpuarray.pyx":2390
        if cuda:
            flags |= GA_USE_CUDA
        if opencl:
            flags |= GA_USE_OPENCL

 */
  "pygpu/gpuarray.pyx":2393
            flags |= GA_USE_OPENCL

        s[0] = source
        l = len(source)
        numargs = <unsigned int>len(types)
 */
  "pygpu/gpuarray.pyx":2394

        s[0] = source
        l = len(source)
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
 */
  "pygpu/gpuarray.pyx":2395
        s[0] = source
        l = len(source)
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
 */
  "pygpu/gpuarray.pyx":2396
        l = len(source)
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError
 */
  "pygpu/gpuarray.pyx":2397
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
 */
    "pygpu/gpuarray.pyx":2398
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
 */
    "pygpu/gpuarray.pyx":2397
        numargs = <unsigned int>len(types)
        self.callbuf = <void **>calloc(len(types), sizeof(void *))
        if self.callbuf == NULL:
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
 */
  "pygpu/gpuarray.pyx":2399
        if self.callbuf == NULL:
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
            raise MemoryError
 */
  "pygpu/gpuarray.pyx":2400
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
            raise MemoryError
        try:
 */
    "pygpu/gpuarray.pyx":2401
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
            raise MemoryError
        try:
            for i in range(numargs):
 */
    "pygpu/gpuarray.pyx":2400
            raise MemoryError
        _types = <int *>calloc(numargs, sizeof(int))
        if _types == NULL:
            raise MemoryError
        try:
 */
  "pygpu/gpuarray.pyx":2402
        if _types == NULL:
            raise MemoryError
        try:
            for i in range(numargs):
   if (types[i] == GpuArray):
 */
    "pygpu/gpuarray.pyx":2403
            raise MemoryError
        try:
            for i in range(numargs):
   if (types[i] == GpuArray):
       _types[i] = GA_BUFFER
 */
      "pygpu/gpuarray.pyx":2404
        try:
            for i in range(numargs):
   if (types[i] == GpuArray):
       _types[i] = GA_BUFFER
   else:
 */
        "pygpu/gpuarray.pyx":2405
            for i in range(numargs):
   if (types[i] == GpuArray):
       _types[i] = GA_BUFFER
   else:
       _types[i] = dtype_to_typecode(types[i])
 */
        "pygpu/gpuarray.pyx":2404
        try:
            for i in range(numargs):
   if (types[i] == GpuArray):
       _types[i] = GA_BUFFER
   else:
 */
      "pygpu/gpuarray.pyx":2407
       _types[i] = GA_BUFFER
   else:
       _types[i] = dtype_to_typecode(types[i])
       self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
       if self.callbuf[i] == NULL:
 */
        "pygpu/gpuarray.pyx":2408
   else:
       _types[i] = dtype_to_typecode(types[i])
       self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
       if self.callbuf[i] == NULL:
           raise MemoryError
 */
        "pygpu/gpuarray.pyx":2409
       _types[i] = dtype_to_typecode(types[i])
       self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
       if self.callbuf[i] == NULL:
           raise MemoryError
            kernel_init(self, self.context.ctx, 1, s, &l,
 */
          "pygpu/gpuarray.pyx":2410
       self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
       if self.callbuf[i] == NULL:
           raise MemoryError
            kernel_init(self, self.context.ctx, 1, s, &l,
           name, numargs, _types, flags)
 */
          "pygpu/gpuarray.pyx":2409
       _types[i] = dtype_to_typecode(types[i])
       self.callbuf[i] = malloc(gpuarray_get_elsize(_types[i]))
       if self.callbuf[i] == NULL:
           raise MemoryError
            kernel_init(self, self.context.ctx, 1, s, &l,
 */
    "pygpu/gpuarray.pyx":2412
           raise MemoryError
            kernel_init(self, self.context.ctx, 1, s, &l,
           name, numargs, _types, flags)
        finally:
            free(_types)
 */
    "pygpu/gpuarray.pyx":2411
       if self.callbuf[i] == NULL:
           raise MemoryError
            kernel_init(self, self.context.ctx, 1, s, &l,
           name, numargs, _types, flags)
        finally:
 */
  "pygpu/gpuarray.pyx":2414
           name, numargs, _types, flags)
        finally:
            free(_types)

    def __call__(self, *args, n=None, gs=None, ls=None, shared=0):
 */
  "pygpu/gpuarray.pyx":2365
        raise RuntimeError, "Cannot pickle GpuKernel object"

    def __cinit__(self, source, name, types, GpuContext context=None,
     have_double=False, have_small=False, have_complex=False,
     have_half=False, cuda=False, opencl=False, *a, **kwa):
 */
"pygpu/gpuarray.pyx":2416
            free(_types)

    def __call__(self, *args, n=None, gs=None, ls=None, shared=0):
        """
        __call__(*args, n=None, gs=None, ls=None, shared=0)
 */
  "pygpu/gpuarray.pyx":2420
        __call__(*args, n=None, gs=None, ls=None, shared=0)
        """
        if n is None and (ls is None or gs is None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, gs, ls, args, shared)
 */
    "pygpu/gpuarray.pyx":2421
        """
        if n is None and (ls is None or gs is None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, gs, ls, args, shared)

 */
    "pygpu/gpuarray.pyx":2420
        __call__(*args, n=None, gs=None, ls=None, shared=0)
        """
        if n is None and (ls is None or gs is None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, gs, ls, args, shared)
 */
  "pygpu/gpuarray.pyx":2422
        if n is None and (ls is None or gs is None):
            raise ValueError, "Must specify size (n) or both gs and ls"
        self.do_call(n, gs, ls, args, shared)

    cdef do_call(self, py_n, py_gs, py_ls, py_args, size_t shared):
 */
  "pygpu/gpuarray.pyx":2416
            free(_types)

    def __call__(self, *args, n=None, gs=None, ls=None, shared=0):
        """
        __call__(*args, n=None, gs=None, ls=None, shared=0)
 */
"pygpu/gpuarray.pyx":2424
        self.do_call(n, gs, ls, args, shared)

    cdef do_call(self, py_n, py_gs, py_ls, py_args, size_t shared):
        cdef size_t n
        cdef size_t gs[3]
 */
  "pygpu/gpuarray.pyx":2434
        cdef unsigned int i

        nd = 0

        if py_ls is None:
 */
  "pygpu/gpuarray.pyx":2436
        nd = 0

        if py_ls is None:
            ls[0] = 0
            nd = 1
 */
    "pygpu/gpuarray.pyx":2437

        if py_ls is None:
            ls[0] = 0
            nd = 1
        else:
 */
    "pygpu/gpuarray.pyx":2438
        if py_ls is None:
            ls[0] = 0
            nd = 1
        else:
            if isinstance(py_ls, int):
 */
    "pygpu/gpuarray.pyx":2436
        nd = 0

        if py_ls is None:
            ls[0] = 0
            nd = 1
 */
  "pygpu/gpuarray.pyx":2440
            nd = 1
        else:
            if isinstance(py_ls, int):
   ls[0] = py_ls
   nd = 1
 */
      "pygpu/gpuarray.pyx":2441
        else:
            if isinstance(py_ls, int):
   ls[0] = py_ls
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
 */
      "pygpu/gpuarray.pyx":2442
            if isinstance(py_ls, int):
   ls[0] = py_ls
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
 */
      "pygpu/gpuarray.pyx":2440
            nd = 1
        else:
            if isinstance(py_ls, int):
   ls[0] = py_ls
   nd = 1
 */
    "pygpu/gpuarray.pyx":2443
   ls[0] = py_ls
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
 */
      "pygpu/gpuarray.pyx":2444
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
   nd = len(py_ls)
 */
        "pygpu/gpuarray.pyx":2445
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
   nd = len(py_ls)

 */
        "pygpu/gpuarray.pyx":2444
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
   nd = len(py_ls)
 */
      "pygpu/gpuarray.pyx":2446
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
   nd = len(py_ls)

   if nd >= 3:
 */
      "pygpu/gpuarray.pyx":2448
   nd = len(py_ls)

   if nd >= 3:
       ls[2] = py_ls[2]
   if nd >= 2:
 */
        "pygpu/gpuarray.pyx":2449

   if nd >= 3:
       ls[2] = py_ls[2]
   if nd >= 2:
       ls[1] = py_ls[1]
 */
        "pygpu/gpuarray.pyx":2448
   nd = len(py_ls)

   if nd >= 3:
       ls[2] = py_ls[2]
   if nd >= 2:
 */
      "pygpu/gpuarray.pyx":2450
   if nd >= 3:
       ls[2] = py_ls[2]
   if nd >= 2:
       ls[1] = py_ls[1]
   if nd >= 1:
 */
        "pygpu/gpuarray.pyx":2451
       ls[2] = py_ls[2]
   if nd >= 2:
       ls[1] = py_ls[1]
   if nd >= 1:
       ls[0] = py_ls[0]
 */
        "pygpu/gpuarray.pyx":2450
   if nd >= 3:
       ls[2] = py_ls[2]
   if nd >= 2:
       ls[1] = py_ls[1]
   if nd >= 1:
 */
      "pygpu/gpuarray.pyx":2452
   if nd >= 2:
       ls[1] = py_ls[1]
   if nd >= 1:
       ls[0] = py_ls[0]
            else:
 */
        "pygpu/gpuarray.pyx":2453
       ls[1] = py_ls[1]
   if nd >= 1:
       ls[0] = py_ls[0]
            else:
   raise TypeError, "ls is not int or list"
 */
        "pygpu/gpuarray.pyx":2452
   if nd >= 2:
       ls[1] = py_ls[1]
   if nd >= 1:
       ls[0] = py_ls[0]
            else:
 */
      "pygpu/gpuarray.pyx":2443
   ls[0] = py_ls
   nd = 1
            elif isinstance(py_ls, (list, tuple)):
   if len(py_ls) > 3:
       raise ValueError, "ls is not of length 3 or less"
 */
    "pygpu/gpuarray.pyx":2455
       ls[0] = py_ls[0]
            else:
   raise TypeError, "ls is not int or list"

        if py_gs is None:
 */
  "pygpu/gpuarray.pyx":2457
   raise TypeError, "ls is not int or list"

        if py_gs is None:
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
 */
    "pygpu/gpuarray.pyx":2458

        if py_gs is None:
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
            gs[0] = 0
 */
      "pygpu/gpuarray.pyx":2459
        if py_gs is None:
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
            gs[0] = 0
        else:
 */
      "pygpu/gpuarray.pyx":2458

        if py_gs is None:
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
            gs[0] = 0
 */
    "pygpu/gpuarray.pyx":2460
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
            gs[0] = 0
        else:
            if isinstance(py_gs, int):
 */
    "pygpu/gpuarray.pyx":2457
   raise TypeError, "ls is not int or list"

        if py_gs is None:
            if nd != 1:
   raise ValueError, "nd mismatch for gs (None)"
 */
  "pygpu/gpuarray.pyx":2462
            gs[0] = 0
        else:
            if isinstance(py_gs, int):
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
 */
      "pygpu/gpuarray.pyx":2463
        else:
            if isinstance(py_gs, int):
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
 */
        "pygpu/gpuarray.pyx":2464
            if isinstance(py_gs, int):
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
 */
        "pygpu/gpuarray.pyx":2463
        else:
            if isinstance(py_gs, int):
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
 */
      "pygpu/gpuarray.pyx":2465
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
 */
      "pygpu/gpuarray.pyx":2462
            gs[0] = 0
        else:
            if isinstance(py_gs, int):
   if nd != 1:
       raise ValueError, "nd mismatch for gs (int)"
 */
    "pygpu/gpuarray.pyx":2466
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
 */
      "pygpu/gpuarray.pyx":2467
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
 */
        "pygpu/gpuarray.pyx":2468
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
       raise ValueError, "nd mismatch for gs (tuple)"
 */
        "pygpu/gpuarray.pyx":2467
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
 */
      "pygpu/gpuarray.pyx":2469
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
       raise ValueError, "nd mismatch for gs (tuple)"

 */
        "pygpu/gpuarray.pyx":2470
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
       raise ValueError, "nd mismatch for gs (tuple)"

   if nd >= 3:
 */
        "pygpu/gpuarray.pyx":2469
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
   if len(py_ls) != nd:
       raise ValueError, "nd mismatch for gs (tuple)"

 */
      "pygpu/gpuarray.pyx":2472
       raise ValueError, "nd mismatch for gs (tuple)"

   if nd >= 3:
       gs[2] = py_gs[2]
   if nd >= 2:
 */
        "pygpu/gpuarray.pyx":2473

   if nd >= 3:
       gs[2] = py_gs[2]
   if nd >= 2:
       gs[1] = py_gs[1]
 */
        "pygpu/gpuarray.pyx":2472
       raise ValueError, "nd mismatch for gs (tuple)"

   if nd >= 3:
       gs[2] = py_gs[2]
   if nd >= 2:
 */
      "pygpu/gpuarray.pyx":2474
   if nd >= 3:
       gs[2] = py_gs[2]
   if nd >= 2:
       gs[1] = py_gs[1]
   if nd >= 1:
 */
        "pygpu/gpuarray.pyx":2475
       gs[2] = py_gs[2]
   if nd >= 2:
       gs[1] = py_gs[1]
   if nd >= 1:
       gs[0] = py_gs[0]
 */
        "pygpu/gpuarray.pyx":2474
   if nd >= 3:
       gs[2] = py_gs[2]
   if nd >= 2:
       gs[1] = py_gs[1]
   if nd >= 1:
 */
      "pygpu/gpuarray.pyx":2476
   if nd >= 2:
       gs[1] = py_gs[1]
   if nd >= 1:
       gs[0] = py_gs[0]
            else:
 */
        "pygpu/gpuarray.pyx":2477
       gs[1] = py_gs[1]
   if nd >= 1:
       gs[0] = py_gs[0]
            else:
   raise TypeError, "gs is not int or list"
 */
        "pygpu/gpuarray.pyx":2476
   if nd >= 2:
       gs[1] = py_gs[1]
   if nd >= 1:
       gs[0] = py_gs[0]
            else:
 */
      "pygpu/gpuarray.pyx":2466
       raise ValueError, "nd mismatch for gs (int)"
   gs[0] = py_gs
            elif isinstance(py_gs, (list, tuple)):
   if len(py_gs) > 3:
       raise ValueError, "gs is not of length 3 or less"
 */
    "pygpu/gpuarray.pyx":2479
       gs[0] = py_gs[0]
            else:
   raise TypeError, "gs is not int or list"

        numargs = self.numargs
 */
  "pygpu/gpuarray.pyx":2481
   raise TypeError, "gs is not int or list"

        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
 */
  "pygpu/gpuarray.pyx":2482

        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
 */
    "pygpu/gpuarray.pyx":2483
        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
 */
    "pygpu/gpuarray.pyx":2482

        numargs = self.numargs
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
 */
  "pygpu/gpuarray.pyx":2484
        if len(py_args) != numargs:
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
 */
  "pygpu/gpuarray.pyx":2485
            raise TypeError, "Expected %d arguments, got %d," % (numargs, len(py_args))
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
 */
    "pygpu/gpuarray.pyx":2486
        kernel_property(self, GA_KERNEL_PROP_TYPES, &types)
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
 */
  "pygpu/gpuarray.pyx":2487
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
 */
    "pygpu/gpuarray.pyx":2488
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
            n = py_n
 */
      "pygpu/gpuarray.pyx":2489
        if py_n is not None:
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
            n = py_n
            kernel_sched(self, n, &gs[0], &ls[0])
 */
      "pygpu/gpuarray.pyx":2488
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
            n = py_n
 */
    "pygpu/gpuarray.pyx":2490
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
            n = py_n
            kernel_sched(self, n, &gs[0], &ls[0])
        kernel_call(self, nd, gs, ls, shared, self.callbuf)
 */
    "pygpu/gpuarray.pyx":2491
   raise ValueError, "n is specified and nd != 1"
            n = py_n
            kernel_sched(self, n, &gs[0], &ls[0])
        kernel_call(self, nd, gs, ls, shared, self.callbuf)

 */
    "pygpu/gpuarray.pyx":2487
        for i in range(numargs):
            self._setarg(i, types[i], py_args[i])
        if py_n is not None:
            if nd != 1:
   raise ValueError, "n is specified and nd != 1"
 */
  "pygpu/gpuarray.pyx":2492
            n = py_n
            kernel_sched(self, n, &gs[0], &ls[0])
        kernel_call(self, nd, gs, ls, shared, self.callbuf)

    cdef _setarg(self, unsigned int index, int typecode, object o):
 */
  "pygpu/gpuarray.pyx":2424
        self.do_call(n, gs, ls, args, shared)

    cdef do_call(self, py_n, py_gs, py_ls, py_args, size_t shared):
        cdef size_t n
        cdef size_t gs[3]
 */
"pygpu/gpuarray.pyx":2494
        kernel_call(self, nd, gs, ls, shared, self.callbuf)

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
 */
  "pygpu/gpuarray.pyx":2495

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
 */
    "pygpu/gpuarray.pyx":2496
    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
 */
      "pygpu/gpuarray.pyx":2497
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
 */
      "pygpu/gpuarray.pyx":2496
    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
 */
    "pygpu/gpuarray.pyx":2498
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2495

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
   raise TypeError, "expected a GpuArray"
 */
    "pygpu/gpuarray.pyx":2499
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
 */
    "pygpu/gpuarray.pyx":2500
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2499
   raise TypeError, "expected a GpuArray"
            self.callbuf[index] = <void *>((<GpuArray>o).ga.data)
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
 */
    "pygpu/gpuarray.pyx":2501
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
 */
    "pygpu/gpuarray.pyx":2502
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2501
        elif typecode == GA_SIZE:
            (<size_t *>self.callbuf[index])[0] = o
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
 */
    "pygpu/gpuarray.pyx":2503
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
 */
    "pygpu/gpuarray.pyx":2504
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2503
        elif typecode == GA_SSIZE:
            (<ssize_t *>self.callbuf[index])[0] = o
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
 */
    "pygpu/gpuarray.pyx":2505
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
 */
    "pygpu/gpuarray.pyx":2506
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2505
        elif typecode == GA_FLOAT:
            (<float *>self.callbuf[index])[0] = o
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
 */
    "pygpu/gpuarray.pyx":2507
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
 */
    "pygpu/gpuarray.pyx":2508
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2507
        elif typecode == GA_DOUBLE:
            (<double *>self.callbuf[index])[0] = o
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
 */
    "pygpu/gpuarray.pyx":2509
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
 */
    "pygpu/gpuarray.pyx":2510
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2509
        elif typecode == GA_BYTE:
            (<signed char *>self.callbuf[index])[0] = o
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
 */
    "pygpu/gpuarray.pyx":2511
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
 */
    "pygpu/gpuarray.pyx":2512
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2511
        elif typecode == GA_UBYTE:
            (<unsigned char *>self.callbuf[index])[0] = o
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
 */
    "pygpu/gpuarray.pyx":2513
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
 */
    "pygpu/gpuarray.pyx":2514
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2513
        elif typecode == GA_SHORT:
            (<short *>self.callbuf[index])[0] = o
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
 */
    "pygpu/gpuarray.pyx":2515
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
 */
    "pygpu/gpuarray.pyx":2516
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2515
        elif typecode == GA_USHORT:
            (<unsigned short *>self.callbuf[index])[0] = o
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
 */
    "pygpu/gpuarray.pyx":2517
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
 */
    "pygpu/gpuarray.pyx":2518
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2517
        elif typecode == GA_INT:
            (<int *>self.callbuf[index])[0] = o
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
 */
    "pygpu/gpuarray.pyx":2519
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
 */
    "pygpu/gpuarray.pyx":2520
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
            (<unsigned long *>self.callbuf[index])[0] = o
 */
    "pygpu/gpuarray.pyx":2519
        elif typecode == GA_UINT:
            (<unsigned int *>self.callbuf[index])[0] = o
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
 */
    "pygpu/gpuarray.pyx":2521
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
            (<unsigned long *>self.callbuf[index])[0] = o
        else:
 */
    "pygpu/gpuarray.pyx":2522
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
            (<unsigned long *>self.callbuf[index])[0] = o
        else:
            raise ValueError("Bad typecode in _setarg: %d "
 */
    "pygpu/gpuarray.pyx":2521
        elif typecode == GA_LONG:
            (<long *>self.callbuf[index])[0] = o
        elif typecode == GA_ULONG:
            (<unsigned long *>self.callbuf[index])[0] = o
        else:
 */
    "pygpu/gpuarray.pyx":2525
        else:
            raise ValueError("Bad typecode in _setarg: %d "
   "(please report this, it is a bug)" % (typecode,))

    property maxlsize:
 */
    "pygpu/gpuarray.pyx":2524
            (<unsigned long *>self.callbuf[index])[0] = o
        else:
            raise ValueError("Bad typecode in _setarg: %d "
   "(please report this, it is a bug)" % (typecode,))

 */
  "pygpu/gpuarray.pyx":2494
        kernel_call(self, nd, gs, ls, shared, self.callbuf)

    cdef _setarg(self, unsigned int index, int typecode, object o):
        if typecode == GA_BUFFER:
            if not isinstance(o, GpuArray):
 */
"pygpu/gpuarray.pyx":2529
    property maxlsize:
        "Maximum local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXLSIZE, &res)
 */
  "pygpu/gpuarray.pyx":2531
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXLSIZE, &res)
            return res

 */
  "pygpu/gpuarray.pyx":2532
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXLSIZE, &res)
            return res

    property preflsize:
 */
  "pygpu/gpuarray.pyx":2529
    property maxlsize:
        "Maximum local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_MAXLSIZE, &res)
 */
"pygpu/gpuarray.pyx":2536
    property preflsize:
        "Preferred multiple for local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_PREFLSIZE, &res)
 */
  "pygpu/gpuarray.pyx":2538
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_PREFLSIZE, &res)
            return res

 */
  "pygpu/gpuarray.pyx":2539
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_PREFLSIZE, &res)
            return res

    property numargs:
 */
  "pygpu/gpuarray.pyx":2536
    property preflsize:
        "Preferred multiple for local size for this kernel"
        def __get__(self):
            cdef size_t res
            kernel_property(self, GA_KERNEL_PROP_PREFLSIZE, &res)
 */
"pygpu/gpuarray.pyx":2543
    property numargs:
        "Number of arguments to kernel"
        def __get__(self):
            cdef unsigned int res
            kernel_property(self, GA_KERNEL_PROP_NUMARGS, &res)
 */
  "pygpu/gpuarray.pyx":2545
        def __get__(self):
            cdef unsigned int res
            kernel_property(self, GA_KERNEL_PROP_NUMARGS, &res)
            return res

 */
  "pygpu/gpuarray.pyx":2546
            cdef unsigned int res
            kernel_property(self, GA_KERNEL_PROP_NUMARGS, &res)
            return res

 */
  "pygpu/gpuarray.pyx":2543
    property numargs:
        "Number of arguments to kernel"
        def __get__(self):
            cdef unsigned int res
            kernel_property(self, GA_KERNEL_PROP_NUMARGS, &res)
 */
  "pygpu/gpuarray.pyx":43
    if isinstance(s, bytes):
        return s
    raise TypeError("Expected a string")

cdef size_t countis(l, object val):
 */
  "pygpu/gpuarray.pyx":564
    cdef GpuContext res

    if dev.startswith('cuda'):
        kind = b"cuda"
        if dev[4:] == '':
 */
  "pygpu/gpuarray.pyx":566
    if dev.startswith('cuda'):
        kind = b"cuda"
        if dev[4:] == '':
            devnum = -1
        else:
 */
  "pygpu/gpuarray.pyx":569
            devnum = -1
        else:
            devnum = int(dev[4:])
        gpucontext_props_cuda_dev(p, devnum)
    elif dev.startswith('opencl'):
 */
  "pygpu/gpuarray.pyx":571
            devnum = int(dev[4:])
        gpucontext_props_cuda_dev(p, devnum)
    elif dev.startswith('opencl'):
        kind = b"opencl"
        devspec = dev[6:].split(':')
 */
  "pygpu/gpuarray.pyx":573
    elif dev.startswith('opencl'):
        kind = b"opencl"
        devspec = dev[6:].split(':')
        if len(devspec) < 2:
            raise ValueError, "OpenCL name incorrect. Should be opencl<int>:<int> instead got: " + dev
 */
  "pygpu/gpuarray.pyx":988
            if arg.ga.nd < ndmin:
   shp = arg.shape
   idx = (1,)*(ndmin-len(shp))
   shp = idx + shp
   arg = arg.reshape(shp)
 */
  "pygpu/gpuarray.pyx":996
        shp = arg.shape
        if len(shp) < ndmin:
            idx = (1,)*(ndmin-len(shp))
            shp = idx + shp
        if order is None or order == 'A':
 */
  "pygpu/gpuarray.pyx":1073
    def __enter__(self):
        if cuda_enter == NULL:
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
 */
  "pygpu/gpuarray.pyx":1075
            raise RuntimeError("cuda_enter not available")
        if cuda_exit == NULL:
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
 */
  "pygpu/gpuarray.pyx":1077
            raise RuntimeError("cuda_exit not available")
        if self.kind != b"cuda":
            raise ValueError("Context manager only works for cuda")
        cuda_enter(self.ctx)
        return self
 */
  "pygpu/gpuarray.pyx":1205

        if isinstance(idx, unicode):
            idx = idx.encode('UTF-8')
        if isinstance(idx, bytes):
            key = idx
 */
  "pygpu/gpuarray.pyx":1270
    def __repr__(self):
        return '\n'.join(" %s : %s" % (name.upper(), getattr(self, name))
            for name in ["c_contiguous", "f_contiguous",
            "owndata", "writeable", "aligned",
            "updateifcopy"])
 */
  "pygpu/gpuarray.pyx":1428
    cdef unsigned int caxis = <unsigned int>compute_axis
    if caxis >= nd:
        raise ValueError("compute_axis is out of bounds")

    cdef size_t *cdims
 */
  "pygpu/gpuarray.pyx":1746
            return bool(numpy.asarray(self))
        else:
            raise ValueError('The truth value of a multi-element array is ambiguous')

    def _empty_like_me(self, dtype=None, order='C'):
 */
  "pygpu/gpuarray.pyx":1786
        if not GpuArray_ISONESEGMENT(&self.ga):
            # For now raise an error, may make it work later
            raise ValueError("transfer() only works for contigous source")
        r = pygpu_empty(self.ga.nd, self.ga.dimensions, self.ga.typecode,
           GA_C_ORDER if GpuArray_IS_C_CONTIGUOUS(&self.ga) else GA_F_ORDER,
 */
  "pygpu/gpuarray.pyx":1909
        if len(params) is 1 and isinstance(params[0], (tuple, list)):
            params = params[0]
        if params is () or params == (None,):
            return pygpu_transpose(self, NULL)
        else:
 */
  "pygpu/gpuarray.pyx":1964
        # is also required for numpy compat.
        try:
            ell_idx = key.index(Ellipsis)
        except ValueError:
            pass
 */
  "pygpu/gpuarray.pyx":1971
            # objects, not counting None entries and the Ellipsis itself
            num_slcs = self.ga.nd - (len(key) - countis(key, None) - 1)
            fill_slices = (slice(None),)num_slcs
            key = key[:ell_idx] + fill_slices + key[ell_idx + 1:]

 */
  "pygpu/gpuarray.pyx":1983
        if len(getitem_idcs) <= 1:
            getitem_idcs = (getitem_idcs +
  (slice(None),)(self.ga.nd - len(getitem_idcs)))

        # Slice into array, then reshape, accommodating for None entries in key
 */
  "pygpu/gpuarray.pyx":2040
       # a[..., 1:] on any array (including 1-dim).  This
       # is also required for numpy compat.
       el = key.index(Ellipsis)
       if isinstance(key, tuple):
           key = (key[:el] +
 */
  "pygpu/gpuarray.pyx":2043
       if isinstance(key, tuple):
           key = (key[:el] +
     (Ellipsis,)*(self.ga.nd - (len(key) - 1)) +
     key[el+1:])
       else:
 */
  "pygpu/gpuarray.pyx":2180
            cdef unsigned int i
            if len(newstrides) != self.ga.nd:
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
 */
  "pygpu/gpuarray.pyx":2182
   raise ValueError("new strides are the wrong length")
            if not strides_ok(self,  newstrides):
   raise ValueError("new strides go outside of allocated memory")
            for i in range(self.ga.nd):
   self.ga.strides[i] = newstrides[i]
 */
  "pygpu/gpuarray.pyx":2230
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
 */
  "pygpu/gpuarray.pyx":2232
   raise TypeError("This is for OpenCL arrays.")
            if self.offset != 0:
   raise ValueError("This array has an offset.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
  "pygpu/gpuarray.pyx":2242
        def __get__(self):
            if self.context.kind != b"opencl":
   raise TypeError("This is for OpenCL arrays.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
  "pygpu/gpuarray.pyx":2252
        def __get__(self):
            if self.context.kind != b"cuda":
   raise TypeError("This is for CUDA arrays.")
            # This wizadry grabs the actual backend pointer since it's
            # guarenteed to be the first element of the gpudata
 */
  "pygpu/gpuarray.pyx":14
from cpython.object cimport Py_EQ, Py_NE

def api_version():
    """api_version()
    """
 */
  "pygpu/gpuarray.pyx":20
    return (GPUARRAY_API_VERSION, 0)

def abi_version():
    """abi_version()
    """
 */
  "pygpu/gpuarray.pyx":54
    return count

def cl_wrap_ctx(size_t ptr):
    """
    cl_wrap_ctx(ptr)
 */
  "pygpu/gpuarray.pyx":72
    return res

def cuda_wrap_ctx(size_t ptr, bint own):
    """
    cuda_wrap_ctx(ptr)
 */
  "pygpu/gpuarray.pyx":102

cdef dict NP_TO_TYPE = {
    np.dtype('bool'): GA_BOOL,
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
 */
  "pygpu/gpuarray.pyx":103
cdef dict NP_TO_TYPE = {
    np.dtype('bool'): GA_BOOL,
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
 */
  "pygpu/gpuarray.pyx":104
    np.dtype('bool'): GA_BOOL,
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
 */
  "pygpu/gpuarray.pyx":105
    np.dtype('int8'): GA_BYTE,
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
 */
  "pygpu/gpuarray.pyx":106
    np.dtype('uint8'): GA_UBYTE,
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
 */
  "pygpu/gpuarray.pyx":107
    np.dtype('int16'): GA_SHORT,
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
 */
  "pygpu/gpuarray.pyx":108
    np.dtype('uint16'): GA_USHORT,
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
 */
  "pygpu/gpuarray.pyx":109
    np.dtype('int32'): GA_INT,
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
 */
  "pygpu/gpuarray.pyx":110
    np.dtype('uint32'): GA_UINT,
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
 */
  "pygpu/gpuarray.pyx":111
    np.dtype('int64'): GA_LONG,
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
 */
  "pygpu/gpuarray.pyx":112
    np.dtype('uint64'): GA_ULONG,
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
 */
  "pygpu/gpuarray.pyx":113
    np.dtype('float32'): GA_FLOAT,
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
    np.dtype('float16'): GA_HALF,
 */
  "pygpu/gpuarray.pyx":114
    np.dtype('float64'): GA_DOUBLE,
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
    np.dtype('float16'): GA_HALF,
}
 */
  "pygpu/gpuarray.pyx":115
    np.dtype('complex64'): GA_CFLOAT,
    np.dtype('complex128'): GA_CDOUBLE,
    np.dtype('float16'): GA_HALF,
}

 */

  "pygpu/gpuarray.pyx":211
    raise ValueError, "don't know how to convert to dtype: %s"%(dtype,)

def dtype_to_ctype(dtype):
    """
    dtype_to_ctype(dtype)
 */
  "pygpu/gpuarray.pyx":489
        raise get_exc(err), gpucontext_error(c.ctx, err)

def set_default_context(GpuContext ctx):
    """
    set_default_context(ctx)
 */
  "pygpu/gpuarray.pyx":515
    default_context = ctx

def get_default_context():
    """
    get_default_context()
 */
  "pygpu/gpuarray.pyx":534
    return isinstance(o, GpuArray)

def count_platforms(kind):
    """
    count_platforms(kind)
 */
  "pygpu/gpuarray.pyx":547
    return platcount

def count_devices(kind, unsigned int platform):
    """
    count_devices(kind, platform)
 */
  "pygpu/gpuarray.pyx":590
    return res

def init(dev, sched='default', single_stream=False, kernel_cache_path=None,
         max_cache_size=sys.maxsize, initial_cache_size=0):
    """
 */
  "pygpu/gpuarray.pyx":660
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":725
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":768
        free(cdims)

def asarray(a, dtype=None, order='A', GpuContext context=None):
    """
    asarray(a, dtype=None, order='A', context=None)
 */
  "pygpu/gpuarray.pyx":797
    cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
    """
    ascontiguousarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":819
    context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
    """
    asfortranarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":841
    context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    """
    may_share_memory(a, b)
 */
  "pygpu/gpuarray.pyx":849
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":931
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":1467
    return 0

def _split(GpuArray a, ind, unsigned int axis):
    """
    _split(a, ind, axis)
 */
  "pygpu/gpuarray.pyx":1505
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
    GpuContext context):
    """
 */
  "pygpu/gpuarray.pyx":1530
cuda_open_ipc_handle = <gpudata *(*)(gpucontext *, GpuArrayIpcMemHandle *, size_t)>gpuarray_get_extension("cuda_open_ipc_handle")

def open_ipc_handle(GpuContext c, bytes hpy, size_t l):
    """
    open_ipc_handle(c, hpy, l)
 */
  



  "pygpu/gpuarray.pyx":211
    raise ValueError, "don't know how to convert to dtype: %s"%(dtype,)

def dtype_to_ctype(dtype):
    """
    dtype_to_ctype(dtype)
 */
  "pygpu/gpuarray.pyx":270


class GpuArrayException(Exception):
    """
    Exception used for most errors related to libgpuarray.
 */
  "pygpu/gpuarray.pyx":275
    """

class UnsupportedException(GpuArrayException):
    pass

 */
  "pygpu/gpuarray.pyx":481
    return default_context

cdef GpuContext default_context = None

cdef int ctx_property(GpuContext c, int prop_id, void *res) except -1:
 */
  "pygpu/gpuarray.pyx":489
        raise get_exc(err), gpucontext_error(c.ctx, err)

def set_default_context(GpuContext ctx):
    """
    set_default_context(ctx)
 */
  "pygpu/gpuarray.pyx":515
    default_context = ctx

def get_default_context():
    """
    get_default_context()
 */
  "pygpu/gpuarray.pyx":534
    return isinstance(o, GpuArray)

def count_platforms(kind):
    """
    count_platforms(kind)
 */
  "pygpu/gpuarray.pyx":547
    return platcount

def count_devices(kind, unsigned int platform):
    """
    count_devices(kind, platform)
 */
  "pygpu/gpuarray.pyx":591

def init(dev, sched='default', single_stream=False, kernel_cache_path=None,
         max_cache_size=sys.maxsize, initial_cache_size=0):
    """
    init(dev, sched='default', single_stream=False, kernel_cache_path=None,
 */
  "pygpu/gpuarray.pyx":590
    return res

def init(dev, sched='default', single_stream=False, kernel_cache_path=None,
         max_cache_size=sys.maxsize, initial_cache_size=0):
    """
 */
  "pygpu/gpuarray.pyx":660
    return pygpu_init(dev, p)

def zeros(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":725
    return 0

def empty(shape, dtype=GA_DOUBLE, order='C', GpuContext context=None,
          cls=None):
    """
 */
  "pygpu/gpuarray.pyx":768
        free(cdims)

def asarray(a, dtype=None, order='A', GpuContext context=None):
    """
    asarray(a, dtype=None, order='A', context=None)
 */
  "pygpu/gpuarray.pyx":797
    cls=GpuArray)

def ascontiguousarray(a, dtype=None, GpuContext context=None):
    """
    ascontiguousarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":819
    context=context)

def asfortranarray(a, dtype=None, GpuArray context=None):
    """
    asfortranarray(a, dtype=None, context=None)
 */
  "pygpu/gpuarray.pyx":841
    context=context)

def may_share_memory(GpuArray a not None, GpuArray b not None):
    """
    may_share_memory(a, b)
 */
  "pygpu/gpuarray.pyx":849
    return array_share(a, b)

def from_gpudata(size_t data, offset, dtype, shape, GpuContext context=None,
    strides=None, writable=True, base=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":931
        free(cstrides)

def array(proto, dtype=None, copy=True, order=None, unsigned int ndmin=0,
          GpuContext context=None, cls=None):
    """
 */
  "pygpu/gpuarray.pyx":1033
cdef void (*cuda_exit)(gpucontext *)

cuda_enter = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_enter")
cuda_exit = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_exit")

 */
  "pygpu/gpuarray.pyx":1034

cuda_enter = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_enter")
cuda_exit = <void (*)(gpucontext *)>gpuarray_get_extension("cuda_exit")

cdef class GpuContext:
 */
  "pygpu/gpuarray.pyx":1467
    return 0

def _split(GpuArray a, ind, unsigned int axis):
    """
    _split(a, ind, axis)
 */
  "pygpu/gpuarray.pyx":1505
    return res

def _concatenate(list al, unsigned int axis, int restype, object cls,
    GpuContext context):
    """
 */
  "pygpu/gpuarray.pyx":1527
cdef gpudata *(*cuda_open_ipc_handle)(gpucontext *, GpuArrayIpcMemHandle *, size_t)

cuda_get_ipc_handle = <int (*)(gpudata *, GpuArrayIpcMemHandle *)>gpuarray_get_extension("cuda_get_ipc_handle")
cuda_open_ipc_handle = <gpudata *(*)(gpucontext *, GpuArrayIpcMemHandle *, size_t)>gpuarray_get_extension("cuda_open_ipc_handle")

def open_ipc_handle(GpuContext c, bytes hpy, size_t l):
    """
    open_ipc_handle(c, hpy, l)
 */

  "pygpu/gpuarray.pyx":1813
        pygpu_sync(self)

    def view(self, object cls=GpuArray):
        """
        view(cls=GpuArray)
 */
  "pygpu/gpuarray.pyx":1
cimport libc.stdio
from libc.stdlib cimport malloc, calloc, free
from cpython.mem cimport PyMem_Malloc, PyMem_Free
 */
