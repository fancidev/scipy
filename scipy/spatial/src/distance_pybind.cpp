// This file contains the C++ implementation of the distance metrics.
//
// This improves performance over a pure Python implementation by eliminating
// the Python loop overhead and in some cases reducing the space complexity.
// The C++ implementation works exclusively with NumPy arrays.
//
// Entry point
// -----------
// For each distance metric <metric>, the entry point has the signature
//
//   xdist_<metric>(return_type, x, y, *, out=None, **kwargs)
//
// where
//
//   - `return_type` specifies the part of result to return (see below);
//   - `x` is a p-by-n array containing p observations each of length n;
//   - `y` is a q-by-n array containing q observations each of length n;
//   - `w` is a metric-specific weight scalar, n-vector, or n-by-n matrix;
//   - `out` is a contiguous output buffer with proper shape and dtype; and
//   - `kwargs` contain extra metric-specific parameters, such as `w` for
//      weight vector and `p` for minkowski order.
//
// Conceptually, the function computes a p-by-q matrix of distances between
// each vector in `x` and `y`.  The `return_type` argument specifies the part
// of this matrix to return; it can be one of the following:
//
//   RETURN_FULL    return the full p-by-q distance matrix; used by 'cdist'
//   RETURN_UPPER   return the upper half of the (square) distance matrix,
//                  excluding the diagonal, as a vector traversed in row-major
//                  order; used by 'pdist'
//   RETURN_DIAG    return the diagonal of the (square) distance matrix as
//                  a vector; reserved for future use
//
// In addition, a few helper functions are exported for use by the Python code
// to perform input validation.
//
// Type promotion
// --------------
// There are three kinds of dtypes used by the computation routines: the
// return type, the working type, and the storage type.  These types are
// determined from the dtype of x, y, and optionally extra dtypes (such as
// that of w).
//
// The return type is the dtype of the returned distance matrix (or vector).
// The return type will always be a *native floating type*, i.e. one of C's
// `float`, `double`, or `long double`.  The *native floating type* for a
// floating dtype is itself it is one of those three types; otherwise, it is
// `float` if the dtype contains no more than 32 bits, `double` if the dtype
// contains more than 32 bits but no more than 64 bits, and `long double`
// otherwise.  The *native floating type* of any other dtype is `double`.
//
// The working type is the type in which computations are performed.  It is
// equal to the return type, unless the return type is `float`, in which case
// the working type is `double`.
//
// The storage type is what x, y, and w (if supplied) are promoted to before
// doing the calculation.  [THIS PARAGRAPH IS TO BE COMPLETED.]
//
// The metrics are classified into three groups according to the domain of
// the vectors for which they are defined.  The type promotion rules are
// defined for each group as follows.
//
//   1. Metric defined for real vectors, including (12):
//
//        braycurtis, canberra, chebyshev, cityblock, correlation, cosine,
//        euclidean, jensenshannon, mahalanobis, minkowski, seuclidean,
//        sqeuclidean
//
//      For these metrics, the return type is the widest *native floating
//      type* of x, y, and w (if supplied).  The storage type is equal to
//      the return type.  Type promotion may lead to loss of precision, for
//      example by converting large `int64` values to `double`.
//
//   2. Metrics defined for boolean vectors, including (7):
//
//        dice, kulczynski1, rogerstanimoto, russellrao, sokalmichener,
//        sokalsneath, yule
//
//      For these metrics, the return type is the *native floating type* of w
//      if it is supplied, or `double` otherwise.  The storage type is `bool`.
//
//   3. Metrics defined for general vectors, including (2):
//
//        hamming, jaccard
//
//      For these metrics, the return type is the *native floating type* of w
//      if it is supplied, or `double` otherwise.  The storage type is the
//      dtype of x and y unchanged, which must agree and which must be a POD
//      type (i.e. not an object).  Floating and complex values are compared
//      by value (IEEE-style); other POD types are compared by bit pattern.
//      Note that the dtype of x and y does not affect the return type even
//      if they are floating.
//

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <numpy/arrayobject.h>
#include <cmath>
#include <cassert>

#include "function_ref.h"
#include "views.h"
#include "distance_metrics.h"

#include <sstream>
#include <string>

namespace py = pybind11;

namespace {

template <typename T>
using DistanceFunc = FunctionRef<
    void(StridedView2D<T>, StridedView2D<const T>, StridedView2D<const T>)>;

template <typename T>
using WeightedDistanceFunc = FunctionRef<
    void(StridedView2D<T>, StridedView2D<const T>,
         StridedView2D<const T>, StridedView2D<const T>)>;

// Validate weights are >= 0
template <typename T>
void validate_weights(const ArrayDescriptor& w, const T* w_data) {
    intptr_t idx[NPY_MAXDIMS] = {0};
    if (w.ndim > NPY_MAXDIMS) {
        throw std::invalid_argument("Too many dimensions");
    }

    intptr_t numiter = 1;
    for (intptr_t ax = 0; ax < w.ndim - 1; ++ax) {
        numiter *= w.shape[ax];
    }

    bool is_valid = true;
    const T* row_ptr = w_data;
    const auto inner_size = w.shape[w.ndim - 1];
    const auto stride = w.strides[w.ndim - 1];

    while (is_valid && numiter > 0) {
        for (intptr_t i = 0; i < inner_size; ++i) {
            if (row_ptr[i * stride] < 0) {
                is_valid = false;
            }
        }

        for (intptr_t ax = w.ndim - 2; ax >= 0; --ax) {
            if (idx[ax] + 1 < w.shape[ax]) {
                ++idx[ax];
                row_ptr += w.strides[ax];
                break;
            } else {
                row_ptr -= idx[ax] * w.strides[ax];
                idx[ax] = 0;
            }
        }
        --numiter;
    }

    if (!is_valid) {
        throw std::invalid_argument("Input weights should be all non-negative");
    }
}

template <typename T>
void pdist_impl(ArrayDescriptor out, T* out_data,
                ArrayDescriptor x, const T* in_data,
                DistanceFunc<T> f) {
    const intptr_t num_rows = x.shape[0], num_cols = x.shape[1];

    StridedView2D<T> out_view;
    out_view.strides = {out.strides[0], 0};
    out_view.shape = {x.shape[0] - 1, x.shape[1]};
    out_view.data = out_data;

    StridedView2D<const T> x_view;
    x_view.strides = {x.strides[0], x.strides[1]};
    x_view.shape = {out_view.shape[0], num_cols};
    x_view.data = in_data + x.strides[0];

    StridedView2D<const T> y_view;
    y_view.strides = {0, x.strides[1]};
    y_view.shape = {out_view.shape[0], num_cols};
    y_view.data = in_data;

    for (intptr_t i = 0; i < num_rows - 1; ++i) {
        f(out_view, x_view, y_view);

        out_view.data += out_view.shape[0] * out_view.strides[0];
        out_view.shape[0] -= 1;
        x_view.shape[0] = y_view.shape[0] = out_view.shape[0];
        x_view.data += x.strides[0];
        y_view.data += x.strides[0];
    }
}

template <typename T>
void pdist_weighted_impl(ArrayDescriptor out, T* out_data,
                         ArrayDescriptor x, const T* x_data,
                         ArrayDescriptor w, const T* w_data,
                         WeightedDistanceFunc<T> f) {
    if (x.ndim != 2) {
        throw std::invalid_argument("x must be 2-dimensional");
    }

    StridedView2D<T> out_view;
    out_view.strides = {out.strides[0], 0};
    out_view.shape = {x.shape[0] - 1, x.shape[1]};
    out_view.data = out_data;

    StridedView2D<const T> w_view;
    w_view.strides = {0, w.strides[0]};
    w_view.shape = out_view.shape;
    w_view.data = w_data;

    StridedView2D<const T> x_view;
    x_view.strides = {x.strides[0], x.strides[1]};
    x_view.shape = out_view.shape;
    x_view.data = x_data + x.strides[0];

    StridedView2D<const T> y_view;
    y_view.strides = {0, x.strides[1]};
    y_view.shape = out_view.shape;
    y_view.data = x_data;

    const intptr_t num_rows = x.shape[0];
    for (intptr_t i = 0; i < num_rows - 1; ++i) {
        f(out_view, x_view, y_view, w_view);

        out_view.data += out_view.shape[0] * out_view.strides[0];
        out_view.shape[0] -= 1;
        x_view.shape[0] = y_view.shape[0] = w_view.shape[0] = out_view.shape[0];
        x_view.data += x.strides[0];
        y_view.data += x.strides[0];
    }
}

template <typename T>
void cdist_impl(ArrayDescriptor out, T* out_data,
                ArrayDescriptor x, const T* x_data,
                ArrayDescriptor y, const T* y_data,
                DistanceFunc<T> f) {

    const auto num_rowsX = x.shape[0];
    const auto num_rowsY = y.shape[0];
    const auto num_cols = x.shape[1];

    StridedView2D<T> out_view;
    out_view.strides = {out.strides[1], 0};
    out_view.shape = {num_rowsY, num_cols};
    out_view.data = out_data;

    StridedView2D<const T> x_view;
    x_view.strides = {0, x.strides[1]};
    x_view.shape = {num_rowsY, num_cols};
    x_view.data = x_data;

    StridedView2D<const T> y_view;
    y_view.strides = {y.strides[0], y.strides[1]};
    y_view.shape = {out_view.shape[0], num_cols};
    y_view.data = y_data;

    for (intptr_t i = 0; i < num_rowsX; ++i) {
        f(out_view, x_view, y_view);

        out_view.data += out.strides[0];
        x_view.data += x.strides[0];
    }
}

template <typename T>
void cdist_weighted_impl(ArrayDescriptor out, T* out_data,
                         ArrayDescriptor x, const T* x_data,
                         ArrayDescriptor y, const T* y_data,
                         ArrayDescriptor w, const T* w_data,
                         WeightedDistanceFunc<T> f) {

    const auto num_rowsX = x.shape[0];
    const auto num_rowsY = y.shape[0];
    const auto num_cols = x.shape[1];

    StridedView2D<T> out_view;
    out_view.strides = {out.strides[1], 0};
    out_view.shape = {num_rowsY, num_cols};
    out_view.data = out_data;

    StridedView2D<const T> x_view;
    x_view.strides = {0, x.strides[1]};
    x_view.shape = {num_rowsY, num_cols};
    x_view.data = x_data;

    StridedView2D<const T> y_view;
    y_view.strides = {y.strides[0], y.strides[1]};
    y_view.shape = {num_rowsY, num_cols};
    y_view.data = y_data;

    StridedView2D<const T> w_view;
    w_view.strides = {0, w.strides[0]};
    w_view.shape = {num_rowsY, num_cols};
    w_view.data = w_data;

    for (intptr_t i = 0; i < num_rowsX; ++i) {
        f(out_view, x_view, y_view, w_view);

        out_view.data += out.strides[0];
        x_view.data += x.strides[0];
    }
}

// Extract shape and stride information from NumPy array. Converts byte-strides
// to element strides, and avoids an extra pointer indirection on access.
ArrayDescriptor get_descriptor(const py::array& arr) {
    const auto ndim = arr.ndim();
    ArrayDescriptor desc(ndim);

    const auto arr_shape = arr.shape();
    desc.shape.assign(arr_shape, arr_shape + ndim);

    desc.element_size = arr.itemsize();
    const auto arr_strides = arr.strides();
    desc.strides.assign(arr_strides, arr_strides + ndim);
    for (intptr_t i = 0; i < ndim; ++i) {
        if (arr_shape[i] <= 1) {
            // Under NumPy's relaxed stride checking, dimensions with
            // 1 or fewer elements are ignored.
            desc.strides[i] = 0;
            continue;
        }

        if (desc.strides[i] % desc.element_size != 0) {
            std::stringstream msg;
            msg << "Arrays must be aligned to element size, but found stride of ";
            msg << desc.strides[i] << " bytes for elements of size " << desc.element_size;
            throw std::runtime_error(msg.str());
        }
        desc.strides[i] /= desc.element_size;
    }
    return desc;
}

// Cast python object to NumPy array of data type T.
// flags can be any NumPy array constructor flags.
template <typename T>
py::array_t<T> npy_asarray(const py::handle& obj, int flags = 0) {
    auto descr = reinterpret_cast<PyArray_Descr*>(
        py::dtype::of<T>().release().ptr());
    auto* arr = PyArray_FromAny(obj.ptr(), descr, 0, 0, flags, nullptr);
    if (arr == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::array_t<T>>(arr);
}

// Cast python object to NumPy array of given dtype.
// flags can be any NumPy array constructor flags.
py::array npy_asarray(const py::handle& obj, py::dtype dtype, int flags = 0) {
    PyObject* arr = PyArray_FromAny(obj.ptr(), dtype.release().ptr(), 0, 0, flags, nullptr);
    if (arr == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::array>(arr);
}

// Cast python object to NumPy array with unspecified dtype.
// flags can be any NumPy array constructor flags.
py::array npy_asarray(const py::handle& obj, int flags = 0) {
    auto* arr = PyArray_FromAny(obj.ptr(), nullptr, 0, 0, flags, nullptr);
    if (arr == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::array>(arr);
}

template <typename scalar_t>
py::array pdist_unweighted(const py::array& out_obj, const py::array& x_obj,
                           DistanceFunc<scalar_t> f) {
    auto x = npy_asarray<scalar_t>(x_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);
    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    {
        py::gil_scoped_release guard;
        pdist_impl(out_desc, out_data, x_desc, x_data, f);
    }
    return std::move(out);
}

template <typename scalar_t>
py::array pdist_weighted(
        const py::array& out_obj, const py::array& x_obj,
        const py::array& w_obj, WeightedDistanceFunc<scalar_t> f) {
    auto x = npy_asarray<scalar_t>(x_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto w = npy_asarray<scalar_t>(w_obj,
                                   NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);
    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    auto w_desc = get_descriptor(w);
    auto w_data = w.data();
    {
        py::gil_scoped_release guard;
        validate_weights(w_desc, w_data);
        pdist_weighted_impl(
            out_desc, out_data, x_desc, x_data, w_desc, w_data, f);
    }
    return std::move(out);
}

template <typename scalar_t>
py::array cdist_unweighted(const py::array& out_obj, const py::array& x_obj,
                           const py::array& y_obj, DistanceFunc<scalar_t> f) {
    auto x = npy_asarray<scalar_t>(x_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto y = npy_asarray<scalar_t>(y_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);

    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    auto y_desc = get_descriptor(y);
    auto y_data = y.data();
    {
        py::gil_scoped_release guard;
        cdist_impl(out_desc, out_data, x_desc, x_data, y_desc, y_data, f);
    }
    return std::move(out);
}

template <typename scalar_t>
py::array cdist_weighted(
        const py::array& out_obj, const py::array& x_obj,
        const py::array& y_obj, const py::array& w_obj,
        WeightedDistanceFunc<scalar_t> f) {
    auto x = npy_asarray<scalar_t>(x_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto y = npy_asarray<scalar_t>(y_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto w = npy_asarray<scalar_t>(w_obj,
                                 NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    auto out = py::cast<py::array_t<scalar_t>>(out_obj);

    auto out_desc = get_descriptor(out);
    auto out_data = out.mutable_data();
    auto x_desc = get_descriptor(x);
    auto x_data = x.data();
    auto y_desc = get_descriptor(y);
    auto y_data = y.data();
    auto w_desc = get_descriptor(w);
    auto w_data = w.data();
    {
        py::gil_scoped_release guard;
        validate_weights(w_desc, w_data);
        cdist_weighted_impl(
            out_desc, out_data, x_desc, x_data, y_desc, y_data, w_desc, w_data, f);
    }
    return std::move(out);
}

py::dtype npy_promote_types(const py::dtype& type1, const py::dtype& type2) {
    PyArray_Descr* descr = PyArray_PromoteTypes(
        reinterpret_cast<PyArray_Descr*>(type1.ptr()),
        reinterpret_cast<PyArray_Descr*>(type2.ptr()));
    if (descr == nullptr) {
        throw py::error_already_set();
    }
    return py::reinterpret_steal<py::dtype>(reinterpret_cast<PyObject*>(descr));
}

template <typename Container>
py::array prepare_out_argument(const py::object& obj, const py::dtype& dtype,
                               const Container& out_shape) {
    if (obj.is_none()) {
        return py::array(dtype, out_shape);
    }

    if (!py::isinstance<py::array>(obj)) {
        throw py::type_error("out argument must be an ndarray");
    }

    py::array out = py::cast<py::array>(obj);
    const auto ndim = out.ndim();
    const auto shape = out.shape();
    auto pao = reinterpret_cast<PyArrayObject*>(out.ptr());

    if (ndim != static_cast<intptr_t>(out_shape.size()) ||
        !std::equal(shape, shape + ndim, out_shape.begin())) {
        throw std::invalid_argument("Output array has incorrect shape.");
    }
    if (!PyArray_ISCONTIGUOUS(pao)) {
        throw std::invalid_argument("Output array must be C-contiguous");
    }
    if (out.dtype().not_equal(dtype)) {
        const py::handle& handle = dtype;
        throw std::invalid_argument("wrong out dtype, expected " +
                                    std::string(py::str(handle)));
    }
    if (!PyArray_ISBEHAVED(pao)) {
        throw std::invalid_argument(
            "out array must be aligned, writable and native byte order");
    }
    return out;
}

py::array prepare_single_weight(const py::object& obj, intptr_t len) {
    py::array weight = npy_asarray(obj);
    if (weight.ndim() != 1) {
        throw std::invalid_argument("Weights must be a vector (ndim = 1)");
    } else if (weight.shape(0) != len) {
        std::stringstream msg;
        msg << "Weights must have same size as input vector. ";
        msg << weight.shape(0) << " vs. " << len << ".";
        throw std::invalid_argument(msg.str());
    }
    return weight;
}

py::dtype common_type(py::dtype type) {
    return type;
}

template <typename... Args>
py::dtype common_type(const py::dtype& type1, const py::dtype& type2,
                      const Args&... tail) {
    return common_type(npy_promote_types(type1, type2), tail...);
}

int dtype_num(const py::dtype& dtype) {
    return reinterpret_cast<const PyArray_Descr*>(
        dtype.ptr())->type_num;
}

py::dtype promote_type_real(const py::dtype& dtype) {
    switch (dtype.kind()) {
    case 'b':
    case 'i':
    case 'u': {
        // Promote integral and boolean types to double
        return py::dtype::template of<double>();
    }
    case 'f': {
        if (dtype_num(dtype) == NPY_LONGDOUBLE) {
            return dtype;
        } else {
            // TODO: Allow float32 output
            return py::dtype::template of<double>();
        }
    }

    default: {
        return dtype;
    }
    }
}

// From a NumPy dtype, run "expression" with scalar_t aliasing the C++ type
#define DISPATCH_DTYPE(dtype, expression)                               \
    do {                                                                \
        const py::dtype& type_obj = dtype;                              \
        switch (dtype_num(type_obj)) {                                  \
        case NPY_HALF:                                                  \
        case NPY_FLOAT: /* TODO: Enable scalar_t=float dispatch */      \
        case NPY_DOUBLE: {                                              \
            using scalar_t = double;                                    \
            expression();                                               \
            break;                                                      \
        }                                                               \
        case NPY_LONGDOUBLE: {                                          \
            using scalar_t = long double;                               \
            expression();                                               \
            break;                                                      \
        }                                                               \
        default: {                                                      \
            const py::handle& handle = type_obj;                        \
            throw std::invalid_argument(                                \
                "Unsupported dtype " + std::string(py::str(handle)));   \
        }                                                               \
        }                                                               \
    } while (0)

template <typename Func>
py::array pdist(const py::object& out_obj, const py::object& x_obj,
                const py::object& w_obj, Func&& f) {
    auto x = npy_asarray(x_obj);
    if (x.ndim() != 2) {
        throw std::invalid_argument("x must be 2-dimensional");
    }

    const intptr_t m = x.shape(1);
    const intptr_t n = x.shape(0);
    std::array<intptr_t, 1> out_shape{{(n * (n - 1)) / 2}};
    if (w_obj.is_none()) {
        auto dtype = promote_type_real(x.dtype());
        auto out = prepare_out_argument(out_obj, dtype, out_shape);
        DISPATCH_DTYPE(dtype, [&]{
            pdist_unweighted<scalar_t>(out, x, f);
        });
        return out;
    }

    auto w = prepare_single_weight(w_obj, m);
    auto dtype = promote_type_real(common_type(x.dtype(), w.dtype()));
    auto out = prepare_out_argument(out_obj, dtype, out_shape);
    DISPATCH_DTYPE(dtype, [&]{
        pdist_weighted<scalar_t>(out, x, w, f);
    });
    return out;
}

template <typename Func>
py::array cdist(const py::object& out_obj, const py::object& x_obj,
                const py::object& y_obj, const py::object& w_obj, Func&& f) {
    auto x = npy_asarray(x_obj);
    auto y = npy_asarray(y_obj);
    if (x.ndim() != 2) {
        throw std::invalid_argument("XA must be a 2-dimensional array.");
    }
    if (y.ndim() != 2) {
        throw std::invalid_argument("XB must be a 2-dimensional array.");
    }
    const intptr_t m = x.shape(1);
    if (m != y.shape(1)) {
        throw std::invalid_argument(
            "XA and XB must have the same number of columns "
            "(i.e. feature dimension).");
    }

    std::array<intptr_t, 2> out_shape{{x.shape(0), y.shape(0)}};
    if (w_obj.is_none()) {
        auto dtype = promote_type_real(common_type(x.dtype(), y.dtype()));
        auto out = prepare_out_argument(out_obj, dtype, out_shape);
        DISPATCH_DTYPE(dtype, [&]{
            cdist_unweighted<scalar_t>(out, x, y, f);
        });
        return out;
    }

    auto w = prepare_single_weight(w_obj, m);
    auto dtype = promote_type_real(
        common_type(x.dtype(), y.dtype(), w.dtype()));
    auto out = prepare_out_argument(out_obj, dtype, out_shape);
    DISPATCH_DTYPE(dtype, [&]{
        cdist_weighted<scalar_t>(out, x, y, w, f);
    });
    return out;
}

////////////////////////////////////////////////////
// NEW STUFF
////////////////////////////////////////////////////

#define RETURN_FULL  ('c')
#define RETURN_UPPER ('p')
#define RETURN_DIAG  ('v') /* for future use */

// Get the *native floating type* for the given dtype.
//
// There is no guarantee that the given dtype is convertible to the returned
// dtype.  An error will occur at the point of conversion if conversion fails
py::dtype get_native_floating_type(const py::dtype& dtype) {
    if (dtype.kind() == 'f') {
        int typenum = dtype.typenum();
        if (typenum == NPY_FLOAT || typenum == NPY_DOUBLE || typenum == NPY_LONGDOUBLE) {
            return dtype;
        }
        auto itemsize = dtype.itemsize();
        typenum = (itemsize <= 4) ? NPY_FLOAT :
                  (itemsize <= 8) ? NPY_DOUBLE : NPY_LONGDOUBLE;
        return py::dtype(typenum);
    } else {
        return py::dtype(NPY_DOUBLE);
    }
}

// Prepare input array x and y for use by a metric defined for real vectors.
//
// For efficiency, w_obj should be an np.array if supplied.
std::pair<py::array, py::array> prepare_input_for_real_metric(
    py::object x_obj, py::object y_obj, py::object w_obj, char extract) {

    // Detect identity to save redundant conversions.
    const bool x_is_y = (x_obj.ptr() == y_obj.ptr());

    // Convert to np.array without requirement on dtype or layout in order
    // to check shape first.
    py::array x = npy_asarray(x_obj);
    if (x.ndim() != 2) {
        throw std::invalid_argument("XA must be a 2-dimensional array.");
    }

    py::array y = x_is_y ? x : npy_asarray(y_obj);
    if (y.ndim() != 2) {
        throw std::invalid_argument("XB must be a 2-dimensional array.");
    }

    if (x.shape(1) != y.shape(1)) {
        throw std::invalid_argument(
            "XA and XB must have the same number of columns "
            "(i.e. feature dimension).");
    }

    if (extract != EXTRACT_FULL) {
        if (x.shape(0) != y.shape(0)) {
            throw std::invalid_argument("XA and XB must have the same number of rows");
        }
    }

    py::dtype dtype = common_type(get_native_floating_type(x.dtype()),
                                  get_native_floating_type(y.dtype()));
    if (!w_obj.is_none()) {
        // w is supplied; also take into account its dtype
        py::array w = npy_asarray(w_obj);
        dtype = common_type(dtype, get_floating_type(w.dtype()));
    }

    // Convert x and y to the desired dtype and layout.
    x = npy_asarray(x, dtype, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    y = x_is_y ? x : npy_asarray(y, dtype, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    return {x, y};
)

// Prepare weight scalar, vector or matrix for use by a metric defined for
// real vectors.
//
// x and y MUST already be prepared.  w must not be None.
//
// desired_ndim is the ndim of the weight to return; 0 for 0D array (scalar),
// 1 for n-vector, 2 for n-by-n square matrix.
//
// `minimum_ndim` is the smallest allowed ndim, that allows conversion according
// to the following rules:
//
//   desired_ndim  ndim  conversion
//   ------------------------------
//   1             0     [w,w,...,w]
//   2             0     diag([w,w,...,w])
//   2             1     diag(w)
//
py::array prepare_weight_for_real_metric(
    py::array x, py::array y, py::object w_obj, int desired_ndim, int minimum_ndim) {

    if (!(desired_ndim >= 0 && desired_ndim <= 2)) {
        throw std::invalid_argument("desired_ndim must be 0, 1 or 2");
    }
    if (!(minimum_ndim >= 0 && minimum_ndim <= desired_ndim)) {
        throw std::invalid_argument("minimum_ndim must be between 0 and desired_ndim");
    }

    py::array w = npy_asarray(w_obj);

    const int ndim = w.ndim();
    if (!(ndim >= minimum_ndim && ndim <= desired_ndim)) {
        std::stringstream msg;
        msg << "Weight must have ndim between " << minimum_ndim << " and " << desired_ndim;
        throw std::invalid_argument(msg.str());
    }

    if (x.ndim() != 2) {
        throw std::invalid_argument("x must be 2-dimensional");
    }
    const intptr_t n = x.shape(1);

    if (ndim == 1) {
        if (w.shape(0) != n) {
            throw std::invalid_argument("weight vector must have the same length "
                                        "as the number of columns in X");
        }
    } else if (ndim == 2) {
        if (w.shape(0) != n || w.shape(1) != n) {
            throw std::invalid_argument("weight matrix must be square and have "
                                        "the same number of columns as X");
        }
    }

    if (minimum_ndim != desired_ndim) {
        throw std::runtime_error("not implemented");
    }

    // Determine the weight dtype, which is the widest native floating type
    // of x, y, and w.
    py::dtype dtype = common_type(get_floating_type(x.dtype()),
                                  get_floating_type(y.dtype()),
                                  get_floating_type(w.dtype()));
    w = npy_asarray(w_obj, dtype, NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED);
    return w;
}

// x, y, and w (if supplied) *must* have already been prepared
py::array prepare_output_for_real_metric(
    py::object x_obj, py::object y_obj, py::object w_obj, char return_type, py::object out_obj)
{
    py::array x = x_obj;
    py::array y = y_obj;
    py::array w = w_obj.is_none() ? py::object{} : w_obj;

    if (!(x.ndim() == 2 && y.ndim() == 2)) {
        throw std::invalid_argument("x and y must be 2-D arrays");
    }

    // Compute output shape.
    const intptr_t p = x.shape(0), q = y.shape(0), n = y.shape(1);
    if (x.shape(1) != n) {
        throw std::invalid_argument("x and y must have the same number of columns");
    }
    py::tuple out_shape;
    if (return_type == RETURN_FULL) {
        out_shape = py::make_tuple(p, q);
    } else if (return_type == RETURN_UPPER) {
        if (p != q) {
            throw std::invalid_argument("x and y must have the same number of rows");
        }
        out_shape = py::make_tuple(p * (p - 1) / 2);
    } else {
        throw std::invalid_argument("invalid return_type");
    }

    // Compute output dtype.
    py::dtype out_dtype = common_type(get_native_floating_type(x.dtype()),
                                      get_native_floating_type(y.dtype()));
    if (w) {
        out_dtype = common_type(out_dtype, get_native_floating_type(w.dtype()));
    }

    // Create output buffer if none is supplied.
    if (out_obj.is_none()) {
        return py::array(dtype, out_shape);
    }

    // Now verify that the supplied `out` argument precisely satisfies our
    // requirement.
    if (!py::isinstance<py::array>(out_obj)) {
        throw py::type_error("out argument must be an ndarray");
    }
    py::array out = out_obj;

    if (!out_shape.equal(out.shape())) {
        throw std::invalid_argument("out array has incorrect shape");
    }
    if (out.dtype().not_equal(dtype)) {
        throw std::invalid_argument("out array has wrong dtype, expected " +
                                    std::string(py::str(dtype)));
    }

    PyArrayObject *pao = reinterpret_cast<PyArrayObject*>(out.ptr());
    if (!PyArray_ISCONTIGUOUS(pao)) {
        throw std::invalid_argument("out array must be C-contiguous");
    }
    if (!PyArray_ISBEHAVED(pao)) {
        throw std::invalid_argument(
            "out array must be aligned, writable and native byte order");
    }
    return out;
}

StridedView2D<void> get_strided_view(const py::array &arr) {

    ArrayDescriptor desc = get_descriptor(arr);
    StridedView2D<void> view;
    view.strides = {desc.strides[0], desc.strides[1]};
    view.shape = {desc.shape[0], desc.shape[1]}; // TODO: check dim
    view.data = arr.data(); // TODO: get mutable
    return view;
}

template <typename T, typename Distance>
void compute_distance(const StridedView2D &x, const StridedView &y,
const StridedView2D &w, StridedView2D &out, char return_type, Distance &&d) {

    const intptr_t p = x.shape[0];
    const intptr_t q = y.shape[0];
    const intptr_t n = x.shape[1];
    if (y.shape[1] != n) {
        throw std::invalid_argument();
    }
    if (w.shape[0] != n || w.shape[1] != n) {
        throw std::invalid_argument();
    }

    if (return_type == RETURN_FULL) {



    }

    // x_view has "logical" shape p-by-n, but the row stride is 0, so the
    // current row automatically broadcasts.
    StridedView2D<const T> x_view;
    x_view.shape = {p, n};
    x_view.strides = {0, x.strides[1]};
    x_view.data = static_cast<const T*>(x.data);

    // y_view maps directly to the actual y.
    StridedView2D<const T> y_view;
    y_view.shape = y.shape;
    y_view.strides = y.strides;
    y_view.data = static_cast<const T*>(y.data);

    // w_view depends on the dimension of w.
    // If w is a scalar or a vector, w_view is also a vector.
    // If w is a matrix, w_view is a square matrix.
    // TODO

    // out_view is an
    StridedView2D<T> out_view;
    out_view.strides = {out.strides[1], 0};
    out_view.shape = {num_rowsY, num_cols};
    out_view.data = out_data;

    for (intptr_t i = 0; i < p; ++i) {
        for (intptr_t j = 0; j < q; ++j) {
            *out++ = d(x[i], y[j])
        }
    }
        f(out_view, x_view, y_view);

        out_view.data += out.strides[0];
        x_view.data += x.strides[0];
    }
}

py::array xdist_mahalanobis(char return_type, py::object x_obj, py::object y_obj,
                            py::object w_obj, py::object out_obj) {

    auto xy = prepare_input_for_real_metric(x_obj, y_obj, w_obj, return_type);
    py::array x = xy.first;
    py::array y = xy.second;
    py::array w = prepare_weight_for_real_metric(x, y, w_obj, 2, 2);
    py::array out = prepare_output_for_real_metric(x, y, w, return_type, out_obj);

    StridedView2D x_view = get_strided_view(x);
    StridedView2D y_view = get_strided_view(y);
    StridedView2D w_view = get_strided_view(w);
    StridedView2D out_view = get_mutable_strided_view(out);

    py::dtype dtype = x.dtype();
    int typenum = dtype.typenum();


    StridedView2D<T> out_view;
    out_view.strides = {out.strides[1], 0};
    out_view.shape = {num_rowsY, num_cols};
    out_view.data = out_data;

    StridedView2D<const void> x_view;
    x_view.shape = {x_desc.shape[0], x_desc.shape[1]};
    x_view.strides = {x_desc.strides[0], x_desc.strides[1]};
    x_view.data = x_data;

    StridedView2D<const T> y_view;
    y_view.strides = {y.strides[0], y.strides[1]};
    y_view.shape = {out_view.shape[0], num_cols};
    y_view.data = y_data;

    // Release the GIL in the below scope.  If nested code wants to call
    // Python, it must acquire the GIL first.
    {
        py::gil_scoped_release guard;
        cdist_impl(out_desc, out_data, x_desc, x_data, y_desc, y_data, f);
        switch (typenum) {
        case NPY_FLOAT:
            MahalanobisDistance<float>::compute(out, x, y, w);
            break;
        case NPY_DOUBLE:


        void cdist_impl(ArrayDescriptor out, T* out_data,
                ArrayDescriptor x, const T* x_data,
                ArrayDescriptor y, const T* y_data,
                DistanceFunc<T> f) {



            MahalanobisDistance<double>::compute(out, x, y, w);


            break;
        case NPY_LONGDOUBLE:
            MahalanobisDistance<long double>::compute(out, x, y, w);
            break;
        default:
            throw std::invalid_argument("unexpected dtype");
        }
    }
    return out;
}

///////////////////////////////////////////////////////////////
// END OF NEW STUFF
///////////////////////////////////////////////////////////////

PYBIND11_MODULE(_distance_pybind, m) {
    if (_import_array() != 0) {
        throw py::error_already_set();
    }
    using namespace pybind11::literals;

    // Helper functions.
    m.def("prepare_input_for_real_metric", prepare_input_for_real_metric,
          "x"_a, "y"_a, "w"_a, "op"_a);
    m.def("prepare_weight_for_real_metric", prepare_weight_for_real_metric,
          "x"_a, "y"_a, "w"_a, "desired_ndim"_a, "minimum_ndim"_a);

    // Individual distance functions.
    m.def("pdist_canberra",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, CanberraDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_hamming",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, HammingDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_dice",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, DiceDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_jaccard",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, JaccardDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_kulczynski1",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, Kulczynski1Distance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_rogerstanimoto",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, RogerstanimotoDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_russellrao",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, RussellRaoDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_sokalmichener",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, SokalmichenerDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_sokalsneath",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, SokalsneathDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_yule",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, YuleDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_chebyshev",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, ChebyshevDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_cityblock",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, CityBlockDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_euclidean",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, EuclideanDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_minkowski",
          [](py::object x, py::object w, py::object out, double p) {
              if (p == 1.0) {
                  return pdist(out, x, w, CityBlockDistance{});
              } else if (p == 2.0) {
                  return pdist(out, x, w, EuclideanDistance{});
              } else if (std::isinf(p)) {
                  return pdist(out, x, w, ChebyshevDistance{});
              } else {
                  return pdist(out, x, w, MinkowskiDistance{p});
              }
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none(), "p"_a=2.0);
    m.def("pdist_sqeuclidean",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, SquareEuclideanDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("pdist_braycurtis",
          [](py::object x, py::object w, py::object out) {
              return pdist(out, x, w, BraycurtisDistance{});
          },
          "x"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_canberra",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, CanberraDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_dice",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, DiceDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_jaccard",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, JaccardDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_kulczynski1",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, Kulczynski1Distance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_hamming",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, HammingDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_rogerstanimoto",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, RogerstanimotoDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_russellrao",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, RussellRaoDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_sokalmichener",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SokalmichenerDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_sokalsneath",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SokalsneathDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_yule",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, YuleDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_chebyshev",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, ChebyshevDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_cityblock",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, CityBlockDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_euclidean",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, EuclideanDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_minkowski",
          [](py::object x, py::object y, py::object w, py::object out,
             double p) {
              if (p == 1.0) {
                  return cdist(out, x, y, w, CityBlockDistance{});
              } else if (p == 2.0) {
                  return cdist(out, x, y, w, EuclideanDistance{});
              } else if (std::isinf(p)) {
                  return cdist(out, x, y, w, ChebyshevDistance{});
              } else {
                  return cdist(out, x, y, w, MinkowskiDistance{p});
              }
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none(), "p"_a=2.0);
    m.def("cdist_sqeuclidean",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, SquareEuclideanDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
    m.def("cdist_braycurtis",
          [](py::object x, py::object y, py::object w, py::object out) {
              return cdist(out, x, y, w, BraycurtisDistance{});
          },
          "x"_a, "y"_a, "w"_a=py::none(), "out"_a=py::none());
}

}  // namespace (anonymous)
