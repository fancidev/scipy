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
//   - `out` is an optional contiguous buffer to store the output; and
//   - `kwargs` contain extra metric-specific parameters, such as `w` for
//      (mahalanobis) weight and `p` for (minkowski) order.
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
// Type conversions
// ----------------
// Two kinds of dtypes are relevant for metric calculation: the *input dtype*
// and the *output dtype*.  The input dtype is the dtype that x, y, and
// optionally extra parameters (such as w) are converted into before doing
// the calculations.  The output dtype is the dtype of the returned distance
// matrix (or vector).
//
// In principle, the input and output dtypes are determined according to the
// values they apply to:
//
//   - Values involved in arithmetic operations are converted to a *native
//     floating dtype*, which is one of `float`, `double`, or `long double`.
//     The *native floating dtype* of a floating dtype is itself if it is
//     one of `float`, `double`, or `long double`; otherwise, it is `float`
//     if the dtype contains no more than 32 bits, `double` if the dtype
//     contains more than 32 bits but no more than 64 bits, and `long double`
//     if the dtype contains more than 64 bits.  The *native floating dtype*
//     of any other dtype is `double`.
//
//   - Values involved in boolean operations are converted to the `bool`
//     dtype.
//
//   - Values only involved in equality comparison are not converted.
//
// Concretely, the input and output dtype of a metric is determined according
// to the domain over which the metric is defined:
//
//   1. For metrics defined for real vectors, including (12):
//
//        braycurtis, canberra, chebyshev, cityblock, correlation, cosine,
//        euclidean, jensenshannon, mahalanobis, minkowski, seuclidean,
//        sqeuclidean,
//
//      the input dtype is the widest *native floating type* of x, y, and
//      optionally extra parameters (such as w) involved in the computation.
//      The output dtype is equal to the input dtype.
//
//   2. For metrics defined for boolean vectors, including (7):
//
//        dice, kulczynski1, rogerstanimoto, russellrao, sokalmichener,
//        sokalsneath, yule,
//
//      the input dtype is `bool`.  The output dtype is the *native floating
//      type* of w if it is supplied, or `double` otherwise.
//
//   3. For metrics defined for general vectors where only equality relation
//      is relevant, including (2):
//
//        hamming, jaccard
//
//      the input dtype is the dtype of x and y unchanged, which must agree
//      and which must be one of a few supported POD dtypes.  The output dtype
//      is the *native floating type* of w if it is supplied, or `double`
//      otherwise.  The input dtype does not affect the output dtype.
//
//      Note: The restriction on input dtype could be relaxed if we use Python
//      equality comparison.
//
// Note 1: Extra parameters (such as w) are typically converted to the output
// dtype to limit the number of C++ template instantiations.
//
// Note 2: For actual computation, a *working dtype* with higher precision may
// be used to improve accuracy.  For example, `float` inputs could be computed
// using a `double` working dtype, and `double` inputs could be computed using
// a double-double working dtype.  The choice of working dtype is up to the
// metric implementation.
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
    PyObject* arr = PyArray_FromAny(obj.ptr(),
        reinterpret_cast<PyArray_Descr*>(dtype.release().ptr()), 0, 0, flags, nullptr);
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
void validate_out_argument(const py::array& out, const py::dtype& dtype,
                           const Container& out_shape) {

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
    validate_out_argument(out, dtype, out_shape);
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

#define SPECIALIZE_FLOAT32 0

enum class JoinMode : char {
    Cross = 'c',
    Upper = 'p',
    Inner = 'r',  // for future use
};

enum class MetricClass : char { // or MetricDomain ?
    Real = 'r',
    Bool = 'b',
    Edit = 'e',
};

enum class WeightShape : char {
    Scalar = 's',
    Vector = 'v',
    Matrix = 'm',
};

// Get the *native floating dtype* for the given dtype.
// TODO: need a map instead
//
// There is no guarantee that the given dtype is convertible to the returned
// dtype.  An error will occur at the point of conversion if it fails.
py::dtype get_native_floating_dtype(const py::dtype& dtype) {
    switch (dtype.kind()) {
    case 'b': // boolean
    case 'i': // signed integer
    case 'u': // unsigned integer
        return py::dtype(NPY_DOUBLE);
    case 'f': { // floating
        int typenum = dtype.num();
        if (typenum == NPY_FLOAT || typenum == NPY_DOUBLE || typenum == NPY_LONGDOUBLE) {
            return dtype;
        }
        auto itemsize = dtype.itemsize();
        typenum = (itemsize <= 4) ? NPY_FLOAT :
                  (itemsize <= 8) ? NPY_DOUBLE : NPY_LONGDOUBLE;
        return py::dtype(typenum);
    }
    default:
    // TODO: Do not throw
        throw std::invalid_argument("unsupported dtype");
    }
}

// Prepare input array x and y.
std::pair<py::array, py::array> prepare_xy(
    py::object x_obj, py::object y_obj, JoinMode join_mode,
    py::object least_dtype_obj, MetricClass metric_class) {

    // Convert to np.array without requirement on dtype or layout in order
    // to check shape first.
    if (x_obj.is_none()) {
        throw std::invalid_argument("X cannot be None");
    }
    py::array x = npy_asarray(x_obj);
    if (x.ndim() != 2) {
        if (y_obj.is_none()) {
            throw std::invalid_argument("X must be a 2-dimensional array.");
        } else {
            throw std::invalid_argument("XA must be a 2-dimensional array.");
        }
    }

    py::array y = y_obj.is_none() ? x :
                  x_obj.is(y_obj) ? x :
                  npy_asarray(y_obj);
    if (y.ndim() != 2) {
        throw std::invalid_argument("XB must be a 2-dimensional array.");
    }

    if (x.shape(1) != y.shape(1)) {
        throw std::invalid_argument(
            "XA and XB must have the same number of columns "
            "(i.e. feature dimension).");
    }

    if (join_mode != JoinMode::Cross) {
        if (x.shape(0) != y.shape(0)) {
            throw std::invalid_argument("XA and XB must have the same number of rows");
        }
    }

    // Compute input dtype according to the metric class
    py::dtype least_dtype = least_dtype_obj.is_none() ? py::object{} : least_dtype_obj;
    py::dtype input_dtype;
    switch (metric_class) {
    case MetricClass::Real:
        input_dtype = common_type(get_native_floating_dtype(x.dtype()),
                                  get_native_floating_dtype(y.dtype()));
        if (least_dtype) {
            input_dtype = common_type(input_dtype, get_native_floating_dtype(least_dtype));
        }
        break;
    case MetricClass::Bool:
        input_dtype = py::dtype(NPY_BOOL);
        break;
    case MetricClass::Edit:
        throw std::invalid_argument("not implemented");
        break;
    default:
        throw std::invalid_argument("invalid metric_class");
    }

    // Convert x and y to the desired dtype and layout.
    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED | NPY_ARRAY_ELEMENTSTRIDES;
    x = npy_asarray(x, input_dtype, flags);
    y = x_obj.is(y_obj) ? x : npy_asarray(y, input_dtype, flags);
    return {x, y};
}

//// Prepare a weight scalar, vector, or matrix.
////
//// Parameters
//// ----------
////   w                Input weight.  Must not be None.
////   weight_shape     Desired weight shape
////   n                Number of columns in x and y; ignored for scalar weight
////   least_dtype_obj  If not None, the dtype of the returned weight object
////                    is at least as wide as the native floating dtype of
////                    least_dtype.
////   promote_shape    If True, the input or vector weight is allowed and a diagonal
////                  weight matrix is constructed from w.
////
//// Return value
//// ------------
//// An n-by-n weight matrix converted from w, with dtype equal to the *native
//// floating dtype* of w or that of least_dtype (if supplied), whichever is
//// wider.
//
//// prepare_w
//py::array prepare_weight_matrix(py::object w_obj,
//                                int n,
//                                py::object least_dtype_obj = py::none(),
//                                bool promote_shape = false) {
//
//    py::array w = npy_asarray(w_obj);
//
//    const int ndim = w.ndim();
//    switch (ndim) {
//    case 0:
//        if (!promote_shape) {
//            throw std::invalid_argument("weight must be a matrix, not scalar");
//        }
//        throw std::invalid_argument("scalar->matrix transform is not implemented");
//        break;
//    case 1:
//        if (!promote_shape) {
//            throw std::invalid_argument("weight must be a matrix, not vector");
//        }
//        if (w.shape(0) != n) {
//            throw std::invalid_argument("weight vector must have the same length "
//                                        "as the number of columns in X");
//        }
//        throw std::invalid_argument("vector->matrix transform is not implemented");
//        break;
//    case 2:
//        if (!(w.shape(0) == n && w.shape(1) == n)) {
//            throw std::invalid_argument("weight matrix must be square and have "
//                                        "the same number of columns as X");
//        }
//        break;
//    default:
//        throw std::invalid_argument(promote_shape ?
//            "weight scalar, vector, or matrix expected" :
//            "weight matrix expected");
//        break;
//    }
//
//    // Determine the weight dtype, which is the wider native floating dtype
//    // of itself and that of least_dtype if supplied.
//    py::dtype weight_dtype = get_native_floating_dtype(w.dtype());
//    if (!least_dtype_obj.is_none()) {
//        py::dtype least_dtype = least_dtype_obj;
//        weight_dtype = common_type(weight_dtype,
//                                   get_native_floating_dtype(least_dtype));
//    }
//
//    // Convert weight matrix to the desired dtype and layout, and
//    // make sure it is contiguous.
//    int flags = NPY_ARRAY_ALIGNED | NPY_ARRAY_NOTSWAPPED |
//                NPY_ARRAY_ELEMENTSTRIDE | NPY_ARRAY_C_CONTIGUOUS;
//    return npy_asarray(w, weight_dtype, flags);
//}

//// x, y, and w (if supplied) *must* have already been prepared
//py::array prepare_output_for_real_metric(
//    py::object x_obj, py::object y_obj, py::object w_obj, char return_type, py::object out_obj)
//{
//    py::array x = x_obj;
//    py::array y = y_obj;
//    py::array w = w_obj.is_none() ? py::object{} : w_obj;
//
//    if (!(x.ndim() == 2 && y.ndim() == 2)) {
//        throw std::invalid_argument("x and y must be 2-D arrays");
//    }
//
//    // Compute output shape.
//    const intptr_t p = x.shape(0), q = y.shape(0), n = y.shape(1);
//    if (x.shape(1) != n) {
//        throw std::invalid_argument("x and y must have the same number of columns");
//    }
//    py::tuple out_shape;
//    if (return_type == RETURN_FULL) {
//        out_shape = py::make_tuple(p, q);
//    } else if (return_type == RETURN_UPPER) {
//        if (p != q) {
//            throw std::invalid_argument("x and y must have the same number of rows");
//        }
//        out_shape = py::make_tuple(p * (p - 1) / 2);
//    } else {
//        throw std::invalid_argument("invalid return_type");
//    }
//
//    // Compute output dtype.
//    py::dtype out_dtype = common_type(get_native_floating_type(x.dtype()),
//                                      get_native_floating_type(y.dtype()));
//    if (w) {
//        out_dtype = common_type(out_dtype, get_native_floating_type(w.dtype()));
//    }
//
//    // Create output buffer if none is supplied.
//    if (out_obj.is_none()) {
//        return py::array(dtype, out_shape);
//    }
//
//    // Now verify that the supplied `out` argument precisely satisfies our
//    // requirement.
//    if (!py::isinstance<py::array>(out_obj)) {
//        throw py::type_error("out argument must be an ndarray");
//    }
//    py::array out = out_obj;
//
//    if (!out_shape.equal(out.shape())) {
//        throw std::invalid_argument("out array has incorrect shape");
//    }
//    if (out.dtype().not_equal(dtype)) {
//        throw std::invalid_argument("out array has wrong dtype, expected " +
//                                    std::string(py::str(dtype)));
//    }
//
//    PyArrayObject *pao = reinterpret_cast<PyArrayObject*>(out.ptr());
//    if (!PyArray_ISCONTIGUOUS(pao)) {
//        throw std::invalid_argument("out array must be C-contiguous");
//    }
//    if (!PyArray_ISBEHAVED(pao)) {
//        throw std::invalid_argument(
//            "out array must be aligned, writable and native byte order");
//    }
//    return out;
//}

//StridedView2D<void> get_strided_view(const py::array &arr) {
//
//    ArrayDescriptor desc = get_descriptor(arr);
//    StridedView2D<void> view;
//    view.strides = {desc.strides[0], desc.strides[1]};
//    view.shape = {desc.shape[0], desc.shape[1]}; // TODO: check dim
//    view.data = arr.data(); // TODO: get mutable
//    return view;
//}

// Check that x, y are well prepared to be used to compute distances.

void validate_prepared_xy(
    const py::array &x,
    const py::array &y,
    JoinMode join_mode,
    const py::dtype &expected_dtype) {

    if (!(x.ndim() == 2 && y.ndim() == 2 && x.shape(1) == y.shape(1))) {
        throw std::invalid_argument("x, y has unexpected shape");
    }
    if (!(join_mode == JoinMode::Cross || x.shape(0) == y.shape(0))) {
        throw std::invalid_argument("x, y has unexpected shape");
    }

    if (!(x.dtype().equal(expected_dtype) && y.dtype().equal(expected_dtype))) {
        throw std::invalid_argument("x, y has unexpected dtype");
    }

    // PyArray_ElementStrides() is not exported, but get_descriptor()
    // performs that check.
}

template <class T>
struct metric_traits {};

template <>
struct metric_traits<EuclideanDistance2> {
    static constexpr MetricClass metric_class = MetricClass::Real;
#if SPECIALIZE_FLOAT32
    using supported_value_types = std::tuple<float, double, long double>;
#else
    using supported_value_types = std::tuple<double, long double>;
#endif
};

template <class T, class U>
struct is_type_in {};

template <class T>
struct is_type_in<T, std::tuple<>> : std::false_type {};

template <class T,  class U, class... Rest>
struct is_type_in<T, std::tuple<U, Rest...>> : is_type_in<T, std::tuple<Rest...>> {};

template <class T,  class... Rest>
struct is_type_in<T, std::tuple<T, Rest...>> : std::true_type {};


// Compute the given metric between rows of 2D arrays x and y using the given
// join mode, and store the result into out.  Elements of the input arrays
// must be of type InputValueType.
template <typename InputValueType, typename Distance>
py::array compute_distance_t(const py::object &out_obj,
                             const py::array &x,
                             const py::array &y,
                             JoinMode join_mode,
                             Distance &&distance) {

    static_assert(is_type_in<
        InputValueType, typename metric_traits<Distance>::supported_value_types>::value,
        "");

    using input_type = InputValueType;
    using span_type = SimpleSpan<const input_type>;
    using output_type = std::invoke_result_t<Distance, span_type, span_type>;

    validate_prepared_xy(x, y, join_mode, py::dtype::of<InputValueType>());

    using index_t = npy_intp;

    const index_t p = x.shape(0);
    const index_t q = y.shape(0);
    const index_t n = x.shape(1);

    std::vector<index_t> out_shape;
    switch (join_mode) {
    case JoinMode::Cross:
        out_shape.push_back(p);
        out_shape.push_back(q);
        break;
    case JoinMode::Upper:
        // assert p == q
        out_shape.push_back(p * (p - 1) / 2);
        break;
    case JoinMode::Inner:
        out_shape.push_back(std::min(p, q));
        break;
    default:
        throw std::invalid_argument("invalid join_mode");
    }

    py::array out = prepare_out_argument(out_obj, py::dtype::of<output_type>(), out_shape);

    const input_type * const x_data = static_cast<const input_type *>(x.data());
    const input_type * const y_data = static_cast<const input_type *>(y.data());

    ArrayDescriptor x_arr = get_descriptor(x);
    ArrayDescriptor y_arr = get_descriptor(y);
    if (!(x_arr.ndim == 2 && y_arr.ndim == 2)) {
        throw std::invalid_argument("invalid descriptor");
    }
    const index_t x_stride_0 = x_arr.strides[0];
    const index_t x_stride_1 = x_arr.strides[1];
    const index_t y_stride_0 = y_arr.strides[0];
    const index_t y_stride_1 = y_arr.strides[1];

    output_type *out_data = static_cast<output_type*>(out.mutable_data());

    // Release the GIL while running the double loop.
    {
        py::gil_scoped_release guard;

        // If x and/or y is not c-contiguous, allocate a contiguous buffer to
        // hold the current row of computation, so that the compiler has more
        // room to optimize and no additional template instantiation is needed
        // for strided input.
        const bool x_contiguous = (x_stride_1 == 1);
        const bool y_contiguous = (y_stride_1 == 1);
        std::vector<input_type> x_buffer(x_contiguous ? 0 : n);
        std::vector<input_type> y_buffer(y_contiguous ? 0 : n);

        const input_type *x_row = x_data;
        for (index_t i = 0; i < p; ++i) {
            if (!x_contiguous) /* unlikely */ {
                for (index_t k = 0; k < n; ++k) {
                    x_buffer[k] = x_row[k * x_stride_1];
                }
            }
            span_type x_span{x_contiguous ? x_row : x_buffer.data(),
                             static_cast<typename span_type::size_type>(n)};

            const index_t j_first = (join_mode == JoinMode::Cross) ? 0 :
                                    (join_mode == JoinMode::Inner) ? i : (i + 1);
            const index_t j_last = (join_mode == JoinMode::Inner) ? (i + 1) : q;

            const input_type *y_row = y_data + y_stride_0 * j_first;
            for (index_t j = j_first; j < j_last; ++j) {
                if (!y_contiguous) /* unlikely */ {
                    for (index_t k = 0; k < n; ++k) {
                        y_buffer[k] = y_row[k * y_stride_1];
                    }
                }
                span_type y_span{y_contiguous ? y_row : y_buffer.data(),
                                 static_cast<typename span_type::size_type>(n)};
                output_type d = distance(x_span, y_span);
                *out_data++ = d; // out MUST be C-Contiguous
                y_row += y_stride_0;
            }
            x_row += x_stride_0;
        }
    }
    return out;
}

// Compute the given metric between rows of 2D arrays x and y using the given
// join mode, and store the result into out.  The dtype of x and y must match
// and must be equal to one of SupportedInputValueTypes.
template <typename Distance>
py::array compute_distance(const py::object &out_obj,
                           const py::object &x_obj,
                           const py::object &y_obj,
                           JoinMode join_mode,
                           Distance &&distance) {

    constexpr MetricClass metric_class = metric_traits<Distance>::metric_class;
    using supported_value_types = typename metric_traits<Distance>::supported_value_types;

    auto xy = prepare_xy(x_obj, y_obj, join_mode, py::none(), metric_class);
    py::array x = xy.first;
    py::array y = xy.second;
    py::dtype input_dtype = x.dtype();

    // Type switches are enclosed in `if constexpr(...)` to avoid unused
    // template instantiations.
//    if constexpr(is_type_in<bool, supported_value_types>::value) {
//        if (input_dtype.equal(py::dtype::of<bool>()) {
//            compute_distance_t<bool>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
#if SPECIALIZE_FLOAT32
    if constexpr(is_type_in<float, supported_value_types>::value) {
        if (input_dtype.equal(py::dtype::of<float>())) {
            return compute_distance_t<float>(out_obj, x, y, join_mode, std::forward<Distance>(distance));
        }
    }
#endif
    if constexpr(is_type_in<double, supported_value_types>::value) {
        if (input_dtype.equal(py::dtype::of<double>())) {
            return compute_distance_t<double>(out_obj, x, y, join_mode, std::forward<Distance>(distance));
        }
    }
    if constexpr(is_type_in<long double, supported_value_types>::value) {
        if (input_dtype.equal(py::dtype::of<long double>())) {
            return compute_distance_t<long double>(out_obj, x, y, join_mode, std::forward<Distance>(distance));
        }
    }
    throw std::invalid_argument("unexpected input dtype");
}

//// Compute the given metric between rows of 2D arrays x and y using the given
//// join mode, and store the result into out.  The dtype of x and y must match
//// and must be equal to one of SupportedInputValueTypes.
//template <MetricClass metric_class, class Distance, class CtorArgs...>
//void compute_weighted_distance(const py::object &out_obj,
//                               const py::object &x_obj, const py::array &y_obj,
//                               JoinMode join_mode, const py::object &w_obj,
//                               WeightShape weight_shape,
//                               CtorArgs &&ctor_args...) {
//
//    py::array w = npy_asarray(w_obj);
//    auto xy = prepare_xy(x_obj, y_obj, join_mode, w.dtype(), metric_class);
//    py::array x = xy.first;
//    py::array y = xy.second;
//
//    if (weight_shape == WeightShape::Matrix) {
//        w = prepare_weight_matrix(...);
//    } else {
//        throw std::invalid_argument("unsupported weight_shape");
//    }
//
//    if constexpr(metric_class == MetricClass::Real) {
//        py::dtype = w.dtype();
//        if (!(x.dtype().equal(dtype) && y.dtype().equal(dtype))) {
//            throw std::invalid_argument("x, y, w must have same dtype for real metric");
//        }
//#if SPECIALIZE_FLOAT32
//        if (dtype.equal(py::dtype::of<float>()) {
//            StridedView2D<float> w_view = ...;
//            using TypedDistance = typename Distance::rebind<float>;
//            TypedDistance typed_distance{w_view, std::forward(ctor_args...)};
//            return compute_distance(out, x, y, join_mode, w_view, std::move(typed_distance));
//            return;
//        }
//    }
//    if constexpr(is_type_in<double, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<double>()) {
//            compute_distance_t<double>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    if constexpr(is_type_in<long double, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<long double>()) {
//            compute_distance_t<long double>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    }
//    }
//    else {
//        static_assert(false, "invalid metric_class");
//    }
//
//    py::dtype input_dtype = x.dtype();
//    if (!input_dtype.equal(y.dtype())) {
//        throw std::invalid_argument("x and y must have the same dtype");
//    }
//
//    // Type switches are enclosed in `if constexpr(...)` to avoid unused
//    // template instantiations.
//    if constexpr(is_type_in<bool, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<bool>()) {
//            compute_distance_t<bool>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    if constexpr(is_type_in<float, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<float>()) {
//            compute_distance_t<float>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    if constexpr(is_type_in<double, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<double>()) {
//            compute_distance_t<double>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    if constexpr(is_type_in<long double, SupportedInputValueTypes>::value) {
//        if (input_dtype.equal(py::dtype::of<long double>()) {
//            compute_distance_t<long double>(out, x, y, join_mode, std::forward(distance));
//            return;
//        }
//    }
//    throw std::invalid_argument("unexpected input dtype");
//}

//py::array xdist_dice(JoinMode join_mode,
//                     py::object x_obj, py::object y_obj=py::none(),
//                     py::object w_obj=py::none(), py::object out_obj=py::none()) {
//
//    // x, y, w, out: py::array
//    auto [x, y] = prepare_xy(x_obj, y_obj, return_type, MetricDomain::Boolean);
//    py::array out = prepare_out(...);
//
//    if (w_obj.is_none()) {
//        compute_distance(out, x, y, join_mode, UnweightedDiceDistance{});
//    } else {
//        py:array w = prepare_weight(w_obj, x, y, join_mode, ...);
//        if (w.dtype().equals(py::dtype::of<double>())) {
//            StridedView2D<double> w_view = w;
//            compute_distance(out, x, y, join_mode, WeightedDiceDistance{w_view});
//        } else if (w.dtype().equals(py::dtype::of<long double>())) {
//            StridedView2D<long double> w_view = w;
//            compute_distance(out, x, y, join_mode, WeightedDiceDistance{w_view});
//        } else {
//            throw std::invalid_argument("unexpected w dtype");
//        }
//    }
//    return out;
//}

py::array xdist_euclidean(char join_mode, py::object x_obj, py::object y_obj,
                          py::object w_obj, py::object out_obj) {
    if (w_obj.is_none()) {
        return compute_distance(out_obj, x_obj, y_obj, static_cast<JoinMode>(join_mode), EuclideanDistance2{});
    } else {
        throw std::invalid_argument("not implemented");
    }
//        py::array w = npy_asarray(w_obj);
//        auto [x, y] = prepare_xy(x_obj, y_obj, join_mode, MetricClass::Real, w.dtype());
//        w = prepare_w(w, weightShape::Vector, x.shape(1), x.dtype());
//        return compute_weighted_distance<EuclideanDistance2>(out, x, y, join_mode, w);
}

//py::array xdist_mahalanobis(JoinMode join_mode, py::object x_obj, py::object y_obj,
//                            py::object w_obj, py::object out_obj) {
//
//    // x, y, w: py::array
//    auto [x, y] = ...
//    auto w = ..., w] = prepare_xyw(x_obj, y_obj, w_obj, return_type,
//                                 MetricClass::Real, WeightShape::Matrix);
//
//    // out no need ? can move to compute
////    py::array out = prepare_output_for_real_metric(x, y, w, return_type, out_obj);
//
//    return compute_weighted_distance<MahalanobisDistance>(out, x, y, join_mode, w);
//}

///////////////////////////////////////////////////////////////
// END OF NEW STUFF
///////////////////////////////////////////////////////////////

PYBIND11_MODULE(_distance_pybind, m) {
    if (_import_array() != 0) {
        throw py::error_already_set();
    }
    using namespace pybind11::literals;

    // Helper functions.
//    m.def("get_native_floating_dtype", get_native_floating_dtype, "dtype"_a);
//    m.def("prepare_input", prepare_input,
//          "x"_a, "y"_a, "return_type"_a, "extra_dtype"_a=py::none(), "metric_class"_a="real");
//    m.def("prepare_weight_matrix", prepare_weight_matrix,
//          "w"_a, "n"_a, "least_dtype"_a=py::none(), "promote_shape"_a=false);

    // Individual distance functions.
//    m.def("xdist_mahalanobis", xdist_mahalanobis,
//          "join_mode"_a, "x"_a, "y"_a, py::kw_only(), "w"_a, "out"_a=py::none());

    m.def("xdist_euclidean", xdist_euclidean,
          "join_mode"_a, "x"_a, "y"_a, py::kw_only(), "w"_a=py::none(), "out"_a=py::none());

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
