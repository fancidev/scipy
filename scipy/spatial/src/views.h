#pragma once

#include <vector>
#include <array>
#include <cstdint>

struct ArrayDescriptor {
    ArrayDescriptor(intptr_t ndim):
        ndim(ndim), shape(ndim, 1), strides(ndim, 0) {
    }

    intptr_t ndim;
    intptr_t element_size;
    std::vector<intptr_t> shape, strides;
};

template <typename T>
struct StridedView2D {
    std::array<intptr_t, 2> shape;
    std::array<intptr_t, 2> strides;
    T* data;

    T& operator()(intptr_t i, intptr_t j) {
        return data[i * strides[0] + j * strides[1]];
    }

    template <typename Other>
    StridedView2D<Other> rebind() {
        return StridedView2D<Other>{shape, strides, static_cast<Other*>(data)};
    }
};

inline void Precondition(bool condition) {
    // do nothing
}

template <typename T, int D>
struct ArrayView {
    static_assert(D >= 1, "D must be a positive integer");

    std::array<intptr_t, D> shape;
    std::array<intptr_t, D> strides;
    T* data;

    size_t size() const {
        size_t k = 1;
        for (size_t i = 0; i < D; i++) {
            k *= shape[i];
        }
        return k;
    }

    typename std::enable_if<D == 1, T&>::type operator[](intptr_t i) {
        Precondition(i >= 0 && i < get<0>(shape));
        return data[i];
    }

    typename std::enable_if<D == 1, const T&>::type operator[](intptr_t i) const {
        Precondition(i >= 0 && i < get<0>(shape));
        return data[i];
    }

    typename std::enable_if<D >= 2, ArrayView<T,D-1>>::type operator[](intptr_t i) {
        Precondition(i >= 0 && i < get<0>(shape));
        ArrayView<T,D-1> result;
        result.shape.assign(shape.begin() + 1, shape.end());
        result.strides.assign(strides.begin() + 1, strides.end());
        result.data = data + strides[0];
        return result;
    }

    typename std::enable_if<D >= 2, ArrayView<const T,D-1>>::type operator[](intptr_t i) const {
        Precondition(i >= 0 && i < get<0>(shape));
        ArrayView<const T,D-1> result;
        result.shape.assign(shape.begin() + 1, shape.end());
        result.strides.assign(strides.begin() + 1, strides.end());
        result.data = data + strides[0];
        return result;
    }

    // maybe use mdspan?
    T& operator()(intptr_t i, intptr_t j) {
        return data[i * strides[0] + j * strides[1]];
    }

//    template <typename Other>
//    StridedView2D<Other> rebind() {
//        return StridedView2D<Other>{shape, strides, static_cast<Other*>(data)};
//    }
};

template <typename T> using MatrixView = ArrayView<T,2>;
template <typename T> using VectorView = ArrayView<T,1>;
