#pragma once

#include <cmath>
#include "views.h"

#ifdef __GNUC__
#define ALWAYS_INLINE inline __attribute__((always_inline))
#define INLINE_LAMBDA __attribute__((always_inline))
#elif defined(_MSC_VER)
#define ALWAYS_INLINE __forceinline
#define INLINE_LAMBDA
#else
#define ALWAYS_INLINE inline
#define INLINE_LAMBDA
#endif

struct Identity {
    template <typename T>
    ALWAYS_INLINE T operator() (T && val) const {
        return std::forward<T>(val);
    }
};

struct Plus {
    template <typename T>
    ALWAYS_INLINE T operator()(T a, T b) const {
        return a + b;
    }
};

// Helper to force a fixed bound loop to be completely unrolled
template <int unroll>
struct ForceUnroll{
    template <typename Func>
    ALWAYS_INLINE void operator()(const Func& f) const {
        ForceUnroll<unroll - 1>{}(f);
        f(unroll - 1);
    }
};

template <>
struct ForceUnroll<1> {
    template <typename Func>
    ALWAYS_INLINE void operator()(const Func& f) const {
        f(0);
    }
};

template <int ilp_factor=4,
          typename T,
          typename TransformFunc,
          typename ProjectFunc = Identity,
          typename ReduceFunc = Plus>
void transform_reduce_2d_(
    StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y,
    const TransformFunc& map,
    const ProjectFunc& project = Identity{},
    const ReduceFunc& reduce = Plus{}) {
    // Result type of calling map
    using AccumulateType = typename std::decay<decltype(
        map(std::declval<T>(), std::declval<T>()))>::type;
    intptr_t xs = x.strides[1], ys = y.strides[1];

    intptr_t i = 0;
    if (xs == 1 && ys == 1) {
        for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
            const T* x_rows[ilp_factor];
            const T* y_rows[ilp_factor];
            ForceUnroll<ilp_factor>{}([&](int k) {
                x_rows[k] = &x(i + k, 0);
                y_rows[k] = &y(i + k, 0);
            });

            AccumulateType dist[ilp_factor] = {};
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                ForceUnroll<ilp_factor>{}([&](int k) {
                    auto val = map(x_rows[k][j], y_rows[k][j]);
                    dist[k] = reduce(dist[k], val);
                });
            }

            ForceUnroll<ilp_factor>{}([&](int k) {
                out(i + k, 0) = project(dist[k]);
            });
        }
    } else {
        for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
            const T* x_rows[ilp_factor];
            const T* y_rows[ilp_factor];
            ForceUnroll<ilp_factor>{}([&](int k) {
                x_rows[k] = &x(i + k, 0);
                y_rows[k] = &y(i + k, 0);
            });

            AccumulateType dist[ilp_factor] = {};
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                auto x_offset = j * xs;
                auto y_offset = j * ys;
                ForceUnroll<ilp_factor>{}([&](int k) {
                    auto val = map(x_rows[k][x_offset], y_rows[k][y_offset]);
                    dist[k] = reduce(dist[k], val);
                });
            }

            ForceUnroll<ilp_factor>{}([&](int k) {
                out(i + k, 0) = project(dist[k]);
            });
        }
    }
    for (; i < x.shape[0]; ++i) {
        const T* x_row = &x(i, 0);
        const T* y_row = &y(i, 0);
        AccumulateType dist = {};
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            auto val = map(x_row[j * xs], y_row[j * ys]);
            dist = reduce(dist, val);
        }
        out(i, 0) = project(dist);
    }
}

template <int ilp_factor=2, typename T,
          typename TransformFunc,
          typename ProjectFunc = Identity,
          typename ReduceFunc = Plus>
void transform_reduce_2d_(
    StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y,
    StridedView2D<const T> w, const TransformFunc& map,
    const ProjectFunc& project = Identity{},
    const ReduceFunc& reduce = Plus{}) {
    intptr_t i = 0;
    intptr_t xs = x.strides[1], ys = y.strides[1], ws = w.strides[1];
    // Result type of calling map
    using AccumulateType = typename std::decay<decltype(
        map(std::declval<T>(), std::declval<T>(), std::declval<T>()))>::type;

    for (; i + (ilp_factor - 1) < x.shape[0]; i += ilp_factor) {
        const T* x_rows[ilp_factor];
        const T* y_rows[ilp_factor];
        const T* w_rows[ilp_factor];
        ForceUnroll<ilp_factor>{}([&](int k) {
            x_rows[k] = &x(i + k, 0);
            y_rows[k] = &y(i + k, 0);
            w_rows[k] = &w(i + k, 0);
        });

        AccumulateType dist[ilp_factor] = {};
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            ForceUnroll<ilp_factor>{}([&](int k) {
                auto val = map(x_rows[k][j * xs], y_rows[k][j * ys], w_rows[k][j * ws]);
                dist[k] = reduce(dist[k], val);
            });
        }

        ForceUnroll<ilp_factor>{}([&](int k) {
            out(i + k, 0) = project(dist[k]);
        });
    }
    for (; i < x.shape[0]; ++i) {
        const T* x_row = &x(i, 0);
        const T* y_row = &y(i, 0);
        const T* w_row = &w(i, 0);
        AccumulateType dist = {};
        for (intptr_t j = 0; j < x.shape[1]; ++j) {
            auto val = map(x_row[j * xs], y_row[j * ys], w_row[j * ws]);
            dist = reduce(dist, val);
        }
        out(i, 0) = project(dist);
    }
}

struct MinkowskiDistance {
    double p_;

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        const T p = static_cast<T>(p_);
        const T invp = static_cast<T>(1.0 / p_);
        transform_reduce_2d_(out, x, y, [p](T x, T y) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return std::pow(diff, p);
        },
        [invp](T x) { return std::pow(x, invp); });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        const T p = static_cast<T>(p_);
        const T invp = static_cast<T>(1.0 / p_);
        transform_reduce_2d_(out, x, y, w, [p](T x, T y, T w) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return w * std::pow(diff, p);
        },
        [invp](T x) { return std::pow(x, invp); });
    }
};

struct EuclideanDistance {
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return diff * diff;
        },
        [](T x) { return std::sqrt(x); });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            auto diff = std::abs(x - y);
            return w * (diff * diff);
        },
        [](T x) { return std::sqrt(x); });
    }
};

struct ChebyshevDistance {
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            return std::abs(x - y);
        },
        Identity{},
        [](T x, T y) { return std::max(x, y); });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        for (intptr_t i = 0; i < x.shape[0]; ++i) {
            T dist = 0;
            for (intptr_t j = 0; j < x.shape[1]; ++j) {
                auto diff = std::abs(x(i, j) - y(i, j));
                if (w(i, j) > 0 && diff > dist) {
                    dist = diff;
                }
            }
            out(i, 0) = dist;
        }
    }
};

struct CityBlockDistance {
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            return std::abs(x - y);
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            return w * std::abs(x - y);
        });
    }
};

struct SquareEuclideanDistance {
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_(out, x, y, [](T x, T y) INLINE_LAMBDA {
            auto diff = x - y;
            return diff * diff;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            auto diff = x - y;
            return w * diff * diff;
        });
    }
};

struct BraycurtisDistance {
    template <typename T>
    struct Acc {
        Acc(): diff(0), sum(0) {}
        T diff, sum;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // dist = abs(x - y).sum() / abs(x + y).sum()
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.diff = std::abs(x - y);
            acc.sum = std::abs(x + y);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.diff / acc.sum;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.diff = a.diff + b.diff;
            acc.sum = a.sum + b.sum;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // dist = (w * abs(x - y)).sum() / (w * abs(x + y)).sum()
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.diff = w * std::abs(x - y);
            acc.sum = w * std::abs(x + y);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.diff / acc.sum;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.diff = a.diff + b.diff;
            acc.sum = a.sum + b.sum;
            return acc;
        });
    }
};

struct CanberraDistance {
    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        // dist = (abs(x - y) / (abs(x) + abs(y))).sum()
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            auto num = std::abs(x - y);
            auto denom = std::abs(x) + std::abs(y);
            // branchless replacement for (denom == 0) ? 0 : num / denom;
            return num / (denom + (denom == 0));
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        // dist = (w * abs(x - y) / (abs(x) + abs(y))).sum()
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            auto num = w * std::abs(x - y);
            auto denom = std::abs(x) + std::abs(y);
            // branchless replacement for (denom == 0) ? 0 : num / denom;
            return num / (denom + (denom == 0));
        });
    }
};

struct HammingDistance {
    template <typename T>
    struct Acc {
        Acc(): nonmatches(0), total(0) {}
        T nonmatches, total;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = x != y;
            acc.total = 1;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / acc.total;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.total = a.total + b.total;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = w * (x != y);
            acc.total = w;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / acc.total;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.total = a.total + b.total;
            return acc;
        });
    }
};

struct DiceDistance {
    template <typename T>
    struct Acc {
        Acc(): nonmatches(0), tt_matches(0) {}
        T nonmatches, tt_matches;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = x * (1.0 - y) + y * (1.0 - x);
            acc.tt_matches = x * y;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / (2*acc.tt_matches + acc.nonmatches);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.tt_matches = a.tt_matches + b.tt_matches;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = w * (x != y);
            acc.tt_matches = w * ((x != 0) & (y != 0));
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.nonmatches / (2*acc.tt_matches + acc.nonmatches);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.nonmatches = a.nonmatches + b.nonmatches;
            acc.tt_matches = a.tt_matches + b.tt_matches;
            return acc;
        });
    }
};

struct JaccardDistance {
    template <typename T>
    struct Acc {
        Acc(): num(0), denom(0) {}
        T num, denom;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = (x != y) & ((x != 0) | (y != 0));
            acc.denom = (x != 0) | (y != 0);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.denom != 0) * (acc.num / (1 * (acc.denom == 0) + acc.denom));
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = a.num + b.num;
            acc.denom = a.denom + b.denom;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = w * ((x != y) & ((x != 0) | (y != 0)));
            acc.denom = w * ((x != 0) | (y != 0));
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.denom != 0) * (acc.num / (1 * (acc.denom == 0) + acc.denom));
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.num = a.num + b.num;
            acc.denom = a.denom + b.denom;
            return acc;
        });
    }
};

struct RogerstanimotoDistance {
    template <typename T>
    struct Acc {
        Acc(): ndiff(0), n(0) {}
        T ndiff, n;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ndiff = (x != 0) != (y != 0);
            acc.n = 1;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.n + acc.ndiff);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ndiff = w * ((x != 0) != (y != 0));
            acc.n = w;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.n + acc.ndiff);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }
};

struct Kulczynski1Distance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0) {}
        T ntt, ndiff;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.ndiff = (x != 0) != (y != 0);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.ntt / acc.ndiff;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.ndiff = w * ((x != 0) != (y != 0));
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return acc.ntt / acc.ndiff;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            return acc;
        });
    }
};

struct RussellRaoDistance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), n(0) {}
        T ntt, n;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.n = 1;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.n - acc.ntt) / acc.n;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.n = a.n + b.n;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.n = w;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (acc.n - acc.ntt) / acc.n;
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.n = a.n + b.n;
            return acc;
        });
    }
};

struct SokalmichenerDistance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0), n(0) {}
        T ntt, ndiff, n;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.ndiff = (x != 0) != (y != 0);
            acc.n = 1;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.ndiff + acc.n);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.ndiff = w * ((x != 0) != (y != 0));
            acc.n = w;
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (acc.ndiff + acc.n);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            acc.n = a.n + b.n;
            return acc;
        });
    }
};

struct SokalsneathDistance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), ndiff(0) {}
        T ntt, ndiff;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<4>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.ndiff = (x != 0) != (y != 0);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (2 * acc.ndiff + acc.ntt);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.ndiff = w * ((x != 0) != (y != 0));
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            return (2 * acc.ndiff) / (2 * acc.ndiff + acc.ntt);
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.ndiff = a.ndiff + b.ndiff;
            return acc;
        });
    }
};

struct YuleDistance {
    template <typename T>
    struct Acc {
        Acc(): ntt(0), nft(0), nff(0), ntf(0) {}
        intptr_t ntt, nft, nff, ntf;
    };

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y) const {
        transform_reduce_2d_<2>(out, x, y, [](T x, T y) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = (x != 0) & (y != 0);
            acc.ntf = (x != 0) & (y == 0);
            acc.nft = (x == 0) & (y != 0);
            acc.nff = (x == 0) & (y == 0);
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            intptr_t half_R = acc.ntf * acc.nft;
            return (2. * half_R) / (acc.ntt * acc.nff + half_R + (half_R == 0));
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.nft = a.nft + b.nft;
            acc.nff = a.nff + b.nff;
            acc.ntf = a.ntf + b.ntf;
            return acc;
        });
    }

    template <typename T>
    void operator()(StridedView2D<T> out, StridedView2D<const T> x, StridedView2D<const T> y, StridedView2D<const T> w) const {
        transform_reduce_2d_<2>(out, x, y, w, [](T x, T y, T w) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = w * ((x != 0) & (y != 0));
            acc.ntf = w * ((x != 0) & (!(y != 0)));
            acc.nft = w * ((!(x != 0)) & (y != 0));
            acc.nff = w * ((!(x != 0)) & (!(y != 0)));
            return acc;
        },
        [](const Acc<T>& acc) INLINE_LAMBDA {
            intptr_t half_R = acc.ntf * acc.nft;
            return (2. * half_R) / (acc.ntt * acc.nff + half_R + (half_R == 0));
        },
        [](const Acc<T>& a, const Acc<T>& b) INLINE_LAMBDA {
            Acc<T> acc;
            acc.ntt = a.ntt + b.ntt;
            acc.nft = a.nft + b.nft;
            acc.nff = a.nff + b.nff;
            acc.ntf = a.ntf + b.ntf;
            return acc;
        });
    }
};

#define SPECIALIZE_FLOAT32 0

// 'assert' like function for code documentation
inline void Precondition(bool condition) {
    // do nothing
}

// Remarks: To simplify the code and to limit the number of template
// instantiations, each distance metric is parameterized by exactly
// one type parameter, T.  The input dtype, output dtype, as well as the
// dtype of any additional parameters (such as w) is a function of T.

// Computes output type of metrics from input type, where appropriate
template <typename T>
struct promoted_floating {
    using type = double;
};

template <>
struct promoted_floating<float> {
    using type = float;
};

template <>
struct promoted_floating<long double> {
    using type = long double;
};

template <typename T>
using promoted_floating_t = typename promoted_floating<T>::type;

template <typename T>
inline T fuzzy_and(T x, T y) {
    return x * y;
}

template <typename T>
inline T fuzzy_xor(T x, T y) {
    return x * (T(1) - y) + y * (T(1) - x);
}

template <typename T>
inline T fuzzy_nor(T x, T y) {
    return (T(1) - x) * (T(1) - y);
}

inline bool fuzzy_and(bool x, bool y) { return x & y; }
inline bool fuzzy_xor(bool x, bool y) { return x ^ y; }
inline bool fuzzy_nor(bool x, bool y) { return ~x & ~y; }

struct UnweightedDiceDistance {

    // todo: move this to compute_distance
    using supported_input_types = std::tuple<bool, double, long double>;

    // Boolean input; "quick" path
    template <class Span>
    std::enable_if_t<std::is_same_v<typename Span::value_type, bool>, double>
    operator()(const Span &x, const Span &y) const {
        Precondition(x.size() == y.size());
        using output_type = double;
        using working_type = std::size_t;
        using index_t = typename Span::size_type;

        const index_t n = x.size();
        working_type n_tt = 0;
        working_type n_tf_ft = 0;
        for (index_t i = 0; i < n; ++i) {
            bool x_i = x[i];
            bool y_i = y[i];
            n_tt += static_cast<working_type>(x_i & y_i);
            n_tf_ft += static_cast<working_type>(x_i ^ y_i);
        }
        // If all values in x and y are zero, nan is returned
        return static_cast<output_type>(n_tf_ft) / static_cast<output_type>(2*n_tt + n_tf_ft);
    }

    // Fuzzy boolean input; "slow" path
    // = 1 - 2*sum(x*y)/(sum(x)+sum(y))
    // x,y are "supposed" to be between 0.0 and 1.0, but this is not checked
    template <class Span>
    typename std::enable_if<!std::is_same<typename Span::value_type, bool>::value,
                            typename Span::value_type>::type
    operator()(const Span &x, const Span &y) const {
        Precondition(x.size() == y.size());
        using output_type = typename Span::value_type;
        using working_type = output_type;
        using index_t = typename Span::size_type;

        const index_t n = x.size();
        working_type s_and = 0;
        working_type s_xor = 0;
        for (index_t i = 0; i < n; ++i) {
            working_type x_i = static_cast<working_type>(x[i]);
            working_type y_i = static_cast<working_type>(y[i]);
            working_type one = 1;
            s_and += fuzzy_and(x_i, y_i);
            s_xor += fuzzy_xor(x_i, y_i);
        }
        // If all values in x and y are zero, nan is returned
        return s_xor / (2*s_and + s_xor);
    }
};

template <typename WeightType /* = std::span<double> */>
struct WeightedDiceDistance {

//    using supported_input_types = std::tuple<bool>;

    WeightType w;

    template <class Span>
    typename std::conditional<std::is_same<typename Span::value_type, bool>::value,
                              double, typename Span::value_type>::type
    operator()(const Span &x, const Span &y) const {
        Precondition(x.size() == w.size() && y.size() == w.size());

        using weight_type = typename WeightType::value_type;
        static_assert(std::is_same<typename Span::value_type, bool>::value ||
                      std::is_same<typename Span::value_type, weight_type>::value,
                      "Non-boolean x and y must have the same type as w");

        using output_type = typename std::conditional<
            std::is_same<typename Span::value_type, bool>::value,
            double, typename Span::value_type>::type;
        using index_t = typename Span::size_type;

        const index_t n = x.size();
        output_type s_and = 0;
        output_type s_xor = 0;
        for (index_t i = 0; i < n; ++i) {
            s_and += w[i] * fuzzy_and(x[i], y[i]);
            s_xor += w[i] * fuzzy_xor(x[i], y[i]);
        }
        // If all values in x and y are zero, nan is returned
        return (s_xor) / (2*s_and + s_xor);
    }
};


// Base class for metrics defined for real (or fuzzy boolean) input vectors.
// The general parameterization supports a weight parameter.
template <typename WeightType>
struct RealDistance {
    using weight_value_type = typename WeightType::value_type;

    // For weighted metrics, constrain the input dtype to be equal to the
    // weight dtype in order to limit the number of template instantiations.
    using supported_input_value_types = std::tuple<weight_value_type>;

    // Store the weight as a member.
    WeightType w;
};

// Specialized for unweighted real metrics.
template <>
struct RealDistance<void> {
    // More variety of input dtypes are supported for unweighted metrics.
#if SPECIALIZE_FLOAT32
    using supported_input_value_types = std::tuple<float, double, long double>;
#else
    using supported_input_value_types = std::tuple<double, long double>;
#endif
};

template <typename WeightType = void>
struct EuclideanDistance : RealDistance<WeightType> {

    // Unweighted
    template <typename Span>
    std::enable_if<std::is_void<WeightType>::value, typename Span::value_type>
    operator()(const Span &x, const Span &y) const {
        Precondition(x.size() == y.size());

        using input_type = typename Span::value_type;
        using output_type = input_type;
        using working_type = typename std::conditional<
            std::is_same<input_type, float>::value, double, input_type>::type;
        using index_t = typename Span::size_type;

        const index_t n = x.size();
        working_type s = 0;
        for (index_t i = 0; i < n; ++i) {
            working_type diff = static_cast<working_type>(x[i]) - static_cast<working_type>(y[i]);
            s += diff * diff;
        }
        return static_cast<output_type>(std::sqrt(s));
    }

    // Weighted [TO BE COMPLETED]
//    template <typename Span>
//    std::enable_if<!std::is_void<WeightType>::value, typename Span::value_type>
//    operator()(const Span &x, const Span &y) const {
//        Precondition(x.size() == y.size());
//
//        using input_type = typename Span::value_type;
//        using output_type = input_type;
//        using working_type = typename std::conditional<
//            std::is_same<input_type, float>::value, double, input_type>::type;
//        using index_t = typename Span::size_type;
//
//        const index_t n = x.size();
//        working_type s = 0;
//        for (index_t i = 0; i < n; ++i) {
//            working_type diff = static_cast<working_type>(x[i]) - static_cast<working_type>(y[i]);
//            s += diff * diff;
//        }
//        return static_cast<output_type>(std::sqrt(s));
//    }
};

// Straightforward implementation of Mahalanobis distance with known n-by-n
// weight matrix.  The time complexity is O(n^2).
template <typename WeightType>
struct MahalanobisDistance : RealDistance<WeightType> {

    template <typename Span>
    typename Span::value_type
    operator()(const Span &x, const Span &y) const {
        Precondition(w.shape(0) == w.shape(1));
        Precondition(x.size() == w.shape(0) && y.size() == w.shape(0));

        using input_type = typename Span::value_type;
        using output_type = input_type;
        using working_type = typename std::conditional<
            std::is_same<input_type, float>::value, double, input_value_type>;
        using index_t = typename Span::size_type;

        const index_t n = x.size();
        working_type s = 0;
        for (index_t i = 0; i < n; ++i) {
            working_type t = 0;
            for (index_t j = 0; j < n; ++j) {
                t += static_cast<working_type>(w(i, j)) * static_cast<working_type>(y[j]);
            }
            s += static_cast<working_type>(x[i]) * t;
        }
        return static_cast<output_type>(std::sqrt(s));
    }
};
