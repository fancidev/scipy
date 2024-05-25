// Functions related to distance metrics.

#pragma once

#include "config.h"
//#include "tools.h"
//#include "error.h"

namespace special {

SPECFUN_HOST_DEVICE inline double js_div(double a, double b) {

    if (std::isnan(a)) {
        return a;
    }
    if (std::isnan(b)) {
        return b;
    }
    if (!(a >= 0 && a <= 1 && b >= 0 && b <= 1)) {
        return std::numeric_limits<double>::infinity();
    }

    const double c = (a+b)/2;
    if (a == 0 || b == 0) {
        return c*std::log(2.0);  // could be -0.0
    }

    const double t = (b-a)/(b+a);
    if (std::abs(t) <= 0.5) {
        return c*( t*std::atanh(t) + 0.5*std::log1p(-t*t) );  // fma?
    } else {
        return 0.5*( a*std::log(a/c) + b*std::log(b/c) );
    }
}

SPECFUN_HOST_DEVICE inline float js_div(float a, float b) {
    return js_div(static_cast<double>(a), static_cast<double>(b));
}

} // namespace special
