# The AAA algorithm for rational approximation [![Build Status](https://travis-ci.com/c-f-h/aaa.svg?branch=master)](https://travis-ci.com/c-f-h/aaa)

This is a Python implementation of the AAA algorithm for rational approximation
described in the paper "The AAA Algorithm for Rational Approximation" by Yuji
Nakatsukasa, Olivier Sète, and Lloyd N. Trefethen, SIAM Journal on Scientific
Computing 2018 40:3, A1494-A1522.

A MATLAB implementation of this algorithm is contained in [Chebfun](http://www.chebfun.org/).
The present Python version is a more or less direct port of the MATLAB version.

## Installation

The implementation is in pure Python and requires only numpy and scipy as
dependencies. Install it using pip:

    pip install aaa-approx

## Usage

Here's an example of how to approximate a function in the interval [0,1]:

    import numpy as np
    from aaa import aaa

    Z = np.linspace(0.0, 1.0, 1000)
    F = np.exp(Z) * np.sin(2*np.pi*Z)

    r = aaa(F, Z, mmax=10)

Instead of the maximum number of terms `mmax`, it's also possible to specify
the error tolerance `tol`.  Both arguments work exactly as in the MATLAB
version.

The returned object `r` is an instance of the class `aaa.BarycentricRational` and can
be called like a function. For instance, you can compute the error on `Z` like this:

    err = F - r(Z)
    print(np.linalg.norm(err, np.inf))

If you are interested in the poles and residues of the computed rational function,
you can query them like

    pol,res = r.polres()

and the zeroes using

    zer = r.zeros()

Finally, the nodes, values and weights used for interpolation (called `zj`, `fj`
and `wj` in the original implementation) can be accessed as properties:

    r.nodes
    r.values
    r.weights

