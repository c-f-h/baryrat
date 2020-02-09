# Barycentric rational approximation [![Build Status](https://travis-ci.com/c-f-h/baryrat.svg?branch=master)](https://travis-ci.com/c-f-h/baryrat)

This is a pure Python package which provides routines for rational and
polynomial approximation through the so-called barycentric representation.
The advantage of this representation is (often significantly) improved
stability over classical approaches.

See the [API documentation](https://baryrat.readthedocs.io/) for an overview of
the available functions.

## Features

### The AAA algorithm

The package includes a Python implementation of the AAA algorithm for rational
approximation described in the paper "The AAA Algorithm for Rational
Approximation" by Yuji Nakatsukasa, Olivier Sète, and Lloyd N. Trefethen, SIAM
Journal on Scientific Computing 2018 40:3, A1494-A1522.
[(doi)](https://doi.org/10.1137/16M1106122)

A MATLAB implementation of this algorithm is contained in
[Chebfun](http://www.chebfun.org/).  The present Python version is a more or
less direct port of the MATLAB version.

The "cleanup" feature for spurious poles and zeros is not currently implemented.

### Further algorithms

The package includes functions for polynomial interpolation, rational
interpolation with either fixed poles or fixed interpolation nodes,
Floater-Hormann interpolation, and more.

## Installation

The implementation is in pure Python and requires only numpy and scipy as
dependencies. Install it using pip:

    pip install baryrat

## Usage

Here's an example of how to approximate a function in the interval [0,1]
using the AAA algorithm:

    import numpy as np
    from baryrat import aaa

    Z = np.linspace(0.0, 1.0, 1000)
    F = np.exp(Z) * np.sin(2*np.pi*Z)

    r = aaa(Z, F, mmax=10)

Instead of the maximum number of terms `mmax`, it's also possible to specify
the error tolerance `tol`.  Both arguments work exactly as in the MATLAB
version.

The returned object `r` is an instance of the class
`baryrat.BarycentricRational` and can be called like a function. For instance,
you can compute the error on `Z` like this:

    err = F - r(Z)
    print(np.linalg.norm(err, np.inf))

If you are interested in the poles and residues of the computed rational function,
you can query them like

    pol, res = r.polres()

and the zeroes using

    zer = r.zeros()

Finally, the nodes, values and weights used for interpolation (called `zj`,
`fj` and `wj` in the original implementation) can be accessed as properties:

    r.nodes
    r.values
    r.weights

