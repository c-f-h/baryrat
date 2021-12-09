# Barycentric rational approximation [![Build Status](https://github.com/c-f-h/baryrat/actions/workflows/python-package.yml/badge.svg)](https://github.com/c-f-h/baryrat/actions/workflows/python-package.yml)

This is a pure Python package which provides routines for rational and
polynomial approximation through the so-called barycentric representation.
The advantage of this representation is (often significantly) improved
stability over classical approaches.

See the [API documentation](https://baryrat.readthedocs.io/) for an overview of
the available functions.

## Features

### Best rational approximation using BRASIL

The package implements the novel BRASIL algorithm for best rational approximation;
see [the paper](https://doi.org/10.1007/s11075-020-01042-0) or
[the preprint](https://www.ricam.oeaw.ac.at/files/reports/20/rep20-37.pdf)
to learn more.

The following example computes the best uniform rational approximation of degree 5
to a given function in the interval [0, pi]:

```python
import numpy as np
import baryrat

def f(x): return np.sin(x) * np.exp(x)
r = baryrat.brasil(f, [0,np.pi], 5)
```

The rational function `r` can then be evaluated at arbitrary nodes, its poles computed,
and more. See the [documentation](https://baryrat.readthedocs.io/) for details.

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

### Extended precision arithmetic

From ``baryrat`` 2.0 forward, most functions in the package support computing in extended precision
using the [`mpmath`](https://mpmath.org/) package. To use this option, first
set the desired number of decimal digits to compute with

```python
from mpmath import mp
mp.dps = 100      # compute using 100-digit precision
```

Arrays of numbers should be represented as numpy arrays with the object datatype.
Don't use the ``mpmath`` matrix feature! For instance, use
`np.array(mp.linspace(0, 1, 100))` to create equispaced points in extended precision.

Most functions will autodetect if you pass such extended precision arrays and
use the corresponding extended precision arithmetic in that case. There is
also a `use_mp` flag for many functions, but it is only required to force
the use of extended precision even when the inputs are in double precision.

Also the `BarycentricRational` class supports having its nodes, values, and
weights stored in extended precision and will operate accordingly, for instance
when computing the poles.

## Installation

The implementation is in pure Python and requires only numpy and scipy as
dependencies. Install it using pip:

    pip install baryrat

## Usage

Here's an example of how to approximate a function in the interval [0,1]
using the AAA algorithm:

```python
import numpy as np
from baryrat import aaa

Z = np.linspace(0.0, 1.0, 1000)
F = np.exp(Z) * np.sin(2*np.pi*Z)

r = aaa(Z, F, mmax=10)
```

Instead of the maximum number of terms `mmax`, it's also possible to specify
the error tolerance `tol`.  Both arguments work exactly as in the MATLAB
version.

The returned object `r` is an instance of the class
`baryrat.BarycentricRational` and can be called like a function. For instance,
you can compute the error on `Z` like this:

```python
err = F - r(Z)
print(np.linalg.norm(err, np.inf))
```

If you are interested in the poles and residues of the computed rational function,
you can query them like

```python
pol, res = r.polres()
```

and the zeroes using

```python
zer = r.zeros()
```

Finally, the nodes, values and weights used for interpolation (called `zj`,
`fj` and `wj` in the original implementation) can be accessed as properties:

```python
r.nodes
r.values
r.weights
```

## Citing ``baryrat``

If you use this package in any published research, please cite the following publication where the package was first introduced:

* C. Hofreither. **An algorithm for best rational approximation based on barycentric rational interpolation.**
  *Numerical Algorithms*, 88(1):365--388, 2021. [(doi)](https://doi.org/10.1007/s11075-020-01042-0)
