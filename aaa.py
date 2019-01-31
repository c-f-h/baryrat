"""A Python implementation of the AAA algorithm for rational approximation.

For more information, see the paper

  The AAA Algorithm for Rational Approximation
  Yuji Nakatsukasa, Olivier Sète, and Lloyd N. Trefethen
  SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522

as well as the Chebfun package <http://www.chebfun.org>. This code is an almost
direct port of the Chebfun implementation of aaa to Python.
"""

import numpy as np
import scipy.linalg

class BarycentricRational:
    """A class representing a rational function in barycentric representation.
    """
    def __init__(self, z, f, w):
        """Barycentric representation of rational function with nodes z, values f and weights w.

        The rational function has the interpolation property r(z_j) = f_j.
        """
        self.nodes = z
        self.values = f
        self.weights = w

    def __call__(self, x):
        """Evaluate rational function at all points of `x`"""
        zj,fj,wj = self.nodes, self.values, self.weights

        xv = np.asanyarray(x).ravel()
        # ignore inf/nan for now
        with np.errstate(divide='ignore', invalid='ignore'):
            C = 1.0 / (xv[:,None] - zj[None,:])
            r = C.dot(wj*fj) / C.dot(wj)

        # for z in zj, the above produces NaN; we check for this
        nans = np.nonzero(np.isnan(r))[0]
        for i in nans:
            # find closest support node from zj
            dist = abs(xv[i] - zj)
            j = np.argmin(dist)
            # if very close, replace r[i] with the value of fj at zj;
            # otherwise, it could be a legitimate NaN
            if dist[j] < 1e-12:
                r[i] = fj[j]

        if np.isscalar(x):
            return r[0]
        else:
            r.shape = x.shape
            return r

    def polres(self):
        """Return the poles and residues of the rational function."""
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        # compute poles
        B = np.eye(m+1)
        B[0,0] = 0
        E = np.block([[0, wj],
                      [np.ones((m,1)), np.diag(zj)]])
        evals = scipy.linalg.eigvals(E, B)
        pol = np.real_if_close(evals[np.isfinite(evals)])

        # compute residues via formula for simple poles of quotients of analytic functions
        C_pol = 1.0 / (pol[:,None] - zj[None,:])
        N_pol = C_pol.dot(fj*wj)
        Ddiff_pol = (-C_pol**2).dot(wj)
        res = N_pol / Ddiff_pol

        return pol, res

    def zeros(self):
        """Return the zeros of the rational function."""
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        B = np.eye(m+1)
        B[0,0] = 0
        E = np.block([[0, wj],
                      [fj[:,None], np.diag(zj)]])
        evals = scipy.linalg.eigvals(E, B)
        return np.real_if_close(evals[np.isfinite(evals)])

################################################################################

def aaa(F, Z, tol=1e-13, mmax=100, return_errors=False):
    """Compute a rational approximation of `F` over the points `Z`.

    The nodes `Z` should be given as an array.

    `F` can be given as a function or as an array of function values over `Z`.

    Returns a `BarycentricRational` instance which can be called to evaluate
    the rational function, and can be queried for the poles, residues, and
    zeros of the function.
    """
    Z = np.asanyarray(Z).ravel()
    if callable(F):
        # allow functions to be passed
        F = F(Z)
    F = np.asanyarray(F).ravel()

    J = list(range(len(F)))
    zj = np.empty(0, dtype=Z.dtype)
    fj = np.empty(0, dtype=F.dtype)
    C = []
    errors = []

    reltol = tol * np.linalg.norm(F, np.inf)

    R = np.mean(F) * np.ones_like(F)

    for m in range(mmax):
        # find largest residual
        jj = np.argmax(abs(F - R))
        zj = np.append(zj, (Z[jj],))
        fj = np.append(fj, (F[jj],))
        J.remove(jj)

        # Cauchy matrix containing the basis functions as columns
        C = 1.0 / (Z[J,None] - zj[None,:])
        # Loewner matrix
        A = (F[J,None] - fj[None,:]) * C

        # compute weights as right singular vector for smallest singular value
        _, _, Vh = np.linalg.svd(A, full_matrices=False)
        wj = Vh[-1, :]

        # approximation: numerator / denominator
        N = C.dot(wj * fj)
        D = C.dot(wj)

        # update residual
        R = F.copy()
        R[J] = N / D

        # check for convergence
        errors.append(np.linalg.norm(F - R, np.inf))
        if errors[-1] <= reltol:
            break

    r = BarycentricRational(zj, fj, wj)
    return (r, errors) if return_errors else r
