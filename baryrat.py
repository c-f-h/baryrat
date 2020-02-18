"""A Python package for barycentric rational approximation.
"""

import numpy as np
import scipy.linalg

def _compute_roots(w, x, use_mp):
    # Cf.:
    # Knockaert, L. (2008). A simple and accurate algorithm for barycentric
    # rational interpolation. IEEE Signal processing letters, 15, 154-157.
    if use_mp:
        from mpmath import mp
        if use_mp is True:
            use_mp = 100
        mp.dps = use_mp

        ak = mp.matrix(w)
        ak /= sum(ak)
        bk = mp.matrix(x)

        M = mp.diag(bk)
        for i in range(M.rows):
            for j in range(M.cols):
                M[i,j] -= ak[i]*bk[j]
        lam = mp.eig(M, left=False, right=False)
        lam = np.array(lam, dtype=complex)
    else:
        # the same procedure in standard double precision
        ak = w / w.sum()
        M = np.diag(x) - np.outer(ak, x)
        lam = scipy.linalg.eigvals(M)

    # remove one simple root
    lam = np.delete(lam, np.argmin(abs(lam)))
    return np.real_if_close(lam)


class BarycentricRational:
    """A class representing a rational function in barycentric representation.
    """
    def __init__(self, z, f, w):
        """Barycentric representation of rational function with nodes z, values f and weights w.

        The rational function has the interpolation property r(z_j) = f_j.
        """
        self.nodes = np.asanyarray(z)
        self.values = np.asanyarray(f)
        self.weights = np.asanyarray(w)

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
            # is xv[i] one of our nodes?
            nodeidx = np.nonzero(xv[i] == zj)[0]
            if len(nodeidx) > 0:
                # then replace the NaN with the value at that node
                r[i] = fj[nodeidx[0]]

        if np.isscalar(x):
            return r[0]
        else:
            r.shape = x.shape
            return r

    def poles(self, use_mp=False):
        """Return the poles of the rational function.

        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result using 100-digit precision arithmetic. If an integer is passed,
        uses that number of digits to compute the result.
        """
        return _compute_roots(self.weights, self.nodes, use_mp=use_mp)

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

    def zeros(self, use_mp=False):
        """Return the zeros of the rational function.

        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result using 100-digit precision arithmetic. If an integer is passed,
        uses that number of digits to compute the result.
        """
        if use_mp:
            return _compute_roots(self.weights*self.values, self.nodes,
                    use_mp=use_mp)
        else:
            zj,fj,wj = self.nodes, self.values, self.weights
            B = np.eye(len(wj) + 1)
            B[0,0] = 0
            E = np.block([[0, wj],
                          [fj[:,None], np.diag(zj)]])
            evals = scipy.linalg.eigvals(E, B)
            return np.real_if_close(evals[np.isfinite(evals)])

    def reciprocal(self):
        """Return a new `BarycentricRational` which is the reciprocal of this one."""
        return BarycentricRational(
                self.nodes.copy(),
                1 / self.values,
                self.weights * self.values)

################################################################################

def aaa(Z, F, tol=1e-13, mmax=100, return_errors=False):
    """Compute a rational approximation of `F` over the points `Z` using the
    AAA algorithm.

    Arguments:
        Z (array): the sampling points of the function. Unlike for interpolation
            algorithms, where a small number of nodes is preferred, since the
            AAA algorithm chooses its support points adaptively, it is better
            to provide a finer mesh over the support.
        F: the function to be approximated; can be given as a function or as an
            array of function values over ``Z``.
        tol: the approximation tolerance
        mmax: the maximum number of iterations/degree of the resulting approximant
        return_errors: if `True`, also return the history of the errors over
            all iterations

    Returns:
        BarycentricRational: an object which can be called to evaluate the
        rational function, and can be queried for the poles, residues, and
        zeros of the function.

    For more information, see the paper

      | The AAA Algorithm for Rational Approximation
      | Yuji Nakatsukasa, Olivier Sete, and Lloyd N. Trefethen
      | SIAM Journal on Scientific Computing 2018 40:3, A1494-A1522

    as well as the Chebfun package <http://www.chebfun.org>. This code is an
    almost direct port of the Chebfun implementation of aaa to Python.
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
        _, _, Vh = np.linalg.svd(A)
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

def interpolate_rat(nodes, values):
    """Compute a rational function which passes through all given node/value
    pairs. The number of nodes must be odd, and they should be passed in
    strictly increasing or strictly decreasing order.
    """
    values = np.asanyarray(values)
    nodes = np.asanyarray(nodes)
    n = len(values) // 2 + 1
    m = n - 1
    if not len(values) == n + m or not len(nodes) == n + m:
        raise ValueError('number of nodes should be odd')
    xa, xb = nodes[0::2], nodes[1::2]
    va, vb = values[0::2], values[1::2]
    B = (vb[:, None] - va[None, :]) / (xb[:, None] - xa[None, :])
    _, _, Vh = np.linalg.svd(B)
    weights = Vh[-1, :]
    assert len(weights) == n
    return BarycentricRational(xa, va, weights)

def interpolate_poly(nodes, values):
    """Compute the interpolating polynomial for the given nodes and values in
    barycentric form.
    """
    n = len(nodes)
    if n != len(values):
        raise ValueError('input arrays should have the same length')
    x = nodes
    weights = np.array([
            1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
            for i in range(n)
    ])
    return BarycentricRational(nodes, values, weights)

def interpolate_with_poles(nodes, values, poles):
    """Compute a rational function which interpolates the given values at the
    given nodes and which has the given poles.

    The arrays ``nodes`` and ``values`` should have length `n`, and
    ``poles`` should have length `n - 1`.
    """
    n = len(nodes)
    if n != len(values) or n != len(poles) + 1:
        raise ValueError('invalid length of arrays')
    nodes = np.asanyarray(nodes)
    values = np.asanyarray(values)
    poles = np.asanyarray(poles)
    # compute Cauchy matrix
    C = 1.0 / (poles[:,None] - nodes[None,:])
    # compute null space
    _, _, Vh = np.linalg.svd(C)
    weights = Vh[-1, :]
    return BarycentricRational(nodes, values, weights)

def floater_hormann(nodes, values, blending):
    """Compute the Floater-Hormann rational interpolant for the given nodes and
    values. See (Floater, Hormann 2007), DOI 10.1007/s00211-007-0093-y.

    The blending parameter (usually called `d` in the literature) is an integer
    between 0 and n (inclusive), where n+1 is the number of interpolation
    nodes. For functions with higher smoothness, the blending parameter may be
    chosen higher. For d=n, the result is the polynomial interpolant.

    Returns an instance of `BarycentricRational`.
    """
    n = len(values) - 1
    if n != len(nodes) - 1:
        raise ValueError('input arrays should have the same length')
    if not (0 <= blending <= n):
        raise ValueError('blending parameter should be between 0 and n')

    weights = np.zeros(n + 1)
    # abbreviations to match the formulas in the literature
    d = blending
    x = nodes
    for i in range(n + 1):
        Ji = range(max(0, i-d), min(i, n-d) + 1)
        weight = 0.0
        for k in Ji:
            weight += np.prod([1.0 / abs(x[i] - x[j])
                    for j in range(k, k+d+1)
                    if j != i])
        weights[i] = (-1.0)**(i-d) * weight
    return BarycentricRational(nodes, values, weights)
