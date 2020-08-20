"""A Python package for barycentric rational approximation.
"""

import numpy as np
import scipy.linalg

__version__ = '1.2.0'

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

    Args:
        z (array): the interpolation nodes
        f (array): the values at the interpolation nodes
        w (array): the weights

    The rational function has the interpolation property r(z_j) = f_j at all
    nodes where w_j != 0.
    """
    def __init__(self, z, f, w):
        self.nodes = np.asanyarray(z)
        self.values = np.asanyarray(f)
        self.weights = np.asanyarray(w)

    def __call__(self, x):
        """Evaluate rational function at all points of `x`."""
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

    def polres(self, use_mp=False):
        """Return the poles and residues of the rational function.

        The ``use_mp`` argument has the same meaning as for ``poles()`` and
        is only used during computation of the poles.
        """
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        # compute poles
        if use_mp:
            pol = self.poles(use_mp=use_mp)
        else:
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

    def gain(self):
        """The gain in a poles-zeros-gain representation of the rational function,
        or equivalently, the value at infinity.
        """
        return np.sum(self.values * self.weights) / np.sum(self.weights)

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


def _piecewise_mesh(nodes, n):
    """Build a mesh over an interval with subintervals described by the array
    ``nodes``. Each subinterval has ``n`` points spaced uniformly between the
    two neighboring nodes.  The final mesh has ``(len(nodes) - 1) * n`` points.
    """
    #z = np.concatenate(([z0], nodes, [z1]))
    M = len(nodes)
    return np.concatenate(tuple(
        np.linspace(nodes[i], nodes[i+1], n, endpoint=(i==M-2))
        for i in range(M - 1)))

def local_maxima_bisect(g, nodes, num_iter=10):
    L, R = nodes[1:-2], nodes[2:-1]
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, (L + R) / 2, R))
    values = g(z[1])
    m = z.shape[1]

    for k in range(num_iter):
        # compute quarter points
        q = np.vstack(((z[0] + z[1]) / 2, (z[1] + z[2])/ 2))
        qval = g(q)

        # move triple of points to be centered on the maximum
        for j in range(m):
            maxk = np.argmax([qval[0,j], values[j], qval[1,j]])
            if maxk == 0:
                z[1,j], z[2,j] = q[0,j], z[1,j]
                values[j] = qval[0,j]
            elif maxk == 1:
                z[0,j], z[2,j] = q[0,j], q[1,j]
            else:
                z[0,j], z[1,j] = z[1,j], q[1,j]
                values[j] = qval[1,j]

    # find maximum per column (usually the midpoint)
    #maxidx = values.argmax(axis=0)
    # select abscissae and values at maxima
    #Z, gZ = z[maxidx, np.arange(m)], values[np.arange(m)]
    Z, gZ = np.empty(m+2), np.empty(m+2)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = values
    # treat the boundary intervals specially since usually the maximum is at the boundary
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def local_maxima_golden(g, nodes, num_iter):
    # vectorized version of golden section search
    golden_mean = (3.0 - np.sqrt(5.0)) / 2   # 0.381966...
    L, R = nodes[1:-2], nodes[2:-1]     # skip boundary intervals (treated below)
    # compute 3 x m array of endpoints and midpoints
    z = np.vstack((L, L + (R-L)*golden_mean, R))
    m = z.shape[1]
    all_m = np.arange(m)
    gB = g(z[1])

    for k in range(num_iter):
        # z[1] = midpoints
        mids = (z[0] + z[2]) / 2

        # compute new nodes according to golden section
        farther_idx = (z[1] <= mids).astype(int) * 2 # either 0 or 2
        X = z[1] + golden_mean * (z[farther_idx, all_m] - z[1])
        gX = g(X)

        for j in range(m):
            x = X[j]
            gx = gX[j]

            b = z[1,j]
            if gx > gB[j]:
                if x > b:
                    z[0,j] = z[1,j]
                else:
                    z[2,j] = z[1,j]
                z[1,j] = x
                gB[j] = gx
            else:
                if x < b:
                    z[0,j] = x
                else:
                    z[2,j] = x

    # prepare output arrays
    Z, gZ = np.empty(m+2), np.empty(m+2)
    Z[1:-1] = z[1, :]
    gZ[1:-1] = gB
    # treat the boundary intervals specially since usually the maximum is at the boundary
    # (no bracket available!)
    Z[0], gZ[0] = _boundary_search(g, nodes[0], nodes[1], num_iter=3)
    Z[-1], gZ[-1] = _boundary_search(g, nodes[-2], nodes[-1], num_iter=3)
    return Z, gZ

def _boundary_search(g, a, c, num_iter):
    X = [a, c]
    Xvals = [g(a), g(c)]
    max_side = 0 if (Xvals[0] >= Xvals[1]) else 1
    other_side = 1 - max_side

    for k in range(num_iter):
        xm = (X[0] + X[1]) / 2
        gm = g(xm)
        if gm < Xvals[max_side]:
            # no new maximum found; shrink interval and iterate
            X[other_side] = xm
            Xvals[other_side] = gm
        else:
            # found a bracket for the minimum
            return _golden_search(g, X[0], X[1], num_iter=num_iter-k)
    return X[max_side], Xvals[max_side]

def _golden_search(g, a, c, num_iter=20):
    golden_mean = 0.5 * (3.0 - np.sqrt(5.0))

    b = (a + c) / 2
    gb = g(b)
    ga, gc = g(a), g(c)
    if not (gb >= ga and gb >= gc):
        # not bracketed - maximum may be at the boundary
        return _boundary_search(g, a, c, num_iter)
    for k in range(num_iter):
        mid = (a + c) / 2
        if b > mid:
            x = b + golden_mean * (a - b)
        else:
            x = b + golden_mean * (c - b)
        gx = g(x)

        if gx > gb:
            # found a larger point, use it as center
            if x > b:
                a = b
            else:
                c = b
            b = x
            gb = gx
        else:
            # point is smaller, use it as boundary
            if x < b:
                a = x
            else:
                c = x
    return b, gb

def local_maxima_sample(g, nodes, N):
    Z = _piecewise_mesh(nodes, N).reshape((-1, N))
    vals = g(Z)
    maxk = vals.argmax(axis=1)
    nn = np.arange(Z.shape[0])
    return Z[nn, maxk], vals[nn, maxk]

def brasil(f, interval, deg, tol=1e-4, maxiter=1000, max_step_size=0.1,
        step_factor=0.1, npi=100, init_steps=100, poly=False, info=False):
    """Best Rational Approximation by Successive Interval Length adjustment.

    Arguments:
        f: the real scalar function to be approximated
        interval: the bounds (a, b) of the approximation interval
        deg: the degree of the numerator and denominator of the rational approximation
        tol: the maximum allowed deviation from equioscillation
        maxiter: the maximum number of iterations
        max_step_size: the maximum allowed step size
        step_factor: factor for adaptive step size choice
        npi: points per interval for error calculation. If `npi < 0`,
            golden section search with `-npi` iterations is used instead of
            sampling
        init_steps: how many steps of the initialization iteration to run
        poly: if true, compute polynomial best approximation instead
        info: whether to return an additional object with details

    Returns:
        BarycentricRational: the computed rational approximation. If `info` is
        True, instead returns a pair containing the approximation and an
        object with additional information (see below).

    The `info` object returned along with the approximation if `info=True` has
    the following members:

    * **converged** (bool): whether the method converged to the desired tolerance **tol**
    * **error** (float): the maximum error of the approximation
    * **deviation** (float): the relative error between the smallest and the largest
      equioscillation peak. The convergence criterion is **deviation** <= **tol**.
    * **nodes** (array): the abscissae of the interpolation nodes  (2*deg + 1)
    * **iterations** (int): the number of iterations used, including the initialization phase
    * **errors** (array): the history of the maximum error over all iterations
    * **deviations** (array): the history of the deviation over all iterations
    * **stepsizes** (array): the history of the adaptive step size over all iterations

    Additional information about the resulting rational function, such as poles,
    residues and zeroes, can be queried from the :class:`BarycentricRational` object
    itself.
    """
    a, b = interval
    assert a < b, 'Invalid interval'
    if poly:
        n = deg + 1
    else:
        n = 2 * deg + 1     # number of interpolation nodes
    errors = []
    stepsize = np.nan

    # start with Chebyshev nodes
    nodes = (1 - np.cos((2*np.arange(1,n+1) - 1) / (2*n) * np.pi)) / 2 * (b - a) + a

    interp = interpolate_poly if poly else interpolate_rat

    for k in range(init_steps + maxiter):
        r = interp(nodes, f(nodes))

        # determine local maxima per interval
        all_nodes = np.concatenate(([a], nodes, [b]))
        errfun = lambda x: abs(f(x) - r(x))
        if npi > 0:
            local_max_x, local_max = local_maxima_sample(errfun, all_nodes, npi)
        else:
            local_max_x, local_max = local_maxima_golden(errfun, all_nodes, num_iter=-npi)

        max_err = local_max.max()
        deviation = max_err / local_max.min() - 1
        errors.append((max_err, deviation, stepsize))

        converged = deviation <= tol
        if converged or k == init_steps + maxiter - 1:
            # convergence or maxiter reached -- return result
            if not converged:
                print('warning: BRASIL did not converge; dev={0:.3}, err={1:.3}'.format(deviation, max_err))
            if info:
                from collections import namedtuple
                Info = namedtuple('Info',
                        'converged error deviation nodes iterations ' +
                        'errors deviations stepsizes')
                errors = np.array(errors)
                return r, Info(
                    converged, max_err, deviation, nodes, k,
                    errors[:,0], errors[:,1], errors[:,2],
                )
            else:
                return r

        if k < init_steps:
            # PHASE 1:
            # move an interpolation node to the point of largest error
            max_intv_i = local_max.argmax()
            max_err_x = local_max_x[max_intv_i]
            # we can't move a node to the boundary, so check for that case
            # and move slightly inwards
            if max_err_x == a:
                max_err_x = (3 * a + nodes[0]) / 4
            elif max_err_x == b:
                max_err_x = (nodes[-1] + 3 * b) / 4
            # find the node to move (neighboring the interval with smallest error)
            min_k = local_max.argmin()
            if min_k == 0:
                min_j = 0
            elif min_k == n:
                min_j = n - 1
            else:
                # of the two nodes on this interval, choose the farther one
                if abs(max_err_x - nodes[min_k-1]) < abs(max_err_x - nodes[min_k]):
                    min_j = min_k
                else:
                    min_j = min_k - 1
            # move the node and re-sort the array
            nodes[min_j] = max_err_x
            nodes.sort()

        else:
            # PHASE 2:
            # global interval size adjustment
            intv_lengths = np.diff(all_nodes)

            mean_err = np.mean(local_max)
            max_dev = abs(local_max - mean_err).max()
            normalized_dev = (local_max - mean_err) / max_dev
            stepsize = min(max_step_size, step_factor * max_dev / mean_err)
            scaling = (1.0 - stepsize)**normalized_dev

            intv_lengths *= scaling
            # rescale so that they add up to b-a again
            intv_lengths *= (b - a) / intv_lengths.sum()
            nodes = np.cumsum(intv_lengths)[:-1] + a
