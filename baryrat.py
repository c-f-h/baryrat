"""A Python package for barycentric rational approximation.
"""

import numpy as np
import scipy.linalg
import math

try:
    import mpmath
except ImportError:
    mpmath = None
else:
    from mpmath import mp, mpf

__version__ = '2.0.0'

def _is_mp_array(x):
    """Checks whether `x` is an ndarray containing mpmath extended precision numbers."""
    return (mpmath
            and x.dtype == 'O'
            and len(x) > 0
            and isinstance(x.flat[0], mpf))

def _q(z, f, w, x):
    """Function which can compute the 'upper' or 'lower' rational function
    in a barycentric rational function.

    `x` may be a number or a column vector.
    """
    return np.sum((f * w) / (x - z), axis=-1)

def _compute_roots(w, x, use_mp):
    # Cf.:
    # Knockaert, L. (2008). A simple and accurate algorithm for barycentric
    # rational interpolation. IEEE Signal processing letters, 15, 154-157.
    if _is_mp_array(w) or _is_mp_array(x):
        use_mp = True

    if use_mp:
        assert mpmath, 'mpmath package is not installed'
        ak = mp.matrix(w)
        ak /= sum(ak)
        bk = mp.matrix(x)

        M = mp.diag(bk)
        for i in range(M.rows):
            for j in range(M.cols):
                M[i,j] -= ak[i]*bk[j]
        lam = np.array(mp.eig(M, left=False, right=False))
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return lam
    else:
        # the same procedure in standard double precision
        ak = w / w.sum()
        M = np.diag(x) - np.outer(ak, x)
        lam = scipy.linalg.eigvals(M)
        # remove one simple root
        lam = np.delete(lam, np.argmin(abs(lam)))
        return np.real_if_close(lam)

def _mp_svd(A, full_matrices=True):
    """Convenience wrapper for mpmath high-precision SVD."""
    assert mpmath, 'mpmath package is not installed'
    AA = mp.matrix(A.tolist())
    U, Sigma, VT = mp.svd(AA, full_matrices=full_matrices)
    return np.array(U.tolist()), np.array(Sigma.tolist()).ravel(), np.array(VT.tolist())

def _mp_qr(A):
    """Convenience wrapper for mpmath high-precision QR decomposition."""
    assert mpmath, 'mpmath package is not installed'
    AA = mp.matrix(A.tolist())
    Q, R = mp.qr(AA, mode='full')
    return np.array(Q.tolist()), np.array(R.tolist())

def _nullspace_vector(A, use_mp=False):
    if _is_mp_array(A):
        use_mp = True

    if A.shape[0] == 0:
        # some LAPACK implementations have trouble with size 0 matrices
        result = np.zeros(A.shape[1])
        result[0] = 1.0
        if use_mp:
            assert mpmath, 'mpmath package is not installed'
            result = np.vectorize(mpf)(result)
        return result

    if use_mp:
        Q, _ = _mp_qr(A.T)
    else:
        Q, _ = scipy.linalg.qr(A.T, mode='full')
    return Q[:, -1].conj()

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
        if not (len(z) == len(f) == len(w)):
            raise ValueError('arrays z, f, and w must have the same length')
        self.nodes = np.asanyarray(z)
        self.values = np.asanyarray(f)
        self.weights = np.asanyarray(w)

    def __call__(self, x):
        """Evaluate rational function at all points of `x`."""
        zj,fj,wj = self.nodes, self.values, self.weights

        xv = np.asanyarray(x).ravel()
        if len(xv) == 0:
            return np.empty(np.shape(x), dtype=xv.dtype)
        D = xv[:,None] - zj[None,:]
        # find indices where x is exactly on a node
        (node_xi, node_zi) = np.nonzero(D == 0)

        one = xv[0] * 0 + 1     # for proper dtype when using mpmath

        with np.errstate(divide='ignore', invalid='ignore'):
            if len(node_xi) == 0:       # no zero divisors
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
            else:
                # set divisor to 1 to avoid division by zero
                D[node_xi, node_zi] = one
                C = np.divide(one, D)
                r = C.dot(wj * fj) / C.dot(wj)
                # fix evaluation at nodes to corresponding fj
                # TODO: this is only correct if wj != 0
                r[node_xi] = fj[node_zi]

        if np.isscalar(x):
            return r[0]
        else:
            r.shape = np.shape(x)
            return r

    def uses_mp(self):
        """Checks whether any of the data of this rational function uses
        ``mpmath`` extended precision.
        """
        return _is_mp_array(self.nodes) or _is_mp_array(self.values) or _is_mp_array(self.weights)

    def eval_deriv(self, x, k=1):
        """Evaluate the `k`-th derivative of this rational function at a scalar
        node `x`, or at each point of an array `x`. Only the cases `k <= 2` are
        currently implemented.

        Note that this function may incur significant numerical error if `x` is
        very close (but not exactly equal) to a node of the barycentric
        rational function.

        References:
            https://doi.org/10.1090/S0025-5718-1986-0842136-8 (C. Schneider and
            W. Werner, 1986)
        """
        if k == 0:
            return self(x)

        # the implementation below assumes scalars, so use numpy to vectorize
        # if we got an array
        if not np.isscalar(x):
            return np.vectorize(lambda X: self.eval_deriv(X, k=k), otypes=[x.dtype])(x)

        # is x one of our nodes?
        nodeidx = np.nonzero(x == self.nodes)[0]
        if len(nodeidx) > 0:
            i = nodeidx[0]      # node index of x
            dx = self.nodes - x
            dx[i] = np.inf   # set i-th summand to 0

            if k == 1:
                # first-order divided differences
                dd = (self(self.nodes) - self(x)) / dx
            elif k == 2:
                # second-order divided differences with nodes (x, x, z_i)
                # (note that repeated nodes correspond to the first derivative)
                dd1 = (self(self.nodes) - self(x)) / dx
                dd = (dd1 - self.eval_deriv(x, k=1)) / dx
            else:
                raise NotImplementedError('derivatives higher than 2 not implemented')
            return -np.sum(dd * self.weights) / self.weights[i] * math.factorial(k)

        else:
            # x is not a node -- use divided differences
            if k == 1:
                # first-order divided differences
                dd = (self(self.nodes) - self(x)) / (self.nodes - x)
            elif k == 2:
                # second-order divided differences with nodes (x, x, z_i)
                # (note that repeated nodes correspond to the first derivative)
                dd1 = (self(self.nodes) - self(x)) / (self.nodes - x)
                dd = (dd1 - self.eval_deriv(x, k=1)) / (self.nodes - x)
            else:
                raise NotImplementedError('derivatives higher than 2 not implemented')
            return BarycentricRational(self.nodes, dd, self.weights)(x) * math.factorial(k)

    def jacobians(self, x):
        """Compute the Jacobians of `r(x)`, where `x` may be a vector of
        evaluation points, with respect to the node, value, and weight vectors.

        The evaluation points `x` may not lie on any of the barycentric nodes
        (unimplemented).

        Returns:
            A triple of arrays with as many rows as `x` has entries and as many
            columns as the barycentric function has nodes, representing the
            Jacobians with respect to :attr:`self.nodes`, :attr:`self.values`,
            and :attr:`self.weights`, respectively.
        """
        z, f, w = self.nodes, self.values, self.weights
        N1 = len(z)
        x_c = np.atleast_2d(x).T      # column vector
        dr_z, dr_f, dr_w = [], [], []
        qz1 = _q(z, 1, w, x_c)
        # build matrices columnwise (j = node index)
        for j in range(N1):
            f_diff = np.subtract(f[j], f)
            x_minus_zj = np.subtract(x, z[j])
            dr_z.append(_q(z, f_diff * w[j], w, x_c) / (x_minus_zj * qz1)**2)
            dr_f.append(np.divide(w[j], (x_minus_zj * qz1)))
            dr_w.append(_q(z, f_diff, w, x_c) / (x_minus_zj * qz1**2))
        return np.column_stack(dr_z), np.column_stack(dr_f), np.column_stack(dr_w)

    @property
    def order(self):
        """The order of the barycentric rational function, that is, the maximum
        degree that its numerator and denominator may have, or the number of
        interpolation nodes minus one.
        """
        return len(self.nodes) - 1

    def poles(self, use_mp=False):
        """Return the poles of the rational function.

        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result. Set `mpmath.mp.dps` to the desired number of decimal digits
        before use.
        """
        return _compute_roots(self.weights, self.nodes, use_mp=use_mp)

    def polres(self, use_mp=False):
        """Return the poles and residues of the rational function.

        If ``use_mp`` is ``True``, uses the ``mpmath`` package to compute the
        result. Set `mpmath.mp.dps` to the desired number of decimal digits
        before use. The ``use_mp`` option will be automatically enabled if
        :meth:`uses_mp` is True.
        """
        zj,fj,wj = self.nodes, self.values, self.weights
        m = len(wj)

        if self.uses_mp():
            use_mp = True

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
        result. Set `mpmath.mp.dps` to the desired number of decimal digits
        before use. The ``use_mp`` option will be automatically enabled if
        :meth:`uses_mp` is True.
        """
        if self.uses_mp():
            use_mp = True

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

    def numerator(self):
        """Return a new :class:`BarycentricRational` which represents the numerator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.values * self.weights / weights, weights)

    def denominator(self):
        """Return a new :class:`BarycentricRational` which represents the denominator polynomial."""
        weights = _polynomial_weights(self.nodes)
        return BarycentricRational(self.nodes.copy(), self.weights / weights, weights)

    def degree_numer(self, tol=1e-12):
        """Compute the true degree of the numerator polynomial.

        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.values * self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree_denom(self, tol=1e-12):
        """Compute the true degree of the denominator polynomial.

        Uses a result from [Berrut, Mittelmann 1997].
        """
        N = len(self.nodes) - 1
        for defect in range(N):
            if abs(np.sum(self.weights * (self.nodes ** defect))) > tol:
                return N - defect
        return 0

    def degree(self, tol=1e-12):
        """Compute the pair `(m,n)` of true degrees of the numerator and denominator."""
        return (self.degree_numer(tol=tol), self.degree_denom(tol=tol))

    def reduce_order(self):
        """Return a new :class:`BarycentricRational` which represents the same rational
        function as this one, but with minimal possible order.

        See (Ionita 2013), PhD thesis.
        """
        # sample at intermediate nodes and compute Loewner matrix
        aux_nodes = (self.nodes[1:] + self.nodes[:-1]) / 2
        aux_v = self(aux_nodes)
        L = (aux_v[:, None] - self.values[None, :]) / (aux_nodes[:, None] - self.nodes[None, :])

        # determine the order as the rank of L (cf. (Ionita 2013))
        order = np.linalg.matrix_rank(L)
        if order == self.order:
            return BarycentricRational(self.nodes.copy(), self.values.copy(), self.weights.copy())
        n = order + 1           # number of nodes in new barycentric function
        scale = 1 if n==1 else int((len(self.nodes) - 1) / (n - 1))    # distribute new nodes over the old ones
        subset = np.arange(0, scale*n, scale)      # choose a subset of n nodes from self.nodes

        # compute Loewner matrix for new subset of nodes
        nodes = self.nodes[subset]
        values = self.values[subset]
        aux_nodes = (nodes[1:] + nodes[:-1]) / 2
        aux_v = self(aux_nodes)
        L = (aux_v[:, None] - values[None, :]) / (aux_nodes[:, None] - self.nodes[None, subset])

        # compute weight vector in nullspace
        w = _nullspace_vector(L)
        return BarycentricRational(nodes, values, w)

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
      | https://doi.org/10.1137/16M1106122

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
        wj = Vh[-1, :].conj()

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

def interpolate_rat(nodes, values, use_mp=False):
    """Compute a rational function which interpolates the given nodes/values.

    Args:
        nodes (array): the interpolation nodes; must have odd length and
            be passed in strictly increasing or decreasing order
        values (array): the values at the interpolation nodes
        use_mp (bool): whether to use ``mpmath`` for extended precision. Is
            automatically enabled if `nodes` or `values` use ``mpmath``.

    Returns:
        BarycentricRational: the rational interpolant. If there are `2n + 1` nodes,
        both the numerator and denominator have degree at most `n`.

    References:
        https://doi.org/10.1109/LSP.2007.913583
    """
    # ref: (Knockaert 2008), doi:10.1109/LSP.2007.913583
    # see also: (Ionita 2013), PhD thesis, Rice U
    values = np.asanyarray(values)
    nodes = np.asanyarray(nodes)
    n = len(values) // 2 + 1
    m = n - 1
    if not len(values) == n + m or not len(nodes) == n + m:
        raise ValueError('number of nodes should be odd')
    xa, xb = nodes[0::2], nodes[1::2]
    va, vb = values[0::2], values[1::2]
    # compute the Loewner matrix
    B = (vb[:, None] - va[None, :]) / (xb[:, None] - xa[None, :])
    # choose a weight vector in the nullspace of B
    weights = _nullspace_vector(B, use_mp=use_mp)
    return BarycentricRational(xa, va, weights)

def _pseudo_equi_nodes(n, k):
    """Choose `k` out of `n` nodes in a quasi-equispaced way."""
    if k > n:
        raise ValueError("k must not be larger than n")
    else:
        return np.rint(np.linspace(0.0, n-1, k)).astype(int)

def _defect_matrix(x, i0, iend, f=None):
    powers_m = np.arange(i0, iend)
    W = x[None, :] ** powers_m[:, None]
    if f is not None:
        W *= f[None, :]
    return W

def _defect_matrix_arnoldi(x, m, f=None):
    # Arnoldi-type orthonormalization of the defect matrix.
    # Based on an idea from Filip et al., 2018, p. A2431.
    # doi: 10.1137/17M1132409
    if m == 0:
        return np.empty((0, len(x)), dtype=x.dtype)
    if f is None:
        f = 0 * x + 1
    f = f / np.linalg.norm(f)
    Q = [f]
    for k in range(1, m):
        q = Q[-1] * x
        for j in range(len(Q)):
            q -= Q[j] * np.inner(q, Q[j])
        q /= np.linalg.norm(q)
        Q.append(q)
    return np.array(Q)

def interpolate_with_degree(nodes, values, deg, use_mp=False):
    """Compute a rational function which interpolates the given nodes/values
    with given degree `m` of the numerator and `n` of the denominator.

    Args:
        nodes (array): the interpolation nodes
        values (array): the values at the interpolation nodes
        deg: a pair `(m, n)` of the degrees of the interpolating rational
            function. The number of interpolation nodes must be `m + n + 1`.
        use_mp (bool): whether to use ``mpmath`` for extended precision. Is
            automatically enabled if `nodes` or `values` use ``mpmath``.

    Returns:
        BarycentricRational: the rational interpolant

    References:
        https://doi.org/10.1016/S0377-0427(96)00163-X
    """
    m, n = deg
    nn = m + n + 1
    if len(nodes) != nn or len(values) != nn:
        raise ValueError('number of interpolation nodes must be m + n + 1')
    if n == 0:
        return interpolate_poly(nodes, values)
    elif m == n:
        return interpolate_rat(nodes, values, use_mp=use_mp)
    else:
        N = max(m, n)       # order of barycentric rational function
        # split given values into primary and secondary nodes
        primary_indices = _pseudo_equi_nodes(nn, N + 1)
        secondary_indices = np.setdiff1d(np.arange(nn), primary_indices, assume_unique=True)
        xp, vp = nodes[primary_indices],   values[primary_indices]
        xs, vs = nodes[secondary_indices], values[secondary_indices]
        # compute Loewner matrix - shape: (m + n - N) x (N + 1)
        L = (vs[:, None] - vp[None, :]) / (xs[:, None] - xp[None, :])
        # add weight constraints for denominator and numerator degree; see (Berrut, Mittelmann 1997)
        # B has shape N x (N + 1)
        B = np.vstack((
            L,
            _defect_matrix_arnoldi(xp, N - n),        # reduce maximum denominator degree by N - n
            _defect_matrix_arnoldi(xp, N - m, vp)     # reduce maximum numerator degree by N - m
        ))
        # choose a weight vector in the nullspace of B
        weights = _nullspace_vector(B, use_mp=use_mp)
        return BarycentricRational(xp, vp, weights)

def _polynomial_weights(x):
    n = len(x)
    return np.array([
            1.0 / np.prod([x[i] - x[j] for j in range(n) if j != i])
            for i in range(n)
    ])

def interpolate_poly(nodes, values):
    """Compute the interpolating polynomial for the given nodes and values in
    barycentric form.
    """
    n = len(nodes)
    if n != len(values):
        raise ValueError('input arrays should have the same length')
    weights = _polynomial_weights(nodes)
    return BarycentricRational(nodes, values, weights)

def interpolate_with_poles(nodes, values, poles, use_mp=False):
    """Compute a rational function which interpolates the given values at the
    given nodes and which has the given poles.

    The arrays ``nodes`` and ``values`` should have length `n`, and
    ``poles`` should have length `n - 1`.
    """
    # ref: (Knockaert 2008), doi:10.1109/LSP.2007.913583
    n = len(nodes)
    if n != len(values) or n != len(poles) + 1:
        raise ValueError('invalid length of arrays')
    nodes = np.asanyarray(nodes)
    values = np.asanyarray(values)
    poles = np.asanyarray(poles)
    # compute Cauchy matrix
    C = 1.0 / (poles[:,None] - nodes[None,:])
    # compute null space
    weights = _nullspace_vector(C, use_mp=use_mp)
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
    Z, gZ = np.empty(m+2, dtype=z.dtype), np.empty(m+2, dtype=gB.dtype)
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

def chebyshev_nodes(num_nodes, interval=(-1.0, 1.0)):
    """Compute `num_nodes` Chebyshev nodes of the first kind in the given interval."""
    # compute nodes in (-1, 1)
    nodes = (1 - np.cos((2*np.arange(1, num_nodes + 1) - 1) / (2*num_nodes) * np.pi))
    # rescale to desired interval
    a, b = interval
    return nodes * ((b - a) / 2) + a


def brasil(f, interval, deg, tol=1e-4, maxiter=1000, max_step_size=0.1,
        step_factor=0.1, npi=-30, init_steps=100, info=False):
    """Best Rational Approximation by Successive Interval Length adjustment.

    Computes best rational or polynomial approximations in the maximum norm by
    the BRASIL algorithm (see reference below).

    References:
        https://doi.org/10.1007/s11075-020-01042-0

    Arguments:
        f: the scalar function to be approximated. Must be able to operate
            on arrays of arguments.
        interval: the bounds `(a, b)` of the approximation interval
        deg: the degree of the numerator `m` and denominator `n` of the
            rational approximation; either an integer (`m=n`) or a pair `(m, n)`.
            If `n = 0`, a polynomial best approximation is computed.
        tol: the maximum allowed deviation from equioscillation
        maxiter: the maximum number of iterations
        max_step_size: the maximum allowed step size
        step_factor: factor for adaptive step size choice
        npi: points per interval for error calculation. If `npi < 0`,
            golden section search with `-npi` iterations is used instead of
            sampling. For high-accuracy results, `npi=-30` is typically a good
            choice.
        init_steps: how many steps of the initialization iteration to run
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

    Note:
        This function supports ``mpmath`` for extended precision. To enable
        this, specify the interval `(a, b)` as `mpf` numbers, e.g.,
        ``interval=(mpf(0), mpf(1))``. Also make sure that the function `f`
        consumes and outputs arrays of `mpf` numbers; the Numpy function
        :func:`numpy.vectorize` may help with this.
    """
    a, b = interval
    assert a < b, 'Invalid interval'

    if np.isscalar(deg):
        m = n = deg
    else:
        if len(deg) != 2:
            raise TypeError("'deg' must be an integer or pair of integers")
        m, n = deg
    nn = m + n + 1      # number of interpolation nodes

    errors = []
    stepsize = np.nan

    # start with Chebyshev nodes
    nodes = chebyshev_nodes(nn, (a, b))

    # choose proper interpolation routine
    if n == 0:
        interp = interpolate_poly
    elif m == n:
        interp = interpolate_rat
    else:
        interp = lambda x,f: interpolate_with_degree(x, f, (m, n))

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
            else:
                # Until now, we have only equilibrated the absolute errors.
                # Check equioscillation property for the signed errors to make
                # sure we actually found the best approximation.
                signed_errors = f(local_max_x) - r(local_max_x)
                # normalize them so that they are all 1 in case of equioscillation
                signed_errors /= (-1)**np.arange(len(signed_errors)) * np.sign(signed_errors[0]) * max_err
                equi_err = abs(1.0 - signed_errors).max()
                if equi_err > tol:
                    print('warning: equioscillation property not satisfied, deviation={0:.3}'.format(equi_err))
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
            elif min_k == nn:
                min_j = nn - 1
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
