import numpy as np
import baryrat
import scipy.interpolate
import flamp
import gmpy2

import pytest

def test_init():
    nodes = [0, 1, 2]
    values = [1, 2, 0]
    weights = [0.5, -1, 0.5]
    r = baryrat.BarycentricRational(nodes, values, weights)
    X = np.linspace(0, 2, 100)
    assert np.allclose(r(X), -3/2*X**2 + 5/2*X + 1)

def test_approx():
    Z = np.linspace(0.0, 1.0, 101)
    def f(z): return np.exp(z)*np.sin(2*np.pi*z)
    F = f(Z)

    r = baryrat.aaa(Z, F, mmax=10)

    assert np.linalg.norm(r(Z) - F, np.inf) < 1e-10, 'insufficient approximation'

    # check invoking with functions
    r2 = baryrat.aaa(Z, f, mmax=10)
    assert np.linalg.norm(r(Z) - r2(Z), np.inf) < 1e-15

    # check that calling r works for scalars, vectors, matrices
    assert np.isscalar(r(0.45))
    assert r(np.ones(7)).shape == (7,)
    assert r(np.ones((3,2))).shape == (3,2)

def test_aaa_complex():
    Z = np.linspace(0.0, 1.0, 101)
    def f(z): return np.exp(2j*np.pi*z)
    F = f(Z)
    r = baryrat.aaa(Z, F, mmax=8)
    assert np.linalg.norm(r(Z) - F, np.inf) < 1e-10, 'insufficient approximation'

def test_reproduction():
    p = [-1.0, -2.0, -3.0]
    def f(z):
        return (z**3 - 2*z**2 + 4*z - 7) / ((z - p[0])*(z - p[1])*(z - p[2]))
    nodes = np.arange(1, 8, dtype=float)
    r = baryrat.aaa(nodes, f(nodes))
    assert np.allclose(f(nodes), r(nodes))
    z = np.linspace(0, 1, 100)
    assert np.allclose(f(z), r(z))

    pol, res = r.polres()
    assert np.allclose(sorted(p), sorted(pol))
    ratfun = sum(res[j] / (z - pol[j]) for j in range(len(pol)))
    bias = ratfun - f(z)
    assert np.allclose(bias, bias[0])   # should be constant

    flamp.set_dps(100)
    pol, res = r.polres(use_mp=True)
    pol = np.real_if_close(np.array(pol, complex))
    assert np.allclose(sorted(p), sorted(pol))
    ratfun = sum(res[j] / (z - pol[j]) for j in range(len(pol)))
    bias = np.array(ratfun - f(z), complex)
    assert np.allclose(bias, bias[0])   # should be constant

def test_polres():
    Z = np.linspace(0.0, 1.0, 101)
    F = np.exp(Z) * np.sin(2*np.pi*Z)
    r = baryrat.aaa(Z, F, mmax=6)
    pol, res = r.polres()

    assert np.allclose(pol,
            np.array([2.26333482+0.j, 0.2338428+0.90087977j,
                0.2338428-0.90087977j, 0.96472415+0.85470621j,
                0.96472415-0.85470621j]))
    assert np.allclose(res,
            np.array([69.08984183+0.j, 20.50747913-9.24908921j,
                20.50747913+9.24908921j, 23.24692682+23.94602455j,
                23.24692682-23.94602455j]))
    polvals = r(pol)
    assert np.min(np.abs(polvals)) > 1e13

    # check that gain == r(inf)
    assert np.allclose(r.gain(), r(1e14))

def test_zeros():
    Z = np.linspace(0.0, 1.0, 101)
    F = np.exp(Z) * np.sin(2*np.pi*Z)
    r = baryrat.aaa(Z, F, mmax=6)

    zer = r.zeros()
    assert np.allclose(zer,
            np.array([-0.38621461,  1.43052691,  0.49999907,  1.,  0.]))
    assert np.allclose(r(zer), 0.0)

    zer2 = r.zeros(use_mp=True)
    zer2 = np.real_if_close(np.array(zer2, complex))
    assert np.allclose(sorted(zer), sorted(zer2))

def test_reciprocal():
    nodes = np.linspace(0, 1, 4)
    r = baryrat.floater_hormann(nodes, np.exp(-nodes), 2)
    rr = r.reciprocal()

    Z = np.linspace(0, 1, 100)
    assert np.allclose(1 / r(Z), rr(Z))

def test_interpolate_rat():
    Z = np.linspace(1, 5, 7)
    F = np.sin(Z)
    r = baryrat.interpolate_rat(Z, F)
    assert np.allclose(r(Z), F)
    X = np.linspace(1, 5, 100)
    err = np.linalg.norm(r(X) - np.sin(X), np.inf)
    assert err < 2e-3
    #
    p, q = r.numerator(), r.denominator()
    assert np.allclose(p(X) / q(X), r(X))
    assert r.degree() == (3, 3)

def test_interpolate_with_degree():
    X = np.linspace(0, 1, 100)
    ##
    def f(x):
        return (x + 3) / ((x + 1) * (x + 2))
    Z = np.linspace(0, 1, 4)
    r = baryrat.interpolate_with_degree(Z, f(Z), (1, 2))
    assert np.allclose(f(X), r(X))
    assert r.order == 2
    assert r.degree() == (1, 2)
    ##
    def f(x):
        return (x * (x + 1) * (x + 2)) / (x + 3)
    Z = np.linspace(0, 1, 5)
    r = baryrat.interpolate_with_degree(Z, f(Z), (3, 1))
    assert np.allclose(f(X), r(X))
    assert r.order == 3
    assert r.degree() == (3, 1)

def test_interpolate_rat_mp():
    flamp.set_dps(100)
    X = flamp.linspace(0, 1, 100)
    ##
    def f(x):
        return (x + 3) / ((x + 1) * (x + 2))
    Z = flamp.linspace(0, 1, 5)
    r = baryrat.interpolate_rat(Z, f(Z))
    assert np.linalg.norm(f(X) - r(X), np.inf) < 1e-90
    assert r.order == 2
    ##
    def f(x):
        return (x * (x + 1) * (x + 2)) / (x + 3)
    Z = flamp.linspace(0, 1, 7)
    r = baryrat.interpolate_rat(Z, f(Z))
    assert np.linalg.norm(f(X) - r(X), np.inf) < 1e-90
    assert r.order == 3
    ## test complex case
    n = 9
    Z = np.array([np.exp(2j * k / n * np.pi) for k in range(0, n)])
    r = baryrat.interpolate_rat(Z, np.exp(Z), use_mp=True)
    X = 1j * flamp.linspace(-1, 1, 100)
    assert abs(flamp.exp(X) - r(X)).max() < 1e-7
    ## same thing with Z already in mpc form
    Z = flamp.exp(1j * flamp.linspace(0, 2 * np.pi, n, endpoint=False))
    r = baryrat.interpolate_rat(Z, flamp.exp(Z))
    X = 1j * flamp.linspace(-1, 1, 100)
    assert abs(flamp.exp(X) - r(X)).max() < 1e-7

def test_interpolate_with_degree_mp():
    flamp.set_dps(100)
    X = flamp.linspace(0, 1, 100)
    ##
    def f(x):
        return (x + 3) / ((x + 1) * (x + 2))
    Z = flamp.linspace(0, 1, 4)
    r = baryrat.interpolate_with_degree(Z, f(Z), (1, 2))
    assert np.linalg.norm(f(X) - r(X), np.inf) < 1e-90
    assert r.order == 2
    assert r.degree() == (1, 2)
    ##
    def f(x):
        return (x * (x + 1) * (x + 2)) / (x + 3)
    Z = flamp.linspace(0, 1, 5)
    r = baryrat.interpolate_with_degree(Z, f(Z), (3, 1))
    assert np.linalg.norm(f(X) - r(X), np.inf) < 1e-90
    assert r.order == 3
    assert r.degree() == (3, 1)

def test_reduce_order():
    nodes = np.linspace(0, 1, 11)
    r = baryrat.interpolate_rat(nodes, np.ones_like(nodes))
    assert r.order == 5
    r2 = r.reduce_order()
    assert r2.order == 0
    X = np.linspace(0, 1, 25)
    assert np.allclose(r2(X), 1.0)
    #
    # another test with full order (no reduction)
    r = baryrat.interpolate_rat(nodes, np.sin(nodes))
    assert r.order == 5
    r2 = r.reduce_order()
    assert r2.order == 5
    X = np.linspace(0, 1, 25)
    assert np.allclose(r2(X), r(X))

def test_interpolate_rat_complex():
    Z = np.linspace(0.0, 1.0, 9)
    def f(z): return np.exp(2j*np.pi*z)
    F = f(Z)
    r = baryrat.interpolate_rat(Z, F)
    assert np.allclose(r(Z), F)     # check interpolation property
    X = np.linspace(0.0, 1.0, 100)
    err = np.linalg.norm(r(X) - f(X), np.inf)
    assert err < 1e-4               # check interpolation error

def test_interpolate_poly():
    Z = np.linspace(1, 5, 7)
    F = np.sin(Z)
    p = baryrat.interpolate_poly(Z, F)
    p1 = scipy.interpolate.lagrange(Z, F)
    X = np.linspace(1, 5, 100)
    assert np.allclose(p(X), p1(X))

def test_interpolate_poly_complex():
    Z = np.linspace(0.0, 1.0, 9)
    def f(z): return np.exp(2j*np.pi*z)
    F = f(Z)
    p = baryrat.interpolate_poly(Z, F)
    assert np.allclose(p(Z), F)     # check interpolation property
    X = np.linspace(0.0, 1.0, 100)
    err = np.linalg.norm(p(X) - f(X), np.inf)
    assert err < 2e-3               # check interpolation error

def test_interpolate_with_poles():
    Z = np.arange(1, 5)
    F = np.sin(Z)
    poles = [-1, -2, -3]
    r = baryrat.interpolate_with_poles(Z, F, poles)
    assert np.allclose(r(Z), F)
    pol, res = r.polres()
    assert np.allclose(sorted(pol), sorted(poles))
    pol1 = r.poles()
    pol2 = r.poles(use_mp=True)
    pol2 = np.real_if_close(np.array(pol2, complex))
    assert np.allclose(sorted(pol1), sorted(poles))
    assert np.allclose(sorted(pol2), sorted(poles))

def test_interpolate_with_poles_mp():
    flamp.set_dps(100)
    Z = flamp.linspace(1.0, 4.0, 4)
    F = flamp.sin(Z)
    poles = [-3, -2, -1]
    r = baryrat.interpolate_with_poles(Z, F, poles)
    assert r.uses_mp()
    assert np.array_equal(r(Z), F)
    pol, res = r.polres()
    pol = np.real_if_close(np.array(pol, complex))
    pol.sort()
    assert np.linalg.norm(pol - poles) < 1e-90

def test_interpolate_floater_hormann():
    n = 10
    Z = np.linspace(-5, 5, n + 1)
    X = np.linspace(-5, 5, 200)
    def f(z): return 1.0 / (1 + z**2)  # Runge's example
    F = f(Z)
    # normalized weights for the equidistant case given in FH2007
    correct_abs_weights = [
        [1, 1, 1, 1],
        [1, 2, 2, 2],
        [1, 3, 4, 4],
        [1, 4, 7, 8]
    ]
    for d in range(4):
        r = baryrat.floater_hormann(Z, F, d)
        assert np.allclose(r(Z), F)
        w = abs(r.weights / r.weights[0]) # normalize
        assert np.allclose(w[:4], correct_abs_weights[d])
        if d == 3:
            err = np.linalg.norm(r(X) - f(X), np.inf)
            assert err < 6.9e-2   # published error in FH2007
    # check that d=n results in polynomial interpolant
    r = baryrat.floater_hormann(Z, F, n)
    p = scipy.interpolate.lagrange(Z, F)
    assert np.allclose(r(X), p(X))

def test_deriv():
    def f(x):
        return (x + 3) / ((x + 1) * (x + 2))
    def df(x):
        return -2 / (x + 1)**2 + 1 / (x + 2)**2
    def d2f(x):
        return 4 / (x + 1)**3 - 2 / (x + 2)**3

    # compute barycentric representation of f
    Z = np.linspace(0, 1, 4)
    r = baryrat.interpolate_with_degree(Z, f(Z), (1, 2))

    X = np.linspace(0, 1, 50)
    X[:len(r.nodes)] = r.nodes  # also test evaluation exactly on the nodes

    assert np.allclose(r.eval_deriv(X), df(X))
    assert np.allclose(r.eval_deriv(X, k=2), d2f(X))

def test_chebnodes():
    n = 7
    nodes = baryrat.chebyshev_nodes(n, (0, 4))
    nodes = (nodes / 2 - 1)     # scale back to (-1,1)
    import scipy.special
    values = scipy.special.eval_chebyt(7, nodes)
    assert abs(values).max() < 1e-14

def test_brasil():
    r, info = baryrat.brasil(np.sqrt, [0,1], 10, tol=1e-5, info=True)
    assert info.converged
    assert info.deviation <= 1e-5
    assert info.error <= 5e-6
    assert(len(info.errors) == info.iterations + 1)

def test_brasil_poly():
    # problem with known error
    # http://www-solar.mcs.st-and.ac.uk/~clare/Lectures/num-analysis/Numan_chap4.pdf
    p, info = baryrat.brasil(np.exp, [0,1], (1,0), tol=1e-12, info=True)
    assert info.converged
    m = np.exp(1) - 1
    theta = np.log(m)
    c = (m + np.exp(1) - m*theta - m) / 2
    E = 1 - c
    assert np.allclose(info.error, E)
    # https://doi.org/10.1007/s10543-009-0240-1
    def f(x): return np.sin(np.exp(x))
    p, info = baryrat.brasil(f, [-1,1], (10,0), npi=-30, info=True)     # use golden section search
    assert np.allclose(info.error, 1.78623400e-6)
    #
    def f(x): return np.sqrt(x + 1)
    p, info = baryrat.brasil(f, [-1,1], (10,0), tol=1e-8, info=True)
    assert np.allclose(info.error, 1.978007008380e-2)

def test_brasil_deg():
    r, info = baryrat.brasil(np.sqrt, [0,1], (10,5), tol=1e-8, info=True)
    assert info.converged
    assert info.deviation <= 1e-8
    assert info.error <= 6e-5
    assert(len(info.errors) == info.iterations + 1)
    #
    r, info = baryrat.brasil(np.sqrt, [0,1], (5,10), tol=1e-8, info=True)
    assert info.converged
    assert info.deviation <= 1e-8
    assert info.error <= 8e-5
    assert(len(info.errors) == info.iterations + 1)

def test_jacobians():
    Z = np.linspace(1, 5, 7)
    r = baryrat.interpolate_rat(Z, np.sin(Z))
    x = np.linspace(2, 4, 3)
    Dz, Df, Dw = r.jacobians(x)
    delta = 1e-6
    # compare to finite differences
    for k in range(len(r.nodes)):
        z_delta = r.nodes.copy()
        z_delta[k] += delta
        r_delta = baryrat.BarycentricRational(z_delta, r.values, r.weights)
        deriv = (r_delta(x) - r(x)) / delta
        assert np.allclose(Dz[:,k], deriv)

        f_delta = r.values.copy()
        f_delta[k] += delta
        r_delta = baryrat.BarycentricRational(r.nodes, f_delta, r.weights)
        deriv = (r_delta(x) - r(x)) / delta
        assert np.allclose(Df[:,k], deriv)

        w_delta = r.weights.copy()
        w_delta[k] += delta
        r_delta = baryrat.BarycentricRational(r.nodes, r.values, w_delta)
        deriv = (r_delta(x) - r(x)) / delta
        assert np.allclose(Dw[:,k], deriv)

def test_bpane():
    def f(x):       return abs(x)
    def f_deriv(x): return np.sign(x)
    interval = (-1.0, 1.0)

    p, info = baryrat.bpane(f, f_deriv, interval, 11, info=True, verbose=0)
    assert info.error < 2.8e-2 and abs(info.lam) < 2.8e-2
    X = np.linspace(*interval, 100)
    assert np.abs(f(X) - p(X)).max() < info.error * (1 + 1e-6)  # allow some tolerance
    assert info.iterations == 11

    # try with numerical derivative
    p, info = baryrat.bpane(f, None, interval, 11, info=True, verbose=0)
    assert info.error < 2.8e-2 and abs(info.lam) < 2.8e-2
    X = np.linspace(*interval, 100)
    assert np.abs(f(X) - p(X)).max() < info.error * (1 + 1e-6)  # allow some tolerance
    assert info.iterations == 11

    # check that failure to converge (even degree for abs) is signaled
    with pytest.raises((RuntimeError, np.linalg.LinAlgError)):
        baryrat.bpane(f, f_deriv, interval, 10, maxiter=100, verbose=0)
