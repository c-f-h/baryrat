import numpy as np
import baryrat
import scipy.interpolate

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
    pol, res = r.polres(use_mp=True)
    assert np.allclose(sorted(p), sorted(pol))

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
    p = baryrat.interpolate_rat(Z, F)
    assert np.allclose(p(Z), F)
    X = np.linspace(1, 5, 100)
    err = np.linalg.norm(p(X) - np.sin(X), np.inf)
    assert err < 2e-3

def test_interpolate_poly():
    Z = np.linspace(1, 5, 7)
    F = np.sin(Z)
    p = baryrat.interpolate_poly(Z, F)
    p1 = scipy.interpolate.lagrange(Z, F)
    X = np.linspace(1, 5, 100)
    assert np.allclose(p(X), p1(X))

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
    assert np.allclose(sorted(pol1), sorted(poles))
    assert np.allclose(sorted(pol2), sorted(poles))

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

def test_brasil():
    def f(x): return np.sqrt(x)
    r, info = baryrat.brasil(f, [0,1], 10, tol=1e-5, info=True)
    assert info.converged
    assert info.deviation <= 1e-5
    assert info.error <= 5e-6
    assert(len(info.errors) == info.iterations + 1)
