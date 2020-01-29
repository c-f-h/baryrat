import numpy as np
import aaa

def test_approx():
    Z = np.linspace(0.0, 1.0, 101)
    def f(z): return np.exp(z)*np.sin(2*np.pi*z)
    F = f(Z)

    r = aaa.aaa(F, Z, mmax=10)

    assert np.linalg.norm(r(Z) - F, np.inf) < 1e-10, 'insufficient approximation'

    # check invoking with functions
    r2 = aaa.aaa(f, Z, mmax=10)
    assert np.linalg.norm(r(Z) - r2(Z), np.inf) < 1e-15

    # check that calling r works for scalars, vectors, matrices
    assert np.isscalar(r(0.45))
    assert r(np.ones(7)).shape == (7,)
    assert r(np.ones((3,2))).shape == (3,2)

def test_polres():
    Z = np.linspace(0.0, 1.0, 101)
    F = np.exp(Z) * np.sin(2*np.pi*Z)
    r = aaa.aaa(F, Z, mmax=6)
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

def test_zeros():
    Z = np.linspace(0.0, 1.0, 101)
    F = np.exp(Z) * np.sin(2*np.pi*Z)
    r = aaa.aaa(F, Z, mmax=6)
    zer = r.zeros()

    assert np.allclose(zer,
            np.array([-0.38621461,  1.43052691,  0.49999907,  1.,  0.]))
    assert np.allclose(r(zer), 0.0)

def test_interpolate():
    Z = np.arange(1, 5)
    F = np.sin(Z)
    poles = [-1, -2, -3]
    r = aaa.interpolate_with_poles(F, Z, poles)
    assert np.allclose(r(Z), F)
    pol, res = r.polres()
    assert np.allclose(sorted(pol), sorted(poles))
