import numpy as np
from functools import reduce

# some helper functions for dealing with matrices and computing fidelity

def tensor(matrices):
    """Compute the tensor product of a list (or array) of matrices"""
    return reduce(np.kron, matrices)


def dots(matrices):
    """Compute the dot product of a list (or array) of matrices"""
    return reduce(np.dot, matrices)


def dot3(A, B, C):
    """Compute the dot product of three matrices"""
    return np.dot(np.dot(A, B), C)


def commutator(A, B):
    """Compute the commutator of two matrices"""
    return np.dot(A, B) - np.dot(B, A)


def lindblad(rho, L, Ld=None):
    """Compute the contribution of one Lindblad term to the master equation""" 
    if Ld is None:
        Ld = L.conj().T
    return dot3(L, rho, Ld) - 0.5*dot3(Ld, L, rho) - 0.5*dot3(rho, Ld, L)


def ket2rho(ket):
    """Convert a state (ket) to a density matrix (rho)."""
    return np.outer(ket, ket.conj())


def sqrtm(A):
    """Compute the matrix square root of a matrix"""
    d, U = np.linalg.eig(A)
    s = np.sqrt(d.astype(complex))
    return dot3(U, np.diag(s), U.conj().T)


def trace_distance(rho, sigma):
    """Compute the trace distance between matrices rho and sigma
    See Nielsen and Chuang, p. 403
    """
    A = rho - sigma
    abs = sqrtm(np.dot(A.conj().T, A))
    return np.real(np.trace(abs)) / 2.0


def fidelity(rho, sigma):
    """Compute the fidelity between matrices rho and sigma
    See Nielsen and Chuang, p. 409
    """
    rhosqrt = sqrtm(rho)
    return np.real(np.trace(sqrtm(dot3(rhosqrt, sigma, rhosqrt))))


def overlap(rho, sigma):
    """Trace of the product of two matrices."""
    # XXX this is only correct if at least one of rhos or sigma is a pure state
    return np.trace(np.dot(rho, sigma))


def concurrence(rho):
    """Concurrence of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    yy = np.array([[ 0, 0, 0,-1],
                   [ 0, 0, 1, 0],
                   [ 0, 1, 0, 0],
                   [-1, 0, 0, 0]], dtype=complex)
    m = dots([rho, yy, rho.conj(), yy])
    eigs = [np.abs(e) for e in np.linalg.eig(m)[0]]
    e = [np.sqrt(x) for x in sorted(eigs, reverse=True)]
    return max(0, e[0] - e[1] - e[2] - e[3])


def eof(rho):
    """Entanglement of formation of a two-qubit density matrix.
    see http://qwiki.stanford.edu/wiki/Entanglement_of_Formation
    """
    def h(x):
        if x <= 0 or x >= 1:
            return 0
        return -x*np.log2(x) - (1-x)*np.log2(1-x)
    C = concurrence(rho)
    arg = max(0, np.sqrt(1-C**2))
    return h((1 + arg)/2.0)

