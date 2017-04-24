"""
This module is for quick construction of various PySCF objects
for Hubbard model
"""

import numpy as np
from pyscf import gto
from pyscf import ao2mo


def hubbard_from_scf(scf_cls, nelec, nsites, u, pbc='y'):
    """
    Construct a Hubbard scf object from a given
    SCF class from pyscf
    :param scf_csf: scf class, such as scf.RHF
    :param nsites: number of sites
    :param nelec: number of electrons
    :param u: interaction strength
    :param pbc:  type of periodic boundary conditions: 'y' - yes, 'n' - no,
    'a' - antiperiodic
    """

    mol = gto.M()
    mol.nelectron = nelec

    #
    # 1D anti-PBC Hubbard model at half filling
    #

    mf = scf_cls(mol)

    h1 = _hubbard_hopping_1d(nsites, pbc)

    eri = np.zeros((nsites, nsites, nsites, nsites))
    for ii in range(nsites):
        eri[ii, ii, ii, ii] = u

    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(nsites)
    mf._eri = ao2mo.restore(8, eri, nsites)
    mf._hubbard_interaction = u

    return mf


def _hubbard_hopping_1d(n, pbc):
    """
    Initialize hopping for 1D Hubbard chain
    :param n: number of sites
    :param pbc: type of periodic boundary conditions: 'y' - yes, 'n' - no,
    'a' - antiperiodic

    >>> t1 = _hubbard_hopping_1d(3,'y')
    >>> t2 = _hubbard_hopping_1d(3,'a')
    >>> t3 = _hubbard_hopping_1d(3,'n')
    >>> np.allclose(t1 + t2 - 2*t3, np.zeros((3,3)))
    True
    """
    if pbc not in set(('y', 'n', 'a')):
        raise ValueError('Unknown pbc setting: {}'.format(pbc))

    t = np.diagflat(-np.ones((n - 1, 1)), 1)
    if n > 2:
        if pbc == 'y':
            t[0, n - 1] = -1.0
        elif pbc == 'a':
            t[0, n - 1] = 1.0

    return t + t.T
