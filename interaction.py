import time
import numpy as np
import tempfile
import h5py
from functools import reduce
from pyscf.lib import logger
from pyscf.lib import pack_tril, unpack_tril
from pyscf import ao2mo

from collections import namedtuple


def ref_ndarray(a):
    return np.array(a, copy=False, order='C')


def _calculate_noncanonical_fock(scf, mo_coeff, mo_occ):
    """
    Calculates Fock matrix in non-canonical basis
    """
    dm = scf.make_rdm1(mo_coeff, mo_occ)
    fockao = scf.get_hcore() + scf.get_veff(scf.mol, dm)
    return reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))


def _assemble_fock(cc, mos=None):
    # Calculate Fock matrix
    FockMatrix = namedtuple('FockMatrix', ('oo', 'ov', 'vv'))
    if mos is None:  # Assume canonical orbitals
        mos = cc.mos
        fock = np.diag(cc.mos.mo_energies)
    else:  # If mo_coeff is not canonical orbitals
        fock = _calculate_noncanonical_fock(cc._scf, mos.mo_coeff,
                                            mos.mo_occ)
    nocc = mos.nocc
    f = FockMatrix(oo=ref_ndarray(fock[:nocc, :nocc]),
                   ov=ref_ndarray(fock[:nocc, nocc:]),
                   vv=ref_ndarray(fock[nocc:, nocc:])
                   )

    return f


class HAM_SPINLESS_FULL_CORE_MUL:
    """
    Mulliken ordered hamiltonian
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos
        nocc = self.mos.nocc
        nmo = self.mos.nmo
        nvir = self.mos.nvir

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        if (cc._scf._eri is not None):
            VFull = namedtuple('VFull', ('oooo', 'ooov', 'oovv', 'ovov',
                                         'voov', 'ovvv', 'vvvv'))

            eri1 = ao2mo.incore.full(cc._scf._eri, self.mos.mo_coeff)
            nvir_pair = nvir * (nvir + 1) // 2

            # Restore first compression over symmetric indices
            eri1 = ao2mo.restore(1, eri1, nmo)
            self.v = VFull(oooo=ref_ndarray(eri1[:nocc, :nocc, :nocc, :nocc]),
                           ooov=ref_ndarray(eri1[:nocc, :nocc, :nocc, nocc:]),
                           oovv=ref_ndarray(eri1[:nocc, :nocc, nocc:, nocc:]),
                           ovov=ref_ndarray(eri1[:nocc, nocc:, :nocc, nocc:]),
                           voov=ref_ndarray(eri1[nocc:, :nocc, :nocc, nocc:]),
                           ovvv=ref_ndarray(eri1[:nocc, nocc:, nocc:, nocc:]),
                           vvvv=ref_ndarray(eri1[nocc:, nocc:, nocc:, nocc:])
                           )
        else:
            raise ValueError('SCF object did not supply AO integrals')

        log.timer('CCSD integral transformation', *cput0)


class HAM_SPINLESS_FULL_CORE_DIR:
    """
    Dirac ordered hamiltonian
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos
        nocc = self.mos.nocc
        nmo = self.mos.nmo
        nvir = self.mos.nvir

        def mulliken_to_dirac(a):
            return a.transpose(0, 2, 1, 3)

        if (cc._scf._eri is not None):
            VFull = namedtuple('VFull', ('oooo', 'ooov', 'oovv', 'ovov',
                                         'voov', 'ovvv', 'vvvv', 'ovvo',
                                         'vvov', 'vvoo', 'ovoo', 'oovo',
                                         'vvvo'))

            eri1 = ao2mo.incore.full(cc._scf._eri, self.mos.mo_coeff)
            nvir_pair = nvir * (nvir + 1) // 2

            # FIXME: need to clean up interaction to having only 7 partitions.
            # FIXME: this will involve fixing rccsd.py
            # Restore first compression over symmetric indices
            eri1 = mulliken_to_dirac(ao2mo.restore(1, eri1, nmo))
            self.v = VFull(oooo=eri1[:nocc, :nocc, :nocc, :nocc],
                           ooov=eri1[:nocc, :nocc, :nocc, nocc:],
                           oovv=eri1[:nocc, :nocc, nocc:, nocc:],
                           ovov=eri1[:nocc, nocc:, :nocc, nocc:],
                           voov=eri1[nocc:, :nocc, :nocc, nocc:],
                           ovvv=eri1[:nocc, nocc:, nocc:, nocc:],
                           vvvv=eri1[nocc:, nocc:, nocc:, nocc:],
                           ovvo=eri1[:nocc, nocc:, nocc:, :nocc],
                           vvov=eri1[nocc:, nocc:, :nocc, nocc:],
                           vvoo=eri1[nocc:, nocc:, :nocc, :nocc],
                           ovoo=eri1[:nocc, nocc:, :nocc, :nocc],
                           oovo=eri1[:nocc, :nocc, nocc:, :nocc],
                           vvvo=eri1[nocc:, nocc:, nocc:, :nocc]
                           )
        else:
            raise ValueError('SCF object did not supply AO integrals')

        log.timer('CCSD integral transformation', *cput0)


class HAM_SPINLESS_RI_CORE:

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos
        nocc = self.mos.nocc
        nmo = self.mos.nmo
        nvir = self.mos.nvir

        if hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            VSymmetricRI = namedtuple(
                'VSymmetricRI', ('poo', 'pov', 'pvv', 'pvo'))

            naux = cc._scf.with_df.get_naoaux()
            Lpnn = np.empty((naux, nmo, nmo))

            mof = np.asarray(self.mos.mo_coeff, order='F')
            ijslice = (0, nmo, 0, nmo)
            Lpq = None
            pq = 0
            for eri1 in cc._scf.with_df.loop():
                Lpq = ao2mo._ao2mo.nr_e2(
                    eri1, mof, ijslice, aosym='s2', out=Lpq).reshape(-1, nmo, nmo)
                npbatch = Lpq.shape[0]
                Lpnn[pq:pq + npbatch, :, :] = Lpq
                pq += npbatch

            self.l = VSymmetricRI(poo=ref_ndarray(Lpnn[:, :nocc, :nocc]),
                                  pov=ref_ndarray(Lpnn[:, :nocc, nocc:]),
                                  pvv=ref_ndarray(Lpnn[:, nocc:, nocc:]),
                                  pvo=ref_ndarray(Lpnn[:, nocc:, :nocc])
                                  )
        else:
            raise ValueError('SCF object did not supply DF AO integrals')

        log.timer('CCSD integral transformation', *cput0)


class HAM_SPINLESS_RI_CORE_HUBBARD:
    """
    Creates RI decomposed Hubbard hamiltonian,
    as this is not easy to supply with PySCF
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        nocc = self.mos.nocc
        nmo = self.mos.nmo
        nvir = self.mos.nvir

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Build RI integrals analytically

        from tcc.utils import khatrirao
        from math import sqrt

        if hasattr(cc._scf, '_hubbard_interaction'):
            u = cc._scf._hubbard_interaction
            if u < 0:
                u12 = 1j * sqrt(-u)
            else:
                u12 = sqrt(u)

            Lpnn = u12 * np.reshape(
                np.transpose(khatrirao(
                    (np.conj((self.mos.mo_coeff).T),
                     self.mos.mo_coeff.T)
                )), (nmo, nmo, nmo)
            )

            VSymmetricRI = namedtuple(
                'VSymmetricRI', ('poo', 'pov', 'pvv', 'pvo'))

            self.l = VSymmetricRI(poo=ref_ndarray(Lpnn[:, :nocc, :nocc]),
                                  pov=ref_ndarray(Lpnn[:, :nocc, nocc:]),
                                  pvv=ref_ndarray(Lpnn[:, nocc:, nocc:]),
                                  pvo=ref_ndarray(Lpnn[:, nocc:, :nocc]))
        else:
            raise ValueError('SCF object did not supply Hubbard interaction')

        log.timer('CCSD integral transformation', *cput0)


class _HAM_SPINLESS_FULL_CORE_DIR_MATFILE:
    """
    Dirac ordered hamiltonian
    """

    def __init__(self, cc, filename='init_ints.mat'):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        self.mos = cc.mos
        nocc = self.mos.nocc
        nmo = self.mos.nmo
        nvir = self.mos.nvir

        from scipy.io import loadmat
        from os.path import isfile

        # Add Fock matrix
        if (isfile(filename)):
            FockMatrix = namedtuple('FockMatrix', ('oo', 'ov', 'vv'))
            fock = loadmat(filename, variable_names=('Fmo'),
                           matlab_compatible=True)['Fmo']
            self.f = FockMatrix(oo=fock[:nocc, :nocc],
                                ov=fock[:nocc, nocc:],
                                vv=fock[nocc:, nocc:]
                                )

        if (isfile(filename)):
            VFull = namedtuple('VFull', ('oooo', 'ooov', 'oovv', 'ovov',
                                         'voov', 'ovvv', 'vvvv'))

            eri1 = loadmat(filename, variable_names=('Imo'),
                           matlab_compatible=True)['Imo']

            self.v = VFull(oooo=eri1[:nocc, :nocc, :nocc, :nocc],
                           ooov=eri1[:nocc, :nocc, :nocc, nocc:],
                           oovv=eri1[:nocc, :nocc, nocc:, nocc:],
                           ovov=eri1[:nocc, nocc:, :nocc, nocc:],
                           voov=eri1[nocc:, :nocc, :nocc, nocc:],
                           ovvv=eri1[:nocc, nocc:, nocc:, nocc:],
                           vvvv=eri1[nocc:, nocc:, nocc:, nocc:],
                           ovvo=eri1[:nocc, nocc:, nocc:, :nocc],
                           vvov=eri1[nocc:, nocc:, :nocc, nocc:],
                           vvoo=eri1[nocc:, nocc:, :nocc, :nocc],
                           ovoo=eri1[:nocc, nocc:, :nocc, :nocc]
                           )
        else:
            raise ValueError('File not found: {}'.format(filename))

        log.timer('CCSD integral transformation', *cput0)

class _HAM_SPINLESS_THC_CORE_MATFILE:
    """
    Mulliken ordered THC hamiltonian with THC decomposed interaction
    The order of factors in THC decomposition is:

    v_{pqrs} = \sum_{m, n} x1_{p,m} x2_{q,m} x5_{m, n} x3_{r,n} x4_{p,n}

    """

    def __init__(self, cc, filename=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        self.mos = cc.mos
        nocc = self.mos.nocc
        nmo = self.mos.nmo

        from scipy.io import loadmat
        from os.path import isfile

        if filename is None:
            raise ValueError('No file provided for THC interaction')

        # Add Fock matrix
        if (isfile(filename)):
            FockMatrix = namedtuple('FockMatrix', ('oo', 'ov', 'vv'))
            fock = loadmat(filename, variable_names=('Fmo'),
                           matlab_compatible=True)['Fmo']
            self.f = FockMatrix(oo=fock[:nocc, :nocc],
                                ov=fock[:nocc, nocc:],
                                vv=fock[nocc:, nocc:]
                                )

        if (isfile(filename)):
            VTHC = namedtuple('VTHC', ('x1', 'x2', 'x3', 'x4',
                                       'x5'))
            THCBlockedFactor = namedtuple('THCBlockedFactor', ('o', 'v'))

            wf1 = loadmat(filename, variable_names=('WFmo'),
                          matlab_compatible=True)['WFmo'][0]

            self.v = VTHC(x1=THCBlockedFactor(o=wf1[0][:nocc, :],
                                              v=wf1[0][nocc:, :]),
                          x2=THCBlockedFactor(o=wf1[1][:nocc, :],
                                              v=wf1[1][nocc:, :]),
                          x3=THCBlockedFactor(o=wf1[2][:nocc, :],
                                              v=wf1[2][nocc:, :]),
                          x4=THCBlockedFactor(o=wf1[3][:nocc, :],
                                              v=wf1[3][nocc:, :]),
                          x5=wf1[4]
            )
        else:
            raise ValueError('File not found: {}'.format(filename))

        log.timer('CCSD integral transformation', *cput0)


class HAM_SPINLESS_THC_CORE_HUBBARD:
    """
    Creates an (analytically) THC decomposed Hubbard hamiltonian
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        nocc = self.mos.nocc
        nao = self.mos.mo_coeff.shape[0]

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Build CPD integrals analytically

        if hasattr(cc._scf, '_hubbard_interaction'):
            u = cc._scf._hubbard_interaction

            CT = (self.mos.mo_coeff) # Why was it done this way?
            THCBlockedFactor = namedtuple('THCBlockedFactor', ('o', 'v'))
            VTHC = namedtuple('VTHC', ('x1', 'x2', 'x3', 'x4',
                                       'x5'))

            self.v = VTHC(x1=THCBlockedFactor(CT[:nocc, :], CT[nocc:, :]),
                          x2=THCBlockedFactor(CT[:nocc, :], CT[nocc:, :]),
                          x3=THCBlockedFactor(CT[:nocc, :], CT[nocc:, :]),
                          x4=THCBlockedFactor(CT[:nocc, :], CT[nocc:, :]),
                          x5=u * np.eye(nao))
        else:
            raise ValueError('SCF object did not supply Hubbard interaction')

        log.timer('CCSD integral transformation', *cput0)
