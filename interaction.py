import time
import numpy
import tempfile
import h5py
from functools import reduce
from pyscf.lib import logger
from pyscf.lib import pack_tril, unpack_tril
from pyscf import ao2mo

from collections import namedtuple


def ref_ndarray(a):
    return numpy.array(a, copy=False, order='C')


def _calculate_noncanonical_fock(scf, mo_coeff, mo_occ):
    """
    Calculates Fock matrix in non-canonical basis
    """
    dm = scf.make_rdm1(mo_coeff, mo_occ)
    fockao = scf.get_hcore() + scf.get_veff(scf.mol, dm)
    return reduce(numpy.dot, (mo_coeff.T, fockao, mo_coeff))


def _assemble_fock(cc, mos=None):
    # Calculate Fock matrix
    FockMatrix = namedtuple('FockMatrix', ('oo', 'ov', 'vv'))
    if mos is None:  # Assume canonical orbitals
        mos = cc.mos
        fock = numpy.diag(cc.mos.mo_energies)
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
                                         'vvov', 'vvoo', 'ovoo'))

            eri1 = ao2mo.incore.full(cc._scf._eri, self.mos.mo_coeff)
            nvir_pair = nvir * (nvir + 1) // 2

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
                           ovoo=eri1[:nocc, nocc:, :nocc, :nocc]
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
            Lpnn = numpy.empty((naux, nmo, nmo))

            mof = numpy.asarray(self.mos.mo_coeff, order='F')
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
