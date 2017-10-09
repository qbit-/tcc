import time
import numpy as np
import tempfile
import h5py
from functools import reduce
from pyscf.lib import logger
from pyscf.lib import pack_tril, unpack_tril
from pyscf import ao2mo
from tcc.tensors import Tensors

from collections import namedtuple


def ref_ndarray(a):
    return np.array(a, copy=False, order='C')



def _calculate_noncanonical_fock(scf, mo_coeff, mo_occ):
    """
    Calculates Fock matrix in non-canonical basis
    """
    dm = scf.make_rdm1(mo_coeff, mo_occ)
    fockao = scf.get_hcore() + scf.get_veff(scf.mol, dm)
    if len(fockao.shape) == 2: # spinless orbitals
        return reduce(np.dot, (mo_coeff.T, fockao, mo_coeff))
    elif len(fockao.shape) == 3: # uhf orbitals
        return np.array([reduce(np.dot, (mo_coeff[0].T, fockao[0], mo_coeff[0])),
                         reduce(np.dot, (mo_coeff[1].T, fockao[1], mo_coeff[1]))])



def _assemble_spinless_fock(cc, mos=None):
    """Assembles spinless fock matrix"""
    if mos is None:  # Assume canonical orbitals
        mos = cc.mos
        fock = np.diag(cc.mos.mo_energies)
    else:  # If mo_coeff is not canonical orbitals
        fock = _calculate_noncanonical_fock(cc._scf, mos.mo_coeff,
                                            mos.mo_occ)
    nocc = mos.nocc
    f = Tensors(oo=ref_ndarray(fock[:nocc, :nocc]),
                ov=ref_ndarray(fock[:nocc, nocc:]),
                vv=ref_ndarray(fock[nocc:, nocc:])
    )

    return f



def _assemble_uhf_fock(cc, mos=None):
    """Assembles UHF fock matrix"""
    if mos is None:  # Assume canonical orbitals
        mos = cc.mos
        fock = np.array([np.diag(cc.mos.a.mo_energies),
                np.diag(cc.mos.b.mo_energies)])
    else:  # If mo_coeff is not canonical orbitals
        fock = _calculate_noncanonical_fock(
            cc._scf,
            np.array((mos.a.mo_coeff,mos.b.mo_coeff)),
            np.array((mos.a.mo_occ, mos.b.mos_occ)))
    nocca = mos.a.nocc
    noccb = mos.b.nocc
    f = Tensors(a=Tensors(oo=ref_ndarray(fock[0][:nocca, :nocca]),
                          ov=ref_ndarray(fock[0][:nocca, nocca:]),
                          vv=ref_ndarray(fock[0][nocca:, nocca:])),
                b=Tensors(oo=ref_ndarray(fock[1][:noccb, :noccb]),
                          ov=ref_ndarray(fock[1][:noccb, noccb:]),
                          vv=ref_ndarray(fock[1][noccb:, noccb:])),
    )
    return f



def _assemble_fock(cc, mos=None):
    """Decides how to asseble the fock matrix"""
    if mos is not None:
        m = mos
    else:
        m = cc.mos
    if hasattr(m, 'a'): # UHF mos will have spin part attributes
        return _assemble_uhf_fock(cc, mos)
    else:
        return _assemble_spinless_fock(cc, mos)



def _assemble_moeri_full_core_mul(scf, mos):
    """
    Gets AO eris, transorms them to MO bases and returns relevant blocks
    in Mulliken order
    """
    nocc = mos.nocc
    nmo = mos.nmo
    nvir = mos.nvir
    
    eri1 = ao2mo.incore.full(scf._eri, mos.mo_coeff)

    # Restore first compression over symmetric indices
    eri1 = ao2mo.restore(1, eri1, nmo)
    return Tensors(oooo=ref_ndarray(eri1[:nocc, :nocc, :nocc, :nocc]),
                     ooov=ref_ndarray(eri1[:nocc, :nocc, :nocc, nocc:]),
                     oovv=ref_ndarray(eri1[:nocc, :nocc, nocc:, nocc:]),
                     ovov=ref_ndarray(eri1[:nocc, nocc:, :nocc, nocc:]),
                     voov=ref_ndarray(eri1[nocc:, :nocc, :nocc, nocc:]),
                     ovvv=ref_ndarray(eri1[:nocc, nocc:, nocc:, nocc:]),
                     vvvv=ref_ndarray(eri1[nocc:, nocc:, nocc:, nocc:])
    )



def _assemble_moeri_full_core_dir(scf, mos):
    """
    Gets AO eris, transorms them to MO bases and returns relevant blocks
    in Dirac order
    """
    nocc = mos.nocc
    nmo = mos.nmo
    nvir = mos.nvir
    
    def mulliken_to_dirac(a):
        return a.transpose(0, 2, 1, 3)

    eri1 = ao2mo.incore.full(scf._eri, mos.mo_coeff)
    
    # FIXME: need to clean up interaction to having only 7 partitions.
    # FIXME: this will involve fixing rccsd.py
    # Restore first compression over symmetric indices
    eri1 = mulliken_to_dirac(ao2mo.restore(1, eri1, nmo))
    return Tensors(oooo=eri1[:nocc, :nocc, :nocc, :nocc],
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



def _assemble_moeri_ri_core(scf, mos):
    """
    Gets RI decomposed eris from scf, transforms them to MO basis
    and takes relevant blocks
    """

    nocc = mos.nocc
    nmo = mos.nmo
    nvir = mos.nvir

    naux = scf.with_df.get_naoaux()
    Lpnn = np.empty((naux, nmo, nmo))

    mof = np.asarray(mos.mo_coeff, order='F')
    ijslice = (0, nmo, 0, nmo)
    Lpq = None
    pq = 0
    for eri1 in scf.with_df.loop():
        Lpq = ao2mo._ao2mo.nr_e2(
            eri1, mof, ijslice, aosym='s2', out=Lpq).reshape(-1, nmo, nmo)
        npbatch = Lpq.shape[0]
        Lpnn[pq:pq + npbatch, :, :] = Lpq
        pq += npbatch

    return Tensors(poo=ref_ndarray(Lpnn[:, :nocc, :nocc]),
                   pov=ref_ndarray(Lpnn[:, :nocc, nocc:]),
                   pvv=ref_ndarray(Lpnn[:, nocc:, nocc:]),
                   pvo=ref_ndarray(Lpnn[:, nocc:, :nocc])
    )



def _assemble_moeri_full_core_hub(scf, mos):
    """
    Builds on-site eris, transforms them to MO basis,
    returns relevant blocks
    """
    nocc = mos.nocc
    nmo = mos.nmo
    nao = mos.mo_coeff.shape[0]

    u = scf._hubbard_interaction

    eri1 = np.zeros((nao, nao, nao, nao))
    for i in range(nao):
        eri1[i, i, i, i] = u
    eri1 = ao2mo.restore(8, eri1, nao)
    eri1 = ao2mo.incore.full(eri1, mos.mo_coeff, compact=False)
    eri1 = ao2mo.restore(1, eri1, nmo)

    # FIXME: need to clean up interaction to having only 7 partitions.
    # FIXME: this will involve fixing rccsd.py
    # Restore first compression over symmetric indices
    return Tensors(oooo=eri1[:nocc, :nocc, :nocc, :nocc],
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



def _assemble_moeri_ri_core_hub(scf, mos):
    """
    Builds RI decomposed Hubbard interaction, transforms it to MO basis
    and returns relevant blocks
    """
    nocc = mos.nocc
    nmo = mos.nmo
    nvir = mos.nvir

    from tcc.utils import khatrirao
    from math import sqrt
    
    # Build RI integrals analytically
    u = scf._hubbard_interaction
    if u < 0:
        u12 = 1j * sqrt(-u)
    else:
        u12 = sqrt(u)

    Lpnn = u12 * np.reshape(
        np.transpose(khatrirao(
            (np.conj((mos.mo_coeff).T),
             mos.mo_coeff.T)
        )), (nmo, nmo, nmo)
    )
    
    return Tensors(poo=ref_ndarray(Lpnn[:, :nocc, :nocc]),
                   pov=ref_ndarray(Lpnn[:, :nocc, nocc:]),
                   pvv=ref_ndarray(Lpnn[:, nocc:, nocc:]),
                   pvo=ref_ndarray(Lpnn[:, nocc:, :nocc]))



def _assemble_moeri_thc_core_hub(scf, mos):
    """
    Builds THC integrals analytically, transfroms them to MO basis,
    returns blocked structure
    """

    nocc = mos.nocc
    nao = mos.mo_coeff.shape[0]

    # Build THC integrals analytically

    u = scf._hubbard_interaction
    CT = (mos.mo_coeff) # Why was it done this way?
    
    return Tensors(x1=Tensors(CT[:nocc, :], CT[nocc:, :]),
                     x2=Tensors(CT[:nocc, :], CT[nocc:, :]),
                     x3=Tensors(CT[:nocc, :], CT[nocc:, :]),
                     x4=Tensors(CT[:nocc, :], CT[nocc:, :]),
                    x5=u * np.eye(nao))


    
class HAM_SPINLESS_FULL_CORE_MUL:
    """
    Mulliken ordered hamiltonian
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get mos
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        if (cc._scf._eri is not None):
            self.v = _assemble_moeri_full_core_mul(cc._scf, self.mos)
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

        # Get mos
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        if (cc._scf._eri is not None):
            self.v = _assemble_moeri_full_core_dir(cc._scf, self.mos)
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

        if hasattr(cc._scf, 'with_df') and cc._scf.with_df:
            
            self.l = _assemble_moeri_ri_core(cc._scf, self.mos)
        else:
            raise ValueError('SCF object did not supply DF AO integrals')

        log.timer('CCSD integral transformation', *cput0)



class HAM_SPINLESS_FULL_CORE_HUBBARD:
    """
    Creates real full Hubbard interaction in MO basis
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        # Build Hubbard integrals analytically

        if hasattr(cc._scf, '_hubbard_interaction'):
            self.v = _assemble_moeri_full_core_hub(cc._scf, self.mos)

        else:
            raise ValueError('SCF object did not supply Hubbard interaction')

        log.timer('CCSD integral transformation', *cput0)



class HAM_SPINLESS_RI_CORE_HUBBARD:
    """
    Creates RI decomposed Hubbard hamiltonian
    """

    def __init__(self, cc, mos=None):
        cput0 = (time.clock(), time.time())
        log = logger.Logger(cc.stdout, cc.verbose)

        # Get sizes
        if mos is None:
            self.mos = cc.mos
        else:
            self.mos = mos

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        if hasattr(cc._scf, '_hubbard_interaction'):
            self.l = _assemble_moeri_ri_core_hub(cc._scf, self.mos) 
        else:
            raise ValueError('SCF object did not supply Hubbard interaction')

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

        # Add Fock matrix
        self.f = _assemble_fock(cc, mos)

        if hasattr(cc._scf, '_hubbard_interaction'):

            self.v = _assemble_moeri_thc_core_hub(cc._scf, self.mos)
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
            fock = loadmat(filename, variable_names=('Fmo'),
                           matlab_compatible=True)['Fmo']
            self.f = Tensors(oo=fock[:nocc, :nocc],
                             ov=fock[:nocc, nocc:],
                             vv=fock[nocc:, nocc:]
            )

        if (isfile(filename)):

            eri1 = loadmat(filename, variable_names=('Imo'),
                           matlab_compatible=True)['Imo']

            self.v = Tensors(oooo=eri1[:nocc, :nocc, :nocc, :nocc],
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
            fock = loadmat(filename, variable_names=('Fmo'),
                           matlab_compatible=True)['Fmo']
            self.f = Tensors(oo=fock[:nocc, :nocc],
                             ov=fock[:nocc, nocc:],
                             vv=fock[nocc:, nocc:]
            )

        if (isfile(filename)):
            wf1 = loadmat(filename, variable_names=('WFmo'),
                          matlab_compatible=True)['WFmo'][0]

            self.v = Tensors(x1=Tensors(o=wf1[0][:nocc, :],
                                        v=wf1[0][nocc:, :]),
                             x2=Tensors(o=wf1[1][:nocc, :],
                                        v=wf1[1][nocc:, :]),
                             x3=Tensors(o=wf1[2][:nocc, :],
                                        v=wf1[2][nocc:, :]),
                             x4=Tensors(o=wf1[3][:nocc, :],
                                        v=wf1[3][nocc:, :]),
                             x5=wf1[4]
            )
        else:
            raise ValueError('File not found: {}'.format(filename))

        log.timer('CCSD integral transformation', *cput0)
