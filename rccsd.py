import time
import numpy as np
from numpy import einsum
from tcc.cc import CC
from collections import namedtuple


def ref_ndarray(a):
    return numpy.array(a, copy=False, order='C')

def _memory_usage_inloop(nocc, nvir):
    """
    Assume nvir > nocc, minimal requirements on memory in loop of update_amps
    """
    v = max(nvir ** 3 * 2 + nvir * nocc ** 2 * 2,
            nvir ** 3 + nocc * nvir ** 2 * 5 + nvir * nocc ** 2 * 2,
            nocc * nvir ** 2 * 9)
    return v * 8 / 1e6


def _mem_usage(nocc, nvir):
    """
    Assume nvir > nocc, minimal requirements on memory
    """
    basic = _memory_usage_inloop(nocc, nvir) * 1e6 / 8 + nocc ** 4
    basic = max(basic, nocc * (nocc + 1) // 2 *
                nvir ** 2) + (nocc * nvir) ** 2 * 2
    basic = basic * 8 / 1e6
    nmo = nocc + nvir
    incore = (max((nmo * (nmo + 1) // 2) ** 2 * 2 * 8 / 1e6, basic) +
              (nocc * nvir ** 3 / 2 + nvir ** 4 / 4 + nocc ** 2 * nvir ** 2 * 2 +
               nocc ** 3 * nvir * 2) * 8 / 1e6)
    outcore = basic
    return incore, outcore, basic

    
class RCCSD(CC):
    """
    This class implements classic RCCSD method
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays
    RCCSD_AMPLITUDES_FULL = namedtuple('RCCSD_AMPLITUDES_FULL',
                                       field_names=('t1', 't2'))
    RCCSD_RHS_FULL = namedtuple('RCCSD_RHS_FULL',
                                field_names=('g1', 'g2'))   
    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None):
        """
        Initialize RCCSD
        """
        # Simply copy some parameters from RHF calculation
        super(RCCSD, self).__init__(mf)

        # Initialize molecular orbitals

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        from tcc.mos import SPINLESS_MOS
        self._mos = SPINLESS_MOS(mo_coeff, mo_energy, mo_occ, frozen)

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_DIR
        return HAM_SPINLESS_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        time0 = time.clock(), time.time()
        e_i = ham.f.oo.diagonal()
        e_a = ham.f.vv.diagonal()
        e_ai = _construct_cc_denom(e_a, e_i, 2, 'dir')
        e_abij = _construct_cc_denom(e_a, e_i, 4, 'dir')

        t1 = ham.f.ov.transpose().conj() / e_ai
        t2 = ham.v.oovv.transpose([2,3,0,1]).conj() / e_abij 

        return self.RCCSD_AMPLITUDES_FULL(t1=ref_ndarray(t1),
                                     t2=ref_ndarray(t2)
        )
        

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        tau0 = (
            2 * einsum("jiab->ijab", h.v.oovv)
            - einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiab->ia", t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        energy = (
            einsum("ai,ia->", t1, tau1)
            + einsum("abij,jiab->", t2, tau0)
        )

        return energy
        
    def update_rhs(self, ham, amps):
        """
        Updates G = < \bar{H} > - Tf
        Automatically generated
        """

    def residual_norm(h, a, g):
        """
        Calculates a norm of the residuals in the CC equations
        """
        
    
def test_mp2_energy():
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8 , (0. , 0.     , 0.)],
        [1 , (0. , -0.757 , 0.587)],
        [1 , (0. , 0.757  , 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz',}
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf() # -76.0267656731
    from tcc.cc import concreter
    CCc = concreter(RCCSD)
    CCobj = CCc(rhf)
    h = CCobj.create_ham()
    a = CCobj.init_amplitudes(h)
    energy = CCobj.calculate_energy(h,a)
    print(energy - 0.204019967288338)

def _get_expanded_shape_tuple(dim, ndim):
    """
    Returns a tuple for reshaping a vector into
    ndim - array
    >>> v = _get_expanded_shape_tuple(1,2)
    >>> v
    (1,-1)
    """

    if (dim >= 0) and (dim < ndim): 
        return (1,)*(dim) + (-1,) + (1,)*(ndim-dim-1)
    else:
        raise ValueError('Invalid dimension: {}, not in 0..{}'.format(dim, ndim-1))

def _construct_cc_denom(fv, fo, ndim, ordering='mul'):
    """
    Builds an energy denominator
    from diagonals of occupied and virtual part of the
    Fock matrix. Denominator is defined as
    ordering == 'dir'
    d_{ab..ij..} = (fv_a + fv_b + .. - fo_i - fo_j - ..)
    ordering == 'mul'
    d_{aibj..} = (fv_a - fo_i + fv_b - fo_j + ..)

    >>> import numpy as np
    >>> a = np.array([1, 2])
    >>> b = np.array([3, 4])
    >>> c = _construct_cc_denom(fv, fo, 4, 'mul')
    >>> d = _construct_cc_denom(fv, fo, 4, 'dir')
    >>> np.allclose(c, d.transpose([0,2,1,3])
    True
    >>> c = _construct_cc_denom(fv, fo, 2, 'mul')
    >>> c[1,0]
    3
    """
    if ndim % 2 != 0:
        raise ValueError('Ndim is not an even integer: {}'.format(d))

    no = len(fo)
    nv = len(fv)
    npair = ndim // 2
    
    if ordering == 'dir':
        vecs = [+fv.reshape(_get_expanded_shape_tuple(ii, ndim))
                for ii in range(npair)]
        vecs.extend([-fo.reshape(_get_expanded_shape_tuple(ii + npair, ndim))
                     for ii in range(npair)])
        
    elif ordering == 'mul':
        vecs = (+fv.reshape(_get_expanded_shape_tuple(2*ii, ndim))
                -fo.reshape(_get_expanded_shape_tuple(2*ii + 1, ndim))
                for ii in range(npair))
    else:
        raise ValueError('Unknown ordering: {}'.format(ordering))

    return sum(vecs)
