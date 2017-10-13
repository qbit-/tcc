import numpy as np
from tcc.cc_solvers import CC
from tcc.denom import cc_denom

from tcc.tensors import Tensors
from tcc._rccsd_mul import (
    _rccsd_mul_calculate_energy,
    _rccsd_mul_calc_residuals
)
from tcc._rccsd_mul import (
    _rccsd_mul_ri_calculate_energy,
    _rccsd_mul_ri_calc_residuals
)

from numpy import einsum


class RCCSD_MUL(CC):
    """
    This implements RCCSD algorithm with Mulliken ordered
    amplitudes and integrals
    """

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None):
        """
        Initialize RCCSD
        """
        # Simply copy some parameters from RHF calculation
        super().__init__(mf)

        # Initialize molecular orbitals

        if mo_energy is None:
            mo_energy = mf.mo_energy
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ

        from tcc.mos import SPINLESS_MOS
        self._mos = SPINLESS_MOS(mo_coeff, mo_energy, mo_occ, frozen)

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_MUL
        return HAM_SPINLESS_FULL_CORE_MUL(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'mul', 'full')
        e_aibj = cc_denom(ham.f, 4, 'mul', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.ovov.transpose().conj() * (- e_aibj)

        return Tensors(t1=t1, t2=t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsd_mul_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates ersiduals of the CC equations
        """

        return _rccsd_mul_calc_residuals(h, a)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        def multiply_by_inverse(x):
            return x * (- cc_denom(h.f, x.ndim, 'mul', 'full'))

        return g.map(multiply_by_inverse)

    def update_rhs(self, h, a, r):
        """
        Calculates CC residuals from RHS and amplitudes
        """

        def divide_by_inverse(x):
            return x / (cc_denom(h.f, x.ndim, 'mul', 'full'))

        return r - a.map(divide_by_inverse)

    def calculate_update(self, h, a):
        """
        Calculate dt
        """
        r = self.calc_residuals(h, a)

        def multiply_by_inverse(x):
            return x * (cc_denom(h.f, x.ndim, 'mul', 'full'))

        return r.map(multiply_by_inverse)


class RCCSD_MUL_RI(RCCSD_MUL):
    """
    This implements RCCSD algorithm with Mulliken ordered
    amplitudes and density fitted integrals
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'mul', 'full')
        e_aibj = cc_denom(ham.f, 4, 'mul', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)

        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()
        t2 = v_vovo * (- e_aibj)

        return Tensors(t1=t1, t2=t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsd_mul_ri_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates residuals of CC equations
        Automatically generated
        """

        return _rccsd_mul_ri_calc_residuals(h, a)


class RCCSD_MUL_RI_HUB(RCCSD_MUL_RI):
    """
    This class implements CCSD RI for Hubbard hamiltonian.
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


def test_mp2_energy():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    cc = RCCSD_MUL_RI(rhf)
    h = cc.create_ham()
    a = cc.init_amplitudes(h)
    energy = cc.calculate_energy(h, a)
    print('E_mp2 - E_cc,init = {:18.12g}'.format(energy - -0.204019967288338))


def test_cc():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'cc-pvdz',
                 'O': 'cc-pvdz', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import classic_solver
    from tcc.rccsd_mul import RCCSD_MUL_RI
    cc = RCCSD_MUL_RI(rhf)
    converged, energy, _ = classic_solver(cc)


def test_cc_hubbard_ri():   # pragma: nocover
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 22, 22, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import residual_diis_solver
    from tcc.rccsd_mul import RCCSD_MUL_RI_HUB
    cc = RCCSD_MUL_RI_HUB(rhf)
    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_res=1e-6, lam=5,
        max_cycle=100)


if __name__ == '__main__':
    test_mp2_energy()
    test_cc()
    test_cc_hubbard_ri()
