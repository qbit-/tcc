import numpy as np
from numpy import einsum
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from tcc.tensors import Tensors
from tcc._rccsdt_ri import (_rccsdt_ri_calculate_energy,
                            _rccsdt_ri_calc_residuals)


class RCCSDT_RI(CC):
    """
    This class implements classic RCCSDT method with
    t1: vo, t2: vvoo, t3: vvvooo ordered amplitudes
    and RI decomposed integrals
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
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2 = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t3 = np.zeros((nvir,) * 3 + (nocc,) * 3)

        return Tensors(t1=t1, t2=t2, t3=t3)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_ri_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_ri_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - a.t2 / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (a.t3 - a.t3.transpose([0, 1, 2, 4, 3, 5])) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        # Symmetrize T3 RHS
        g3 = ((+ g.t3
               + g.t3.transpose([1, 2, 0, 4, 5, 3])
               + g.t3.transpose([2, 0, 1, 5, 3, 4])
               + g.t3.transpose([0, 2, 1, 3, 5, 4])
               + g.t3.transpose([2, 1, 0, 5, 4, 3])
               + g.t3.transpose([1, 0, 2, 4, 3, 5])
               ) / 12)

        # Symmetrize T2 RHS
        g2 = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))

        # Solve
        t2 = g2 * (- cc_denom(h.f, g.t2.ndim, 'dir', 'full'))
        t3 = g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))

        # Symmetrize amplitudes
        t2 = 1 / 2 * (t2 + t2.transpose([1, 0, 3, 2]))
        t3 = ((+ t3
               + t3.transpose([1, 2, 0, 4, 5, 3])
               + t3.transpose([2, 0, 1, 5, 3, 4])
               + t3.transpose([0, 2, 1, 3, 5, 4])
               + t3.transpose([2, 1, 0, 5, 4, 3])
               + t3.transpose([1, 0, 2, 4, 3, 5])) / 6)

        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=t2,
            t3=t3)

    def calculate_update(self, h, a):
        """
        Update amplitudes
        """
        r = self.calc_residuals(h, a)
        g = self.update_rhs(h, a, r)
        return self.solve_amps(h, a, g)


class RCCSDT_RI_HUB(RCCSDT_RI):
    """
    This class implements classic RCCSDT method with
    t1: vo, t2: vvoo, t3: vvvooo ordered amplitudes
    and RI decomposed integrals for Hubbard hamiltonian
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


class RCCSDT_UNIT_RI(CC):
    """
    This class implements classic RCCSDT method with
    t1: vo, t2: vvoo, t3: vvvooo ordered amplitudes
    and RI decomposed integrals. Unitary group residuals are
    formed for T3 from opposite spin residuals
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
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        v_vovo = einsum("pia,pjb->aibj", ham.l.pov, ham.l.pov).conj()

        t2 = v_vovo.transpose([0, 2, 1, 3]) * (- e_abij)
        t3 = np.zeros((nvir,) * 3 + (nocc,) * 3)

        return Tensors(t1=t1, t2=t2, t3=t3)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_ri_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_ri_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - a.t2 / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (a.t3 - a.t3.transpose([0, 1, 2, 4, 3, 5])) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        # Apply n_body symmetry, which builds all
        # other spin parts of the unitary group residual
        g3s = (+ g.t3
               - g.t3.transpose([0, 1, 2, 3, 5, 4])
               + g.t3.transpose([0, 1, 2, 4, 5, 3]))
        g3 = ((+ g.t3
               + g.t3.transpose([1, 2, 0, 4, 5, 3])
               + g.t3.transpose([2, 0, 1, 5, 3, 4])
               + g.t3.transpose([0, 2, 1, 3, 5, 4])
               + g.t3.transpose([2, 1, 0, 5, 4, 3])
               + g.t3.transpose([1, 0, 2, 4, 3, 5])
               + 2 * g3s) / 12)

        g2 = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))

        t2 = g2 * (- cc_denom(h.f, g.t2.ndim, 'dir', 'full'))
        t3 = g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))

        # Symmetrize
        t2 = 1 / 2 * (t2 + t2.transpose([1, 0, 3, 2]))
        t3 = ((+ t3
               + t3.transpose([1, 2, 0, 4, 5, 3])
               + t3.transpose([2, 0, 1, 5, 3, 4])
               + t3.transpose([0, 2, 1, 3, 5, 4])
               + t3.transpose([2, 1, 0, 5, 4, 3])
               + t3.transpose([1, 0, 2, 4, 3, 5])) / 6)

        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=t2,
            t3=t3)

    def calculate_update(self, h, a):
        """
        Update amplitudes
        """
        r = self.calc_residuals(h, a)
        g = self.update_rhs(h, a, r)
        return self.solve_amps(h, a, g)


class RCCSDT_UNIT_RI_HUB(RCCSDT_UNIT_RI):
    """
    This class implements classic RCCSDT method with
    t1: vo, t2: vvoo, t3: vvvooo ordered amplitudes
    and RI decomposed integrals for Hubbard hamiltonian
    """

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE_HUBBARD
        return HAM_SPINLESS_RI_CORE_HUBBARD(self)


def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()

    from tcc.rccsdt_ri import RCCSDT_UNIT_RI
    from tcc.cc_solvers import (classic_solver,
                                update_diis_solver)
    cc = RCCSDT_UNIT_RI(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)
    cc._converged = False
    converged, energy, amps = update_diis_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy - -1.304738e-01))


def test_cc_unit():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()

    from tcc.rccsdt_ri import RCCSDT_UNIT_RI
    from tcc.cc_solvers import (classic_solver,
                                update_diis_solver)
    cc = RCCSDT_UNIT_RI(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)
    cc._converged = False
    converged, energy, amps = update_diis_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy - -1.304738e-01))


if __name__ == '__main__':
    test_cc()
    test_cc_unit()
