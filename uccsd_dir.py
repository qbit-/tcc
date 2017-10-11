import numpy as np
from tcc.cc_solvers import CC
from tcc.denom import cc_denom_spin

from tcc.tensors import Tensors
from tcc._uccsd_dir import (
    _uccsd_calculate_energy,
    _uccsd_calc_residuals
)


class UCCSD(CC):
    """
    This class implements classic UCCSD method
    with vvoo ordered amplitudes and Dirac ordered integrals
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

        from tcc.mos import UHF_MOS
        self._mos = UHF_MOS(mo_coeff, mo_energy, mo_occ, frozen)

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_UHF_FULL_CORE_DIR
        return HAM_UHF_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_a_ai = cc_denom_spin(ham.f.a, ham.f.b, 1, 2, 'dir', 'full')
        e_b_ai = cc_denom_spin(ham.f.a, ham.f.b, 0, 2, 'dir', 'full')

        t1_a = ham.f.a.ov.transpose().conj() * (- e_a_ai)
        t1_b = ham.f.b.ov.transpose().conj() * (- e_b_ai)

        e_a_abij = cc_denom_spin(ham.f.a, ham.f.b, 2, 4, 'dir', 'full')
        e_b_abij = cc_denom_spin(ham.f.a, ham.f.b, 0, 4, 'dir', 'full')

        t2_aa = ham.v.aaaa.oovv.transpose([2, 3, 0, 1]).conj() * (- e_a_abij)
        t2_bb = ham.v.aaaa.oovv.transpose([2, 3, 0, 1]).conj() * (- e_b_abij)

        e_ab_abij = cc_denom_spin(ham.f.a, ham.f.b, 1, 4, 'dir', 'full')
        t2_ab = ham.v.abab.oovv.transpose([2, 3, 0, 1]).conj() * (- e_ab_abij)
        return Tensors(
            t1=Tensors(a=t1_a, b=t1_b),
            t2=Tensors(aa=t2_aa, bb=t2_bb, ab=t2_ab))

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """
        return _uccsd_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        return _uccsd_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        return Tensors(
            t1=Tensors(
                a=r.t1.a - a.t1.a / cc_denom_spin(
                    h.f.a, h.f.b, 1, 2, 'dir', 'full'),
                b=r.t1.b - a.t1.b / cc_denom_spin(
                    h.f.a, h.f.b, 0, 2, 'dir', 'full'),
            ),
            t2=Tensors(
                aa=r.t2.aa - 1 / 2 * (
                    + a.t2.aa + a.t2.aa.transpose([1, 0, 3, 2])
                    - a.t2.aa.transpose([1, 0, 2, 3])
                    - a.t2.aa.transpose([0, 1, 3, 2]))
                / cc_denom_spin(
                    h.f.a, h.f.b, 2, 4, 'dir', 'full'),
                bb=r.t2.bb - 1 / 2 * (
                    + a.t2.bb + a.t2.bb.transpose([1, 0, 3, 2])
                    - a.t2.bb.transpose([1, 0, 2, 3])
                    - a.t2.bb.transpose([0, 1, 3, 2]))
                / cc_denom_spin(
                    h.f.a, h.f.b, 0, 4, 'dir', 'full'),
                ab=r.t2.ab - a.t2.ab / cc_denom_spin(
                    h.f.a, h.f.b, 1, 4, 'dir', 'full')
            )
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        tensor
        """
        
        # g2_aa = (+ g.t2.aa
        #          + g.t2.aa.transpose([1, 0, 3, 2])
        #          - g.t2.aa.transpose([0, 1, 3, 2])
        #          - g.t2.aa.transpose([1, 0, 2, 3])) / 4
        # g2_bb = (+ g.t2.bb
        #          + g.t2.bb.transpose([1, 0, 3, 2])
        #          - g.t2.bb.transpose([0, 1, 3, 2])
        #          - g.t2.bb.transpose([1, 0, 2, 3])) / 4
        g2_ab = (g.t2.ab + g.t2.ab.transpose([1, 0, 3, 2])) / 2

        g2_aa = g.t2.aa
        g2_bb = g.t2.bb
        # g2_ab = g.t2.ab

        return Tensors(
            t1=Tensors(
                a=g.t1.a * (- cc_denom_spin(
                    h.f.a, h.f.b, 1, 2, 'dir', 'full')),
                b=g.t1.b * (- cc_denom_spin(
                    h.f.a, h.f.b, 0, 2, 'dir', 'full'))
            ),
            t2=Tensors(
                aa=g2_aa * (- cc_denom_spin(
                    h.f.a, h.f.b, 2, 4, 'dir', 'full')),
                bb=g2_bb * (- cc_denom_spin(
                    h.f.a, h.f.b, 0, 4, 'dir', 'full')),
                ab=g2_ab * (- cc_denom_spin(
                    h.f.a, h.f.b, 1, 4, 'dir', 'full')),
            ))

    def calculate_update(self, h, a):
        """
        Calculate approximate gradient of T
        """
        r = self.calc_residuals(h, a)
        r2_aa = (+ r.t2.aa
                 + r.t2.aa.transpose([1, 0, 3, 2])
                 - r.t2.aa.transpose([0, 1, 3, 2])
                 - r.t2.aa.transpose([1, 0, 2, 3])) / 4
        r2_bb = (+ r.t2.bb
                 + r.t2.bb.transpose([1, 0, 3, 2])
                 - r.t2.bb.transpose([0, 1, 3, 2])
                 - r.t2.bb.transpose([1, 0, 2, 3])) / 4
        r2_ab = (r.t2.ab + r.t2.ab.transpose([1, 0, 3, 2])) / 2
        dt = Tensors(t1=Tensors(a=r.t1.a, b=r.t1.b),
                     t2=Tensors(aa=r2_aa, bb=r2_bb, ab=r2_ab))
        return Tensors(
            t1=Tensors(
                a=r.t1 * (- cc_denom_spin(
                    h.f.a, h.f.b, 1, 2, 'dir', 'full')),
                b=g.t1 * (- cc_denom_spin(
                    h.f.a, h.f.b, 0, 2, 'dir', 'full'))
            ),
            t2=Tensors(
                aa=g2_aa * (cc_denom_spin(
                    h.f.a, h.f.b, 2, 4, 'dir', 'full')),
                bb=g2_bb * (cc_denom_spin(
                    h.f.a, h.f.b, 0, 4, 'dir', 'full')),
                ab=g2_ab * (cc_denom_spin(
                    h.f.a, h.f.b, 1, 4, 'dir', 'full')),
            ))


    
def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    uhf = scf.UHF(mol)
    uhf.scf()  # -76.0267656731
    rhf = scf.RHF(mol)
    rhf.scf()  # -76.0267656731

    from tcc.uccsd_dir import UCCSD
    from tcc.rccsd import RCCSD
    cc1 = UCCSD(uhf)
    cc2 = RCCSD(rhf)

    from tcc.cc_solvers import classic_solver

    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-8,
        max_cycle=100)

    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-8,
        max_cycle=100)
