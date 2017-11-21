import numpy as np
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from tcc.tensors import Tensors
from tcc._rccsdt import (_rccsdt_calculate_energy,
                         _rccsdt_calc_residuals)


class RCCSDT(CC):
    """
    This class implements classic RCCSDT method with
    vvvooo ordered amplitudes and Dirac ordered integrals
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays

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
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_DIR
        return HAM_SPINLESS_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')
        nocc = self.mos.nocc
        nvir = self.mos.nvir

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)
        t3 = np.zeros((nvir,) * 3 + (nocc,) * 3)

        return Tensors(t1=t1, t2=t2, t3=t3)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        return _rccsdt_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_calc_residuals(h, a)

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

        g3 = 1 / 9 * (+ g.t3
                      + g.t3.transpose([2, 0, 1, 3, 4, 5])
                      + g.t3.transpose([1, 2, 0, 3, 4, 5])
                      + g.t3.transpose([0, 1, 2, 5, 3, 4])
                      + g.t3.transpose([1, 0, 2, 4, 3, 5])
                      + g.t3.transpose([2, 1, 0, 4, 3, 5])
                      + g.t3.transpose([2, 0, 1, 4, 5, 3])
                      + g.t3.transpose([0, 1, 2, 4, 5, 3])
                      + g.t3.transpose([1, 2, 0, 4, 5, 3]))

        g2 = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))
        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=g2 * (- cc_denom(h.f, g.t2.ndim, 'dir', 'full')),
            t3=g3 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))
        )

    def calculate_gradient(self, h, a):
        """
        Calculate approximate gradient of T
        """
        r = self.calc_residuals(h, a)
        r3 = 1 / 9 * (+ r.t3
                      + r.t3.transpose([2, 0, 1, 3, 4, 5])
                      + r.t3.transpose([1, 2, 0, 3, 4, 5])
                      + r.t3.transpose([0, 1, 2, 5, 3, 4])
                      + r.t3.transpose([1, 0, 2, 4, 3, 5])
                      + r.t3.transpose([2, 1, 0, 4, 3, 5])
                      + r.t3.transpose([2, 0, 1, 4, 5, 3])
                      + r.t3.transpose([0, 1, 2, 4, 5, 3])
                      + r.t3.transpose([1, 2, 0, 4, 5, 3]))

        dt = Tensors(t1=r.t1, t2=r.t2, t3=r3)

        def multiply_by_inverse(x):
            return x * (cc_denom(h.f, x.ndim, 'dir', 'full'))

        return dt.map(multiply_by_inverse)


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
    rhf.scf()

    from tcc.rccsdt import RCCSDT
    from tcc.cc_solvers import classic_solver
    cc = RCCSDT(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-8,
        max_cycle=100)

    print('dE: {}'.format(energy - -1.298894e-01))


def test_compare_to_aq():  # pragma: nocover
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
    rhf.scf()  # -76.0267656731

    # load reference arrays
    import h5py
    import numpy as np
    f1 = h5py.File('data/test_references/aq_ccsdt_amps.h5', 'r')
    # use amplitudes from the last iteration
    num_steps = int(len(f1.keys()) / 4)
    t1 = f1['t1_' + str(num_steps)][()].T
    t2 = f1['t2_' + str(num_steps)][()].T
    t3 = f1['t3_' + str(num_steps)][()].T
    t3a = f1['t3b_' + str(num_steps)][()].T
    f1.close()

    f1 = h5py.File('data/test_references/aq_ccsdt_mos.h5', 'r')
    CA = np.hstack((f1['cI'][()].T, f1['cA'][()].T))
    CB = np.hstack((f1['ci'][()].T, f1['ca'][()].T))
    f1.close()

    # permute AO indices to match pyscf order
    from tcc.utils import perm_matrix
    perm = [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12]
    m = perm_matrix(perm)
    CA_perm = m.dot(CA)

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import (step_solver, classic_solver,
                                residual_diis_solver)
    from tcc.rccsdt import RCCSDT
    from tcc.rccsd import RCCSD
    cc = RCCSDT(rhf, mo_coeff=CA_perm)

    from tcc.tensors import Tensors
    h = cc.create_ham()
    t3s = (+ t3
           - t3.transpose([0, 1, 2, 4, 3, 5])
           + t3.transpose([0, 1, 2, 5, 3, 4])
           - t3.transpose([0, 1, 2, 3, 5, 4])
           + t3.transpose([0, 1, 2, 4, 5, 3])
           - t3.transpose([0, 1, 2, 5, 4, 3])
           + t3
           - t3.transpose([0, 1, 2, 4, 3, 5])
           + t3.transpose([0, 2, 1, 3, 5, 4])
           - t3.transpose([0, 2, 1, 4, 5, 3])
           + t3.transpose([2, 1, 0, 5, 4, 3])
           - t3.transpose([2, 1, 0, 5, 3, 4])) / 6
    test_amps = Tensors(t1=t1, t2=t2, t3=t3s)
    r = cc.calc_residuals(h, test_amps)

    # converged, energy, amps = step_solver(
    #     cc, amps=test_amps, conv_tol_energy=1e-14, use_optimizer='momentum',
    #     optimizer_kwargs=dict(beta=0.6, alpha=0.01),
    #     max_cycle=300)
    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-12,
        max_cycle=300)

    print('max r1: {}'.format(np.max(r.t1)))
    print('max r2: {}'.format(np.max(r.t2)))
    print('max r3: {}'.format(np.max(r.t3)))


if __name__ == '__main__':
    test_cc()
    test_compare_to_aq()
