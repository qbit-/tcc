import numpy as np
from .cc_solvers import CC
from .denom import cc_denom

from .tensors import Tensors
from ._rccsd import (_rccsd_calculate_energy,
                     _rccsd_calc_residuals)

from ._rccsd import (_rccsd_unit_calculate_energy,
                     _rccsd_unit_calc_residuals)


class RCCSD(CC):
    """
    This class implements classic RCCSD method
    with vvoo ordered amplitudes
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
        from tcc.interaction import HAM_SPINLESS_FULL_CORE_DIR
        return HAM_SPINLESS_FULL_CORE_DIR(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        t2 = ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)

        return Tensors(t1=t1, t2=t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """

        return _rccsd_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """

        return _rccsd_calc_residuals(h, a)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        raise NotIMplemented('Bug is likely here')
        multiply_by_inverse = lambda x: x * \
            (- cc_denom(h.f, x.ndim, 'dir', 'full'))
        return g.map(multiply_by_inverse)

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        divide_by_inverse = lambda x: x / \
            (- cc_denom(h.f, x.ndim, 'dir', 'full'))
        return (r - a).map(divide_by_inverse)

    def calculate_update(self, h, a):
        """
        Solving for new T amlitudes using RHS and denominator
        """
        r = self.calc_residuals(h, a)
        multiply_by_inverse = lambda x: x * \
            (cc_denom(h.f, x.ndim, 'dir', 'full'))
        return r.map(multiply_by_inverse)


class RCCSD_UNIT(RCCSD):
    """
    RCCSD equations based entirely on the unitary group operators
    """

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """

        return _rccsd_unit_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates residuals of the CC equations
        """

        return _rccsd_unit_calc_residuals(h, a)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return Tensors(
            t1=g.t1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full'),
            t2=(2 * g.t2 + g.t2.transpose([0, 1, 3, 2])
                ) / (- 6) * cc_denom(h.f, 4, 'dir', 'full')
        )

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = 2 * ham.f.ov.transpose().conj() * (- e_ai)
        t2 = 2 * ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)

        return Tensors(t1=t1, t2=t2)

    def update_rhs(self, h, a, r):
        """
        Calculates RHS of the fixed point iteration of CC equations
        """
        return Tensors(
            t1=r.t1 - 2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - 2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                           ) / cc_denom(h.f, 4, 'dir', 'full')
        )


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
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    CCobj = RCCSD(rhf)
    h = CCobj.create_ham()
    a = CCobj.init_amplitudes(h)
    energy = CCobj.calculate_energy(h, a)
    print('E_mp2 - E_cc,init = {:18.12g}'.format(energy - -0.204019967288338))


def test_cc_unitary():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g', }
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)

    converged, energy, _ = residual_diis_solver(
        cc, ndiis=5, conv_tol_energy=-1, conv_tol_res=1e-10)


def test_cc_hubbard():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 6, 6, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import root_solver, residual_diis_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)
    # converged, energy, _ = residual_diis_solver(
    #     cc, ndiis=5, conv_tol_res=1e-6, lam=5,
    #     max_cycle=100)
    converged, energy, _ = root_solver(
        cc, conv_tol=1e-6)


def test_cc_step():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0., -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g', }
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import step_solver, classic_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf)

    converged, energy, _ = step_solver(
        cc, conv_tol_energy=-1, use_optimizer='adamax',
        optimizer_kwargs=dict(alpha=0.01, beta=0.85, gamma=0.9), max_cycle=500)


def compare_to_aq(): # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., -0.10277433)],
        [1, (0., -1.18603436,  0.81555159)],
        [1, (0., 1.18603436,  0.81555159)]]
    mol.unit = 'Bohr'
    mol.basis = {'H': '3-21g',
                 'O': '3-21g', }
    mol.build()
    rhf = scf.RHF(mol)
    # rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import step_solver, classic_solver
    from tcc.rccsd import RCCSD_UNIT
    cc = RCCSD_UNIT(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-14, max_cycle=200)

    import h5py
    f = h5py.File('amplitude_dump.h5', 'r')
    t1 = f['t1_19'][()].T
    t2 = f['t2_19'][()].T    
    f.close()
if __name__ == '__main__':
    # test_mp2_energy()
    # test_cc_hubbard()
    # test_cc_unitary()
    test_cc_step()
