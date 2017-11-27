import numpy as np
from tcc.cc_solvers import CC
from tcc.denom import cc_denom

from tcc.tensors import Tensors
from tcc._rccsd import (_rccsd_calculate_energy,
                        _rccsd_calc_residuals)

from tcc._rccsd import (_rccsd_unit_calculate_energy,
                        _rccsd_unit_calc_residuals)

from tcc._rccsd import (_rccsd_ri_calculate_energy,
                        _rccsd_ri_calc_residuals)


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

    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        def divide_by_inverse(x):
            return x / (cc_denom(h.f, x.ndim, 'dir', 'full'))

        g = r - a.map(divide_by_inverse)
        # Apply symmetrization to the RHS, so we sstay always with
        # tensors having an n_body symmetry and hence have always a
        # contracting algorithm. See RCCSDT notes for more discussion.
        # This should be done with a separate function
        # g['t2'] = 1 / 2 * (g.t2 + g.t2.transpose([1, 0, 3, 2]))

        return g

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        tensor
        """

        def multiply_by_inverse(x):
            return x * (- cc_denom(h.f, x.ndim, 'dir', 'full'))

        return g.map(multiply_by_inverse)

    def calculate_gradient(self, h, a):
        """
        Calculate dt
        """
        r = self.calc_residuals(h, a)
        # Symmetrize
        # r['t2'] = 1 / 2 * (r.t2 + r.t2.transpose([1, 0, 3, 2]))

        def multiply_by_inverse(x):
            return x * (cc_denom(h.f, x.ndim, 'dir', 'full'))

        return r.map(multiply_by_inverse)

    def calculate_update(self, h, a):
        """
        Calculate new amplitudes in one shot
        """
        r = self.calc_residuals(h, a)
        # Symmetrize
        # r['t2'] = 1 / 2 * (r.t2 + r.t2.transpose([1, 0, 3, 2]))

        def multiply_by_inverse(x):
            return x * (- cc_denom(h.f, x.ndim, 'dir', 'full'))

        def divide_by_inverse(x):
            return x / (cc_denom(h.f, x.ndim, 'dir', 'full'))

        return (r - a.map(divide_by_inverse)).map(multiply_by_inverse)


class RCCSD_UNIT(RCCSD):
    """
    RCCSD equations based entirely on the unitary group operators
    """

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = 2 * ham.f.ov.transpose().conj() * (- e_ai)
        t2 = 2 * ham.v.oovv.transpose([2, 3, 0, 1]).conj() * (- e_abij)

        return Tensors(t1=t1, t2=t2)

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

    def update_rhs(self, h, a, r):
        """
        Calculates RHS of the fixed point iteration of CC equations
        """

        return Tensors(
            t1=r.t1 - 2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - 2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                           ) / cc_denom(h.f, 4, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        # Solve
        t2 = ((2 * g.t2 + g.t2.transpose([0, 1, 3, 2]))
              / (- 6) * cc_denom(h.f, 4, 'dir', 'full'))
        # Symmetrize
        t2 = 1 / 2 * (t2 + t2.transpose([1, 0, 3, 2]))

        return Tensors(
            t1=g.t1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full'),
            t2=t2
        )


class RCCSD_DIR_RI(RCCSD):
    """
    This class implements RCCSD with Dirac ordered amplitudes:
    t1: vo, t2: vvoo, usual abba residuals and RI decomposed
    integrals.
    """

    def create_ham(self):
        """
        Create RI Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')
        e_abij = cc_denom(ham.f, 4, 'dir', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)

        v_vvoo = np.einsum("pia,pjb->abij", ham.l.pov, ham.l.pov).conj()
        t2 = v_vvoo * (- e_abij)

        return Tensors(t1=t1, t2=t2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        """

        return _rccsd_ri_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates residuals of the CC equations
        """

        return _rccsd_ri_calc_residuals(h, a)


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
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD_UNIT, RCCSD
    cc1 = RCCSD_UNIT(rhf)
    cc2 = RCCSD(rhf)

    converged1, energy2, _ = residual_diis_solver(
        cc1, ndiis=5, conv_tol_energy=-1, conv_tol_res=1e-10)

    converged2, energy2, _ = residual_diis_solver(
        cc2, ndiis=5, conv_tol_energy=-1, conv_tol_res=1e-10)


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
    converged, energy, _ = root_solver(
        cc, conv_tol=1e-6)


def test_cc_step():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import step_solver, classic_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf)

    converged1, energy1, _ = classic_solver(
        cc, conv_tol_energy=1e-10, conv_tol_res=1e-10,
        max_cycle=20)
    cc._converged = False

    converged2, energy2, _ = step_solver(
        cc, conv_tol_energy=1e-10,
        beta=0.5,
        max_cycle=100)


def test_cc_step_diis():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]

    mol.basis = {'H': 'sto-3g',
                 'O': 'sto-3g', }
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()  # -76.0267656731

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import update_diis_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf)

    converged1, energy1, _ = residual_diis_solver(
        cc, conv_tol_energy=1e-10, conv_tol_res=1e-10,
        max_cycle=100, lam=1)
    cc._converged = False

    converged2, energy2, _ = update_diis_solver(
        cc, conv_tol_energy=1e-10, conv_tol_res=1e-10,
        beta=0,
        max_cycle=100)


def compare_to_aq():  # pragma: nocover
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
    f1 = h5py.File('data/test_references/aq_ccsd_amps.h5', 'r')
    # use amplitudes from the last iteration
    num_steps = int(len(f1.keys()) / 2)
    t1 = f1['t1_' + str(num_steps)][()].T
    t2 = f1['t2_' + str(num_steps)][()].T
    f1.close()

    f1 = h5py.File('data/test_references/aq_ccsd_mos.h5', 'r')
    CA = np.hstack((f1['cI'][()].T, f1['cA'][()].T))
    CB = np.hstack((f1['ci'][()].T, f1['ca'][()].T))
    f1.close()

    # permute AO indices to match pyscf order
    perm = [0, 1, 2, 4, 5, 3, 7, 8, 6, 9, 10, 11, 12]
    from tcc.utils import perm_matrix
    m = perm_matrix(perm)
    CA_perm = m.dot(CA)

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import step_solver, classic_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf, mo_coeff=CA_perm)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-14, conv_tol_res=1e-10,
        max_cycle=200)

    print('dt1: {}'.format(np.max(t1 - amps.t1)))
    print('dt2: {}'.format(np.max(t2 - amps.t2)))

    from tcc.tensors import Tensors
    test_amps = Tensors(t1=t1, t2=t2)
    h = cc.create_ham()
    r = cc.calc_residuals(h, test_amps)

    print('max r1: {}'.format(np.max(r.t1)))
    print('max r2: {}'.format(np.max(r.t2)))


def test_cc_ri():  # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = [
        ['O', (0., 0., 0.)],
        ['H', (0., -0.757, 0.587)],
        ['H', (0., 0.757, 0.587)]]

    mol.basis = 'sto-3g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()  # -76.0267656731

    rhf_ri = scf.density_fit(scf.RHF(mol))
    rhf_ri.scf()

    from tcc.cc_solvers import residual_diis_solver
    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD_DIR_RI, RCCSD
    cc1 = RCCSD_DIR_RI(rhf_ri)
    cc2 = RCCSD(rhf)

    converged1, energy2, _ = classic_solver(
        cc1, conv_tol_energy=-1)

    converged2, energy2, _ = classic_solver(
        cc2, conv_tol_energy=-1)


def test_compare_to_hirata():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.atom = """
    # H2O
    H    0.000000000000000   1.079252144093028   1.474611055780858
    O    0.000000000000000   0.000000000000000   0.000000000000000
    H    0.000000000000000   1.079252144093028  -1.474611055780858
    """
    mol.unit = 'Bohr'
    mol.basis = {
        'H': gto.basis.parse(
            """
            H         S   
                      3.42525091         0.15432897
                      0.62391373         0.53532814
                      0.16885540         0.44463454
            """
        ),
        'O': gto.basis.parse(
            """
            O         S   
                    130.70932000         0.15432897
                     23.80886100         0.53532814
                      6.44360830         0.44463454
            O         S   
                      5.03315130        -0.09996723
                      1.16959610         0.39951283
                      0.38038900         0.70011547
            O         P   
                      5.03315130         0.15591627
                      1.16959610         0.60768372
                      0.38038900         0.39195739
            """
        ),
    }
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()

    from tcc.cc_solvers import classic_solver, update_diis_solver
    from tcc.rccsd import RCCSD
    cc1 = RCCSD(rhf)

    converged, energy, amps = classic_solver(
        cc1, conv_tol_energy=1e-10, lam=3, conv_tol_res=1e-10,
        max_cycle=200)

    print('E_cc: {}'.format(energy))
    print('E_tot: {}'.format(rhf.e_tot + energy))
    print('delta E: {}'.format(energy - -0.0501273286))


def test_show_cc_divergence():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 10, 10, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD
    cc = RCCSD(rhf)
    converged, energy, _ = classic_solver(
        cc, conv_tol_energy=1e-12, conv_tol_res=1e-12,
        max_cycle=200)
    # print('dE: {}, norm: {}'.format(cc._dEs[-1], cc._dnorms[-1]))
    import numpy as np
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.semilogy(np.abs(cc._dEs))
    ax.semilogy(cc._dnorms)
    plt.xlabel('Number of iterations')
    plt.ylabel('log($x$)')
    plt.ylim(None, 8)
    plt.legend(['$|dE|$', '$|{}^2 T - {}^2 T_{symmetric}|$'])
    plt.title(
        'Unstable RCCSD algorithm on 1D Hubbard, 10 sites, U = 2, PBC')
    plt.show()


def test_unit_equivalence():
    """
    Shows that two unitary residual formulations of CC
    are equivalent
    """
    from pyscf import gto
    from pyscf import scf
    from tcc.hubbard import hubbard_from_scf
    rhf = hubbard_from_scf(scf.RHF, 10, 10, 3, 'y')
    rhf.damp = -4.0
    rhf.scf()

    from tcc.cc_solvers import classic_solver
    from tcc.rccsd import RCCSD, RCCSD_UNIT
    cc1 = RCCSD(rhf)
    converged1, energy1, amps1 = classic_solver(
        cc1, conv_tol_energy=1e-12, conv_tol_res=1e-12,
        max_cycle=120)

    cc2 = RCCSD_UNIT(rhf)
    converged2, energy2, amps2 = classic_solver(
        cc2, conv_tol_energy=1e-12, conv_tol_res=1e-12,
        max_cycle=120)
    print('cc1: E: {}, norm: {}'.format(energy1, cc1._dnorms[-1]))
    print('cc2: E: {}, norm: {}'.format(energy2, cc2._dnorms[-1]))


if __name__ == '__main__':
    # test_mp2_energy()
    # test_cc_hubbard()
    # test_cc_unitary()
    # test_cc_step()
    # compare_to_aq()
    # test_compare_to_hirata()
    test_show_cc_divergence()
    # test_unit_equivalence()
