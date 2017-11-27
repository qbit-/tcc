import numpy as np
from tcc.denom import cc_denom
from tcc.cc_solvers import CC
from tcc.tensors import Tensors
from tcc._rccsdt import (_rccsdt_calculate_energy,
                         _rccsdt_calc_residuals)

from tcc._rccsdt import (_rccsdt_unit_calculate_energy,
                         _rccsdt_unit_calc_residuals)

from tcc._rccsdt import (_rccsdt_unit_anti_calculate_energy,
                         _rccsdt_unit_anti_calc_residuals)


class RCCSDT(CC):
    """
    This class implements classic RCCSDT method with
    vvvooo, vvoo, vo ordered amplitudes and Dirac ordered integrals
    Residuals are (alpha), (alpha beta), (alpha alpha beta) for
    t1, t2, t3 respectively. This code reproduces results of
    So Hirata, but I suspect that it is not correct, and
    yeilds only T3(alpha, alpha, beta) amplitude block
    of UCCSDT instead of RCCSDT.
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

        # This version coincides with So
        # Hirata. To do full unitary residuals uncomment blocks
        # %1 and %2 below

        # g3s = (+ g.t3     # %1
        #        - g.t3.transpose([0, 1, 2, 3, 5, 4])
        #        + g.t3.transpose([0, 1, 2, 4, 5, 3]))

        # Apply n_body symmetry, which builds all
        # other spin parts of the unitary group residual
        g3 = ((+ g.t3
               + g.t3.transpose([1, 2, 0, 4, 5, 3])
               + g.t3.transpose([2, 0, 1, 5, 3, 4])
               + g.t3.transpose([0, 2, 1, 3, 5, 4])
               + g.t3.transpose([2, 1, 0, 5, 4, 3])
               + g.t3.transpose([1, 0, 2, 4, 3, 5])
               # + 2 * g3s) / 12   # %2
               ) / 6)

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
        # Apply n_body symmetry
        r3 = (+ r.t3
              + r.t3.transpose([1, 2, 0, 4, 5, 3])
              + r.t3.transpose([2, 0, 1, 5, 3, 4])
              + r.t3.transpose([0, 2, 1, 3, 5, 4])
              + r.t3.transpose([2, 1, 0, 5, 4, 3])
              + r.t3.transpose([1, 0, 2, 4, 3, 5])) / 6

        dt = Tensors(t1=r.t1, t2=r.t2, t3=r3)

        def multiply_by_inverse(x):
            return x * (cc_denom(h.f, x.ndim, 'dir', 'full'))

        return dt.map(multiply_by_inverse)


class RCCSDT_UNIT(CC):
    """
    This class implements classic RCCSDT method with
    vvvooo, vvoo, vo ordered amplitudes and Dirac ordered integrals
    Residuals are defined with respect to unitary group generators
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

        return _rccsdt_unit_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_unit_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - 2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                           ) / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - (+ 4 * a.t3.transpose([0, 1, 2, 4, 3, 5])
                       + 4 * a.t3.transpose([0, 1, 2, 5, 4, 3])
                       + 4 * a.t3.transpose([0, 1, 2, 3, 5, 4])
                       - 2 * a.t3.transpose([0, 1, 2, 5, 3, 4])
                       - 2 * a.t3.transpose([0, 1, 2, 4, 5, 3])
                       - 8 * a.t3) /
            cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        # Extract an n_body part of the T3 RHS (and hence residual)
        g3 = (+ g.t3
              + g.t3.transpose([1, 2, 0, 4, 5, 3])
              + g.t3.transpose([2, 0, 1, 5, 3, 4])
              + g.t3.transpose([0, 2, 1, 3, 5, 4])
              + g.t3.transpose([2, 1, 0, 5, 4, 3])
              + g.t3.transpose([1, 0, 2, 4, 3, 5])) / 6

        t3 = (g3 / 12
              * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full')))
        # Apply n_body symmetry - may this be skipped because of the above?
        # t3 = (+ t3
        #       + t3.transpose([1, 2, 0, 4, 5, 3])
        #       + t3.transpose([2, 0, 1, 5, 3, 4])
        #       + t3.transpose([0, 2, 1, 3, 5, 4])
        #       + t3.transpose([2, 1, 0, 5, 4, 3])
        #       + t3.transpose([1, 0, 2, 4, 3, 5])) / 6

        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=(2 * g.t2 + g.t2.transpose([0, 1, 3, 2])
                ) / (- 6) * cc_denom(h.f, 4, 'dir', 'full'),
            t3=t3
        )

    def calculate_update(self, h, a):
        """
        Do normal update
        """
        r = self.calc_residuals(h, a)
        g = self.update_rhs(h, a, r)
        return self.solve_amps(h, a, g)


class RCCSDT_UNIT_ANTI(CC):
    """
    This class implements classic RCCSDT method with
    vvvooo, vvoo, vo ordered amplitudes and Dirac ordered integrals
    Residuals are defined with respect to unitary group generators.
    In this class T3 amplitudes are completely antisymmetric,
    which is more restrictive than a usual `n_body` symmetry
    of RCCSDT amplitudes. This is by no way a correct RCCSDT,
    procedure because by making T3
    antisymmetric we forget about spin labels and treat all particles as
    spinless fermions in triple excitations.
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

        return _rccsdt_unit_anti_calculate_energy(h, a)

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """

        return _rccsdt_unit_anti_calc_residuals(h, a)

    def update_rhs(self, h, a, r):
        """
        Calculates right hand side of CC equations
        """
        return Tensors(
            t1=r.t1 - a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            t2=r.t2 - 2 * (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                           ) / cc_denom(h.f, 4, 'dir', 'full'),
            t3=r.t3 - 24 * a.t3 / cc_denom(h.f, 6, 'dir', 'full')
        )

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        t3 = g.t3 / 24 * (- cc_denom(h.f, g.t3.ndim, 'dir', 'full'))
        # Antisymmetrize over all indices (upper and lower separately)
        t3 = 1 / 6 * (+ t3
                      - t3.transpose([1, 0, 2, 3, 4, 5])
                      + t3.transpose([1, 2, 0, 3, 4, 5])
                      - t3.transpose([2, 1, 0, 3, 4, 5])
                      + t3.transpose([2, 0, 1, 3, 4, 5])
                      - t3.transpose([0, 2, 1, 3, 4, 5]))
        t3 = 1 / 6 * (+ t3
                      - t3.transpose([0, 1, 2, 4, 3, 5])
                      + t3.transpose([0, 1, 2, 4, 5, 3])
                      - t3.transpose([0, 1, 2, 5, 4, 3])
                      + t3.transpose([0, 1, 2, 5, 3, 4])
                      - t3.transpose([0, 1, 2, 3, 5, 4]))

        return Tensors(
            t1=g.t1 * (- cc_denom(h.f, g.t1.ndim, 'dir', 'full')),
            t2=(2 * g.t2 + g.t2.transpose([0, 1, 3, 2])
                ) / 6 * (- cc_denom(h.f, 4, 'dir', 'full')),
            t3=t3
        )

    def calculate_update(self, h, a):
        """
        Do normal update
        """
        r = self.calc_residuals(h, a)
        g = self.update_rhs(h, a, r)
        return self.solve_amps(h, a, g)


def test_cc_anti():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.unit = 'Angstrom'
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = '3-21g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()

    from tcc.rccsdt import RCCSDT_UNIT_ANTI
    from tcc.cc_solvers import classic_solver
    cc = RCCSDT_UNIT_ANTI(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-10, conv_tol_res=1e-10,
        lam=3,
        max_cycle=100)

    h = cc.create_ham()
    res = cc.calc_residuals(h, amps)
    r3 = res.t3
    # Extract antisymmetric part only
    r3 = 1 / 6 * (+ r3
                  - r3.transpose([1, 0, 2, 3, 4, 5])
                  + r3.transpose([1, 2, 0, 3, 4, 5])
                  - r3.transpose([2, 1, 0, 3, 4, 5])
                  + r3.transpose([2, 0, 1, 3, 4, 5])
                  - r3.transpose([0, 2, 1, 3, 4, 5]))
    r3 = 1 / 6 * (+ r3
                  - r3.transpose([0, 1, 2, 4, 3, 5])
                  + r3.transpose([0, 1, 2, 4, 5, 3])
                  - r3.transpose([0, 1, 2, 5, 4, 3])
                  + r3.transpose([0, 1, 2, 5, 3, 4])
                  - r3.transpose([0, 1, 2, 3, 5, 4]))

    import numpy as np
    norms = res.map(np.linalg.norm)
    print('r1: {}, r2: {}, r3: {}, r3_antisym: {}'.format(
        norms.t1, norms.t2, norms.t3, np.linalg.norm(r3)))
    # The energy should be higher than in the correct RCCSDT
    # due to more restrictions on the symmetry of T3 than needed
    print('dE: {}'.format(energy - -1.311811e-01))


def test_cc():   # pragma: nocover
    from pyscf import gto
    from pyscf import scf
    mol = gto.Mole()
    mol.unit = 'Angstrom'
    mol.atom = [
        [8, (0., 0., 0.)],
        [1, (0.,  -0.757, 0.587)],
        [1, (0., 0.757, 0.587)]]
    mol.basis = '3-21g'
    mol.build()
    rhf = scf.RHF(mol)
    rhf.scf()

    from tcc.rccsdt import RCCSDT_UNIT
    from tcc.cc_solvers import classic_solver
    cc = RCCSDT_UNIT(rhf)

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-12, conv_tol_res=1e-12,
        lam=3,
        max_cycle=100)

    h = cc.create_ham()
    res = cc.calc_residuals(h, amps)
    r3 = res.t3
    # Apply n_body symmetry
    r3 = (+ r3
          + r3.transpose([1, 2, 0, 4, 5, 3])
          + r3.transpose([2, 0, 1, 5, 3, 4])
          + r3.transpose([0, 2, 1, 3, 5, 4])
          + r3.transpose([2, 1, 0, 5, 4, 3])
          + r3.transpose([1, 0, 2, 4, 3, 5])) / 6

    import numpy as np
    norms = res.map(np.linalg.norm)
    print('r1: {}, r2: {}, r3: {}, r3_nbody: {}'.format(
        norms.t1, norms.t2, norms.t3, np.linalg.norm(r3)))
    print('dE: {}'.format(energy - -1.311811e-01))


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

    converged, energy, amps = classic_solver(
        cc, conv_tol_energy=1e-10, conv_tol_res=1e-10,
        max_cycle=300)

    print('delta E: {}'.format(energy - -0.1311305308))


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
    from tcc.rccsdt import RCCSDT, RCCSDT_UNIT, RCCSDT_UNIT_ANTI
    cc1 = RCCSDT(rhf)

    converged, energy, amps = classic_solver(
        cc1, conv_tol_energy=1e-11, lam=3, conv_tol_res=1e-11,
        max_cycle=200)

    print('E_cc: {}'.format(energy))
    print('E_tot: {}'.format(rhf.e_tot + energy))
    print('delta E: {}'.format(energy - -0.0502322580))


if __name__ == '__main__':
    test_cc()
    # test_compare_to_aq()
    # test_compare_to_hirata()
