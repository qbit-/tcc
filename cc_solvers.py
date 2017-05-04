import abc
import time
from pyscf import lib
from pyscf.lib import logger
import numpy as np
from scipy.optimize import minimize, root
from tcc.diis import diis_multiple


def residual_diis_solver(cc, amps=None, max_cycle=50,
                         conv_tol_energy=1e-6, conv_tol_res=1e-5,
                         lam=1, ndiis=5,
                         diis_energy_tol=1e-4, max_memory=None,
                         verbose=logger.INFO):
    """
    Carry on a CC calculation
    :param cc: cc object
    :param amps:  amplitudes container
    :param max_cycle: number of cycles
    :param conv_tol_energy: convergence tolerance for energy
    :param conv_tol_res: convergence tolerance for amplitudes
    :param lam: damping factor
    :param ndiis: number of diis vectors
    :param diis_energy_tol: energy tolerance to start diis iterations
    :param max_memory: maximal amount of memory to use
    :param verbose: verbosity level
    :rtype: converged, energy, amplitudes
    """
    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    cput_cycle = cput_total = (time.clock(), time.time())

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    diis = diis_multiple(len(amps), ndiis)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy
    old_amps = amps
    old_energy = energy

    for istep in range(max_cycle):

        if diis.ready:
            amps = cc.types.AMPLITUDES_TYPE(*diis.predict())

        rhs = cc.update_rhs(ham, amps)
        amps = cc.solve_amps(ham, amps, rhs)
        if lam != 1:
            amps = damp_amplitudes(cc, amps, old_amps, lam)
        diis.push_variable(amps)

        rhs = cc.update_rhs(ham, amps)
        res = cc.calc_residuals(ham, amps, rhs)
        diis.push_predictor(res)

        norm_res = np.array([
            np.linalg.norm(res[ii]) for ii in range(len(res))
        ])

        energy = cc.calculate_energy(ham, amps)

        cput_cycle = log.timer('CC iter', *cput_cycle)
        log.info('istep = %d  E(%s) = %.6e'
                 ' dE = %.6e  max(|r|) = %.6e',
                 istep, cc.method_name,
                 energy, energy - old_energy, np.max(np.abs(norm_res)))

        log.debug('%s', ', '.join('|{}| = {:.6e}'.format(field_name, val)
                                  for field_name, val
                                  in zip(res._fields, norm_res)))

        if abs(energy - old_energy) < conv_tol_energy:
            cc._converged = True
            break
        if np.max(np.abs(norm_res)) < conv_tol_res:
            cc._converged = True
            break
        old_amps = amps
        old_energy = energy

    cc._energy_corr = energy
    cc._energy_tot = cc._scf.energy_tot() + energy
    cc._amps = amps
    log.timer('CC done', *cput_total)

    return cc._converged, energy, amps


def classic_solver(cc, amps=None, max_cycle=50,
                   conv_tol_energy=1e-6, conv_tol_amps=1e-5, lam=1,
                   max_memory=None, verbose=logger.INFO):
    """
    Carry on a CC calculation

    :param cc: cc object
    :param amps:  amplitudes container
    :param max_cycle: number of cycles
    :param conv_tol_energy: convergence tolerance for energy
    :param conv_tol_amps: convergence tolerance for amplitudes
    :param lam: damping factor
    :param max_memory: maximal amount of memory to use
    :param verbose: verbosity level
    :rtype: converged, energy, amplitudes
    """
    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    cput_cycle = cput_total = (time.clock(), time.time())

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy

    for istep in range(max_cycle):
        rhs = cc.update_rhs(ham, amps)
        new_amps = cc.solve_amps(ham, amps, rhs)
        if lam != 1:
            new_amps = damp_amplitudes(cc, new_amps, amps, lam)

        new_energy = cc.calculate_energy(ham, new_amps)

        norm_diff_amps = [np.linalg.norm(new_amps[ii] - amps[ii])
                          for ii in range(len(amps))]

        cput_cycle = log.timer('CC iter', *cput_cycle)
        log.info('istep = %d  E(%s) = %.6e'
                 ' dE = %.6e  max(|T|) = %.6e',
                 istep, cc.method_name,
                 new_energy, new_energy - energy, np.max(
                     np.abs(norm_diff_amps)))

        log.debug('%s', ', '.join('|{}| = {:.6e}'.format(field_name, val)
                                  for field_name, val in zip(
            amps._fields, norm_diff_amps)))

        if abs(new_energy - energy) < conv_tol_energy:
            cc._converged = True
            break
        if np.max(np.abs(norm_diff_amps)) < conv_tol_amps:
            cc._converged = True
            break

        energy = new_energy
        amps = new_amps

    cc._energy_corr = energy
    cc._energy_tot = cc._scf.energy_tot() + energy
    cc._amps = amps
    log.timer('CC done', *cput_total)

    return cc._converged, energy, amps


def damp_amplitudes(cc, amps, amps_old, lam):
    """
    Calculates amplitudes damped with factor lam, as

    T = T_new / lam + (lam - 1) / lam * T_old
    :param cc: cc object
    :param amps:  amplitudes container
    :param amps_old:  amplitudes container
    :rtype: amplitudes container
    """
    return cc.types.AMPLITUDES_TYPE(*(
        amps[ii] / lam + (lam - 1) / lam * amps_old[ii]
        for ii in range(len(amps))
    ))


def root_solver(cc, amps=None, method='krylov', options=None, conv_tol=1e-5,
                max_memory=None, verbose=logger.INFO):
    """
    Solves CC equations using root finding functions from scipy

    :param cc: cc object
    :param amps:  amplitudes container
    :param method: str, optional. method of scipy.optimize.root to call.
    :param options: dict, optional. Extra options to give to
    scipy.optimize.root
    :param conv_tol: convergence tolerance for root method.
    For detailed control read help of scipy.optimize.root
    :param max_memory: maximal amount of memory to use
    :param verbose: verbosity level
    :rtype: converged, energy, amplitudes
    """
    from tcc.utils import (merge_np_container,
                           np_container_structure,
                           unmerge_np_container)

    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy

    amps_structure = np_container_structure(amps)

    def residuals(x):
        amps = unmerge_np_container(cc.types.AMPLITUDES_TYPE,
                                    amps_structure, x)
        rhs = cc.update_rhs(ham, amps)
        res = cc.calc_residuals(ham, amps, rhs)
        return merge_np_container(res)

    istep = 1

    def log_residuals(x, res):
        nonlocal istep
        energy = cc.calculate_energy(ham,
                                     unmerge_np_container(
                                         cc.types.AMPLITUDES_TYPE,
                                         amps_structure,
                                         x))
        log.info('istep = %d  E(%s) = %.6e'
                 ' |T| = %.6e |R| = %.6e',
                 istep, cc.method_name,
                 energy,
                 np.linalg.norm(x),
                 np.linalg.norm(res))
        istep += 1

    if verbose >= logger.INFO:
        callback = log_residuals
    else:
        callback = None

    result = root(
        fun=residuals,
        x0=merge_np_container(amps),
        tol=conv_tol,
        method='krylov',
        options=options,
        callback=callback
    )

    amps = unmerge_np_container(cc.types.AMPLITUDES_TYPE,
                                amps_structure, result.x)
    cc._converged = result.success
    cc._energy_corr = cc.calculate_energy(ham, amps)
    cc._energy_tot = cc._scf.energy_tot() + cc._energy_corr

    return cc._converged, cc._energy_corr, amps


def concreter(abclass):
    """
    >>> import abc
    >>> class Abstract(metaclass=abc.ABCMeta):
    ...     @abc.abstractmethod
    ...     def bar(self):
    ...        return None

    >>> c = concreter(Abstract)
    >>> c.__name__
    'dummy_concrete_Abstract'
    >>> c().bar() # doctest: +ELLIPSIS
    0.0
    >>> concreter(tuple)
    <class 'tuple'>
    """
    if "__abstractmethods__" not in abclass.__dict__:
        return abclass
    new_dict = abclass.__dict__.copy()
    for abstractmethod in abclass.__abstractmethods__:
        # replace each abc method or property with an identity function:
        new_dict[abstractmethod] = lambda x, *args, **kw: 0.0
    # creates a new class, with the overriden ABCs:
    return type("dummy_concrete_%s" % abclass.__name__, (abclass,), new_dict)


class CC(abc.ABC):
    """
    This is an abstract class implementing a CC calculation.
    Normally, subclasses need to override properties
    :py:attr:`ham`  and methods :py:attr:`init_amps`,
    :py:attr:`update_rhs`, :py:attr:`solve_amps`, :py:attr:`energy`
    """

    def __init__(self, mf):
        """
        Initialize CC by copying values from the HF calculation

        Parameters
        ----------

        mf
           SCF object containing previous SCF calculation
        frozen
           list of frozen orbitals
        mo_energy
           list of orbital energies (Fock eigenvalues)
        mo_coeff
           MO coefficients (Fock eigenvectors)
        """

        # The following should not be modified
        self._mol = mf.mol
        self._scf = mf
        self._converged = False
        self._energy_corr = None
        self._energy_tot = None
        self._amps = None
        self._zeta = None

        # Those are parameters to modify
        self.verbose = self._mol.verbose
        self.max_memory = mf.max_memory
        self.stdout = self._mol.stdout

    @property
    def energy_corr(self):
        return self._energy_corr

    @property
    def energy_tot(self):
        return self._energy_tot

    @abc.abstractproperty
    def mos(self):
        """
        Returns a constructor for a MOS object
        """

    @abc.abstractproperty
    def types(self):
        """
        Contains type definitions for parameters of the method
        """

    @abc.abstractproperty
    def method_name(self):
        """
        Returns the name of a prticular method
        """

    @abc.abstractclassmethod
    def create_ham(self):
        """
        One and two electron integrals in the MO basis along
        with the transformation matrix
        """

    @abc.abstractclassmethod
    def init_amplitudes(self, ham):
        """
        Initialize amplitudes
        """

    @abc.abstractclassmethod
    def update_rhs(self, ham, amps):
        """
        Iteration of Coupled Cluster. Update right hand side of the
        equations
        """

    @abc.abstractclassmethod
    def solve_amps(self, ham, amps, rhs):
        """
        Calculate new amplitudes
        """

    @abc.abstractclassmethod
    def calculate_energy(self, ham, amps):
        """
        Calculate Coupled Cluster energy
        """

    @abc.abstractclassmethod
    def calc_residuals(self, ham, amps, rhs):
        """
        Calculates residuals of CC equations
        """


class CC_lagrangian(abc.ABC):
    """
    This is an abstract class implementing a CC calculation
    by lagrangian minimization. 
    Normally, subclasses need to override properties
    :py:attr:`ham`  and methods :py:attr:`init_amps`,
    :py:attr:`lagrangian`, :py:attr:`gradient`, :py:attr:`calculate_energy`
    """

    def __init__(self, mf):
        """
        Initialize CC by copying values from the HF calculation

        Parameters
        ----------

        mf
           SCF object containing previous SCF calculation
        frozen
           list of frozen orbitals
        mo_energy
           list of orbital energies (Fock eigenvalues)
        mo_coeff
           MO coefficients (Fock eigenvectors)
        """

        # The following should not be modified
        self._mol = mf.mol
        self._scf = mf
        self._converged = False
        self._energy_corr = None
        self._energy_tot = None
        self._amps = None
        self._zeta = None

        # Those are parameters to modify
        self.verbose = self._mol.verbose
        self.max_memory = mf.max_memory
        self.stdout = self._mol.stdout

    @property
    def energy_corr(self):
        return self._energy_corr

    @property
    def energy_tot(self):
        return self._energy_tot

    @abc.abstractproperty
    def mos(self):
        """
        Returns a constructor for a MOS object
        """

    @abc.abstractproperty
    def types(self):
        """
        Contains type definitions for parameters of the method
        """

    @abc.abstractproperty
    def method_name(self):
        """
        Returns the name of a prticular method
        """

    @abc.abstractclassmethod
    def create_ham(self):
        """
        One and two electron integrals in the MO basis along
        with the transformation matrix
        """

    @abc.abstractclassmethod
    def init_amplitudes(self, ham):
        """
        Initialize amplitudes
        """

    @abc.abstractclassmethod
    def calculate_lagrangian(self, ham, amps):
        """
        Calculates CC lagrangian value with current amplitudes
        """

    @abc.abstractclassmethod
    def lagrangian_gradient(self, ham, amps):
        """
        Calculate gradient of CC lagrangian
        """

    @abc.abstractclassmethod
    def calculate_energy(self, ham, amps):
        """
        Calculate Coupled Cluster energy
        """
