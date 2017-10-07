import abc
import time
from pyscf.lib import logger
import numpy as np
from scipy.optimize import root, minimize
from tcc.diis import diis_multiple
import tcc.tensors as tensors
from tcc.tensors import Tensors


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
            amps = Tensors(diis.predict())

        res = cc.calc_residuals(ham, amps)
        rhs = cc.update_rhs(ham, amps, res)
        amps = cc.solve_amps(ham, amps, rhs)
        if lam != 1:
            amps = damp_amplitudes(cc, amps, old_amps, lam)
        diis.push_variable(amps)

        res = cc.calc_residuals(ham, amps)
        diis.push_predictor(res)

        norm_res = res.map(np.linalg.norm).to_shallow_dict()
        max_key = max(norm_res.keys(),
                      key=(lambda x: abs(norm_res[x])))
        max_val = norm_res[max_key]

        energy = cc.calculate_energy(ham, amps)

        cput_cycle = log.timer('CC iter', *cput_cycle)
        log.info('istep = %d  E(%s) = %.6e'
                 ' dE = %.6e  max(|r|) = %.6e (%s)',
                 istep, cc.method_name,
                 energy, energy - old_energy,
                 max_val, max_key)

        log.debug('%s', ', '.join('|{}| = {:.6e}'.format(field_name, val)
                                  for field_name, val in norm_res.items()))

        dE = energy - old_energy
        old_amps = amps
        old_energy = energy

        if abs(dE) < conv_tol_energy:
            cc._converged = True
        if abs(max_val) < conv_tol_res:
            cc._converged = True
        if cc._converged:
            log.note('Converged in %d steps. E(%s) = %.6e'
                     ' dE = %.6e  max(|r|) = %.6e (%s)',
                     istep, cc.method_name,
                     energy, dE,
                     max_val, max_key)
            break

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
    div_tol_energy = 1e20

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
        res = cc.calc_residuals(ham, amps)
        rhs = cc.update_rhs(ham, amps, res)
        new_amps = cc.solve_amps(ham, amps, rhs)
        if lam != 1:
            new_amps = damp_amplitudes(cc, new_amps, amps, lam)

        new_energy = cc.calculate_energy(ham, new_amps)

        norm_diff_amps = (
            new_amps - amps).map(np.linalg.norm).to_shallow_dict()
        max_key = max(norm_diff_amps.keys(),
                      key=(lambda x: abs(norm_diff_amps[x])))
        max_val = norm_diff_amps[max_key]

        cput_cycle = log.timer('CC iter', *cput_cycle)
        log.info('istep = %d  E(%s) = %.6e'
                 ' dE = %.6e  max(|T|) = %.6e (%s)',
                 istep, cc.method_name,
                 new_energy, new_energy - energy,
                 max_val, max_key)

        log.debug('%s', ', '.join('|{}| = {:.6e}'.format(field_name, val)
                                  for field_name, val in norm_diff_amps.items()))

        dE = new_energy - energy
        if abs(dE) < conv_tol_energy:
            cc._converged = True
        if abs(max_val) < conv_tol_amps:
            cc._converged = True
        if abs(dE) > div_tol_energy:
            cc._converged = False
            energy = np.nan
            break

        energy = new_energy
        amps = new_amps
        if cc._converged:
            log.note('Converged in %d steps E(%s) = %.6e'
                     ' dE = %.6e  max(|dT|) = %.6e (%s)',
                     istep, cc.method_name,
                     new_energy, dE,
                     max_val,
                     max_key)
            break

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
    if hasattr(cc, 'damp_amplitudes'):
        return cc.damp_amplitudes(amps, amps_old, lam)
    else:
        return amps / lam + amps_old * (lam - 1) / lam


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

    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    cput_cycle = cput_total = (time.clock(), time.time())

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy

    amps_structure = amps.struct()

    def residuals(x):
        amps.update_from_vector(x)
        res = cc.calc_residuals(ham, amps)
        return res.flatten()

    istep = 1

    def log_residuals(x, res):
        nonlocal istep
        nonlocal cput_cycle
        energy = cc.calculate_energy(
            ham, tensors.from_vec(x, amps_structure)
        )
        log.info('istep = %d  E(%s) = %.6e'
                 ' |T| = %.6e |R| = %.6e',
                 istep, cc.method_name,
                 energy,
                 np.linalg.norm(x),
                 np.linalg.norm(res))
        cput_cycle = log.timer('CC iter', *cput_cycle)
        istep += 1

    if verbose >= logger.INFO:
        callback = log_residuals
    else:
        callback = None

    result = root(
        fun=residuals,
        x0=amps.flatten(),
        tol=conv_tol,
        method=method,
        options=options,
        callback=callback
    )

    amps = tensors.from_vec(result.x, amps_structure)
    converged = result.success
    energy = cc.calculate_energy(ham, amps)

    cc._energy_corr = energy
    cc._energy_tot = cc._scf.energy_tot() + energy
    cc._amps = amps
    log.timer('CC done', *cput_total)

    return converged, cc._energy_corr, amps


def lagrangian_solver(cc, amps=None, method='L-BFGS-B',
                      options=None, conv_tol=1e-5,
                      max_memory=None, verbose=logger.INFO):
    """
    Solves CC equations using minimization functions from scipy

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

    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    cput_cycle = cput_total = (time.clock(), time.time())

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy

    amps_structure = amps.struct()

    def gradient(x):
        amps = tensors.from_vec(x, amps_structure)
        res = cc.calc_residuals(ham, amps)
        return res.flatten()

    def constraint(x):
        amps = tensors.from_vec(x, amps_structure)
        return cc.calc_lagrangian(ham, amps)

    def gradient_constraint(x):
        amps = tensors.from_vec(x, amps_structure)
        return cc.calc_lagrangian(ham, amps)

    istep = 1

    def log_residuals(x):
        nonlocal istep
        nonlocal cput_cycle
        energy = cc.calculate_energy(
            ham, tensors.from_vec(x, amps_structure))
        log.info('istep = %d  E(%s) = %.6e'
                 ' |T| = %.6e',
                 istep, cc.method_name,
                 energy,
                 np.linalg.norm(x),)
        cput_cycle = log.timer('CC iter', *cput_cycle)
        istep += 1

    if verbose >= logger.INFO:
        callback = log_residuals
    else:
        callback = None

    result = minimize(
        fun=lagrangian,
        x0=amps.flatten(),
        tol=conv_tol,
        method=method,
        options=options,
        callback=callback
    )

    amps = tensors.from_vec(result.x, amps_structure)
    converged = result.success
    energy = cc.calculate_energy(ham, amps)

    cc._energy_corr = energy
    cc._energy_tot = cc._scf.energy_tot() + energy
    cc._amps = amps
    log.timer('CC done', *cput_total)

    return converged, cc._energy_corr, amps


def step_solver(cc, amps=None, max_cycle=50,
                conv_tol_energy=1e-6, conv_tol_amps=1e-5,
                div_tol_energy=1e20, alpha=1,
                use_optimizer='momentum', optimizer_kwargs={},
                max_memory=None, verbose=logger.INFO):
    """
    Solves CC equations in a gradient-descent fashion.
    At each iteration an update is added to the amplitudes.
    This update is generated by the specified optimizer.

    :param cc: cc object
    :param amps:  amplitudes container
    :param max_cycle: number of cycles
    :param conv_tol_energy: convergence tolerance for energy
    :param conv_tol_amps: convergence tolerance for amplitudes
    :param div_tol_energy: energy divergence threshold
    :param use_optimizer: optimizer to use
    :param optimizer_kwargs: dictionary passed to optimizer
    :param max_memory: maximal amount of memory to use
    :param verbose: verbosity level
    :rtype: converged, energy, amplitudes
    """

    # Copy everything needed from the HF

    if max_memory is None:
        max_memory = cc.max_memory

    log = logger.Logger(cc.stdout, verbose)

    cput_cycle = cput_total = (time.clock(), time.time())

    ham = cc.create_ham()

    if amps is None:
        amps = cc.init_amplitudes(ham)

    energy = cc.calculate_energy(ham, amps)
    cc._emp2 = energy

    import tcc.optimizers as optimizers
    optimizer = optimizers.initialize(use_optimizer, amps, **optimizer_kwargs)

    for istep in range(max_cycle):
        step = cc.calculate_update(ham, amps)
        new_amps = amps + optimizer.update(step)

        new_energy = cc.calculate_energy(ham, new_amps)

        norm_step = step.map(np.linalg.norm).to_shallow_dict()
        max_key = max(norm_step.keys(),
                      key=(lambda x: abs(norm_step[x])))
        max_val = norm_step[max_key]

        cput_cycle = log.timer('CC iter', *cput_cycle)
        log.info('istep = %d  E(%s) = %.6e'
                 ' dE = %.6e  max(|T|) = %.6e (%s)',
                 istep, cc.method_name,
                 new_energy, new_energy - energy,
                 max_val, max_key)

        log.debug('%s', ', '.join('|{}| = {:.6e}'.format(field_name, val)
                                  for field_name, val in norm_step.items()))

        if abs(new_energy - energy) < conv_tol_energy:
            cc._converged = True
        if abs(max_val) < conv_tol_amps:
            cc._converged = True
        if abs(new_energy - energy) > div_tol_energy:
            cc._converged = False
            energy = np.nan
            break

        energy = new_energy
        amps = new_amps
        if cc._converged:
            log.note('Converged in %d steps E(%s) = %.6e'
                     ' dE = %.6e  max(|dT|) = %.6e (%s)',
                     istep, cc.method_name,
                     new_energy, new_energy - energy,
                     max_val,
                     max_key)
            break

    cc._energy_corr = energy
    cc._energy_tot = cc._scf.energy_tot() + energy
    cc._amps = amps
    log.timer('CC done', *cput_total)

    return cc._converged, energy, amps


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

    @property
    def mos(self):
        """
        Returns a constructor for a MOS object
        """
        return self._mos

    @property
    def method_name(self):
        """
        Returns the name of a prticular method
        """
        return self.__class__.__name__

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
    def update_rhs(self, ham, amps, res):
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
    def calc_residuals(self, ham, amps):
        """
        Calculates residuals of CC equations
        """
