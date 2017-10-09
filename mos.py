import numpy

"""
This module implements everything related to the MO transformation
matrix
"""


class SPINLESS_MOS():
    """
    Convenience class for molecular orbitals and
    their various blocks

    :param mo_coeff: MO coefficients (rectangular numpy matrix)
    :param mo_energy: HF energies of MO coefficients (1d array)
    :param mo_occ: MO occupation numbers - (1d array)
    :param frozen: frozen orbitals. If int, freeze # of frozen orbitals, if vector
    of ints, freeze selected orbitals. Default None.

    >>> import numpy as np
    >>> m = SPINLESS_MOS(np.random.rand(4,3), np.ones(3), [2, 2, 0])
    >>> m.nocc
    2
    >>> m.nvir
    1
    >>> m.nmo
    3
    >>> np.max(np.concatenate((m.occ_coeff, m.vir_coeff), axis=1) - m.mo_coeff)
    0.0
    >>> np.max(np.concatenate((m.occ_energies, m.vir_energies))-m.mo_energies)
    0.0
    >>> m = SPINLESS_MOS(np.random.rand(4,4), np.ones(4), [1.0,1.0,0,0], [0,3])
    >>> m.nmo
    2
    >>> np.max(m.mo_coeff_frozen - m._mo_coeff_full[:,m.frozen_index])
    0.0
    >>> np.sum(m.mo_occ_full)-np.sum(m.mo_occ)-np.sum(m.mo_occ_frozen)
    0.0
    """

    def __init__(self, mo_coeff, mo_energy, mo_occ, frozen=None):
        self._mo_coeff_full = numpy.asarray(mo_coeff)
        self._mo_energy_full = numpy.asarray(mo_energy)
        self._mo_occ_full = numpy.asarray(mo_occ)

        # Set occupied_index
        self._occupied_index = numpy.logical_not(
            numpy.ones(mo_energy.size, dtype=numpy.bool)
        )

        nonzero_occupieds = numpy.nonzero(mo_occ)
        self._occupied_index[nonzero_occupieds] = True

        # Set frozen index
        self._active_index = numpy.ones(mo_energy.size, dtype=numpy.bool)
        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                self._active_index[:frozen] = False
            elif len(frozen) > 0:
                self._active_index[numpy.asarray(frozen)] = False

    @property
    def nmo(self):
        """
        Returns the number of active MOs
        """
        return len(
            numpy.ones(self._mo_coeff_full.shape[1])[self._active_index]
        )

    @property
    def nocc(self):
        """
        Returns the number of active occupieds
        """
        return len(
            numpy.ones(
                self._mo_coeff_full.shape[1]
            )[self._active_index & self._occupied_index])

    @property
    def nvir(self):
        """
        Returns the number of active virtuals
        """
        return len(
            numpy.ones(
                self._mo_coeff_full.shape[1]
            )[self._active_index &
              numpy.logical_not(self._occupied_index)])

    @property
    def nocc_frozen(self):
        """
        Returns the number of frozen occupieds
        """
        return len(
            numpy.ones(
                self._mo_coeff_full.shape[1]
            )[numpy.logical_not(self._active_index)
              & self._occupied_index])

    @property
    def nvir_frozen(self):
        """
        Returns the number of frozen virtuals
        """
        return len(
            numpy.ones(
                self._mo_coeff_full.shape[1]
            )[numpy.logical_not(self._active_index)
              & numpy.logical_not(self._occupied_index)])

    @property
    def mo_coeff(self):
        """
        Returns coefficients for active orbitals
        """
        return self._mo_coeff_full[:, self._active_index]

    @property
    def occ_coeff(self):
        """
        Returns coefficients for active occupieds
        """
        return self._mo_coeff_full[:, self._active_index &
                                   self._occupied_index]

    @property
    def vir_coeff(self):
        """
        Returns coefficients for active virtuals
        """
        return self._mo_coeff_full[:, self._active_index &
                                   numpy.logical_not(self._occupied_index)]

    @property
    def mo_coeff_frozen(self):
        """
        Returns coefficients for frozen orbitals
        """
        return self._mo_coeff_full[:, numpy.logical_not(self._active_index)]

    @property
    def frozen_index(self):
        """
        Returns indices of frozen orbitals
        """
        return numpy.logical_not(self._active_index)

    @property
    def occupied_index(self):
        """
        Returns indices of occupied orbitals
        """
        return self._occupied_index

    @property
    def mo_energies(self):
        """
        Returns active mo energies
        """
        return self._mo_energy_full[self._active_index]

    @property
    def mo_energies_frozen(self):
        """
        Returns frozen core mo energies
        """
        return self._mo_energy_full[numpy.logical_not(self._active_index)]

    @property
    def occ_energies(self):
        """
        Returns energies of active occupied orbitals
        """
        return self._mo_energy_full[self._active_index & self._occupied_index]

    @property
    def vir_energies(self):
        """
        Returns energies of active virtual orbitals
        """
        return self._mo_energy_full[self._active_index &
                                    numpy.logical_not(self._occupied_index)]

    @property
    def mo_occ(self):
        """
        Returns occupation numbers for active orbitals
        """
        return self._mo_occ_full[self._active_index]

    @property
    def mo_occ_frozen(self):
        """
        Returns occupation numbers for frozen orbitals
        """
        return self._mo_occ_full[numpy.logical_not(self._active_index)]

    @property
    def mo_occ_full(self):
        """
        Returns occupation numbers for all orbitals
        """
        return self._mo_occ_full

class UHF_MOS():
    """
    This class implements molecular orbitals with explicit spin labels (UHF case)
    This is just a wrapper for the SPINLESS_MOS class

    :param mo_coeff: MO coefficients (ndarray of shape (2, N_basis, N_mos))
    :param mo_energy: HF energies of MO coefficients (ndarray of shape (2, N_mos))
    :param mo_occ: MO occupation numbers (ndarray of shape (2, N_mos))
    :param frozen: frozen orbitals. If int, freeze # of frozen orbitals, if vector
    of ints, freeze selected orbitals, if ndarray of shape (2, *)
    freese selected orbitals for alpha and beta. Default None.

    >>> import numpy as np
    >>> m = UHF_MOS(np.random.rand(2,4,3), np.ones((2, 3)), np.vstack(([2, 2, 0], [2, 2, 0])))
    >>> m.a.nocc
    2
    >>> m.b.nvir
    1
    >>> m.a.nmo
    3
    """

    def __init__(self, mo_coeff, mo_energy, mo_occ, frozen=None):
        mo_coeff_full = numpy.asarray(mo_coeff)
        mo_energy_full = numpy.asarray(mo_energy)
        mo_occ_full = numpy.asarray(mo_occ)

        # Check arguments
        if len(mo_coeff_full.shape) != 3:
            raise ValueError('Wrong shape of MO coefficients: {}'.format(
                mo_coeff_full.shape))
        for arr in [mo_energy_full, mo_occ_full]:
            if len(arr) != 2:
                raise ValueError('Wrong shape of MO energies/occupations: {}'.format(
                    arr.shape))

        if frozen is not None:
            if isinstance(frozen, (int, numpy.integer)):
                freeze_a = np.asarray(range(frozen))
                freeze_b = np.asarray(range(frozen))
            else:
                ff = numpy.asarray(frozen)
                if len(ff.shape) == 1:
                    freeze_a = ff
                    freeze_b = ff
                elif len(ff.shape) == 2:
                    freeze_a = ff[0, :]
                    freeze_b = ff[1, :]
                else:
                    raise ValueError(
                        'Wrong frozen vector shape: {}'.format(ff.shape))
        else:
            freeze_a = None
            freeze_b = None

        # Set occupied_index
        self.a = SPINLESS_MOS(mo_coeff_full[0, :],
                              mo_energy_full[0, :],
                              mo_occ_full[0, :],
                              freeze_a)
        self.b = SPINLESS_MOS(mo_coeff_full[1, :],
                              mo_energy_full[1, :],
                              mo_occ_full[1, :],
                              freeze_b)

