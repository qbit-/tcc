import numpy

"""
This module implements evarything related to the MO transformation
matrix
"""


class SPINLESS_MOS(object):
    """
    Convenience class for molecular orbitals and
    their various blocks

    Input:
    mo_coeff    MO coefficients (rectangular numpy matrix)
    mo_energy   HF energies of MO coefficients (1d array)
    mo_occ      MO occupation numbers - (1d array)

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
