import numpy as np
from numpy import einsum
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace


class RCCSD_EXT(CC):
    """
    This class implements classic RCCSD method
    with vvoo ordered amplitudes and also solves for deexcitations
    """
    #  These are containers used by all  methods of this class
    # to pass numpy arrays

    types = SimpleNamespace()

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

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple('RCCSD_AMPLITUDES',
                                                field_names=('t1', 'z1',
                                                             't2', 'z2'))
        self.types.RHS_TYPE = namedtuple('RCCSD_RHS',
                                         field_names=('gt1', 'gz1',
                                                      'gt2', 'gz2'))
        self.types.RESIDUALS_TYPE = namedtuple('RCCSD_RESIDUALS',
                                               field_names=('rt1', 'rz1',
                                                            'rt2', 'rz2'))

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD_EXT'

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
        z1 = t1.conj()
        z2 = t2.conj()
        return self.types.AMPLITUDES_TYPE(t1, z1, t2, z2)

    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """

        tau0 = (
            - einsum("jiab->ijab", h.v.oovv)
            + 2 * einsum("jiba->ijab", h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiba->ia", a.t1, tau0)
            + 2 * einsum("ia->ia", h.f.ov)
        )

        energy = (
            einsum("ai,ia->", a.t1, tau1)
            + einsum("abij,jiba->", a.t2, tau0)
        )

        return energy

    def update_rhs(self, h, a, r):
        return self.types.RHS_TYPE(
            gt1=r.rt1 - 2 * a.t1 / cc_denom(h.f, 2, 'dir', 'full'),
            gt2=r.rt2 - (2 * a.t2 - a.t2.transpose([0, 1, 3, 2])
                         ) / cc_denom(h.f, 4, 'dir', 'full'),
            gz1=r.rz1 - 2 * a.z1 / cc_denom(h.f, 2, 'dir', 'full'),
            gz2=r.rz2 - (2 * a.z2 - a.t2.transpose([0, 1, 3, 2])
                         ) / cc_denom(h.f, 4, 'dir', 'full')
        )

    def calc_residuals(self, h, a):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        tau0 = (
            einsum("bj,jiba->ia", a.t1, h.v.oovv)
        )

        tau1 = (
            einsum("bj,jiab->ia", a.t1, h.v.oovv)
        )

        tau2 = (
            einsum("caik,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau3 = (
            einsum("abki,kjab->ij", a.t2, h.v.oovv)
        )

        tau4 = (
            einsum("ci,iacb->ab", a.t1, h.v.ovvv)
        )

        tau5 = (
            einsum("caik,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau6 = (
            einsum("abik,kjba->ij", a.t2, h.v.oovv)
        )

        tau7 = (
            einsum("ak,ikja->ij", a.t1, h.v.ooov)
        )

        tau8 = (
            einsum("caij,jicb->ab", a.t2, h.v.oovv)
        )

        tau9 = (
            einsum("acki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau10 = (
            einsum("acij,jicb->ab", a.t2, h.v.oovv)
        )

        tau11 = (
            einsum("ai,ja->ij", a.t1, tau1)
        )

        tau12 = (
            einsum("acik,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau13 = (
            einsum("ai,ja->ij", a.t1, tau0)
        )

        tau14 = (
            einsum("abki,kjba->ij", a.t2, h.v.oovv)
        )

        tau15 = (
            einsum("ia,aj->ij", h.f.ov, a.t1)
        )

        tau16 = (
            einsum("acki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau17 = (
            einsum("abik,kjab->ij", a.t2, h.v.oovv)
        )

        tau18 = (
            einsum("caki,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau19 = (
            einsum("ci,iabc->ab", a.t1, h.v.ovvv)
        )

        tau20 = (
            einsum("acij,jibc->ab", a.t2, h.v.oovv)
        )

        tau21 = (
            einsum("caki,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau22 = (
            einsum("caij,jibc->ab", a.t2, h.v.oovv)
        )

        tau23 = (
            einsum("ak,kija->ij", a.t1, h.v.ooov)
        )

        tau24 = (
            einsum("acik,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau25 = (
            einsum("caik,jkbc->ijab", a.t2, tau2)
        )

        tau26 = (
            einsum("acki,jkbc->ijab", a.t2, tau9)
        )

        tau27 = (
            einsum("ijab->ijab", tau25)
            + einsum("ijab->ijab", tau26)
        )

        tau28 = (
            einsum("al,likj->ijka", a.t1, h.v.oooo)
        )

        tau29 = (
            einsum("ak,kijb->ijab", a.t1, tau28)
        )

        tau30 = (
            einsum("di,badc->iabc", a.t1, h.v.vvvv)
        )

        tau31 = (
            einsum("ci,jabc->ijab", a.t1, tau30)
        )

        tau32 = (
            einsum("caki,jkbc->ijab", a.t2, tau18)
        )

        tau33 = (
            einsum("acik,jkbc->ijab", a.t2, tau24)
        )

        tau34 = (
            einsum("acik,jkbc->ijab", a.t2, tau12)
        )

        tau35 = (
            einsum("caik,jkbc->ijab", a.t2, tau5)
        )

        tau36 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau37 = (
            einsum("ai,jkla->ijkl", a.t1, tau36)
        )

        tau38 = (
            einsum("al,ijkl->ijka", a.t1, tau37)
        )

        tau39 = (
            einsum("ak,ijkb->ijab", a.t1, tau38)
        )

        tau40 = (
            einsum("caki,jkbc->ijab", a.t2, tau21)
        )

        tau41 = (
            einsum("acki,jkbc->ijab", a.t2, tau16)
        )

        tau42 = (
            4 * einsum("ijab->ijab", tau29)
            + 4 * einsum("ijab->ijab", tau31)
            - 2 * einsum("ijab->ijab", tau32)
            - 2 * einsum("ijab->ijab", tau33)
            + 4 * einsum("ijab->ijab", tau34)
            + einsum("ijab->ijab", tau35)
            + 4 * einsum("ijab->ijab", tau39)
            + 4 * einsum("baji->ijab", h.v.vvoo)
            + 4 * einsum("ijab->ijab", tau40)
            + einsum("ijab->ijab", tau41)
        )

        tau43 = (
            einsum("acki,jkbc->ijab", a.t2, tau18)
        )

        tau44 = (
            einsum("abli,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau45 = (
            einsum("ak,ikjb->ijab", a.t1, tau44)
        )

        tau46 = (
            einsum("ki,abkj->ijab", tau15, a.t2)
        )

        tau47 = (
            einsum("ik,abkj->ijab", tau17, a.t2)
        )

        tau48 = (
            einsum("acik,kbcj->ijab", a.t2, h.v.ovvo)
        )

        tau49 = (
            einsum("bc,acij->ijab", tau4, a.t2)
        )

        tau50 = (
            einsum("cdij,badc->ijab", a.t2, h.v.vvvv)
        )

        tau51 = (
            einsum("abij,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau52 = (
            einsum("abkl,ijkl->ijab", a.t2, tau51)
        )

        tau53 = (
            einsum("al,ijkl->ijka", a.t1, tau51)
        )

        tau54 = (
            einsum("ak,ijkb->ijab", a.t1, tau53)
        )

        tau55 = (
            einsum("ik,abkj->ijab", tau6, a.t2)
        )

        tau56 = (
            einsum("ak,kbij->ijab", a.t1, h.v.ovoo)
        )

        tau57 = (
            einsum("kb,baij->ijka", tau1, a.t2)
        )

        tau58 = (
            einsum("ak,ijkb->ijab", a.t1, tau57)
        )

        tau59 = (
            einsum("bcij,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau60 = (
            einsum("ak,ijkb->ijab", a.t1, tau59)
        )

        tau61 = (
            einsum("cbkj,ikac->ijab", a.t2, tau12)
        )

        tau62 = (
            einsum("acik,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau63 = (
            einsum("bi,jkab->ijka", a.t1, tau21)
        )

        tau64 = (
            einsum("ak,ijkb->ijab", a.t1, tau63)
        )

        tau65 = (
            einsum("ib,bajk->ijka", h.f.ov, a.t2)
        )

        tau66 = (
            einsum("ak,kijb->ijab", a.t1, tau65)
        )

        tau67 = (
            einsum("bi,jkab->ijka", a.t1, tau18)
        )

        tau68 = (
            einsum("ak,ijkb->ijab", a.t1, tau67)
        )

        tau69 = (
            einsum("daij,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau70 = (
            einsum("ci,jabc->ijab", a.t1, tau69)
        )

        tau71 = (
            einsum("ci,jabc->ijab", a.t1, h.v.ovvv)
        )

        tau72 = (
            einsum("bi,jkab->ijka", a.t1, tau71)
        )

        tau73 = (
            einsum("ak,ijkb->ijab", a.t1, tau72)
        )

        tau74 = (
            einsum("cbkj,ikac->ijab", a.t2, tau2)
        )

        tau75 = (
            einsum("ki,abkj->ijab", h.f.oo, a.t2)
        )

        tau76 = (
            einsum("acki,jkbc->ijab", a.t2, tau21)
        )

        tau77 = (
            einsum("acik,jkbc->ijab", a.t2, tau5)
        )

        tau78 = (
            einsum("adji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau79 = (
            einsum("ci,jabc->ijab", a.t1, tau78)
        )

        tau80 = (
            einsum("bckj,ikac->ijab", a.t2, tau24)
        )

        tau81 = (
            einsum("kj,abik->ijab", tau23, a.t2)
        )

        tau82 = (
            einsum("bc,acij->ijab", tau8, a.t2)
        )

        tau83 = (
            einsum("ik,abkj->ijab", tau11, a.t2)
        )

        tau84 = (
            einsum("jk,abik->ijab", tau3, a.t2)
        )

        tau85 = (
            einsum("abij,jidc->abcd", a.t2, h.v.oovv)
        )

        tau86 = (
            einsum("di,abdc->iabc", a.t1, tau85)
        )

        tau87 = (
            einsum("cj,iabc->ijab", a.t1, tau86)
        )

        tau88 = (
            einsum("jk,abik->ijab", tau17, a.t2)
        )

        tau89 = (
            einsum("caik,kbcj->ijab", a.t2, h.v.ovvo)
        )

        tau90 = (
            einsum("ackj,ikbc->ijab", a.t2, tau2)
        )

        tau91 = (
            einsum("ac,cbij->ijab", tau20, a.t2)
        )

        tau92 = (
            einsum("abjk,kjic->iabc", a.t2, h.v.ooov)
        )

        tau93 = (
            einsum("ci,jabc->ijab", a.t1, tau92)
        )

        tau94 = (
            einsum("acik,jkbc->ijab", a.t2, tau2)
        )

        tau95 = (
            einsum("bc,acij->ijab", tau22, a.t2)
        )

        tau96 = (
            einsum("bc,acij->ijab", tau20, a.t2)
        )

        tau97 = (
            einsum("abkl,lkji->ijab", a.t2, h.v.oooo)
        )

        tau98 = (
            einsum("caki,kbcj->ijab", a.t2, h.v.ovvo)
        )

        tau99 = (
            einsum("bi,jkab->ijka", a.t1, tau16)
        )

        tau100 = (
            einsum("ak,ijkb->ijab", a.t1, tau99)
        )

        tau101 = (
            einsum("acki,kbcj->ijab", a.t2, h.v.ovvo)
        )

        tau102 = (
            einsum("bckj,ikac->ijab", a.t2, tau12)
        )

        tau103 = (
            einsum("caki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau104 = (
            einsum("kj,abik->ijab", tau7, a.t2)
        )

        tau105 = (
            einsum("jk,abik->ijab", tau14, a.t2)
        )

        tau106 = (
            einsum("cbkj,ikac->ijab", a.t2, tau24)
        )

        tau107 = (
            einsum("bi,jkab->ijka", a.t1, tau12)
        )

        tau108 = (
            einsum("ak,ijkb->ijab", a.t1, tau107)
        )

        tau109 = (
            einsum("ik,abkj->ijab", tau13, a.t2)
        )

        tau110 = (
            einsum("ac,cbij->ijab", h.f.vv, a.t2)
        )

        tau111 = (
            einsum("bi,jkab->ijka", a.t1, tau24)
        )

        tau112 = (
            einsum("ak,ijkb->ijab", a.t1, tau111)
        )

        tau113 = (
            einsum("bi,jabk->ijka", a.t1, h.v.ovvo)
        )

        tau114 = (
            einsum("ak,ikjb->ijab", a.t1, tau113)
        )

        tau115 = (
            einsum("bail,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau116 = (
            einsum("ak,ikjb->ijab", a.t1, tau115)
        )

        tau117 = (
            einsum("kb,baij->ijka", tau0, a.t2)
        )

        tau118 = (
            einsum("ak,ijkb->ijab", a.t1, tau117)
        )

        tau119 = (
            einsum("bc,acij->ijab", tau19, a.t2)
        )

        tau120 = (
            einsum("jk,abik->ijab", tau6, a.t2)
        )

        tau121 = (
            einsum("cbkj,ikac->ijab", a.t2, tau5)
        )

        tau122 = (
            einsum("ac,cbij->ijab", tau10, a.t2)
        )

        tau123 = (
            einsum("bc,acij->ijab", tau10, a.t2)
        )

        tau124 = (
            einsum("bi,jkab->ijka", a.t1, tau5)
        )

        tau125 = (
            einsum("ak,ijkb->ijab", a.t1, tau124)
        )

        tau126 = (
            einsum("ijab->ijab", tau43)
            + 2 * einsum("ijab->ijab", tau45)
            - 2 * einsum("ijab->ijab", tau46)
            + einsum("ijab->ijab", tau47)
            + 4 * einsum("ijab->ijab", tau48)
            + 4 * einsum("ijab->ijab", tau49)
            + 2 * einsum("ijab->ijab", tau50)
            + einsum("ijab->ijab", tau52)
            + 2 * einsum("ijab->ijab", tau54)
            - 2 * einsum("ijab->ijab", tau55)
            - 4 * einsum("ijab->ijab", tau56)
            + 2 * einsum("ijab->ijab", tau58)
            - 2 * einsum("ijab->ijab", tau60)
            + 4 * einsum("ijab->ijab", tau61)
            - 2 * einsum("ijab->ijab", tau62)
            - 4 * einsum("ijab->ijab", tau64)
            - 2 * einsum("ijab->ijab", tau66)
            + 2 * einsum("ijab->ijab", tau68)
            - 2 * einsum("ijab->ijab", tau70)
            - 4 * einsum("ijab->ijab", tau73)
            + einsum("ijab->ijab", tau74)
            - 2 * einsum("ijab->ijab", tau75)
            - 2 * einsum("ijab->ijab", tau76)
            - 2 * einsum("ijab->ijab", tau77)
            - 2 * einsum("ijab->ijab", tau79)
            + einsum("ijab->ijab", tau80)
            + 2 * einsum("ijab->ijab", tau81)
            + einsum("ijab->ijab", tau82)
            + 2 * einsum("ijab->ijab", tau83)
            - 2 * einsum("ijab->ijab", tau84)
            + 2 * einsum("ijab->ijab", tau87)
            + einsum("ijab->ijab", tau88)
            - 2 * einsum("ijab->ijab", tau89)
            + einsum("ijab->ijab", tau90)
            + einsum("ijab->ijab", tau91)
            + 2 * einsum("ijab->ijab", tau93)
            + einsum("ijab->ijab", tau94)
            - 2 * einsum("ijab->ijab", tau95)
            + einsum("ijab->ijab", tau96)
            + 2 * einsum("ijab->ijab", tau97)
            + 4 * einsum("ijab->ijab", tau98)
            + 2 * einsum("ijab->ijab", tau100)
            - 2 * einsum("ijab->ijab", tau101)
            - 2 * einsum("ijab->ijab", tau102)
            - 2 * einsum("ijab->ijab", tau103)
            - 4 * einsum("ijab->ijab", tau104)
            + einsum("ijab->ijab", tau105)
            - 2 * einsum("ijab->ijab", tau106)
            - 4 * einsum("ijab->ijab", tau108)
            - 4 * einsum("ijab->ijab", tau109)
            + 2 * einsum("ijab->ijab", tau110)
            + 2 * einsum("ijab->ijab", tau112)
            - 4 * einsum("ijab->ijab", tau114)
            + 2 * einsum("ijab->ijab", tau116)
            - 4 * einsum("ijab->ijab", tau118)
            - 2 * einsum("ijab->ijab", tau119)
            - 2 * einsum("ijab->ijab", tau120)
            - 2 * einsum("ijab->ijab", tau121)
            - 2 * einsum("ijab->ijab", tau122)
            - 2 * einsum("ijab->ijab", tau123)
            + 2 * einsum("ijab->ijab", tau125)
        )

        tau127 = (
            einsum("abjk,jkic->iabc", a.t2, h.v.ooov)
        )

        tau128 = (
            einsum("ci,jabc->ijab", a.t1, tau127)
        )

        tau129 = (
            einsum("jk,abki->ijab", tau3, a.t2)
        )

        tau130 = (
            einsum("abil,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau131 = (
            einsum("ak,ikjb->ijab", a.t1, tau130)
        )

        tau132 = (
            einsum("kb,abij->ijka", tau1, a.t2)
        )

        tau133 = (
            einsum("ak,ijkb->ijab", a.t1, tau132)
        )

        tau134 = (
            einsum("ib,abjk->ijka", h.f.ov, a.t2)
        )

        tau135 = (
            einsum("ak,kijb->ijab", a.t1, tau134)
        )

        tau136 = (
            einsum("bc,caij->ijab", tau19, a.t2)
        )

        tau137 = (
            einsum("bail,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau138 = (
            einsum("ak,ikjb->ijab", a.t1, tau137)
        )

        tau139 = (
            einsum("caik,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau140 = (
            einsum("kj,abki->ijab", tau23, a.t2)
        )

        tau141 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau142 = (
            einsum("al,ijlk->ijka", a.t1, tau141)
        )

        tau143 = (
            einsum("ak,ikjb->ijab", a.t1, tau142)
        )

        tau144 = (
            einsum("bcij,kacb->ijka", a.t2, h.v.ovvv)
        )

        tau145 = (
            einsum("ak,ijkb->ijab", a.t1, tau144)
        )

        tau146 = (
            einsum("ki,abjk->ijab", tau15, a.t2)
        )

        tau147 = (
            einsum("bc,caij->ijab", tau22, a.t2)
        )

        tau148 = (
            einsum("kj,abki->ijab", tau7, a.t2)
        )

        tau149 = (
            einsum("bi,jkab->ijka", a.t1, tau9)
        )

        tau150 = (
            einsum("ak,ijkb->ijab", a.t1, tau149)
        )

        tau151 = (
            einsum("adij,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau152 = (
            einsum("ci,jabc->ijab", a.t1, tau151)
        )

        tau153 = (
            einsum("ac,bcij->ijab", h.f.vv, a.t2)
        )

        tau154 = (
            einsum("bc,caij->ijab", tau8, a.t2)
        )

        tau155 = (
            einsum("ci,abjc->ijab", a.t1, h.v.vvov)
        )

        tau156 = (
            einsum("ik,abjk->ijab", tau13, a.t2)
        )

        tau157 = (
            einsum("ki,abjk->ijab", h.f.oo, a.t2)
        )

        tau158 = (
            einsum("abli,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau159 = (
            einsum("ak,ikjb->ijab", a.t1, tau158)
        )

        tau160 = (
            einsum("adij,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau161 = (
            einsum("ci,jabc->ijab", a.t1, tau160)
        )

        tau162 = (
            einsum("daij,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau163 = (
            einsum("ci,jabc->ijab", a.t1, tau162)
        )

        tau164 = (
            einsum("daji,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau165 = (
            einsum("ci,jabc->ijab", a.t1, tau164)
        )

        tau166 = (
            einsum("bc,caij->ijab", tau4, a.t2)
        )

        tau167 = (
            einsum("acki,kbjc->ijab", a.t2, h.v.ovov)
        )

        tau168 = (
            einsum("bi,jkab->ijka", a.t1, tau2)
        )

        tau169 = (
            einsum("ak,ijkb->ijab", a.t1, tau168)
        )

        tau170 = (
            einsum("adji,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau171 = (
            einsum("ci,jabc->ijab", a.t1, tau170)
        )

        tau172 = (
            einsum("abil,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau173 = (
            einsum("ak,ikjb->ijab", a.t1, tau172)
        )

        tau174 = (
            einsum("ik,abjk->ijab", tau11, a.t2)
        )

        tau175 = (
            einsum("bali,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau176 = (
            einsum("ak,ikjb->ijab", a.t1, tau175)
        )

        tau177 = (
            einsum("kb,abij->ijka", tau0, a.t2)
        )

        tau178 = (
            einsum("ak,ijkb->ijab", a.t1, tau177)
        )

        tau179 = (
            einsum("daji,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau180 = (
            einsum("ci,jabc->ijab", a.t1, tau179)
        )

        tau181 = (
            einsum("bi,jakb->ijka", a.t1, h.v.ovov)
        )

        tau182 = (
            einsum("ak,ikjb->ijab", a.t1, tau181)
        )

        tau183 = (
            einsum("jk,abki->ijab", tau14, a.t2)
        )

        tau184 = (
            einsum("ackj,ikbc->ijab", a.t2, tau5)
        )

        tau185 = (
            einsum("bali,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau186 = (
            einsum("ak,ikjb->ijab", a.t1, tau185)
        )

        tau187 = (
            einsum("abkl,ijlk->ijab", a.t2, tau51)
        )

        tau188 = (
            2 * einsum("ijab->ijab", tau128)
            - 2 * einsum("ijab->ijab", tau129)
            - 4 * einsum("ijab->ijab", tau131)
            + 2 * einsum("ijab->ijab", tau133)
            - 2 * einsum("ijab->ijab", tau135)
            - 2 * einsum("ijab->ijab", tau136)
            + 2 * einsum("ijab->ijab", tau138)
            - 2 * einsum("ijab->ijab", tau139)
            + 2 * einsum("ijab->ijab", tau140)
            + 4 * einsum("ijab->ijab", tau143)
            - 2 * einsum("ijab->ijab", tau145)
            - 2 * einsum("ijab->ijab", tau146)
            - 2 * einsum("ijab->ijab", tau147)
            - 4 * einsum("ijab->ijab", tau148)
            + 2 * einsum("ijab->ijab", tau150)
            - 2 * einsum("ijab->ijab", tau152)
            + 2 * einsum("ijab->ijab", tau153)
            + einsum("ijab->ijab", tau154)
            + 4 * einsum("ijab->ijab", tau155)
            - 4 * einsum("ijab->ijab", tau156)
            - 2 * einsum("ijab->ijab", tau157)
            + 2 * einsum("ijab->ijab", tau159)
            + 4 * einsum("ijab->ijab", tau161)
            - 2 * einsum("ijab->ijab", tau163)
            + 4 * einsum("ijab->ijab", tau165)
            + 4 * einsum("ijab->ijab", tau166)
            - 2 * einsum("ijab->ijab", tau167)
            + 2 * einsum("ijab->ijab", tau169)
            - 2 * einsum("ijab->ijab", tau171)
            + 2 * einsum("ijab->ijab", tau173)
            + 2 * einsum("ijab->ijab", tau174)
            - 4 * einsum("ijab->ijab", tau176)
            - 4 * einsum("ijab->ijab", tau178)
            - 2 * einsum("ijab->ijab", tau180)
            - 4 * einsum("ijab->ijab", tau182)
            + einsum("ijab->ijab", tau183)
            + einsum("ijab->ijab", tau184)
            + 2 * einsum("ijab->ijab", tau186)
            + einsum("ijab->ijab", tau187)
        )

        rt1 = (
            - einsum("bcij,jabc->ai", a.t2, h.v.ovvv)
            + einsum("bj,ijab->ai", a.t1, tau2)
            - 2 * einsum("aj,ij->ai", a.t1, tau3)
            + 4 * einsum("bi,ab->ai", a.t1, tau4)
            - 2 * einsum("ji,aj->ai", h.f.oo, a.t1)
            - 2 * einsum("bj,ijab->ai", a.t1, tau5)
            - 2 * einsum("aj,ij->ai", a.t1, tau6)
            - 4 * einsum("aj,ji->ai", a.t1, tau7)
            + 2 * einsum("bcji,jabc->ai", a.t2, h.v.ovvv)
            + einsum("bi,ab->ai", a.t1, tau8)
            - 2 * einsum("abjk,jkib->ai", a.t2, h.v.ooov)
            + einsum("bj,ijab->ai", a.t1, tau9)
            - 2 * einsum("bi,ab->ai", a.t1, tau10)
            - 2 * einsum("bj,jaib->ai", a.t1, h.v.ovov)
            + 2 * einsum("bcij,jacb->ai", a.t2, h.v.ovvv)
            + einsum("abjk,kjib->ai", a.t2, h.v.ooov)
            + 2 * einsum("aj,ij->ai", a.t1, tau11)
            + 4 * einsum("bj,ijab->ai", a.t1, tau12)
            - 4 * einsum("aj,ij->ai", a.t1, tau13)
            + einsum("aj,ij->ai", a.t1, tau14)
            - einsum("jb,abji->ai", h.f.ov, a.t2)
            - 2 * einsum("aj,ji->ai", a.t1, tau15)
            - 2 * einsum("bj,ijab->ai", a.t1, tau16)
            + einsum("aj,ij->ai", a.t1, tau17)
            - 2 * einsum("bj,ijab->ai", a.t1, tau18)
            - 2 * einsum("bi,ab->ai", a.t1, tau19)
            + 2 * einsum("jb,baji->ai", h.f.ov, a.t2)
            + einsum("bi,ab->ai", a.t1, tau20)
            + 2 * einsum("jb,abij->ai", h.f.ov, a.t2)
            + 4 * einsum("bj,jabi->ai", a.t1, h.v.ovvo)
            - einsum("bcji,jacb->ai", a.t2, h.v.ovvv)
            + 4 * einsum("bj,ijab->ai", a.t1, tau21)
            - 2 * einsum("bi,ab->ai", a.t1, tau22)
            - 2 * einsum("bajk,kjib->ai", a.t2, h.v.ooov)
            + 2 * einsum("aj,ji->ai", a.t1, tau23)
            - 2 * einsum("bj,ijab->ai", a.t1, tau24)
            + 2 * einsum("ab,bi->ai", h.f.vv, a.t1)
            + einsum("bajk,jkib->ai", a.t2, h.v.ooov)
            + 2 * einsum("ia->ai", h.f.ov.conj())
            - einsum("jb,baij->ai", h.f.ov, a.t2)
        )

        rt2 = (
            einsum("jiab->abij", tau27) / 2
            - einsum("jiba->abij", tau27) / 4
            - einsum("ijba->abij", tau42) / 4
            + einsum("ijab->abij", tau42) / 2
            + einsum("ijab->abij", tau126) / 2
            - einsum("ijba->abij", tau126) / 4
            - einsum("jiab->abij", tau126) / 4
            + einsum("jiba->abij", tau126) / 2
            - einsum("ijab->abij", tau188) / 4
            + einsum("ijba->abij", tau188) / 2
            + einsum("jiab->abij", tau188) / 2
            - einsum("jiba->abij", tau188) / 4
        )
        tau0 = (
            einsum("abki,bajk->ij", a.t2, a.z2)
        )

        tau1 = (
            einsum("bi,kjba->ijka", a.t1, h.v.oovv)
        )

        tau2 = (
            einsum("bj,jiba->ia", a.t1, h.v.oovv)
        )

        tau3 = (
            einsum("caij,cbij->ab", a.t2, a.z2)
        )

        tau4 = (
            einsum("caik,bcjk->ijab", a.t2, a.z2)
        )

        tau5 = (
            einsum("kjcb,kica->ijab", tau4, h.v.oovv)
        )

        tau6 = (
            einsum("acik,bcjk->ijab", a.t2, a.z2)
        )

        tau7 = (
            einsum("kjcb,kiac->ijab", tau6, h.v.oovv)
        )

        tau8 = (
            einsum("daij,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau9 = (
            einsum("abij,bakl->ijkl", a.t2, a.z2)
        )

        tau10 = (
            einsum("klij,lkba->ijab", tau9, h.v.oovv)
        )

        tau11 = (
            einsum("ai,jkla->ijkl", a.t1, h.v.ooov)
        )

        tau12 = (
            einsum("kijl,ablk->ijab", tau11, a.z2)
        )

        tau13 = (
            einsum("abik,abjk->ij", a.t2, a.z2)
        )

        tau14 = (
            einsum("bj,jiab->ia", a.t1, h.v.oovv)
        )

        tau15 = (
            einsum("caik,cbjk->ijab", a.t2, a.z2)
        )

        tau16 = (
            einsum("kjcb,kiac->ijab", tau15, h.v.oovv)
        )

        tau17 = (
            einsum("jilk,abkl->ijab", h.v.oooo, a.z2)
        )

        tau18 = (
            einsum("caij,cbji->ab", a.t2, a.z2)
        )

        tau19 = (
            einsum("acij,bcji->ab", a.t2, a.z2)
        )

        tau20 = (
            einsum("ia,aj->ij", h.f.ov, a.t1)
        )

        tau21 = (
            einsum("acik,bckj->ijab", a.t2, a.z2)
        )

        tau22 = (
            einsum("kjcb,kica->ijab", tau21, h.v.oovv)
        )

        tau23 = (
            einsum("kjcb,kiac->ijab", tau4, h.v.oovv)
        )

        tau24 = (
            einsum("icka,bcjk->ijab", h.v.ovov, a.z2)
        )

        tau25 = (
            einsum("acik,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau26 = (
            einsum("bi,bajk->ijka", a.t1, a.z2)
        )

        tau27 = (
            einsum("kljb,kila->ijab", tau26, h.v.ooov)
        )

        tau28 = (
            einsum("idab,cdjk->ijkabc", h.v.ovvv, a.z2)
        )

        tau29 = (
            einsum("ck,kijabc->ijab", a.t1, tau28)
        )

        tau30 = (
            einsum("caik,cbkj->ijab", a.t2, a.z2)
        )

        tau31 = (
            einsum("abik,abkj->ij", a.t2, a.z2)
        )

        tau32 = (
            einsum("caki,cbjk->ijab", a.t2, a.z2)
        )

        tau33 = (
            einsum("icka,cbjk->ijab", h.v.ovov, a.z2)
        )

        tau34 = (
            einsum("abki,abkj->ij", a.t2, a.z2)
        )

        tau35 = (
            einsum("bali,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau36 = (
            einsum("acij,bcij->ab", a.t2, a.z2)
        )

        tau37 = (
            einsum("caik,kjcb->ijab", a.t2, h.v.oovv)
        )

        tau38 = (
            einsum("kica,bcjk->ijab", tau37, a.z2)
        )

        tau39 = (
            einsum("kica,cbkj->ijab", tau25, a.z2)
        )

        tau40 = (
            einsum("abki,abjk->ij", a.t2, a.z2)
        )

        tau41 = (
            einsum("caij,bcji->ab", a.t2, a.z2)
        )

        tau42 = (
            einsum("bi,ab->ia", a.t1, tau41)
        )

        tau43 = (
            einsum("bali,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau44 = (
            einsum("acik,cbkj->ijab", a.t2, a.z2)
        )

        tau45 = (
            einsum("kjcb,kiac->ijab", tau44, h.v.oovv)
        )

        tau46 = (
            einsum("adij,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau47 = (
            einsum("cj,icab->ijab", a.z1, h.v.ovvv)
        )

        tau48 = (
            einsum("caij,bcij->ab", a.t2, a.z2)
        )

        tau49 = (
            einsum("acij,cbij->ab", a.t2, a.z2)
        )

        tau50 = (
            einsum("bi,ab->ia", a.t1, tau49)
        )

        tau51 = (
            einsum("caki,cbkj->ijab", a.t2, a.z2)
        )

        tau52 = (
            einsum("ai,aj->ij", a.t1, a.z1)
        )

        tau53 = (
            einsum("kjcb,kica->ijab", tau15, h.v.oovv)
        )

        tau54 = (
            einsum("kijl,abkl->ijab", tau11, a.z2)
        )

        tau55 = (
            einsum("icka,cbkj->ijab", h.v.ovov, a.z2)
        )

        tau56 = (
            einsum("acik,cbjk->ijab", a.t2, a.z2)
        )

        tau57 = (
            einsum("kjcb,kiac->ijab", tau56, h.v.oovv)
        )

        tau58 = (
            einsum("bcij,kacb->ijka", a.t2, h.v.ovvv)
        )

        tau59 = (
            einsum("caik,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau60 = (
            einsum("kica,cbkj->ijab", tau59, a.z2)
        )

        tau61 = (
            einsum("bj,baij->ia", a.z1, a.t2)
        )

        tau62 = (
            einsum("ak,ikja->ij", a.t1, h.v.ooov)
        )

        tau63 = (
            einsum("abik,kjba->ij", a.t2, h.v.oovv)
        )

        tau64 = (
            einsum("abki,kjab->ij", a.t2, h.v.oovv)
        )

        tau65 = (
            einsum("abki,bakj->ij", a.t2, a.z2)
        )

        tau66 = (
            einsum("kjcb,kiac->ijab", tau21, h.v.oovv)
        )

        tau67 = (
            einsum("bi,ab->ia", a.t1, tau18)
        )

        tau68 = (
            einsum("kica,cbjk->ijab", tau25, a.z2)
        )

        tau69 = (
            einsum("kjcb,kica->ijab", tau6, h.v.oovv)
        )

        tau70 = (
            einsum("acij,cbji->ab", a.t2, a.z2)
        )

        tau71 = (
            einsum("icak,cbjk->ijab", h.v.ovvo, a.z2)
        )

        tau72 = (
            einsum("abik,bajk->ij", a.t2, a.z2)
        )

        tau73 = (
            einsum("ai,ja->ij", a.t1, tau14)
        )

        tau74 = (
            einsum("abij,lkba->ijkl", a.t2, h.v.oovv)
        )

        tau75 = (
            einsum("klij,abkl->ijab", tau74, a.z2)
        )

        tau76 = (
            einsum("bcij,kabc->ijka", a.t2, h.v.ovvv)
        )

        tau77 = (
            einsum("caik,bckj->ijab", a.t2, a.z2)
        )

        tau78 = (
            einsum("icak,cbkj->ijab", h.v.ovvo, a.z2)
        )

        tau79 = (
            einsum("kljb,ikla->ijab", tau26, h.v.ooov)
        )

        tau80 = (
            einsum("bail,jlkb->ijka", a.t2, h.v.ooov)
        )

        tau81 = (
            einsum("kica,cbjk->ijab", tau59, a.z2)
        )

        tau82 = (
            einsum("abik,bakj->ij", a.t2, a.z2)
        )

        tau83 = (
            einsum("lkia,kljb->ijab", tau1, tau26)
        )

        tau84 = (
            einsum("bail,ljkb->ijka", a.t2, h.v.ooov)
        )

        tau85 = (
            einsum("acik,kjbc->ijab", a.t2, h.v.oovv)
        )

        tau86 = (
            einsum("kica,cbkj->ijab", tau85, a.z2)
        )

        tau87 = (
            einsum("bi,ab->ia", a.t1, tau36)
        )

        tau88 = (
            einsum("bj,abij->ia", a.z1, a.t2)
        )

        tau89 = (
            einsum("kjcb,kiac->ijab", tau77, h.v.oovv)
        )

        tau90 = (
            einsum("ci,iacb->ab", a.t1, h.v.ovvv)
        )

        tau91 = (
            einsum("kjcb,kiac->ijab", tau30, h.v.oovv)
        )

        tau92 = (
            einsum("caij,jicb->ab", a.t2, h.v.oovv)
        )

        tau93 = (
            einsum("kica,bckj->ijab", tau85, a.z2)
        )

        tau94 = (
            einsum("bi,ab->ia", a.t1, tau3)
        )

        tau95 = (
            einsum("idab,dcjk->ijkabc", h.v.ovvv, a.z2)
        )

        tau96 = (
            einsum("ck,ikjacb->ijab", a.t1, tau95)
        )

        tau97 = (
            einsum("bi,ab->ia", a.t1, tau19)
        )

        tau98 = (
            einsum("kica,bcjk->ijab", tau59, a.z2)
        )

        tau99 = (
            einsum("caki,bckj->ijab", a.t2, a.z2)
        )

        tau100 = (
            einsum("abij,abkl->ijkl", a.t2, a.z2)
        )

        tau101 = (
            einsum("klij,lkba->ijab", tau100, h.v.oovv)
        )

        tau102 = (
            einsum("bi,abjk->ijka", a.t1, a.z2)
        )

        tau103 = (
            einsum("lkia,kljb->ijab", tau1, tau102)
        )

        tau104 = (
            einsum("kica,cbjk->ijab", tau37, a.z2)
        )

        tau105 = (
            einsum("icka,bckj->ijab", h.v.ovov, a.z2)
        )

        tau106 = (
            einsum("kica,bckj->ijab", tau59, a.z2)
        )

        tau107 = (
            einsum("kica,cbjk->ijab", tau85, a.z2)
        )

        tau108 = (
            einsum("caki,bcjk->ijab", a.t2, a.z2)
        )

        tau109 = (
            einsum("ak,kija->ij", a.t1, h.v.ooov)
        )

        tau110 = (
            einsum("kica,bcjk->ijab", tau25, a.z2)
        )

        tau111 = (
            einsum("kica,cbkj->ijab", tau37, a.z2)
        )

        tau112 = (
            einsum("bi,ab->ia", a.t1, tau70)
        )

        tau113 = (
            einsum("kjcb,kica->ijab", tau56, h.v.oovv)
        )

        tau114 = (
            einsum("adij,jbdc->iabc", a.t2, h.v.ovvv)
        )

        tau115 = (
            einsum("icak,bcjk->ijab", h.v.ovvo, a.z2)
        )

        tau116 = (
            einsum("klij,ablk->ijab", tau74, a.z2)
        )

        tau117 = (
            einsum("lika,kljb->ijab", tau1, tau26)
        )

        tau118 = (
            einsum("ai,jkla->ijkl", a.t1, tau102)
        )

        tau119 = (
            einsum("klij,lkba->ijab", tau118, h.v.oovv)
        )

        tau120 = (
            einsum("kjcb,kica->ijab", tau44, h.v.oovv)
        )

        tau121 = (
            einsum("abik,kjab->ij", a.t2, h.v.oovv)
        )

        tau122 = (
            einsum("dcba,cdij->ijab", h.v.vvvv, a.z2)
        )

        tau123 = (
            einsum("icak,bckj->ijab", h.v.ovvo, a.z2)
        )

        tau124 = (
            einsum("lika,kljb->ijab", tau1, tau102)
        )

        tau125 = (
            einsum("daij,jbcd->iabc", a.t2, h.v.ovvv)
        )

        tau126 = (
            einsum("kica,bckj->ijab", tau37, a.z2)
        )

        tau127 = (
            einsum("bi,ab->ia", a.t1, tau48)
        )

        tau128 = (
            einsum("ck,kijabc->ijab", a.t1, tau95)
        )

        tau129 = (
            einsum("kjlb,ikla->ijab", tau26, h.v.ooov)
        )

        tau130 = (
            einsum("kica,bckj->ijab", tau25, a.z2)
        )

        tau131 = (
            einsum("acij,jicb->ab", a.t2, h.v.oovv)
        )

        tau132 = (
            einsum("kica,bcjk->ijab", tau85, a.z2)
        )

        tau133 = (
            einsum("kjlb,kila->ijab", tau26, h.v.ooov)
        )

        tau134 = (
            einsum("ck,ikjacb->ijab", a.t1, tau28)
        )

        tau135 = (
            einsum("ai,ja->ij", a.t1, tau2)
        )

        tau136 = (
            einsum("kjcb,kica->ijab", tau77, h.v.oovv)
        )

        tau137 = (
            einsum("abki,kjba->ij", a.t2, h.v.oovv)
        )

        tau138 = (
            einsum("lkia,kjlb->ijab", tau1, tau26)
        )

        tau139 = (
            einsum("ci,iabc->ab", a.t1, h.v.ovvv)
        )

        tau140 = (
            einsum("acij,jibc->ab", a.t2, h.v.oovv)
        )

        tau141 = (
            einsum("ck,ijkacb->ijab", a.t1, tau95)
        )

        tau142 = (
            einsum("caij,jibc->ab", a.t2, h.v.oovv)
        )

        tau143 = (
            einsum("kjcb,kica->ijab", tau30, h.v.oovv)
        )

        tau144 = (
            einsum("ck,ijkacb->ijab", a.t1, tau28)
        )

        tau145 = (
            einsum("kj,kiba->ijab", tau0, h.v.oovv)
        )

        tau146 = (
            einsum("ka,kijb->ijab", h.f.ov, tau26)
        )

        tau147 = (
            einsum("cb,jica->ijab", tau48, h.v.oovv)
        )

        tau148 = (
            einsum("ki,abkj->ijab", tau135, a.z2)
        )

        tau149 = (
            einsum("kjlb,ikla->ijab", tau102, h.v.ooov)
        )

        tau150 = (
            einsum("cb,jica->ijab", tau70, h.v.oovv)
        )

        tau151 = (
            einsum("cb,jica->ijab", tau18, h.v.oovv)
        )

        tau152 = (
            einsum("ik,abkj->ijab", tau109, a.z2)
        )

        tau153 = (
            einsum("ca,cbij->ijab", tau142, a.z2)
        )

        tau154 = (
            einsum("cb,jica->ijab", tau36, h.v.oovv)
        )

        tau155 = (
            einsum("lika,kjlb->ijab", tau1, tau26)
        )

        tau156 = (
            einsum("ka,kijb->ijab", tau14, tau26)
        )

        tau157 = (
            einsum("ik,abkj->ijab", tau20, a.z2)
        )

        tau158 = (
            einsum("ai,jkla->ijkl", a.t1, tau1)
        )

        tau159 = (
            einsum("lkji,abkl->ijab", tau158, a.z2)
        )

        tau160 = (
            einsum("cb,jica->ijab", tau49, h.v.oovv)
        )

        tau161 = (
            einsum("ki,abkj->ijab", tau64, a.z2)
        )

        tau162 = (
            einsum("ik,abkj->ijab", tau62, a.z2)
        )

        tau163 = (
            einsum("ck,ikjcab->ijab", a.t1, tau95)
        )

        tau164 = (
            einsum("kj,kiba->ijab", tau82, h.v.oovv)
        )

        tau165 = (
            einsum("ca,cbij->ijab", tau140, a.z2)
        )

        tau166 = (
            einsum("ki,abkj->ijab", tau73, a.z2)
        )

        tau167 = (
            einsum("ka,kijb->ijab", tau2, tau26)
        )

        tau168 = (
            einsum("ca,cbij->ijab", tau131, a.z2)
        )

        tau169 = (
            einsum("kj,kiba->ijab", tau13, h.v.oovv)
        )

        tau170 = (
            einsum("kj,kiba->ijab", tau72, h.v.oovv)
        )

        tau171 = (
            einsum("kj,kiba->ijab", tau52, h.v.oovv)
        )

        tau172 = (
            einsum("bk,kija->ijab", a.z1, tau1)
        )

        tau173 = (
            einsum("ca,cbij->ijab", tau90, a.z2)
        )

        tau174 = (
            einsum("ki,abkj->ijab", tau63, a.z2)
        )

        tau175 = (
            einsum("kljb,kila->ijab", tau102, h.v.ooov)
        )

        tau176 = (
            einsum("kj,kiba->ijab", tau40, h.v.oovv)
        )

        tau177 = (
            einsum("ki,abkj->ijab", tau121, a.z2)
        )

        tau178 = (
            einsum("cb,jica->ijab", tau41, h.v.oovv)
        )

        tau179 = (
            einsum("ca,cbij->ijab", tau139, a.z2)
        )

        tau180 = (
            einsum("kj,kiba->ijab", tau65, h.v.oovv)
        )

        tau181 = (
            einsum("kj,kiba->ijab", tau31, h.v.oovv)
        )

        tau182 = (
            einsum("cb,jica->ijab", tau3, h.v.oovv)
        )

        tau183 = (
            einsum("ik,abkj->ijab", h.f.oo, a.z2)
        )

        tau184 = (
            einsum("ck,ijkcab->ijab", a.t1, tau28)
        )

        tau185 = (
            einsum("kj,kiba->ijab", tau34, h.v.oovv)
        )

        tau186 = (
            einsum("kjlb,kila->ijab", tau102, h.v.ooov)
        )

        tau187 = (
            einsum("lkia,kjlb->ijab", tau1, tau102)
        )

        tau188 = (
            einsum("ca,cbij->ijab", h.f.vv, a.z2)
        )

        tau189 = (
            einsum("lika,kjlb->ijab", tau1, tau102)
        )

        tau190 = (
            einsum("ca,cbij->ijab", tau92, a.z2)
        )

        tau191 = (
            einsum("cb,jica->ijab", tau19, h.v.oovv)
        )

        tau192 = (
            einsum("ki,abkj->ijab", tau137, a.z2)
        )

        tau193 = (
            - 2 * einsum("ijab->ijab", tau145)
            - 2 * einsum("ijab->ijab", tau55)
            - 2 * einsum("ijab->ijab", tau111)
            - 2 * einsum("ijab->ijab", tau146)
            - 2 * einsum("ijab->ijab", tau134)
            + einsum("ijab->ijab", tau147)
            + einsum("ijab->ijab", tau53)
            + 2 * einsum("ijab->ijab", tau124)
            - 2 * einsum("ijab->ijab", tau38)
            - 4 * einsum("ijab->ijab", tau148)
            + einsum("ijab->ijab", tau57)
            - 2 * einsum("ijab->ijab", tau71)
            + 2 * einsum("ijab->ijab", tau149)
            + einsum("ijab->ijab", tau75)
            - 2 * einsum("ijab->ijab", tau150)
            + einsum("ijab->ijab", tau126)
            + einsum("ijab->ijab", tau151)
            + einsum("ijab->ijab", tau66)
            + 2 * einsum("ijab->ijab", tau152)
            - 2 * einsum("ijab->ijab", tau29)
            - 2 * einsum("ijab->ijab", tau153)
            - 2 * einsum("ijab->ijab", tau154)
            + 2 * einsum("ijab->ijab", tau155)
            + 2 * einsum("ijab->ijab", tau156)
            + 4 * einsum("ijab->ijab", tau47)
            + einsum("ijab->ijab", tau104)
            - 2 * einsum("ijab->ijab", tau157)
            + 2 * einsum("ijab->ijab", tau159)
            + einsum("ijab->ijab", tau160)
            - 2 * einsum("ijab->ijab", tau161)
            + 4 * einsum("ijab->ijab", tau78)
            + 4 * einsum("ijab->ijab", tau120)
            + 2 * einsum("ijab->ijab", tau122)
            + 4 * einsum("ijab->ijab", tau144)
            - 4 * einsum("ijab->ijab", tau162)
            - 4 * einsum("ijab->ijab", tau27)
            - 2 * einsum("ijab->ijab", tau163)
            - 2 * einsum("ijab->ijab", tau164)
            + einsum("ijab->ijab", tau165)
            + 2 * einsum("ijab->ijab", tau166)
            + einsum("ijab->ijab", tau107)
            + 4 * einsum("ijab->ijab", tau39)
            - 2 * einsum("ijab->ijab", tau45)
            + einsum("ijab->ijab", tau60)
            + 2 * einsum("ijab->ijab", tau133)
            - 4 * einsum("ijab->ijab", tau167)
            + 2 * einsum("ijab->ijab", tau12)
            + 4 * einsum("ijab->ijab", tau96)
            - 2 * einsum("ijab->ijab", tau168)
            - 2 * einsum("ijab->ijab", tau169)
            + einsum("ijab->ijab", tau93)
            + einsum("ijab->ijab", tau170)
            + 8 * einsum("ia,bj->ijab", tau2, a.z1)
            + einsum("ijab->ijab", tau98)
            - 2 * einsum("ijab->ijab", tau143)
            - 2 * einsum("ijab->ijab", tau7)
            + einsum("ijab->ijab", tau91)
            + 4 * einsum("ijab->ijab", tau115)
            - 4 * einsum("ijab->ijab", tau171)
            - 4 * einsum("ijab->ijab", tau172)
            + 4 * einsum("ijab->ijab", tau173)
            - 2 * einsum("ijab->ijab", tau174)
            + 2 * einsum("ijab->ijab", tau175)
            + 4 * einsum("ijab->ijab", tau110)
            - 2 * einsum("ijab->ijab", tau22)
            + 2 * einsum("ijab->ijab", tau83)
            + 2 * einsum("ijab->ijab", tau17)
            + 2 * einsum("ijab->ijab", tau119)
            + einsum("ijab->ijab", tau176)
            + einsum("ijab->ijab", tau177)
            - 2 * einsum("ijab->ijab", tau178)
            - 2 * einsum("ijab->ijab", tau179)
            + einsum("ijab->ijab", tau180)
            + einsum("ijab->ijab", tau181)
            - 2 * einsum("ijab->ijab", tau141)
            - 2 * einsum("ijab->ijab", tau130)
            - 4 * einsum("ijab->ijab", tau117)
            - 2 * einsum("ijab->ijab", tau182)
            + 4 * einsum("ia,bj->ijab", h.f.ov, a.z1)
            - 2 * einsum("ijab->ijab", tau183)
            - 2 * einsum("ijab->ijab", tau184)
            - 4 * einsum("ia,bj->ijab", tau14, a.z1)
            - 2 * einsum("ijab->ijab", tau132)
            - 2 * einsum("ijab->ijab", tau113)
            - 2 * einsum("ijab->ijab", tau185)
            - 2 * einsum("ijab->ijab", tau5)
            - 2 * einsum("ijab->ijab", tau68)
            - 4 * einsum("ijab->ijab", tau186)
            + 2 * einsum("ijab->ijab", tau187)
            + einsum("ijab->ijab", tau101)
            - 2 * einsum("ijab->ijab", tau86)
            + 2 * einsum("ijab->ijab", tau188)
            - 4 * einsum("ijab->ijab", tau189)
            + einsum("ijab->ijab", tau190)
            + einsum("ijab->ijab", tau191)
            + einsum("ijab->ijab", tau192)
            + 4 * einsum("ijab->ijab", tau69)
            + einsum("ijab->ijab", tau23)
            + 2 * einsum("ijab->ijab", tau79)
            - 2 * einsum("ijab->ijab", tau24)
            - 2 * einsum("ijab->ijab", tau123)
            + einsum("ijab->ijab", tau136)
        )

        tau194 = (
            einsum("ik,abjk->ijab", h.f.oo, a.z2)
        )

        tau195 = (
            einsum("ck,ijkcab->ijab", a.t1, tau95)
        )

        tau196 = (
            einsum("kljb,ikla->ijab", tau102, h.v.ooov)
        )

        tau197 = (
            einsum("ki,abjk->ijab", tau63, a.z2)
        )

        tau198 = (
            einsum("ca,bcij->ijab", tau90, a.z2)
        )

        tau199 = (
            einsum("bk,ijka->ijab", a.z1, h.v.ooov)
        )

        tau200 = (
            einsum("ki,abjk->ijab", tau121, a.z2)
        )

        tau201 = (
            einsum("ca,bcij->ijab", tau142, a.z2)
        )

        tau202 = (
            einsum("ki,abjk->ijab", tau73, a.z2)
        )

        tau203 = (
            einsum("ik,abjk->ijab", tau20, a.z2)
        )

        tau204 = (
            einsum("ka,kijb->ijab", tau14, tau102)
        )

        tau205 = (
            einsum("ca,bcij->ijab", tau92, a.z2)
        )

        tau206 = (
            einsum("ca,bcij->ijab", tau131, a.z2)
        )

        tau207 = (
            einsum("ik,abjk->ijab", tau62, a.z2)
        )

        tau208 = (
            einsum("ka,kijb->ijab", tau2, tau102)
        )

        tau209 = (
            einsum("ki,abjk->ijab", tau137, a.z2)
        )

        tau210 = (
            einsum("ck,ikjcab->ijab", a.t1, tau28)
        )

        tau211 = (
            einsum("ik,abjk->ijab", tau109, a.z2)
        )

        tau212 = (
            einsum("ca,bcij->ijab", tau139, a.z2)
        )

        tau213 = (
            einsum("ca,bcij->ijab", tau140, a.z2)
        )

        tau214 = (
            einsum("ca,bcij->ijab", h.f.vv, a.z2)
        )

        tau215 = (
            einsum("ki,abjk->ijab", tau135, a.z2)
        )

        tau216 = (
            einsum("ki,abjk->ijab", tau64, a.z2)
        )

        tau217 = (
            einsum("ka,kijb->ijab", h.f.ov, tau102)
        )

        tau218 = (
            - 2 * einsum("ijab->ijab", tau194)
            - 2 * einsum("ijab->ijab", tau195)
            + 2 * einsum("ijab->ijab", tau196)
            + einsum("ijab->ijab", tau116)
            - 2 * einsum("ijab->ijab", tau197)
            - 2 * einsum("ijab->ijab", tau105)
            + 4 * einsum("ijab->ijab", tau198)
            - 4 * einsum("ijab->ijab", tau199)
            + einsum("ijab->ijab", tau200)
            - 2 * einsum("ijab->ijab", tau201)
            + 2 * einsum("ijab->ijab", tau202)
            - 2 * einsum("ijab->ijab", tau33)
            - 2 * einsum("ijab->ijab", tau203)
            + einsum("ijab->ijab", tau81)
            + einsum("ijab->ijab", tau10)
            + 2 * einsum("ijab->ijab", tau204)
            + einsum("ijab->ijab", tau205)
            - 2 * einsum("ijab->ijab", tau206)
            + 2 * einsum("ijab->ijab", tau54)
            - 4 * einsum("ijab->ijab", tau207)
            + 2 * einsum("ijab->ijab", tau103)
            - 4 * einsum("ijab->ijab", tau208)
            - 2 * einsum("ijab->ijab", tau128)
            + einsum("ijab->ijab", tau16)
            + einsum("ijab->ijab", tau209)
            - 2 * einsum("ijab->ijab", tau210)
            + 2 * einsum("ijab->ijab", tau211)
            + 2 * einsum("ijab->ijab", tau129)
            - 2 * einsum("ijab->ijab", tau212)
            + 2 * einsum("ijab->ijab", tau138)
            + einsum("ijab->ijab", tau89)
            + einsum("ijab->ijab", tau213)
            + 2 * einsum("ijab->ijab", tau214)
            + einsum("ijab->ijab", tau106)
            - 4 * einsum("ijab->ijab", tau215)
            - 2 * einsum("ijab->ijab", tau216)
            - 2 * einsum("ijab->ijab", tau217)
        )

        rz1 = (
            - 2 * einsum("kj,jika->ai", tau0, tau1)
            - 2 * einsum("ib,ba->ai", tau2, tau3)
            + einsum("bj,jiab->ai", a.t1, tau5)
            + einsum("bj,jiab->ai", a.t1, tau7)
            - 2 * einsum("ji,ja->ai", tau0, tau2)
            + einsum("jbca,bcij->ai", tau8, a.z2) / 2
            - einsum("bj,ijab->ai", a.t1, tau10) / 2
            - einsum("bj,ijba->ai", a.t1, tau12)
            + einsum("ji,ja->ai", tau13, tau14)
            + einsum("bj,jiab->ai", a.t1, tau16)
            - einsum("bj,ijba->ai", a.t1, tau17)
            - einsum("bc,icab->ai", tau18, h.v.ovvv)
            - einsum("ib,ba->ai", tau14, tau19) / 2
            - 2 * einsum("ibkj,bajk->ai", h.v.ovoo, a.z2)
            - 2 * einsum("ij,aj->ai", tau20, a.z1)
            + einsum("bj,jiab->ai", a.t1, tau22)
            - einsum("bj,ijba->ai", a.t1, tau23) / 2
            + einsum("bj,ijba->ai", a.t1, tau24)
            + 4 * einsum("bj,jiba->ai", a.z1, tau25)
            + 2 * einsum("bj,jiab->ai", a.t1, tau27)
            + einsum("bj,ijba->ai", a.t1, tau29)
            - einsum("jibc,jcba->ai", tau30, h.v.ovvv)
            + einsum("jk,jika->ai", tau31, h.v.ooov)
            - einsum("jkba,jikb->ai", tau32, h.v.ooov) / 2
            - 2 * einsum("bj,jiab->ai", a.t1, tau33)
            - 2 * einsum("kj,jika->ai", tau34, tau1)
            + 2 * einsum("bj,jiba->ai", a.t1, tau17)
            + einsum("jikb,bakj->ai", tau35, a.z2)
            - einsum("bc,icba->ai", tau36, h.v.ovvv)
            + einsum("bj,jiab->ai", a.t1, tau38)
            + einsum("ibkj,abjk->ai", h.v.ovoo, a.z2)
            - 2 * einsum("bj,ijba->ai", a.t1, tau39)
            + einsum("ja,ji->ai", tau2, tau40)
            - 2 * einsum("jb,jiba->ai", tau42, h.v.oovv)
            + 2 * einsum("jibc,jcba->ai", tau6, h.v.ovvv)
            - einsum("jikb,abjk->ai", tau43, a.z2) / 2
            + einsum("bj,ijba->ai", a.t1, tau45)
            + einsum("jbca,cbji->ai", tau46, a.z2) / 2
            + einsum("jikb,abkj->ai", tau43, a.z2)
            + einsum("bj,ijba->ai", a.t1, tau16)
            - 2 * einsum("bj,ijba->ai", a.t1, tau47)
            + einsum("ib,ba->ai", h.f.ov, tau48) / 2
            + einsum("jb,jiba->ai", tau50, h.v.oovv)
            + einsum("jb,jiab->ai", tau42, h.v.oovv)
            + einsum("jkba,jikb->ai", tau51, h.v.ooov)
            - 2 * einsum("ja,ji->ai", h.f.ov, tau52)
            + einsum("ib,ba->ai", tau14, tau3)
            - einsum("bj,ijba->ai", a.t1, tau53) / 2
            - einsum("bj,jiba->ai", a.t1, tau54)
            + einsum("bj,ijba->ai", a.t1, tau55)
            - einsum("bj,jiab->ai", a.t1, tau57) / 2
            - einsum("jkib,abkj->ai", tau58, a.z2)
            - einsum("bj,ijba->ai", a.t1, tau60) / 2
            - 2 * einsum("jb,jiba->ai", tau61, h.v.oovv)
            + einsum("bj,jiab->ai", a.t1, tau24)
            - 4 * einsum("ij,aj->ai", tau62, a.z1)
            + einsum("kj,jika->ai", tau40, tau1)
            - 2 * einsum("ji,aj->ai", tau63, a.z1)
            - 2 * einsum("ji,aj->ai", tau64, a.z1)
            + einsum("jk,jika->ai", tau65, h.v.ooov)
            - einsum("bj,jiab->ai", a.t1, tau66) / 2
            - 2 * einsum("kj,jika->ai", tau13, tau1)
            + einsum("jbca,cbji->ai", tau8, a.z2) / 2
            - einsum("kj,jkia->ai", tau40, tau1) / 2
            - einsum("jbca,cbij->ai", tau46, a.z2)
            + einsum("jb,jiba->ai", tau67, h.v.oovv)
            + einsum("bj,jiab->ai", a.t1, tau68)
            - 2 * einsum("bj,jiab->ai", a.t1, tau69)
            - einsum("jbca,cbij->ai", tau8, a.z2)
            + einsum("jkba,jikb->ai", tau15, h.v.ooov)
            + 2 * einsum("bc,icab->ai", tau70, h.v.ovvv)
            - einsum("bj,jiab->ai", a.t1, tau17)
            + einsum("ja,ji->ai", h.f.ov, tau31) / 2
            + einsum("bj,ijba->ai", a.t1, tau71)
            + einsum("kj,jkia->ai", tau34, tau1)
            - einsum("kj,jkia->ai", tau72, tau1) / 2
            - 2 * einsum("ji,ja->ai", tau13, tau2)
            + einsum("jkli,kjla->ai", tau9, h.v.ooov)
            - einsum("ja,ji->ai", h.f.ov, tau0)
            + 2 * einsum("ji,aj->ai", tau73, a.z1)
            - einsum("bc,icba->ai", tau70, h.v.ovvv)
            - einsum("kj,jkia->ai", tau31, tau1) / 2
            + einsum("bc,icba->ai", tau49, h.v.ovvv) / 2
            + einsum("bj,jiba->ai", a.t1, tau75)
            - einsum("jkib,abjk->ai", tau76, a.z2)
            - einsum("jkba,ijkb->ai", tau77, h.v.ooov) / 2
            - 2 * einsum("bj,ijba->ai", a.t1, tau78)
            - einsum("bcja,cbji->ai", h.v.vvov, a.z2)
            + einsum("bj,jiab->ai", a.t1, tau71)
            - einsum("jibc,jcab->ai", tau6, h.v.ovvv)
            + 2 * einsum("bj,jiab->ai", a.t1, tau54)
            - einsum("bj,jiab->ai", a.t1, tau79)
            - einsum("ja,ji->ai", tau14, tau72) / 2
            - einsum("jkba,jikb->ai", tau30, h.v.ooov) / 2
            + einsum("jikb,abkj->ai", tau80, a.z2)
            - einsum("kj,jkia->ai", tau65, tau1) / 2
            - 2 * einsum("ib,ba->ai", tau2, tau70)
            - einsum("jibc,jcba->ai", tau4, h.v.ovvv)
            + einsum("bj,ijba->ai", a.t1, tau81)
            - 2 * einsum("ja,ji->ai", tau2, tau82)
            + einsum("bj,ijba->ai", a.t1, tau38)
            + einsum("bj,ijba->ai", a.t1, tau22)
            + einsum("bj,ijba->ai", a.t1, tau7)
            - einsum("bj,jiab->ai", a.t1, tau83)
            - einsum("jikb,abkj->ai", tau84, a.z2) / 2
            + einsum("bj,jiab->ai", a.t1, tau86)
            - 2 * einsum("jb,jiba->ai", tau87, h.v.oovv)
            - 2 * einsum("jb,jiab->ai", tau88, h.v.oovv)
            + einsum("bj,ijba->ai", a.t1, tau89)
            + 4 * einsum("ba,bi->ai", tau90, a.z1)
            - einsum("bj,ijba->ai", a.t1, tau91) / 2
            - 2 * einsum("bj,jiba->ai", a.z1, tau85)
            + 2 * einsum("bj,ijab->ai", a.t1, tau17)
            + einsum("bc,icba->ai", tau18, h.v.ovvv) / 2
            + einsum("ba,bi->ai", tau92, a.z1)
            - einsum("bj,ijba->ai", a.t1, tau93) / 2
            - einsum("ib,ba->ai", h.f.ov, tau41)
            + einsum("jibc,jcab->ai", tau21, h.v.ovvv) / 2
            - einsum("jkil,kjla->ai", tau9, h.v.ooov) / 2
            - einsum("ib,ba->ai", tau14, tau49) / 2
            + einsum("jb,jiab->ai", tau94, h.v.oovv)
            - 2 * einsum("bj,ijba->ai", a.t1, tau96)
            - 2 * einsum("jk,jika->ai", tau13, h.v.ooov)
            + einsum("jb,jiba->ai", tau97, h.v.oovv)
            + einsum("bj,jiba->ai", a.z1, tau59)
            - einsum("bj,ijba->ai", a.t1, tau98) / 2
            - einsum("bj,jiab->ai", a.t1, tau75) / 2
            + einsum("jkba,ijkb->ai", tau99, h.v.ooov)
            + einsum("jibc,jcab->ai", tau30, h.v.ovvv) / 2
            - einsum("bj,jiab->ai", a.t1, tau101) / 2
            - 2 * einsum("bj,jiba->ai", a.t1, tau29)
            + einsum("ja,ji->ai", tau2, tau65)
            - einsum("ib,ba->ai", h.f.ov, tau70)
            - einsum("ja,ji->ai", tau14, tau40) / 2
            + einsum("jkil,kjla->ai", tau100, h.v.ooov)
            + 2 * einsum("bcja,bcji->ai", h.v.vvov, a.z2)
            - einsum("bj,jiab->ai", a.t1, tau53) / 2
            + einsum("jibc,jcab->ai", tau4, h.v.ovvv) / 2
            + 2 * einsum("bj,ijba->ai", a.t1, tau103)
            - einsum("ja,ji->ai", h.f.ov, tau34)
            - 2 * einsum("bj,ijba->ai", a.t1, tau69)
            - einsum("bj,ijba->ai", a.t1, tau104) / 2
            - 2 * einsum("bj,ijba->ai", a.t1, tau105)
            + 2 * einsum("bc,icab->ai", tau3, h.v.ovvv)
            + einsum("bj,jiab->ai", a.t1, tau106)
            + einsum("ibjk,bajk->ai", h.v.ovoo, a.z2)
            - einsum("bj,jiab->ai", a.t1, tau107) / 2
            + einsum("jkba,jikb->ai", tau108, h.v.ooov)
            + 2 * einsum("ij,aj->ai", tau109, a.z1)
            - 2 * einsum("bj,ijba->ai", a.t1, tau33)
            - 2 * einsum("bj,ijba->ai", a.t1, tau110)
            + einsum("jk,ijka->ai", tau34, h.v.ooov)
            - 2 * einsum("ja,ji->ai", tau2, tau34)
            + einsum("ja,ji->ai", tau14, tau34)
            - 2 * einsum("bj,jiba->ai", a.z1, tau37)
            + einsum("kj,jika->ai", tau72, tau1)
            - einsum("bc,icab->ai", tau49, h.v.ovvv)
            + einsum("jibc,jcba->ai", tau77, h.v.ovvv) / 2
            + einsum("kj,jika->ai", tau31, tau1)
            + einsum("bj,jiab->ai", a.t1, tau111)
            - einsum("jbca,bcji->ai", tau46, a.z2)
            + einsum("jb,jiab->ai", tau112, h.v.oovv)
            - einsum("bc,icab->ai", tau19, h.v.ovvv)
            + einsum("bj,ijba->ai", a.t1, tau106)
            - einsum("ja,ji->ai", tau14, tau31) / 2
            - 2 * einsum("ibjk,abjk->ai", h.v.ovoo, a.z2)
            + einsum("bj,jiab->ai", a.t1, tau55)
            - einsum("jk,ijka->ai", tau72, h.v.ooov) / 2
            + einsum("jkib,bajk->ai", tau76, a.z2) / 2
            + 2 * einsum("ba,bi->ai", h.f.vv, a.z1)
            - 2 * einsum("bj,ijab->ai", a.t1, tau29)
            - 2 * einsum("ib,ba->ai", tau2, tau41)
            - 2 * einsum("jikb,bajk->ai", tau35, a.z2)
            - 2 * einsum("bj,ibja->ai", a.z1, h.v.ovov)
            - einsum("bj,ijba->ai", a.t1, tau83)
            + einsum("bj,ijba->ai", a.t1, tau113)
            - einsum("jk,ijka->ai", tau40, h.v.ooov) / 2
            + 2 * einsum("jbca,bcji->ai", tau114, a.z2)
            + einsum("bj,jiab->ai", a.t1, tau113)
            - einsum("jikb,bakj->ai", tau80, a.z2) / 2
            - 2 * einsum("bj,ijba->ai", a.t1, tau115)
            + 2 * einsum("ia->ai", h.f.ov)
            - einsum("bj,jiba->ai", a.t1, tau116) / 2
            + einsum("ja,ji->ai", tau2, tau31)
            + einsum("bj,jiab->ai", a.t1, tau89)
            + 2 * einsum("bj,ijba->ai", a.t1, tau117)
            - 2 * einsum("bj,jiab->ai", a.t1, tau115)
            + einsum("bj,jiab->ai", a.t1, tau116)
            + einsum("jikb,bajk->ai", tau80, a.z2)
            + einsum("bc,icba->ai", tau19, h.v.ovvv) / 2
            + einsum("jkil,jkla->ai", tau9, h.v.ooov)
            - einsum("jkba,jikb->ai", tau4, h.v.ooov) / 2
            - 2 * einsum("bj,jiab->ai", a.t1, tau39)
            + einsum("ja,ji->ai", tau2, tau72)
            - einsum("bj,ijba->ai", a.t1, tau119)
            - einsum("jb,jiab->ai", tau97, h.v.oovv) / 2
            - einsum("bcja,bcij->ai", h.v.vvov, a.z2)
            + 2 * einsum("bj,jiba->ai", a.t1, tau12)
            - 2 * einsum("bj,ijba->ai", a.t1, tau120)
            + 4 * einsum("jb,jiba->ai", tau88, h.v.oovv)
            - 2 * einsum("ia->ai", tau14)
            - einsum("ja,ji->ai", h.f.ov, tau13)
            + einsum("ji,aj->ai", tau121, a.z1)
            + 2 * einsum("bc,icab->ai", tau41, h.v.ovvv)
            + 4 * einsum("bj,ibaj->ai", a.z1, h.v.ovvo)
            + einsum("ib,ba->ai", h.f.ov, tau49) / 2
            + einsum("bj,ijba->ai", a.t1, tau5)
            + 4 * einsum("bj,ijab->ai", a.t1, tau47)
            + 2 * einsum("bj,ijab->ai", a.t1, tau122)
            + einsum("bj,jiab->ai", a.t1, tau123)
            - einsum("bj,ijba->ai", a.t1, tau66) / 2
            + einsum("jibc,jcba->ai", tau15, h.v.ovvv) / 2
            - einsum("bj,ijba->ai", a.t1, tau124)
            - 2 * einsum("kj,jika->ai", tau82, tau1)
            - einsum("jkli,kjla->ai", tau100, h.v.ooov) / 2
            + einsum("ib,ba->ai", tau14, tau36)
            + einsum("jbca,cbij->ai", tau125, a.z2) / 2
            - einsum("bj,jiab->ai", a.t1, tau23) / 2
            + einsum("bj,jiab->ai", a.t1, tau29)
            - 2 * einsum("jb,jiba->ai", tau112, h.v.oovv)
            + 4 * einsum("ia->ai", tau2)
            - einsum("bj,jiab->ai", a.t1, tau126) / 2
            - einsum("bj,jiab->ai", a.t1, tau91) / 2
            - 4 * einsum("ja,ji->ai", tau2, tau52)
            - einsum("jkil,jkla->ai", tau100, h.v.ooov) / 2
            + 2 * einsum("bj,ijba->ai", a.t1, tau54)
            - einsum("ja,ji->ai", h.f.ov, tau82)
            - einsum("jikb,bakj->ai", tau43, a.z2) / 2
            + einsum("ja,ji->ai", h.f.ov, tau40) / 2
            - einsum("bj,ijba->ai", a.t1, tau101) / 2
            + einsum("jikb,abjk->ai", tau84, a.z2)
            - einsum("jb,jiab->ai", tau127, h.v.oovv) / 2
            - einsum("jbca,bcji->ai", tau8, a.z2)
            + einsum("ja,ji->ai", h.f.ov, tau65) / 2
            + einsum("jk,jika->ai", tau40, h.v.ooov)
            - 2 * einsum("jb,jiba->ai", tau94, h.v.oovv)
            + einsum("jikb,bakj->ai", tau84, a.z2)
            + 2 * einsum("bcja,cbij->ai", h.v.vvov, a.z2)
            - 2 * einsum("jk,jika->ai", tau34, h.v.ooov)
            + einsum("jibc,jcab->ai", tau56, h.v.ovvv) / 2
            - einsum("bj,jiab->ai", a.t1, tau98) / 2
            + 2 * einsum("jk,kjia->ai", tau52, tau1)
            - 2 * einsum("bj,ijba->ai", a.t1, tau128)
            + einsum("jb,jiab->ai", tau61, h.v.oovv)
            + einsum("jk,ijka->ai", tau82, h.v.ooov)
            + einsum("jk,jika->ai", tau72, h.v.ooov)
            - einsum("bj,jiab->ai", a.t1, tau93) / 2
            + 2 * einsum("ja,ji->ai", tau14, tau52)
            + einsum("bj,ijab->ai", a.t1, tau75)
            + einsum("jkba,ijkb->ai", tau4, h.v.ooov)
            - 2 * einsum("jkba,ijkb->ai", tau108, h.v.ooov)
            + einsum("bj,ijba->ai", a.t1, tau116)
            + einsum("ib,ba->ai", tau14, tau41)
            + 2 * einsum("bj,jiab->ai", a.t1, tau129)
            + einsum("ba,ib->ai", tau19, tau2)
            + einsum("ba,ib->ai", tau18, tau2)
            - einsum("bj,ijab->ai", a.t1, tau54)
            + einsum("bj,ijba->ai", a.t1, tau130)
            + einsum("bj,ijba->ai", a.t1, tau123)
            + einsum("jikb,bajk->ai", tau43, a.z2)
            - einsum("jbca,bcij->ai", tau114, a.z2)
            - 2 * einsum("ba,bi->ai", tau131, a.z1)
            + einsum("jkba,jikb->ai", tau77, h.v.ooov)
            - einsum("ja,ji->ai", tau14, tau65) / 2
            - einsum("jkib,bajk->ai", tau58, a.z2)
            + einsum("jikb,abjk->ai", tau35, a.z2)
            + 2 * einsum("jk,ijka->ai", tau52, h.v.ooov)
            - einsum("ib,ba->ai", h.f.ov, tau36)
            - 2 * einsum("jikb,abkj->ai", tau35, a.z2)
            + einsum("bj,jiab->ai", a.t1, tau132)
            - einsum("jibc,jcba->ai", tau21, h.v.ovvv)
            - 2 * einsum("jk,jika->ai", tau82, h.v.ooov)
            - 2 * einsum("bj,jiab->ai", a.t1, tau105)
            - einsum("bj,jiab->ai", a.t1, tau133)
            + einsum("bj,ijab->ai", a.t1, tau128)
            + einsum("ji,ja->ai", tau0, tau14)
            + einsum("jkba,ijkb->ai", tau32, h.v.ooov)
            + einsum("jbca,bcij->ai", tau46, a.z2) / 2
            + einsum("jk,ijka->ai", tau13, h.v.ooov)
            + einsum("jkli,jkla->ai", tau100, h.v.ooov)
            - einsum("jibc,jcab->ai", tau15, h.v.ovvv)
            + einsum("jkib,bakj->ai", tau58, a.z2) / 2
            - einsum("jk,ijka->ai", tau65, h.v.ooov) / 2
            + einsum("jkib,abjk->ai", tau58, a.z2) / 2
            + einsum("bj,ijba->ai", a.t1, tau10)
            + einsum("bj,ijba->ai", a.t1, tau111)
            - einsum("ib,ba->ai", tau14, tau18) / 2
            - 2 * einsum("bj,jiab->ai", a.t1, tau78)
            + einsum("jb,jiab->ai", tau87, h.v.oovv)
            - einsum("jbca,bcij->ai", tau125, a.z2)
            - einsum("jkli,jkla->ai", tau9, h.v.ooov) / 2
            - einsum("bj,ijab->ai", a.t1, tau116) / 2
            + einsum("bj,ijba->ai", a.t1, tau134)
            - einsum("jb,jiab->ai", tau67, h.v.oovv) / 2
            - einsum("bj,ijba->ai", a.t1, tau126) / 2
            - einsum("bj,jiab->ai", a.t1, tau60) / 2
            - einsum("jkib,bakj->ai", tau76, a.z2)
            - 4 * einsum("ji,aj->ai", tau135, a.z1)
            + einsum("ib,ba->ai", tau2, tau49)
            + einsum("bj,ijba->ai", a.t1, tau68)
            - einsum("jb,jiab->ai", tau50, h.v.oovv) / 2
            - einsum("bj,jiab->ai", a.t1, tau136) / 2
            - 2 * einsum("jk,jika->ai", tau0, h.v.ooov)
            - einsum("bj,jiab->ai", a.t1, tau104) / 2
            - einsum("jk,ijka->ai", tau31, h.v.ooov) / 2
            - einsum("jibc,jcba->ai", tau56, h.v.ovvv)
            + 2 * einsum("bj,jiba->ai", a.t1, tau119)
            + einsum("bj,ijba->ai", a.t1, tau86)
            + einsum("kj,jkia->ai", tau0, tau1)
            - einsum("bc,icab->ai", tau48, h.v.ovvv)
            + einsum("kj,jkia->ai", tau13, tau1)
            + einsum("ib,ba->ai", h.f.ov, tau18) / 2
            + einsum("jk,ijka->ai", tau0, h.v.ooov)
            - 2 * einsum("bj,jiab->ai", a.t1, tau128)
            + einsum("jkba,ijkb->ai", tau30, h.v.ooov)
            - einsum("jkba,jikb->ai", tau99, h.v.ooov) / 2
            + einsum("ja,ji->ai", h.f.ov, tau72) / 2
            + einsum("ib,ba->ai", tau2, tau48)
            + einsum("ja,ji->ai", tau14, tau82)
            - einsum("bj,ijba->ai", a.t1, tau75) / 2
            + einsum("jb,jiba->ai", tau127, h.v.oovv)
            + 2 * einsum("bc,icab->ai", tau36, h.v.ovvv)
            + einsum("ji,aj->ai", tau137, a.z1)
            - einsum("jkba,ijkb->ai", tau15, h.v.ooov) / 2
            - 2 * einsum("ib,ba->ai", tau2, tau36)
            - einsum("bj,jiab->ai", a.t1, tau12)
            + 2 * einsum("bj,jiab->ai", a.t1, tau138)
            + einsum("bj,jiab->ai", a.t1, tau130)
            - 2 * einsum("ba,bi->ai", tau139, a.z1)
            + einsum("jbca,bcji->ai", tau125, a.z2) / 2
            + einsum("bc,icba->ai", tau48, h.v.ovvv) / 2
            + einsum("ba,bi->ai", tau140, a.z1)
            + 2 * einsum("jibc,jcba->ai", tau44, h.v.ovvv)
            + einsum("bj,jiab->ai", a.t1, tau45)
            - einsum("jikb,abjk->ai", tau80, a.z2) / 2
            - einsum("bj,ijba->ai", a.t1, tau107) / 2
            + einsum("bj,ijba->ai", a.t1, tau141)
            - einsum("bj,ijba->ai", a.t1, tau57) / 2
            + einsum("bj,jiab->ai", a.t1, tau81)
            + einsum("bj,jiba->ai", a.t1, tau101)
            - einsum("bc,icba->ai", tau41, h.v.ovvv)
            + einsum("bj,ijab->ai", a.t1, tau101)
            + 2 * einsum("bj,jiba->ai", a.t1, tau122)
            + einsum("bj,jiab->ai", a.t1, tau10)
            + einsum("ib,ba->ai", tau14, tau70)
            - 2 * einsum("ba,bi->ai", tau142, a.z1)
            - einsum("jikb,bajk->ai", tau84, a.z2) / 2
            + einsum("ib,ba->ai", h.f.ov, tau19) / 2
            - 2 * einsum("bj,jiab->ai", a.t1, tau110)
            + einsum("kj,jika->ai", tau65, tau1)
            - 4 * einsum("jk,jika->ai", tau52, h.v.ooov)
            + einsum("bj,jiab->ai", a.t1, tau143)
            + einsum("bj,ijba->ai", a.t1, tau132)
            - einsum("bc,icba->ai", tau3, h.v.ovvv)
            - einsum("bj,jiab->ai", a.t1, tau122)
            - 2 * einsum("ij,aj->ai", h.f.oo, a.z1)
            + 2 * einsum("bj,ijab->ai", a.t1, tau12)
            + einsum("jkib,abkj->ai", tau76, a.z2) / 2
            - 2 * einsum("jkba,ijkb->ai", tau51, h.v.ooov)
            + einsum("bj,ijba->ai", a.t1, tau143)
            - einsum("jbca,cbji->ai", tau114, a.z2)
            - einsum("jibc,jcab->ai", tau77, h.v.ovvv)
            + einsum("bj,jiba->ai", a.t1, tau128)
            - einsum("bj,ijba->ai", a.t1, tau136) / 2
            - einsum("jibc,jcab->ai", tau44, h.v.ovvv)
            - 2 * einsum("bj,ijba->ai", a.t1, tau144)
            - 2 * einsum("bj,jiab->ai", a.t1, tau120)
            + 2 * einsum("jbca,cbij->ai", tau114, a.z2)
            - 4 * einsum("jk,kija->ai", tau52, tau1)
            - einsum("ib,ba->ai", tau14, tau48) / 2
            - einsum("bj,ijba->ai", a.t1, tau122)
            - einsum("jbca,cbji->ai", tau125, a.z2)
            + einsum("kj,jkia->ai", tau82, tau1)
            - einsum("ib,ba->ai", h.f.ov, tau3)
            - einsum("bj,jiba->ai", a.t1, tau10) / 2
        )

        rz2 = (
            - einsum("jiab->abij", h.v.oovv)
            + 2 * einsum("jiba->abij", h.v.oovv)
            + einsum("ijab->abij", tau193) / 2
            - einsum("ijba->abij", tau193) / 4
            - einsum("jiab->abij", tau193) / 4
            + einsum("jiba->abij", tau193) / 2
            - einsum("ijab->abij", tau218) / 4
            + einsum("ijba->abij", tau218) / 2
            + einsum("jiab->abij", tau218) / 2
            - einsum("jiba->abij", tau218) / 4
        )

        return self.types.RESIDUALS_TYPE(rt1, rz1, rt2, rz2)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """

        return self.types.AMPLITUDES_TYPE(
            t1=g.gt1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full'),
            t2=(2 * g.gt2 + g.gt2.transpose([0, 1, 3, 2])
                ) / (- 3) * cc_denom(h.f, 4, 'dir', 'full'),
            z1=g.gz1 / (- 2) * cc_denom(h.f, 2, 'dir', 'full'),
            z2=(2 * g.gz2 + g.gz2.transpose([0, 1, 3, 2])
                ) / (- 3) * cc_denom(h.f, 4, 'dir', 'full')

        )


def test_root_solver():
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
    rhf.scf()  # -74.963063129719586

    from tcc.cc_solvers import root_solver
    from tcc.cc_solvers import classic_solver, residual_diis_solver
    from tcc.rccsd_dir_ext import RCCSD_EXT
    from tcc.rccsd import RCCSD_UNIT
    cc1 = RCCSD_EXT(rhf)
    cc2 = RCCSD_UNIT(rhf)

    converged1, energy1, amps1 = root_solver(cc1)
    converged2, energy2, amps2 = root_solver(cc2)

    print('|t1| = {}'.format(
        np.linalg.norm((amps1[0] - amps2[0]).flatten())
    ))
    print('|t2| = {}'.format(
        np.linalg.norm((amps1[2] - amps2[1]).flatten())
    ))

    ampi = cc1.types.AMPLITUDES_TYPE(
        t1=amps2.t1,
        t2=amps2.t2,
        z1=np.zeros_like(amps2.t1),
        z2=np.zeros_like(amps2.t2)
    )

    converged1, energy1, amps = classic_solver(
        cc1, conv_tol_energy=-1, amps=ampi, max_cycle=1)

    print('|z1| = {}'.format(
        np.linalg.norm((amps[1] - amps1[1]).flatten())
    ))
    print('|z2| = {}'.format(
        np.linalg.norm((amps[3] - amps1[3]).flatten())
    ))


if __name__ == '__main__':
    test_root_solver()
