import numpy as np
from numpy import einsum
from tcc.cc_solvers import CC
from tcc.denom import cc_denom
from collections import namedtuple
from types import SimpleNamespace
from tcc.utils import cpd_initialize

class RCCSD_CPD(CC):
    """
    This class implements classic RCCSD method
    with CPD decomposed amplitudes
    """
    types = SimpleNamespace()

    def __init__(self, mf, frozen=[], mo_energy=None, mo_coeff=None,
                 mo_occ=None, rankt=None):
        """
        Initialize RCCSD
        :param rankt: rank of the CPD decomposition of amplitudes
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

        # initialize sizes

        if rankt is None:
            self.rankt = np.min((self._mos.nocc, self._mos.nvir))
        else:
            self.rankt = rankt

        # Add some type definitions
        self.types.AMPLITUDES_TYPE = namedtuple(
            'RCCSD_AMPLITUDES',
            field_names=('t1', 'z1',
                         'x1', 'x2', 'x3', 'x4',
                         'y1', 'y2', 'y3', 'y4'))

        self.types.RHS_TYPE = namedtuple(
            'RCCSD_RHS',
            field_names=('gt1', 'gz1',
                         'gx1', 'gx2', 'gx3', 'gx4',
                         'gy1', 'gy2', 'gy3', 'gy4'))

        self.types.RESIDUALS_TYPE = namedtuple(
            'RCCSD_RESIDUALS',
            field_names=('rt1', 'rz1',
                         'rx1', 'rx2', 'rx3', 'rx4',
                         'ry1', 'ry2', 'ry3', 'ry4'))

    @property
    def mos(self):
        '''
        MOs object
        '''
        return self._mos

    @property
    def method_name(self):
        return 'RCCSD_CPD'

    def create_ham(self):
        """
        Create full Hamiltonian (in core)
        """
        from tcc.interaction import HAM_SPINLESS_RI_CORE
        return HAM_SPINLESS_RI_CORE(self)

    def init_amplitudes(self, ham):
        """
        Initialize amplitudes from interaction
        """
        e_ai = cc_denom(ham.f, 2, 'dir', 'full')

        t1 = ham.f.ov.transpose().conj() * (- e_ai)
        z1 = ham.f.ov.transpose() * (- e_ai)

        no = self.mos.nocc
        nv = self.mos.nvir

        xs = cpd_initialize((nv, ) * 2 + (no, ) * 2, self.rankt)
        ys = cpd_initialize((nv, ) * 2 + (no, ) * 2, self.rankt)

        return self.types.AMPLITUDES_TYPE(t1, z1, *xs, *ys)


    def calculate_energy(self, h, a):
        """
        Calculate RCCSD energy
        Automatically generated
        """
        tau0 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )

        tau1 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )

        tau2 = (
            einsum("oia,ojb->ijab", h.l.pov, h.l.pov)
        )

        tau3 = (
            einsum("bj,ijba->ia", a.t1, tau2)
        )

        tau4 = (
            einsum("bp,ijab->pija", a.x2, tau2)
        )

        tau5 = (
            einsum("ap,pija->pij", a.x1, tau4)
        )

        tau6 = (
            einsum("jp,pji->pi", a.x3, tau5)
        )

        tau7 = (
            einsum("bp,ijab->pija", a.x1, tau2)
        )

        tau8 = (
            einsum("ap,pija->pij", a.x2, tau7)
        )

        tau9 = (
            einsum("jp,pji->pi", a.x3, tau8)
        )

        energy = (
            2 * einsum("o,o->", tau0, tau1)
            - einsum("ai,ia->", a.t1, tau3)
            + 2 * einsum("pi,ip->", tau6, a.x4)
            - einsum("pi,ip->", tau9, a.x4)
            + 2 * einsum("ia,ai->", h.f.ov, a.t1)
        )

        return energy


    def calc_residuals(self, h, a):
        """
        Calculates CC residuals for CC equations
        """
        
        tau0 = (
            einsum("oia,ojb->ijab", h.l.pov, h.l.pov)
        )
    
        tau1 = (
            einsum("bp,ijab->pija", a.x1, tau0)
        )
    
        tau2 = (
            einsum("ai,pjka->pijk", a.t1, tau1)
        )
    
        tau3 = (
            einsum("kp,pijk->pij", a.x3, tau2)
        )
    
        tau4 = (
            einsum("jp,pij->pi", a.x4, tau3)
        )
    
        tau5 = (
            einsum("ia,ap->pi", h.f.ov, a.x2)
        )
    
        tau6 = (
            einsum("pi,ip->p", tau5, a.x3)
        )
    
        tau7 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )
    
        tau8 = (
            einsum("oij,oka->ijka", h.l.poo, h.l.pov)
        )
    
        tau9 = (
            einsum("ap,ijka->pijk", a.x1, tau8)
        )
    
        tau10 = (
            einsum("kp,pijk->pij", a.x3, tau9)
        )
    
        tau11 = (
            einsum("jp,pji->pi", a.x4, tau10)
        )
    
        tau12 = (
            einsum("ia,aj->ij", h.f.ov, a.t1)
        )
    
        tau13 = (
            einsum("o,oia->ia", tau7, h.l.pov)
        )
    
        tau14 = (
            einsum("ai,ja->ij", a.t1, tau13)
        )
    
        tau15 = (
            einsum("ia,ap->pi", tau13, a.x1)
        )
    
        tau16 = (
            einsum("pi,ip->p", tau15, a.x3)
        )
    
        tau17 = (
            einsum("bj,jiab->ia", a.t1, tau0)
        )
    
        tau18 = (
            einsum("ia,ap->pi", tau17, a.x2)
        )
    
        tau19 = (
            einsum("pi,ip->p", tau18, a.x4)
        )
    
        tau20 = (
            einsum("ap,pija->pij", a.x2, tau1)
        )
    
        tau21 = (
            einsum("jp,pij->pi", a.x4, tau20)
        )
    
        tau22 = (
            einsum("ai,pi->pa", a.t1, tau21)
        )
    
        tau23 = (
            einsum("oia,obc->iabc", h.l.pov, h.l.pvv)
        )
    
        tau24 = (
            einsum("cp,iabc->piab", a.x2, tau23)
        )
    
        tau25 = (
            einsum("bp,piba->pia", a.x1, tau24)
        )
    
        tau26 = (
            einsum("ip,pia->pa", a.x4, tau25)
        )
    
        tau27 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )
    
        tau28 = (
            einsum("o,oij->ij", tau27, h.l.poo)
        )
    
        tau29 = (
            einsum("pi,ip->p", tau18, a.x3)
        )
    
        tau30 = (
            einsum("ia,ap->pi", tau17, a.x1)
        )
    
        tau31 = (
            einsum("pi,ip->p", tau30, a.x3)
        )
    
        tau32 = (
            einsum("bp,ijab->pija", a.x2, tau0)
        )
    
        tau33 = (
            einsum("ai,pjka->pijk", a.t1, tau32)
        )
    
        tau34 = (
            einsum("kp,pijk->pij", a.x3, tau33)
        )
    
        tau35 = (
            einsum("jp,pij->pi", a.x4, tau34)
        )
    
        tau36 = (
            einsum("ap,ijka->pijk", a.x2, tau8)
        )
    
        tau37 = (
            einsum("kp,pijk->pij", a.x3, tau36)
        )
    
        tau38 = (
            einsum("jp,pji->pi", a.x4, tau37)
        )
    
        tau39 = (
            einsum("kp,pijk->pij", a.x4, tau36)
        )
    
        tau40 = (
            einsum("jp,pji->pi", a.x3, tau39)
        )
    
        tau41 = (
            einsum("ak,kija->ij", a.t1, tau8)
        )
    
        tau42 = (
            einsum("ap,pija->pij", a.x1, tau32)
        )
    
        tau43 = (
            einsum("jp,pij->pi", a.x4, tau42)
        )
    
        tau44 = (
            einsum("ai,pi->pa", a.t1, tau43)
        )
    
        tau45 = (
            einsum("ia,ap->pi", h.f.ov, a.x1)
        )
    
        tau46 = (
            einsum("pi,ip->p", tau45, a.x4)
        )
    
        tau47 = (
            einsum("kp,pijk->pij", a.x4, tau9)
        )
    
        tau48 = (
            einsum("jp,pji->pi", a.x3, tau47)
        )
    
        tau49 = (
            einsum("kp,pijk->pij", a.x4, tau33)
        )
    
        tau50 = (
            einsum("jp,pij->pi", a.x3, tau49)
        )
    
        tau51 = (
            einsum("oij,oab->ijab", h.l.poo, h.l.pvv)
        )
    
        tau52 = (
            einsum("o,oab->ab", tau7, h.l.pvv)
        )
    
        tau53 = (
            einsum("jp,pij->pi", a.x3, tau42)
        )
    
        tau54 = (
            einsum("ai,pi->pa", a.t1, tau53)
        )
    
        tau55 = (
            einsum("ip,pia->pa", a.x3, tau25)
        )
    
        tau56 = (
            einsum("jp,pij->pi", a.x3, tau20)
        )
    
        tau57 = (
            einsum("ai,pi->pa", a.t1, tau56)
        )
    
        tau58 = (
            einsum("pi,ip->p", tau45, a.x3)
        )
    
        tau59 = (
            einsum("pi,ip->p", tau15, a.x4)
        )
    
        tau60 = (
            einsum("ai,ja->ij", a.t1, tau17)
        )
    
        tau61 = (
            einsum("cp,iabc->piab", a.x1, tau23)
        )
    
        tau62 = (
            einsum("bp,piba->pia", a.x2, tau61)
        )
    
        tau63 = (
            einsum("ip,pia->pa", a.x4, tau62)
        )
    
        tau64 = (
            einsum("ip,pia->pa", a.x3, tau62)
        )
    
        tau65 = (
            einsum("ia,ap->pi", tau13, a.x2)
        )
    
        tau66 = (
            einsum("pi,ip->p", tau65, a.x4)
        )
    
        tau67 = (
            einsum("ci,iabc->ab", a.t1, tau23)
        )
    
        tau68 = (
            einsum("kp,pijk->pij", a.x4, tau2)
        )
    
        tau69 = (
            einsum("jp,pij->pi", a.x3, tau68)
        )
    
        tau70 = (
            einsum("pi,ip->p", tau5, a.x4)
        )
    
        tau71 = (
            einsum("pi,ip->p", tau30, a.x4)
        )
    
        tau72 = (
            einsum("pi,ip->p", tau65, a.x3)
        )
    
        tau73 = (
            einsum("ai,ap->pi", a.t1, a.y2)
        )
    
        tau74 = (
            einsum("ci,jabc->ijab", a.t1, tau23)
        )
    
        tau75 = (
            einsum("bi,jkba->ijka", a.t1, tau74)
        )
    
        tau76 = (
            einsum("kp,kija->pija", a.y4, tau75)
        )
    
        tau77 = (
            einsum("jp,pjia->pia", a.y3, tau76)
        )
    
        tau78 = (
            einsum("ap,aq->pq", a.x2, a.y2)
        )
    
        tau79 = (
            einsum("ip,iq->pq", a.x3, a.y3)
        )
    
        tau80 = (
            einsum("ij,jp->pi", tau41, a.x4)
        )
    
        tau81 = (
            einsum("pi,iq->pq", tau80, a.y4)
        )
    
        tau82 = (
            einsum("ip,iq->pq", a.x3, a.y4)
        )
    
        tau83 = (
            einsum("jq,pij->pqi", a.y3, tau47)
        )
    
        tau84 = (
            einsum("qi,pqi->pq", tau73, tau83)
        )
    
        tau85 = (
            einsum("pi,qi->pq", tau5, tau73)
        )
    
        tau86 = (
            einsum("ip,iq->pq", a.x4, a.y4)
        )
    
        tau87 = (
            einsum("ap,aq->pq", a.x1, a.y2)
        )
    
        tau88 = (
            einsum("ip,iq->pq", a.x4, a.y3)
        )
    
        tau89 = (
            einsum("kq,pikj->pqij", a.y4, tau36)
        )
    
        tau90 = (
            einsum("jp,pqji->pqi", a.x3, tau89)
        )
    
        tau91 = (
            einsum("ai,pqi->pqa", a.t1, tau90)
        )
    
        tau92 = (
            einsum("aq,pija->pqij", a.x2, tau1)
        )
    
        tau93 = (
            einsum("jp,pqij->pqi", a.x4, tau92)
        )
    
        tau94 = (
            einsum("ip,pqi->pq", a.x3, tau93)
        )
    
        tau95 = (
            einsum("rq,rq,rq,pr->pq", tau82, tau87, tau88, tau94)
        )
    
        tau96 = (
            einsum("ij,jp->pi", h.f.oo, a.y4)
        )
    
        tau97 = (
            einsum("qi,ip->pq", tau96, a.x4)
        )
    
        tau98 = (
            einsum("kq,pikj->pqij", a.y3, tau9)
        )
    
        tau99 = (
            einsum("jp,pqji->pqi", a.x3, tau98)
        )
    
        tau100 = (
            einsum("qi,pqi->pq", tau73, tau99)
        )
    
        tau101 = (
            einsum("jq,pij->pqi", a.x4, tau42)
        )
    
        tau102 = (
            einsum("ip,pqi->pq", a.x3, tau101)
        )
    
        tau103 = (
            einsum("rp,rq->pq", tau102, tau88)
        )
    
        tau104 = (
            einsum("jq,pij->pqi", a.y4, tau10)
        )
    
        tau105 = (
            einsum("ai,pqi->pqa", a.t1, tau104)
        )
    
        tau106 = (
            einsum("ap,qija->pqij", a.x2, tau32)
        )
    
        tau107 = (
            einsum("jp,qpij->pqi", a.x4, tau106)
        )
    
        tau108 = (
            einsum("ip,qpi->pq", a.x3, tau107)
        )
    
        tau109 = (
            einsum("pr,rq,rq->pq", tau108, tau82, tau87)
        )
    
        tau110 = (
            einsum("ij,jp->pi", tau60, a.x4)
        )
    
        tau111 = (
            einsum("pi,iq->pq", tau110, a.y4)
        )
    
        tau112 = (
            einsum("ai,jkla->ijkl", a.t1, tau8)
        )
    
        tau113 = (
            einsum("lp,ijkl->pijk", a.x4, tau112)
        )
    
        tau114 = (
            einsum("kq,pijk->pqij", a.y4, tau113)
        )
    
        tau115 = (
            einsum("jp,pqij->pqi", a.x3, tau114)
        )
    
        tau116 = (
            einsum("iq,pqi->pq", a.y3, tau115)
        )
    
        tau117 = (
            einsum("pi,qi->pq", tau45, tau73)
        )
    
        tau118 = (
            einsum("rp,rq,rq->pq", tau108, tau86, tau87)
        )
    
        tau119 = (
            einsum("bp,ijba->pija", a.x1, tau0)
        )
    
        tau120 = (
            einsum("ai,pjka->pijk", a.t1, tau119)
        )
    
        tau121 = (
            einsum("kp,pijk->pij", a.x4, tau120)
        )
    
        tau122 = (
            einsum("jq,pji->pqi", a.y3, tau121)
        )
    
        tau123 = (
            einsum("qi,pqi->pq", tau73, tau122)
        )
    
        tau124 = (
            einsum("pi,iq->pq", tau80, a.y3)
        )
    
        tau125 = (
            einsum("ij,jp->pi", tau28, a.y3)
        )
    
        tau126 = (
            einsum("qi,ip->pq", tau125, a.x4)
        )
    
        tau127 = (
            einsum("bj,ijba->ia", a.t1, tau0)
        )
    
        tau128 = (
            einsum("ia,ap->pi", tau127, a.x2)
        )
    
        tau129 = (
            einsum("pi,qi->pq", tau128, tau73)
        )
    
        tau130 = (
            einsum("iq,pqi->pq", a.x3, tau93)
        )
    
        tau131 = (
            einsum("pr,rq,rq->pq", tau130, tau86, tau87)
        )
    
        tau132 = (
            einsum("ap,qija->pqij", a.x1, tau32)
        )
    
        tau133 = (
            einsum("jp,pqij->pqi", a.x4, tau132)
        )
    
        tau134 = (
            einsum("iq,pqi->pq", a.x4, tau133)
        )
    
        tau135 = (
            einsum("rp,rq,rq->pq", tau134, tau78, tau82)
        )
    
        tau136 = (
            einsum("oia,obj->ijab", h.l.pov, h.l.pvo)
        )
    
        tau137 = (
            einsum("bi,jkba->ijka", a.t1, tau136)
        )
    
        tau138 = (
            einsum("kp,ijka->pija", a.y4, tau137)
        )
    
        tau139 = (
            einsum("jp,pjia->pia", a.y3, tau138)
        )
    
        tau140 = (
            einsum("lp,ijkl->pijk", a.x3, tau112)
        )
    
        tau141 = (
            einsum("kq,pijk->pqij", a.y4, tau140)
        )
    
        tau142 = (
            einsum("jp,pqij->pqi", a.x4, tau141)
        )
    
        tau143 = (
            einsum("iq,pqi->pq", a.y3, tau142)
        )
    
        tau144 = (
            einsum("pi,iq->pq", tau21, a.x3)
        )
    
        tau145 = (
            einsum("rp,rq->pq", tau144, tau79)
        )
    
        tau146 = (
            einsum("bp,ijab->pija", a.y2, tau136)
        )
    
        tau147 = (
            einsum("ap,qija->pqij", a.x1, tau146)
        )
    
        tau148 = (
            einsum("jq,pqij->pqi", a.y3, tau147)
        )
    
        tau149 = (
            einsum("ip,pqi->pq", a.x4, tau148)
        )
    
        tau150 = (
            einsum("rq,rp->pq", tau78, tau94)
        )
    
        tau151 = (
            einsum("jq,pqij->pqi", a.x4, tau132)
        )
    
        tau152 = (
            einsum("iq,pqi->pq", a.x3, tau151)
        )
    
        tau153 = (
            einsum("pr,rq->pq", tau152, tau87)
        )
    
        tau154 = (
            einsum("rp,rq,rq->pq", tau130, tau78, tau82)
        )
    
        tau155 = (
            einsum("kq,pijk->pqij", a.y3, tau113)
        )
    
        tau156 = (
            einsum("jp,pqij->pqi", a.x3, tau155)
        )
    
        tau157 = (
            einsum("iq,pqi->pq", a.y4, tau156)
        )
    
        tau158 = (
            einsum("jq,pji->pqi", a.y3, tau68)
        )
    
        tau159 = (
            einsum("ai,pqi->pqa", a.t1, tau158)
        )
    
        tau160 = (
            einsum("bi,pjba->pija", a.t1, tau61)
        )
    
        tau161 = (
            einsum("jp,pija->pia", a.x4, tau160)
        )
    
        tau162 = (
            einsum("iq,pia->pqa", a.y4, tau161)
        )
    
        tau163 = (
            einsum("lp,ijlk->pijk", a.y3, tau112)
        )
    
        tau164 = (
            einsum("kp,pkij->pij", a.y4, tau163)
        )
    
        tau165 = (
            einsum("pj,pij->pi", tau73, tau164)
        )
    
        tau166 = (
            einsum("bi,jkab->ijka", a.t1, tau0)
        )
    
        tau167 = (
            einsum("ai,jkla->ijkl", a.t1, tau166)
        )
    
        tau168 = (
            einsum("lp,ijkl->pijk", a.x4, tau167)
        )
    
        tau169 = (
            einsum("kp,pijk->pij", a.x3, tau168)
        )
    
        tau170 = (
            einsum("jq,pij->pqi", a.y4, tau169)
        )
    
        tau171 = (
            einsum("iq,pqi->pq", a.y3, tau170)
        )
    
        tau172 = (
            einsum("cp,icab->piab", a.x2, tau23)
        )
    
        tau173 = (
            einsum("bi,pjab->pija", a.t1, tau172)
        )
    
        tau174 = (
            einsum("jp,pija->pia", a.x4, tau173)
        )
    
        tau175 = (
            einsum("iq,pia->pqa", a.y4, tau174)
        )
    
        tau176 = (
            einsum("bq,piab->pqia", a.y2, tau24)
        )
    
        tau177 = (
            einsum("ai,pqja->pqij", a.t1, tau176)
        )
    
        tau178 = (
            einsum("jp,pqij->pqi", a.x3, tau177)
        )
    
        tau179 = (
            einsum("iq,pqi->pq", a.y3, tau178)
        )
    
        tau180 = (
            einsum("qi,pia->pqa", tau73, tau62)
        )
    
        tau181 = (
            einsum("cp,iacb->piab", a.y2, tau23)
        )
    
        tau182 = (
            einsum("bp,qiba->pqia", a.x2, tau181)
        )
    
        tau183 = (
            einsum("ai,pqja->pqij", a.t1, tau182)
        )
    
        tau184 = (
            einsum("jp,pqij->pqi", a.x3, tau183)
        )
    
        tau185 = (
            einsum("iq,pqi->pq", a.y4, tau184)
        )
    
        tau186 = (
            einsum("ij,jp->pi", h.f.oo, a.y3)
        )
    
        tau187 = (
            einsum("qi,ip->pq", tau186, a.x4)
        )
    
        tau188 = (
            einsum("bp,qiba->pqia", a.x1, tau181)
        )
    
        tau189 = (
            einsum("ai,pqja->pqij", a.t1, tau188)
        )
    
        tau190 = (
            einsum("jp,pqij->pqi", a.x4, tau189)
        )
    
        tau191 = (
            einsum("iq,pqi->pq", a.y3, tau190)
        )
    
        tau192 = (
            einsum("bp,ijba->pija", a.x1, tau136)
        )
    
        tau193 = (
            einsum("jq,pija->pqia", a.y3, tau192)
        )
    
        tau194 = (
            einsum("ip,pqia->pqa", a.x3, tau193)
        )
    
        tau195 = (
            einsum("ba,bp->pa", tau52, a.y2)
        )
    
        tau196 = (
            einsum("qa,ap->pq", tau195, a.x1)
        )
    
        tau197 = (
            einsum("ba,bp->pa", tau67, a.x1)
        )
    
        tau198 = (
            einsum("kq,pikj->pqij", a.y3, tau36)
        )
    
        tau199 = (
            einsum("jp,pqji->pqi", a.x4, tau198)
        )
    
        tau200 = (
            einsum("qi,pqi->pq", tau73, tau199)
        )
    
        tau201 = (
            einsum("bq,piab->pqia", a.y2, tau61)
        )
    
        tau202 = (
            einsum("ai,pqja->pqij", a.t1, tau201)
        )
    
        tau203 = (
            einsum("jp,pqij->pqi", a.x3, tau202)
        )
    
        tau204 = (
            einsum("iq,pqi->pq", a.y4, tau203)
        )
    
        tau205 = (
            einsum("rp,rq->pq", tau102, tau86)
        )
    
        tau206 = (
            einsum("ij,jp->pi", tau14, a.x3)
        )
    
        tau207 = (
            einsum("pi,iq->pq", tau206, a.y4)
        )
    
        tau208 = (
            einsum("ip,pqi->pq", a.x3, tau133)
        )
    
        tau209 = (
            einsum("rp,rq->pq", tau208, tau78)
        )
    
        tau210 = (
            einsum("ia,ap->pi", tau127, a.x1)
        )
    
        tau211 = (
            einsum("pi,qi->pq", tau210, tau73)
        )
    
        tau212 = (
            einsum("ai,pi->pa", a.t1, tau65)
        )
    
        tau213 = (
            einsum("jq,pji->pqi", a.y4, tau3)
        )
    
        tau214 = (
            einsum("ai,pqi->pqa", a.t1, tau213)
        )
    
        tau215 = (
            einsum("iq,pqi->pq", a.x3, tau133)
        )
    
        tau216 = (
            einsum("rp,rq,rq->pq", tau215, tau78, tau79)
        )
    
        tau217 = (
            einsum("ij,jp->pi", tau14, a.x4)
        )
    
        tau218 = (
            einsum("pi,iq->pq", tau217, a.y4)
        )
    
        tau219 = (
            einsum("qj,pji->pqi", tau73, tau42)
        )
    
        tau220 = (
            einsum("ai,pqi->pqa", a.t1, tau219)
        )
    
        tau221 = (
            einsum("jq,pij->pqi", a.y4, tau37)
        )
    
        tau222 = (
            einsum("ai,pqi->pqa", a.t1, tau221)
        )
    
        tau223 = (
            einsum("ab,bp->pa", h.f.vv, a.x1)
        )
    
        tau224 = (
            einsum("bp,ijba->pija", a.x2, tau0)
        )
    
        tau225 = (
            einsum("ai,pjka->pijk", a.t1, tau224)
        )
    
        tau226 = (
            einsum("kp,pijk->pij", a.x3, tau225)
        )
    
        tau227 = (
            einsum("jq,pji->pqi", a.y3, tau226)
        )
    
        tau228 = (
            einsum("ai,pqi->pqa", a.t1, tau227)
        )
    
        tau229 = (
            einsum("jp,pqij->pqi", a.x3, tau132)
        )
    
        tau230 = (
            einsum("iq,pqi->pq", a.x3, tau229)
        )
    
        tau231 = (
            einsum("rp,rq,rq->pq", tau230, tau78, tau88)
        )
    
        tau232 = (
            einsum("jq,pij->pqi", a.y4, tau47)
        )
    
        tau233 = (
            einsum("ai,pqi->pqa", a.t1, tau232)
        )
    
        tau234 = (
            einsum("pi,iq->pq", tau217, a.y3)
        )
    
        tau235 = (
            einsum("bp,ijab->pija", a.x1, tau51)
        )
    
        tau236 = (
            einsum("aq,pija->pqij", a.y2, tau235)
        )
    
        tau237 = (
            einsum("jq,pqij->pqi", a.y3, tau236)
        )
    
        tau238 = (
            einsum("ip,pqi->pq", a.x3, tau237)
        )
    
        tau239 = (
            einsum("ip,pqi->pq", a.x4, tau101)
        )
    
        tau240 = (
            einsum("rp,rq->pq", tau239, tau79)
        )
    
        tau241 = (
            einsum("ap,qija->pqij", a.x1, tau1)
        )
    
        tau242 = (
            einsum("jp,qpij->pqi", a.x4, tau241)
        )
    
        tau243 = (
            einsum("ip,qpi->pq", a.x3, tau242)
        )
    
        tau244 = (
            einsum("rp,rq,rq->pq", tau243, tau78, tau88)
        )
    
        tau245 = (
            einsum("ip,pqi->pq", a.x3, tau242)
        )
    
        tau246 = (
            einsum("pr,rq,rq,rq->pq", tau245, tau78, tau82, tau88)
        )
    
        tau247 = (
            einsum("cp,icab->piab", a.x1, tau23)
        )
    
        tau248 = (
            einsum("bi,pjab->pija", a.t1, tau247)
        )
    
        tau249 = (
            einsum("jp,pija->pia", a.x3, tau248)
        )
    
        tau250 = (
            einsum("iq,pia->pqa", a.y3, tau249)
        )
    
        tau251 = (
            einsum("jp,pqji->pqi", a.x4, tau98)
        )
    
        tau252 = (
            einsum("ai,pqi->pqa", a.t1, tau251)
        )
    
        tau253 = (
            einsum("ai,pjka->pijk", a.t1, tau146)
        )
    
        tau254 = (
            einsum("kp,pijk->pij", a.y3, tau253)
        )
    
        tau255 = (
            einsum("jp,pji->pi", a.y4, tau254)
        )
    
        tau256 = (
            einsum("oai,obj->ijab", h.l.pvo, h.l.pvo)
        )
    
        tau257 = (
            einsum("bp,ijab->pija", a.y2, tau256)
        )
    
        tau258 = (
            einsum("jp,pija->pia", a.y4, tau257)
        )
    
        tau259 = (
            einsum("oij,okl->ijkl", h.l.poo, h.l.poo)
        )
    
        tau260 = (
            einsum("lp,ijkl->pijk", a.y4, tau259)
        )
    
        tau261 = (
            einsum("kp,pikj->pij", a.y3, tau260)
        )
    
        tau262 = (
            einsum("pj,pij->pi", tau73, tau261)
        )
    
        tau263 = (
            einsum("jq,pij->pqi", a.y3, tau39)
        )
    
        tau264 = (
            einsum("ai,pqi->pqa", a.t1, tau263)
        )
    
        tau265 = (
            einsum("iq,pqi->pq", a.x3, tau101)
        )
    
        tau266 = (
            einsum("rp,rq,rq->pq", tau265, tau82, tau88)
        )
    
        tau267 = (
            einsum("ai,pqi->pqa", a.t1, tau99)
        )
    
        tau268 = (
            einsum("pr,rq,rq,rq->pq", tau208, tau82, tau87, tau88)
        )
    
        tau269 = (
            einsum("jp,pqij->pqi", a.x4, tau106)
        )
    
        tau270 = (
            einsum("ip,pqi->pq", a.x3, tau269)
        )
    
        tau271 = (
            einsum("rp,rq->pq", tau270, tau87)
        )
    
        tau272 = (
            einsum("jp,pqij->pqi", a.x3, tau106)
        )
    
        tau273 = (
            einsum("iq,pqi->pq", a.x3, tau272)
        )
    
        tau274 = (
            einsum("pr,rq,rq->pq", tau273, tau87, tau88)
        )
    
        tau275 = (
            einsum("qi,pqi->pq", tau73, tau213)
        )
    
        tau276 = (
            einsum("pi,qi->pq", tau65, tau73)
        )
    
        tau277 = (
            einsum("jp,pqij->pqi", a.x3, tau189)
        )
    
        tau278 = (
            einsum("iq,pqi->pq", a.y3, tau277)
        )
    
        tau279 = (
            einsum("jp,qpij->pqi", a.x3, tau241)
        )
    
        tau280 = (
            einsum("ip,qpi->pq", a.x3, tau279)
        )
    
        tau281 = (
            einsum("rp,rq,rq->pq", tau280, tau78, tau88)
        )
    
        tau282 = (
            einsum("oab,ocd->abcd", h.l.pvv, h.l.pvv)
        )
    
        tau283 = (
            einsum("dp,abdc->pabc", a.y2, tau282)
        )
    
        tau284 = (
            einsum("ci,pabc->piab", a.t1, tau283)
        )
    
        tau285 = (
            einsum("bi,pjab->pija", a.t1, tau284)
        )
    
        tau286 = (
            einsum("jp,pija->pia", a.y4, tau285)
        )
    
        tau287 = (
            einsum("rp,rq,rq->pq", tau273, tau86, tau87)
        )
    
        tau288 = (
            einsum("oij,oak->ijka", h.l.poo, h.l.pvo)
        )
    
        tau289 = (
            einsum("kp,ijka->pija", a.y4, tau288)
        )
    
        tau290 = (
            einsum("jp,pija->pia", a.y3, tau289)
        )
    
        tau291 = (
            einsum("iq,pqi->pq", a.y4, tau190)
        )
    
        tau292 = (
            einsum("pr,rq,rq,rq->pq", tau270, tau82, tau87, tau88)
        )
    
        tau293 = (
            einsum("qi,ip->pq", tau186, a.x3)
        )
    
        tau294 = (
            einsum("iq,pia->pqa", a.y3, tau174)
        )
    
        tau295 = (
            einsum("ip,pqi->pq", a.x3, tau107)
        )
    
        tau296 = (
            einsum("pr,rq,rq,rq->pq", tau295, tau79, tau86, tau87)
        )
    
        tau297 = (
            einsum("ab,bp->pa", tau52, a.x2)
        )
    
        tau298 = (
            einsum("qi,pia->pqa", tau73, tau25)
        )
    
        tau299 = (
            einsum("jq,pji->pqi", a.y4, tau169)
        )
    
        tau300 = (
            einsum("iq,pqi->pq", a.y3, tau299)
        )
    
        tau301 = (
            einsum("pj,pji->pi", tau73, tau261)
        )
    
        tau302 = (
            einsum("bp,ijba->pija", a.x2, tau136)
        )
    
        tau303 = (
            einsum("jq,pija->pqia", a.y3, tau302)
        )
    
        tau304 = (
            einsum("ip,pqia->pqa", a.x3, tau303)
        )
    
        tau305 = (
            einsum("bi,pjba->pija", a.t1, tau24)
        )
    
        tau306 = (
            einsum("jp,pija->pia", a.x3, tau305)
        )
    
        tau307 = (
            einsum("iq,pia->pqa", a.y4, tau306)
        )
    
        tau308 = (
            einsum("jq,pqij->pqi", a.x3, tau132)
        )
    
        tau309 = (
            einsum("ip,pqi->pq", a.x3, tau308)
        )
    
        tau310 = (
            einsum("pr,rq,rq->pq", tau309, tau87, tau88)
        )
    
        tau311 = (
            einsum("kp,pijk->pij", a.x4, tau225)
        )
    
        tau312 = (
            einsum("jq,pji->pqi", a.y4, tau311)
        )
    
        tau313 = (
            einsum("qi,pqi->pq", tau73, tau312)
        )
    
        tau314 = (
            einsum("ai,pqi->pqa", a.t1, tau122)
        )
    
        tau315 = (
            einsum("pr,rq,rq->pq", tau309, tau86, tau87)
        )
    
        tau316 = (
            einsum("ip,pqi->pq", a.x3, tau151)
        )
    
        tau317 = (
            einsum("pr,rq,rq->pq", tau316, tau79, tau87)
        )
    
        tau318 = (
            einsum("pi,iq->pq", tau43, a.x3)
        )
    
        tau319 = (
            einsum("rp,rq->pq", tau318, tau79)
        )
    
        tau320 = (
            einsum("ij,jp->pi", tau28, a.y4)
        )
    
        tau321 = (
            einsum("qi,ip->pq", tau320, a.x3)
        )
    
        tau322 = (
            einsum("jp,pqij->pqi", a.x4, tau177)
        )
    
        tau323 = (
            einsum("iq,pqi->pq", a.y3, tau322)
        )
    
        tau324 = (
            einsum("jq,pji->pqi", a.y4, tau34)
        )
    
        tau325 = (
            einsum("qi,pqi->pq", tau73, tau324)
        )
    
        tau326 = (
            einsum("jp,pija->pia", a.x4, tau248)
        )
    
        tau327 = (
            einsum("iq,pia->pqa", a.y4, tau326)
        )
    
        tau328 = (
            einsum("jq,pji->pqi", a.y3, tau3)
        )
    
        tau329 = (
            einsum("ai,pqi->pqa", a.t1, tau328)
        )
    
        tau330 = (
            einsum("rp,rq,rq->pq", tau265, tau79, tau86)
        )
    
        tau331 = (
            einsum("rp,rq,rq,rq->pq", tau152, tau78, tau79, tau86)
        )
    
        tau332 = (
            einsum("qj,pij->pqi", tau73, tau42)
        )
    
        tau333 = (
            einsum("ai,pqi->pqa", a.t1, tau332)
        )
    
        tau334 = (
            einsum("ap,ijka->pijk", a.y2, tau288)
        )
    
        tau335 = (
            einsum("kp,pijk->pij", a.y4, tau334)
        )
    
        tau336 = (
            einsum("jp,pij->pi", a.y3, tau335)
        )
    
        tau337 = (
            einsum("rp,rq->pq", tau239, tau82)
        )
    
        tau338 = (
            einsum("ij,jp->pi", tau12, a.y4)
        )
    
        tau339 = (
            einsum("qi,ip->pq", tau338, a.x3)
        )
    
        tau340 = (
            einsum("bp,ijba->pija", a.y2, tau51)
        )
    
        tau341 = (
            einsum("ai,pjka->pijk", a.t1, tau340)
        )
    
        tau342 = (
            einsum("kp,pijk->pij", a.y3, tau341)
        )
    
        tau343 = (
            einsum("jp,pji->pi", a.y4, tau342)
        )
    
        tau344 = (
            einsum("kp,pijk->pij", a.y4, tau341)
        )
    
        tau345 = (
            einsum("jp,pji->pi", a.y3, tau344)
        )
    
        tau346 = (
            einsum("ap,qija->pqij", a.x2, tau146)
        )
    
        tau347 = (
            einsum("jq,pqij->pqi", a.y4, tau346)
        )
    
        tau348 = (
            einsum("ip,pqi->pq", a.x4, tau347)
        )
    
        tau349 = (
            einsum("qi,ip->pq", tau338, a.x4)
        )
    
        tau350 = (
            einsum("jq,pij->pqi", a.x4, tau20)
        )
    
        tau351 = (
            einsum("ip,pqi->pq", a.x3, tau350)
        )
    
        tau352 = (
            einsum("rp,rq->pq", tau351, tau86)
        )
    
        tau353 = (
            einsum("kq,pikj->pqij", a.y4, tau9)
        )
    
        tau354 = (
            einsum("jp,pqji->pqi", a.x4, tau353)
        )
    
        tau355 = (
            einsum("qi,pqi->pq", tau73, tau354)
        )
    
        tau356 = (
            einsum("jq,pij->pqi", a.y3, tau10)
        )
    
        tau357 = (
            einsum("qi,pqi->pq", tau73, tau356)
        )
    
        tau358 = (
            einsum("jq,pqij->pqi", a.y3, tau346)
        )
    
        tau359 = (
            einsum("ip,pqi->pq", a.x4, tau358)
        )
    
        tau360 = (
            einsum("qi,ip->pq", tau320, a.x4)
        )
    
        tau361 = (
            einsum("ab,bp->pa", h.f.vv, a.x2)
        )
    
        tau362 = (
            einsum("jp,pija->pia", a.x3, tau173)
        )
    
        tau363 = (
            einsum("iq,pia->pqa", a.y3, tau362)
        )
    
        tau364 = (
            einsum("ip,pqia->pqa", a.x4, tau303)
        )
    
        tau365 = (
            einsum("bp,ijab->pija", a.x2, tau51)
        )
    
        tau366 = (
            einsum("jq,pija->pqia", a.y3, tau365)
        )
    
        tau367 = (
            einsum("ip,pqia->pqa", a.x3, tau366)
        )
    
        tau368 = (
            einsum("jp,pjia->pia", a.y4, tau285)
        )
    
        tau369 = (
            einsum("ai,pi->pa", a.t1, tau128)
        )
    
        tau370 = (
            einsum("ip,pqia->pqa", a.x4, tau366)
        )
    
        tau371 = (
            einsum("pi,iq->pq", tau43, a.x4)
        )
    
        tau372 = (
            einsum("rp,rq->pq", tau371, tau82)
        )
    
        tau373 = (
            einsum("aq,pija->pqij", a.y2, tau365)
        )
    
        tau374 = (
            einsum("jq,pqij->pqi", a.y4, tau373)
        )
    
        tau375 = (
            einsum("ip,pqi->pq", a.x3, tau374)
        )
    
        tau376 = (
            einsum("qi,pqi->pq", tau73, tau251)
        )
    
        tau377 = (
            einsum("ip,qpi->pq", a.x3, tau269)
        )
    
        tau378 = (
            einsum("pr,rq,rq->pq", tau377, tau79, tau87)
        )
    
        tau379 = (
            einsum("ba,bp->pa", tau67, a.x2)
        )
    
        tau380 = (
            einsum("pa,aq->pq", tau379, a.y2)
        )
    
        tau381 = (
            einsum("iq,pia->pqa", a.y4, tau249)
        )
    
        tau382 = (
            einsum("pr,rq,rq,rq->pq", tau208, tau79, tau86, tau87)
        )
    
        tau383 = (
            einsum("lp,ijlk->pijk", a.y4, tau112)
        )
    
        tau384 = (
            einsum("kp,pkij->pij", a.y3, tau383)
        )
    
        tau385 = (
            einsum("pj,pij->pi", tau73, tau384)
        )
    
        tau386 = (
            einsum("pa,aq->pq", tau361, a.y2)
        )
    
        tau387 = (
            einsum("ip,qpi->pq", a.x4, tau107)
        )
    
        tau388 = (
            einsum("pr,rq,rq->pq", tau387, tau82, tau87)
        )
    
        tau389 = (
            einsum("qi,pqi->pq", tau73, tau263)
        )
    
        tau390 = (
            einsum("ap,pqia->pqi", a.x1, tau176)
        )
    
        tau391 = (
            einsum("ai,pqi->pqa", a.t1, tau390)
        )
    
        tau392 = (
            einsum("iq,pia->pqa", a.y4, tau362)
        )
    
        tau393 = (
            einsum("rp,rq,rq->pq", tau309, tau78, tau86)
        )
    
        tau394 = (
            einsum("ai,pi->pa", a.t1, tau5)
        )
    
        tau395 = (
            einsum("iq,pqi->pq", a.y4, tau277)
        )
    
        tau396 = (
            einsum("jq,pqij->pqi", a.x4, tau92)
        )
    
        tau397 = (
            einsum("iq,pqi->pq", a.x3, tau396)
        )
    
        tau398 = (
            einsum("rp,rq,rq,rq->pq", tau397, tau78, tau79, tau86)
        )
    
        tau399 = (
            einsum("jq,pji->pqi", a.y4, tau121)
        )
    
        tau400 = (
            einsum("ai,pqi->pqa", a.t1, tau399)
        )
    
        tau401 = (
            einsum("pa,aq->pq", tau223, a.y2)
        )
    
        tau402 = (
            einsum("pr,rq,rq,rq->pq", tau295, tau82, tau87, tau88)
        )
    
        tau403 = (
            einsum("rp,rq,rq->pq", tau243, tau78, tau86)
        )
    
        tau404 = (
            einsum("pr,rq,rq->pq", tau134, tau79, tau87)
        )
    
        tau405 = (
            einsum("ab,bp->pa", tau52, a.x1)
        )
    
        tau406 = (
            einsum("dp,abcd->pabc", a.x1, tau282)
        )
    
        tau407 = (
            einsum("cq,pabc->pqab", a.y2, tau406)
        )
    
        tau408 = (
            einsum("bp,pqab->pqa", a.x2, tau407)
        )
    
        tau409 = (
            einsum("pr,rq,rq->pq", tau230, tau86, tau87)
        )
    
        tau410 = (
            einsum("ai,pi->pa", a.t1, tau45)
        )
    
        tau411 = (
            einsum("rp,rq->pq", tau245, tau78)
        )
    
        tau412 = (
            einsum("ap,pqia->pqi", a.x2, tau201)
        )
    
        tau413 = (
            einsum("ai,pqi->pqa", a.t1, tau412)
        )
    
        tau414 = (
            einsum("jq,pij->pqi", a.y3, tau37)
        )
    
        tau415 = (
            einsum("ai,pqi->pqa", a.t1, tau414)
        )
    
        tau416 = (
            einsum("rp,rq,rq->pq", tau377, tau86, tau87)
        )
    
        tau417 = (
            einsum("ij,jp->pi", tau41, a.x3)
        )
    
        tau418 = (
            einsum("pi,iq->pq", tau417, a.y4)
        )
    
        tau419 = (
            einsum("qi,pqi->pq", tau73, tau221)
        )
    
        tau420 = (
            einsum("ai,pqi->pqa", a.t1, tau356)
        )
    
        tau421 = (
            einsum("lp,ijkl->pijk", a.y3, tau259)
        )
    
        tau422 = (
            einsum("kp,qijk->pqij", a.x4, tau421)
        )
    
        tau423 = (
            einsum("jq,pqij->pqi", a.y4, tau422)
        )
    
        tau424 = (
            einsum("ip,pqi->pq", a.x3, tau423)
        )
    
        tau425 = (
            einsum("jp,pqij->pqi", a.x4, tau202)
        )
    
        tau426 = (
            einsum("iq,pqi->pq", a.y3, tau425)
        )
    
        tau427 = (
            einsum("kp,pijk->pij", a.x3, tau120)
        )
    
        tau428 = (
            einsum("jq,pji->pqi", a.y3, tau427)
        )
    
        tau429 = (
            einsum("qi,pqi->pq", tau73, tau428)
        )
    
        tau430 = (
            einsum("qi,ip->pq", tau96, a.x3)
        )
    
        tau431 = (
            einsum("kp,pijk->pij", a.y4, tau253)
        )
    
        tau432 = (
            einsum("jp,pji->pi", a.y3, tau431)
        )
    
        tau433 = (
            einsum("jp,pqji->pqi", a.x3, tau198)
        )
    
        tau434 = (
            einsum("qi,pqi->pq", tau73, tau433)
        )
    
        tau435 = (
            einsum("jq,pji->pqi", a.y3, tau311)
        )
    
        tau436 = (
            einsum("ai,pqi->pqa", a.t1, tau435)
        )
    
        tau437 = (
            einsum("jq,pji->pqi", a.y4, tau49)
        )
    
        tau438 = (
            einsum("ai,pqi->pqa", a.t1, tau437)
        )
    
        tau439 = (
            einsum("jp,pqij->pqi", a.x4, tau183)
        )
    
        tau440 = (
            einsum("iq,pqi->pq", a.y4, tau439)
        )
    
        tau441 = (
            einsum("rp,rq,rq->pq", tau316, tau78, tau86)
        )
    
        tau442 = (
            einsum("iq,pqi->pq", a.y4, tau322)
        )
    
        tau443 = (
            einsum("ip,pqi->pq", a.x4, tau151)
        )
    
        tau444 = (
            einsum("rp,rq,rq->pq", tau443, tau78, tau79)
        )
    
        tau445 = (
            einsum("jp,pija->pia", a.x3, tau160)
        )
    
        tau446 = (
            einsum("iq,pia->pqa", a.y4, tau445)
        )
    
        tau447 = (
            einsum("pr,rq,rq->pq", tau108, tau79, tau87)
        )
    
        tau448 = (
            einsum("qi,pqi->pq", tau73, tau90)
        )
    
        tau449 = (
            einsum("ij,jp->pi", tau60, a.x3)
        )
    
        tau450 = (
            einsum("pi,iq->pq", tau449, a.y3)
        )
    
        tau451 = (
            einsum("iq,pqi->pq", a.x4, tau269)
        )
    
        tau452 = (
            einsum("rp,rq,rq->pq", tau451, tau82, tau87)
        )
    
        tau453 = (
            einsum("rp,rq,rq->pq", tau309, tau78, tau88)
        )
    
        tau454 = (
            einsum("jq,pji->pqi", a.y4, tau427)
        )
    
        tau455 = (
            einsum("ai,pqi->pqa", a.t1, tau454)
        )
    
        tau456 = (
            einsum("rp,rq->pq", tau295, tau87)
        )
    
        tau457 = (
            einsum("rp,rq,rq->pq", tau230, tau78, tau86)
        )
    
        tau458 = (
            einsum("jp,pqij->pqi", a.x4, tau241)
        )
    
        tau459 = (
            einsum("ip,qpi->pq", a.x3, tau458)
        )
    
        tau460 = (
            einsum("pr,rq,rq->pq", tau459, tau78, tau79)
        )
    
        tau461 = (
            einsum("qi,pqi->pq", tau73, tau227)
        )
    
        tau462 = (
            einsum("ip,pqi->pq", a.x3, tau358)
        )
    
        tau463 = (
            einsum("pr,rq,rq->pq", tau130, tau87, tau88)
        )
    
        tau464 = (
            einsum("jq,pij->pqi", a.x3, tau42)
        )
    
        tau465 = (
            einsum("ip,pqi->pq", a.x3, tau464)
        )
    
        tau466 = (
            einsum("rp,rq->pq", tau465, tau86)
        )
    
        tau467 = (
            einsum("jq,pija->pqia", a.y4, tau235)
        )
    
        tau468 = (
            einsum("ip,pqia->pqa", a.x4, tau467)
        )
    
        tau469 = (
            einsum("rp,rq->pq", tau371, tau79)
        )
    
        tau470 = (
            einsum("ai,pqi->pqa", a.t1, tau324)
        )
    
        tau471 = (
            einsum("rq,rq,rq,pr->pq", tau79, tau86, tau87, tau94)
        )
    
        tau472 = (
            einsum("bi,pjab->pija", a.t1, tau181)
        )
    
        tau473 = (
            einsum("ai,pjka->pijk", a.t1, tau472)
        )
    
        tau474 = (
            einsum("kp,pikj->pij", a.y4, tau473)
        )
    
        tau475 = (
            einsum("jp,pji->pi", a.y3, tau474)
        )
    
        tau476 = (
            einsum("pi,iq->pq", tau449, a.y4)
        )
    
        tau477 = (
            einsum("qi,pqi->pq", tau73, tau158)
        )
    
        tau478 = (
            einsum("rp,rq,rq->pq", tau387, tau79, tau87)
        )
    
        tau479 = (
            einsum("iq,pqi->pq", a.x3, tau350)
        )
    
        tau480 = (
            einsum("rp,rq,rq->pq", tau479, tau79, tau86)
        )
    
        tau481 = (
            einsum("pr,rq,rq->pq", tau443, tau79, tau87)
        )
    
        tau482 = (
            einsum("jp,pija->pia", a.x4, tau305)
        )
    
        tau483 = (
            einsum("iq,pia->pqa", a.y3, tau482)
        )
    
        tau484 = (
            einsum("ip,qpi->pq", a.x4, tau242)
        )
    
        tau485 = (
            einsum("rp,rq,rq->pq", tau484, tau78, tau79)
        )
    
        tau486 = (
            einsum("jq,pija->pqia", a.y4, tau192)
        )
    
        tau487 = (
            einsum("ip,pqia->pqa", a.x4, tau486)
        )
    
        tau488 = (
            einsum("jq,pija->pqia", a.y4, tau365)
        )
    
        tau489 = (
            einsum("ip,pqia->pqa", a.x3, tau488)
        )
    
        tau490 = (
            einsum("jp,pqji->pqi", a.x4, tau89)
        )
    
        tau491 = (
            einsum("ai,pqi->pqa", a.t1, tau490)
        )
    
        tau492 = (
            einsum("rp,rq->pq", tau318, tau82)
        )
    
        tau493 = (
            einsum("pr,rq,rq->pq", tau443, tau82, tau87)
        )
    
        tau494 = (
            einsum("iq,pqi->pq", a.y4, tau178)
        )
    
        tau495 = (
            einsum("ip,pqi->pq", a.x4, tau237)
        )
    
        tau496 = (
            einsum("qi,pqi->pq", tau73, tau328)
        )
    
        tau497 = (
            einsum("ip,pqi->pq", a.x3, tau347)
        )
    
        tau498 = (
            einsum("kq,pijk->pqij", a.y3, tau140)
        )
    
        tau499 = (
            einsum("jp,pqij->pqi", a.x4, tau498)
        )
    
        tau500 = (
            einsum("iq,pqi->pq", a.y4, tau499)
        )
    
        tau501 = (
            einsum("qi,ip->pq", tau125, a.x3)
        )
    
        tau502 = (
            einsum("iq,pqi->pq", a.x4, tau458)
        )
    
        tau503 = (
            einsum("pr,rq,rq->pq", tau502, tau78, tau79)
        )
    
        tau504 = (
            einsum("pa,aq->pq", tau197, a.y2)
        )
    
        tau505 = (
            einsum("jq,pij->pqi", a.y4, tau39)
        )
    
        tau506 = (
            einsum("ai,pqi->pqa", a.t1, tau505)
        )
    
        tau507 = (
            einsum("ij,jp->pi", tau12, a.y3)
        )
    
        tau508 = (
            einsum("qi,ip->pq", tau507, a.x3)
        )
    
        tau509 = (
            einsum("jq,pji->pqi", a.y4, tau226)
        )
    
        tau510 = (
            einsum("ai,pqi->pqa", a.t1, tau509)
        )
    
        tau511 = (
            einsum("rp,rq,rq->pq", tau459, tau78, tau86)
        )
    
        tau512 = (
            einsum("iq,pia->pqa", a.y3, tau306)
        )
    
        tau513 = (
            einsum("pi,iq->pq", tau206, a.y3)
        )
    
        tau514 = (
            einsum("pi,iq->pq", tau417, a.y3)
        )
    
        tau515 = (
            einsum("rp,rq,rq->pq", tau479, tau82, tau88)
        )
    
        tau516 = (
            einsum("jq,pija->pqia", a.y4, tau302)
        )
    
        tau517 = (
            einsum("ip,pqia->pqa", a.x3, tau516)
        )
    
        tau518 = (
            einsum("iq,pia->pqa", a.y3, tau161)
        )
    
        tau519 = (
            einsum("jp,qpij->pqi", a.x3, tau106)
        )
    
        tau520 = (
            einsum("ip,qpi->pq", a.x3, tau519)
        )
    
        tau521 = (
            einsum("rp,rq,rq->pq", tau520, tau87, tau88)
        )
    
        tau522 = (
            einsum("iq,pqi->pq", a.y3, tau439)
        )
    
        tau523 = (
            einsum("jq,pqij->pqi", a.y4, tau147)
        )
    
        tau524 = (
            einsum("ip,pqi->pq", a.x3, tau523)
        )
    
        tau525 = (
            einsum("ip,pqi->pq", a.x3, tau396)
        )
    
        tau526 = (
            einsum("rp,rq,rq->pq", tau525, tau78, tau86)
        )
    
        tau527 = (
            einsum("rp,rq,rq->pq", tau443, tau78, tau82)
        )
    
        tau528 = (
            einsum("rp,rq,rq->pq", tau134, tau78, tau79)
        )
    
        tau529 = (
            einsum("pi,qi->pq", tau15, tau73)
        )
    
        tau530 = (
            einsum("rp,rq,rq->pq", tau377, tau87, tau88)
        )
    
        tau531 = (
            einsum("oai,obc->iabc", h.l.pvo, h.l.pvv)
        )
    
        tau532 = (
            einsum("cp,iacb->piab", a.y2, tau531)
        )
    
        tau533 = (
            einsum("bi,pjab->pija", a.t1, tau532)
        )
    
        tau534 = (
            einsum("jp,pija->pia", a.y3, tau533)
        )
    
        tau535 = (
            einsum("pr,rq,rq->pq", tau520, tau86, tau87)
        )
    
        tau536 = (
            einsum("qi,ip->pq", tau507, a.x4)
        )
    
        tau537 = (
            einsum("kp,qijk->pqij", a.x4, tau260)
        )
    
        tau538 = (
            einsum("jq,pqij->pqi", a.y3, tau537)
        )
    
        tau539 = (
            einsum("ip,pqi->pq", a.x3, tau538)
        )
    
        tau540 = (
            einsum("jp,pqji->pqi", a.x3, tau353)
        )
    
        tau541 = (
            einsum("qi,pqi->pq", tau73, tau540)
        )
    
        tau542 = (
            einsum("rp,rq->pq", tau465, tau88)
        )
    
        tau543 = (
            einsum("jq,pji->pqi", a.y3, tau34)
        )
    
        tau544 = (
            einsum("qi,pqi->pq", tau73, tau543)
        )
    
        tau545 = (
            einsum("ai,pi->pa", a.t1, tau210)
        )
    
        tau546 = (
            einsum("rp,rq,rq,rq->pq", tau152, tau78, tau82, tau88)
        )
    
        tau547 = (
            einsum("jq,pija->pqia", a.y3, tau235)
        )
    
        tau548 = (
            einsum("ip,pqia->pqa", a.x3, tau547)
        )
    
        tau549 = (
            einsum("jq,pqij->pqi", a.y3, tau373)
        )
    
        tau550 = (
            einsum("ip,pqi->pq", a.x4, tau549)
        )
    
        tau551 = (
            einsum("rp,rq,rq->pq", tau108, tau87, tau88)
        )
    
        tau552 = (
            einsum("iq,pia->pqa", a.y3, tau445)
        )
    
        tau553 = (
            einsum("jq,pqij->pqi", a.y4, tau236)
        )
    
        tau554 = (
            einsum("ip,pqi->pq", a.x4, tau553)
        )
    
        tau555 = (
            einsum("cp,icab->piab", a.y2, tau531)
        )
    
        tau556 = (
            einsum("bi,pjab->pija", a.t1, tau555)
        )
    
        tau557 = (
            einsum("jp,pija->pia", a.y4, tau556)
        )
    
        tau558 = (
            einsum("kp,pkij->pij", a.y4, tau473)
        )
    
        tau559 = (
            einsum("jp,pji->pi", a.y3, tau558)
        )
    
        tau560 = (
            einsum("rp,rq->pq", tau351, tau88)
        )
    
        tau561 = (
            einsum("rp,rq,rq->pq", tau525, tau78, tau88)
        )
    
        tau562 = (
            einsum("pi,iq->pq", tau110, a.y3)
        )
    
        tau563 = (
            einsum("kp,ijka->pija", a.y3, tau137)
        )
    
        tau564 = (
            einsum("jp,pjia->pia", a.y4, tau563)
        )
    
        tau565 = (
            einsum("jp,pqij->pqi", a.x3, tau241)
        )
    
        tau566 = (
            einsum("iq,pqi->pq", a.x3, tau565)
        )
    
        tau567 = (
            einsum("pr,rq,rq->pq", tau566, tau78, tau88)
        )
    
        tau568 = (
            einsum("qi,pqi->pq", tau73, tau399)
        )
    
        tau569 = (
            einsum("ip,pqi->pq", a.x3, tau458)
        )
    
        tau570 = (
            einsum("rp,rq->pq", tau569, tau78)
        )
    
        tau571 = (
            einsum("pr,rq->pq", tau397, tau87)
        )
    
        tau572 = (
            einsum("pj,pji->pi", tau73, tau164)
        )
    
        tau573 = (
            einsum("kp,ikja->pija", a.y4, tau75)
        )
    
        tau574 = (
            einsum("jp,pjia->pia", a.y3, tau573)
        )
    
        tau575 = (
            einsum("jp,pija->pia", a.y3, tau556)
        )
    
        tau576 = (
            einsum("ip,pqi->pq", a.x4, tau374)
        )
    
        tau577 = (
            einsum("ai,pqi->pqa", a.t1, tau354)
        )
    
        tau578 = (
            einsum("dp,abcd->pabc", a.x2, tau282)
        )
    
        tau579 = (
            einsum("cq,pabc->pqab", a.y2, tau578)
        )
    
        tau580 = (
            einsum("bp,pqab->pqa", a.x1, tau579)
        )
    
        tau581 = (
            einsum("rp,rq,rq->pq", tau316, tau78, tau88)
        )
    
        tau582 = (
            einsum("rp,rq,rq->pq", tau459, tau78, tau88)
        )
    
        tau583 = (
            einsum("ai,pi->pa", a.t1, tau15)
        )
    
        tau584 = (
            einsum("ai,pqi->pqa", a.t1, tau433)
        )
    
        tau585 = (
            einsum("pi,iq->pq", tau53, a.x3)
        )
    
        tau586 = (
            einsum("rp,rq->pq", tau585, tau88)
        )
    
        tau587 = (
            einsum("pr,rq,rq->pq", tau134, tau82, tau87)
        )
    
        tau588 = (
            einsum("kp,ijka->pija", a.y3, tau288)
        )
    
        tau589 = (
            einsum("jp,pija->pia", a.y4, tau588)
        )
    
        tau590 = (
            einsum("qi,pqi->pq", tau73, tau232)
        )
    
        tau591 = (
            einsum("pr,rq,rq,rq->pq", tau245, tau78, tau79, tau86)
        )
    
        tau592 = (
            einsum("pr,rq,rq,rq->pq", tau569, tau78, tau82, tau88)
        )
    
        tau593 = (
            einsum("ip,pqia->pqa", a.x4, tau516)
        )
    
        tau594 = (
            einsum("lp,iljk->pijk", a.y4, tau167)
        )
    
        tau595 = (
            einsum("kp,pkij->pij", a.y3, tau594)
        )
    
        tau596 = (
            einsum("aj,pij->pia", a.t1, tau595)
        )
    
        tau597 = (
            einsum("pr,rq,rq->pq", tau243, tau78, tau82)
        )
    
        tau598 = (
            einsum("iq,pia->pqa", a.y4, tau482)
        )
    
        tau599 = (
            einsum("iq,pqi->pq", a.y3, tau184)
        )
    
        tau600 = (
            einsum("rp,rq->pq", tau585, tau86)
        )
    
        tau601 = (
            einsum("ip,pqia->pqa", a.x4, tau547)
        )
    
        tau602 = (
            einsum("pr,rq,rq->pq", tau215, tau86, tau87)
        )
    
        tau603 = (
            einsum("kp,pijk->pij", a.y3, tau334)
        )
    
        tau604 = (
            einsum("jp,pij->pi", a.y4, tau603)
        )
    
        tau605 = (
            einsum("qi,pqi->pq", tau73, tau505)
        )
    
        tau606 = (
            einsum("bi,jkab->ijka", a.t1, tau51)
        )
    
        tau607 = (
            einsum("kp,ijka->pija", a.y4, tau606)
        )
    
        tau608 = (
            einsum("jp,pjia->pia", a.y3, tau607)
        )
    
        tau609 = (
            einsum("jp,pija->pia", a.y3, tau257)
        )
    
        tau610 = (
            einsum("pr,rq,rq,rq->pq", tau270, tau79, tau86, tau87)
        )
    
        tau611 = (
            einsum("jq,pji->pqi", a.y4, tau68)
        )
    
        tau612 = (
            einsum("ai,pqi->pqa", a.t1, tau611)
        )
    
        tau613 = (
            einsum("ip,pqi->pq", a.x4, tau523)
        )
    
        tau614 = (
            einsum("pr,rq,rq->pq", tau316, tau82, tau87)
        )
    
        tau615 = (
            einsum("ai,pqi->pqa", a.t1, tau83)
        )
    
        tau616 = (
            einsum("qi,pqi->pq", tau73, tau104)
        )
    
        tau617 = (
            einsum("pr,rq,rq->pq", tau525, tau79, tau87)
        )
    
        tau618 = (
            einsum("pr,rq,rq->pq", tau215, tau87, tau88)
        )
    
        tau619 = (
            einsum("pr,rq,rq->pq", tau525, tau82, tau87)
        )
    
        tau620 = (
            einsum("ip,pqia->pqa", a.x4, tau193)
        )
    
        tau621 = (
            einsum("ip,pqia->pqa", a.x3, tau486)
        )
    
        tau622 = (
            einsum("rp,rq,rq->pq", tau502, tau78, tau82)
        )
    
        tau623 = (
            einsum("ip,pqia->pqa", a.x4, tau488)
        )
    
        tau624 = (
            einsum("ai,pqi->pqa", a.t1, tau312)
        )
    
        tau625 = (
            einsum("iq,pia->pqa", a.y3, tau326)
        )
    
        tau626 = (
            einsum("pr,rq,rq->pq", tau377, tau82, tau87)
        )
    
        tau627 = (
            einsum("pr,rq,rq,rq->pq", tau569, tau78, tau79, tau86)
        )
    
        tau628 = (
            einsum("ip,pqia->pqa", a.x3, tau467)
        )
    
        tau629 = (
            einsum("qi,pqi->pq", tau73, tau490)
        )
    
        tau630 = (
            einsum("jp,pija->pia", a.y4, tau533)
        )
    
        tau631 = (
            einsum("qi,pqi->pq", tau73, tau414)
        )
    
        tau632 = (
            einsum("iq,pqi->pq", a.y3, tau203)
        )
    
        tau633 = (
            einsum("qa,ap->pq", tau195, a.x2)
        )
    
        tau634 = (
            einsum("qi,pqi->pq", tau73, tau611)
        )
    
        tau635 = (
            einsum("ai,pqi->pqa", a.t1, tau543)
        )
    
        tau636 = (
            einsum("ip,pqi->pq", a.x3, tau549)
        )
    
        tau637 = (
            einsum("qi,pqi->pq", tau73, tau509)
        )
    
        tau638 = (
            einsum("rp,rq,rq->pq", tau130, tau78, tau79)
        )
    
        tau639 = (
            einsum("rp,rq,rq->pq", tau215, tau78, tau82)
        )
    
        tau640 = (
            einsum("pr,rq,rq->pq", tau280, tau78, tau86)
        )
    
        tau641 = (
            einsum("rp,rq->pq", tau144, tau82)
        )
    
        tau642 = (
            einsum("jq,pji->pqi", a.y3, tau49)
        )
    
        tau643 = (
            einsum("ai,pqi->pqa", a.t1, tau642)
        )
    
        tau644 = (
            einsum("rp,rq,rq,rq->pq", tau397, tau78, tau82, tau88)
        )
    
        tau645 = (
            einsum("ai,pqi->pqa", a.t1, tau540)
        )
    
        tau646 = (
            einsum("pr,rq,rq->pq", tau243, tau78, tau79)
        )
    
        tau647 = (
            einsum("pr,rq,rq->pq", tau451, tau79, tau87)
        )
    
        tau648 = (
            einsum("ip,pqi->pq", a.x3, tau553)
        )
    
        tau649 = (
            einsum("pj,pji->pi", tau73, tau384)
        )
    
        tau650 = (
            einsum("pr,rq,rq->pq", tau230, tau87, tau88)
        )
    
        tau651 = (
            einsum("rp,rq,rq->pq", tau566, tau78, tau86)
        )
    
        tau652 = (
            einsum("qi,pqi->pq", tau73, tau437)
        )
    
        tau653 = (
            einsum("ai,pqi->pqa", a.t1, tau199)
        )
    
        tau654 = (
            einsum("ai,pqi->pqa", a.t1, tau428)
        )
    
        tau655 = (
            einsum("pj,pij->pi", tau73, tau595)
        )
    
        tau656 = (
            einsum("kp,ijka->pija", a.y3, tau606)
        )
    
        tau657 = (
            einsum("jp,pjia->pia", a.y4, tau656)
        )
    
        tau658 = (
            einsum("pr,rq,rq->pq", tau484, tau78, tau82)
        )
    
        tau659 = (
            einsum("iq,pqi->pq", a.y4, tau425)
        )
    
        tau660 = (
            einsum("qi,pqi->pq", tau73, tau454)
        )
    
        tau661 = (
            einsum("ip,pqi->pq", a.x3, tau148)
        )
    
        tau662 = (
            einsum("qi,pqi->pq", tau73, tau642)
        )
    
        tau663 = (
            einsum("qi,pqi->pq", tau73, tau435)
        )
    
        tau664 = (
            einsum("pr,rq,rq->pq", tau459, tau78, tau82)
        )
    
        tau665 = (
            einsum("ap,aq->pq", a.x1, a.y1)
        )
    
        tau666 = (
            einsum("pr,rq,rq->pq", tau377, tau665, tau82)
        )
    
        tau667 = (
            einsum("ap,aq->pq", a.x2, a.y1)
        )
    
        tau668 = (
            einsum("pr,rq,rq->pq", tau130, tau665, tau86)
        )
    
        tau669 = (
            einsum("rp,rq->pq", tau208, tau667)
        )
    
        tau670 = (
            einsum("bp,ijab->pija", a.y1, tau136)
        )
    
        tau671 = (
            einsum("ai,pjka->pijk", a.t1, tau670)
        )
    
        tau672 = (
            einsum("kp,pijk->pij", a.y4, tau671)
        )
    
        tau673 = (
            einsum("jp,pji->pi", a.y3, tau672)
        )
    
        tau674 = (
            einsum("ai,ap->pi", a.t1, a.y1)
        )
    
        tau675 = (
            einsum("qi,pia->pqa", tau674, tau62)
        )
    
        tau676 = (
            einsum("pi,qi->pq", tau128, tau674)
        )
    
        tau677 = (
            einsum("qi,pqi->pq", tau674, tau251)
        )
    
        tau678 = (
            einsum("qi,pqi->pq", tau674, tau227)
        )
    
        tau679 = (
            einsum("rp,rq->pq", tau569, tau667)
        )
    
        tau680 = (
            einsum("pr,rq,rq->pq", tau309, tau665, tau86)
        )
    
        tau681 = (
            einsum("bp,ijba->pija", a.y1, tau51)
        )
    
        tau682 = (
            einsum("ai,pjka->pijk", a.t1, tau681)
        )
    
        tau683 = (
            einsum("kp,pijk->pij", a.y3, tau682)
        )
    
        tau684 = (
            einsum("jp,pji->pi", a.y4, tau683)
        )
    
        tau685 = (
            einsum("rp,rq->pq", tau270, tau665)
        )
    
        tau686 = (
            einsum("pr,rq,rq->pq", tau215, tau665, tau88)
        )
    
        tau687 = (
            einsum("qi,pqi->pq", tau674, tau642)
        )
    
        tau688 = (
            einsum("ap,qija->pqij", a.x1, tau670)
        )
    
        tau689 = (
            einsum("jq,pqij->pqi", a.y3, tau688)
        )
    
        tau690 = (
            einsum("ip,pqi->pq", a.x4, tau689)
        )
    
        tau691 = (
            einsum("pi,qi->pq", tau15, tau674)
        )
    
        tau692 = (
            einsum("pr,rq,rq->pq", tau243, tau667, tau82)
        )
    
        tau693 = (
            einsum("rp,rq,rq->pq", tau309, tau667, tau86)
        )
    
        tau694 = (
            einsum("cp,iacb->piab", a.y1, tau23)
        )
    
        tau695 = (
            einsum("bp,qiba->pqia", a.x1, tau694)
        )
    
        tau696 = (
            einsum("ai,pqja->pqij", a.t1, tau695)
        )
    
        tau697 = (
            einsum("jp,pqij->pqi", a.x4, tau696)
        )
    
        tau698 = (
            einsum("iq,pqi->pq", a.y4, tau697)
        )
    
        tau699 = (
            einsum("pr,rq,rq->pq", tau230, tau665, tau86)
        )
    
        tau700 = (
            einsum("rp,rq,rq->pq", tau377, tau665, tau88)
        )
    
        tau701 = (
            einsum("rp,rq,rq->pq", tau273, tau665, tau86)
        )
    
        tau702 = (
            einsum("pr,rq,rq->pq", tau387, tau665, tau82)
        )
    
        tau703 = (
            einsum("qi,pqi->pq", tau674, tau232)
        )
    
        tau704 = (
            einsum("qi,pqi->pq", tau674, tau354)
        )
    
        tau705 = (
            einsum("bq,piab->pqia", a.y1, tau24)
        )
    
        tau706 = (
            einsum("ai,pqja->pqij", a.t1, tau705)
        )
    
        tau707 = (
            einsum("jp,pqij->pqi", a.x4, tau706)
        )
    
        tau708 = (
            einsum("iq,pqi->pq", a.y3, tau707)
        )
    
        tau709 = (
            einsum("rp,rq,rq->pq", tau443, tau667, tau79)
        )
    
        tau710 = (
            einsum("qi,pqi->pq", tau674, tau490)
        )
    
        tau711 = (
            einsum("jp,pqij->pqi", a.x3, tau706)
        )
    
        tau712 = (
            einsum("iq,pqi->pq", a.y4, tau711)
        )
    
        tau713 = (
            einsum("cp,pacb->pab", a.x1, tau578)
        )
    
        tau714 = (
            einsum("bq,pba->pqa", a.y1, tau713)
        )
    
        tau715 = (
            einsum("pr,rq->pq", tau397, tau665)
        )
    
        tau716 = (
            einsum("qi,pqi->pq", tau674, tau454)
        )
    
        tau717 = (
            einsum("rp,rq,rq->pq", tau243, tau667, tau88)
        )
    
        tau718 = (
            einsum("ap,pqia->pqi", a.x1, tau705)
        )
    
        tau719 = (
            einsum("ai,pqi->pqa", a.t1, tau718)
        )
    
        tau720 = (
            einsum("jp,pqij->pqi", a.x3, tau696)
        )
    
        tau721 = (
            einsum("iq,pqi->pq", a.y3, tau720)
        )
    
        tau722 = (
            einsum("rp,rq,rq->pq", tau215, tau667, tau79)
        )
    
        tau723 = (
            einsum("ap,qija->pqij", a.x2, tau670)
        )
    
        tau724 = (
            einsum("jq,pqij->pqi", a.y4, tau723)
        )
    
        tau725 = (
            einsum("ip,pqi->pq", a.x3, tau724)
        )
    
        tau726 = (
            einsum("ap,ijka->pijk", a.y1, tau288)
        )
    
        tau727 = (
            einsum("kp,pijk->pij", a.y4, tau726)
        )
    
        tau728 = (
            einsum("jp,pij->pi", a.y3, tau727)
        )
    
        tau729 = (
            einsum("cp,iacb->piab", a.y1, tau531)
        )
    
        tau730 = (
            einsum("bi,pjab->pija", a.t1, tau729)
        )
    
        tau731 = (
            einsum("jp,pija->pia", a.y4, tau730)
        )
    
        tau732 = (
            einsum("pi,qi->pq", tau5, tau674)
        )
    
        tau733 = (
            einsum("iq,pqi->pq", a.y4, tau720)
        )
    
        tau734 = (
            einsum("rp,rq,rq->pq", tau459, tau667, tau86)
        )
    
        tau735 = (
            einsum("cp,icab->piab", a.y1, tau531)
        )
    
        tau736 = (
            einsum("bi,pjab->pija", a.t1, tau735)
        )
    
        tau737 = (
            einsum("jp,pija->pia", a.y4, tau736)
        )
    
        tau738 = (
            einsum("qi,pqi->pq", tau674, tau611)
        )
    
        tau739 = (
            einsum("bi,pjab->pija", a.t1, tau694)
        )
    
        tau740 = (
            einsum("ai,pjka->pijk", a.t1, tau739)
        )
    
        tau741 = (
            einsum("kp,pkij->pij", a.y4, tau740)
        )
    
        tau742 = (
            einsum("jp,pji->pi", a.y3, tau741)
        )
    
        tau743 = (
            einsum("pr,rq,rq->pq", tau566, tau667, tau88)
        )
    
        tau744 = (
            einsum("qi,pqi->pq", tau674, tau540)
        )
    
        tau745 = (
            einsum("qi,pqi->pq", tau674, tau428)
        )
    
        tau746 = (
            einsum("qj,pij->pqi", tau674, tau42)
        )
    
        tau747 = (
            einsum("ai,pqi->pqa", a.t1, tau746)
        )
    
        tau748 = (
            einsum("pr,rq,rq->pq", tau108, tau665, tau79)
        )
    
        tau749 = (
            einsum("rp,rq,rq->pq", tau108, tau665, tau88)
        )
    
        tau750 = (
            einsum("rq,rp->pq", tau667, tau94)
        )
    
        tau751 = (
            einsum("kp,pijk->pij", a.y4, tau682)
        )
    
        tau752 = (
            einsum("jp,pji->pi", a.y3, tau751)
        )
    
        tau753 = (
            einsum("qi,pqi->pq", tau674, tau414)
        )
    
        tau754 = (
            einsum("pi,qi->pq", tau210, tau674)
        )
    
        tau755 = (
            einsum("ba,bp->pa", tau52, a.y1)
        )
    
        tau756 = (
            einsum("qa,ap->pq", tau755, a.x2)
        )
    
        tau757 = (
            einsum("dp,dabc->pabc", a.y1, tau282)
        )
    
        tau758 = (
            einsum("ci,pabc->piab", a.t1, tau757)
        )
    
        tau759 = (
            einsum("bi,pjba->pija", a.t1, tau758)
        )
    
        tau760 = (
            einsum("jp,pjia->pia", a.y4, tau759)
        )
    
        tau761 = (
            einsum("jq,pqij->pqi", a.y3, tau723)
        )
    
        tau762 = (
            einsum("ip,pqi->pq", a.x4, tau761)
        )
    
        tau763 = (
            einsum("pr,rq,rq->pq", tau309, tau665, tau88)
        )
    
        tau764 = (
            einsum("aq,pija->pqij", a.y1, tau365)
        )
    
        tau765 = (
            einsum("jq,pqij->pqi", a.y4, tau764)
        )
    
        tau766 = (
            einsum("ip,pqi->pq", a.x4, tau765)
        )
    
        tau767 = (
            einsum("rp,rq,rq->pq", tau566, tau667, tau86)
        )
    
        tau768 = (
            einsum("pr,rq,rq->pq", tau443, tau665, tau82)
        )
    
        tau769 = (
            einsum("qi,pqi->pq", tau674, tau543)
        )
    
        tau770 = (
            einsum("pi,qi->pq", tau45, tau674)
        )
    
        tau771 = (
            einsum("qj,pji->pqi", tau674, tau42)
        )
    
        tau772 = (
            einsum("ai,pqi->pqa", a.t1, tau771)
        )
    
        tau773 = (
            einsum("qi,pqi->pq", tau674, tau213)
        )
    
        tau774 = (
            einsum("qi,pqi->pq", tau674, tau435)
        )
    
        tau775 = (
            einsum("bq,piab->pqia", a.y1, tau61)
        )
    
        tau776 = (
            einsum("ap,pqia->pqi", a.x2, tau775)
        )
    
        tau777 = (
            einsum("ai,pqi->pqa", a.t1, tau776)
        )
    
        tau778 = (
            einsum("pj,pji->pi", tau674, tau384)
        )
    
        tau779 = (
            einsum("pr,rq,rq->pq", tau280, tau667, tau86)
        )
    
        tau780 = (
            einsum("pr,rq,rq->pq", tau243, tau667, tau79)
        )
    
        tau781 = (
            einsum("rp,rq,rq->pq", tau377, tau665, tau86)
        )
    
        tau782 = (
            einsum("rp,rq,rq->pq", tau230, tau667, tau86)
        )
    
        tau783 = (
            einsum("pr,rq,rq->pq", tau484, tau667, tau82)
        )
    
        tau784 = (
            einsum("qi,pqi->pq", tau674, tau324)
        )
    
        tau785 = (
            einsum("qi,pqi->pq", tau674, tau312)
        )
    
        tau786 = (
            einsum("ip,pqi->pq", a.x3, tau761)
        )
    
        tau787 = (
            einsum("bp,qiba->pqia", a.x2, tau694)
        )
    
        tau788 = (
            einsum("ai,pqja->pqij", a.t1, tau787)
        )
    
        tau789 = (
            einsum("jp,pqij->pqi", a.x3, tau788)
        )
    
        tau790 = (
            einsum("iq,pqi->pq", a.y4, tau789)
        )
    
        tau791 = (
            einsum("bp,ijba->pija", a.y1, tau256)
        )
    
        tau792 = (
            einsum("jp,pija->pia", a.y3, tau791)
        )
    
        tau793 = (
            einsum("aq,pija->pqij", a.y1, tau235)
        )
    
        tau794 = (
            einsum("jq,pqij->pqi", a.y3, tau793)
        )
    
        tau795 = (
            einsum("ip,pqi->pq", a.x3, tau794)
        )
    
        tau796 = (
            einsum("jp,pija->pia", a.y3, tau736)
        )
    
        tau797 = (
            einsum("rp,rq,rq->pq", tau459, tau667, tau88)
        )
    
        tau798 = (
            einsum("rp,rq,rq->pq", tau215, tau667, tau82)
        )
    
        tau799 = (
            einsum("qi,pia->pqa", tau674, tau25)
        )
    
        tau800 = (
            einsum("rp,rq,rq->pq", tau502, tau667, tau82)
        )
    
        tau801 = (
            einsum("pr,rq,rq,rq->pq", tau245, tau667, tau79, tau86)
        )
    
        tau802 = (
            einsum("pr,rq,rq->pq", tau316, tau665, tau79)
        )
    
        tau803 = (
            einsum("iq,pqi->pq", a.y4, tau707)
        )
    
        tau804 = (
            einsum("ai,pqja->pqij", a.t1, tau775)
        )
    
        tau805 = (
            einsum("jp,pqij->pqi", a.x4, tau804)
        )
    
        tau806 = (
            einsum("iq,pqi->pq", a.y4, tau805)
        )
    
        tau807 = (
            einsum("rp,rq,rq->pq", tau309, tau667, tau88)
        )
    
        tau808 = (
            einsum("pj,pij->pi", tau674, tau384)
        )
    
        tau809 = (
            einsum("pr,rq,rq->pq", tau451, tau665, tau79)
        )
    
        tau810 = (
            einsum("rp,rq,rq->pq", tau443, tau667, tau82)
        )
    
        tau811 = (
            einsum("pr,rq,rq,rq->pq", tau208, tau665, tau79, tau86)
        )
    
        tau812 = (
            einsum("pj,pji->pi", tau674, tau164)
        )
    
        tau813 = (
            einsum("rp,rq,rq->pq", tau130, tau667, tau79)
        )
    
        tau814 = (
            einsum("kp,pikj->pij", a.y4, tau740)
        )
    
        tau815 = (
            einsum("jp,pji->pi", a.y3, tau814)
        )
    
        tau816 = (
            einsum("rp,rq,rq->pq", tau108, tau665, tau86)
        )
    
        tau817 = (
            einsum("pa,aq->pq", tau361, a.y1)
        )
    
        tau818 = (
            einsum("rp,rq,rq->pq", tau230, tau667, tau88)
        )
    
        tau819 = (
            einsum("rp,rq,rq->pq", tau316, tau667, tau86)
        )
    
        tau820 = (
            einsum("pr,rq,rq->pq", tau134, tau665, tau82)
        )
    
        tau821 = (
            einsum("pa,aq->pq", tau197, a.y1)
        )
    
        tau822 = (
            einsum("qi,pqi->pq", tau674, tau509)
        )
    
        tau823 = (
            einsum("rq,rq,rq,pr->pq", tau665, tau79, tau86, tau94)
        )
    
        tau824 = (
            einsum("jp,pqij->pqi", a.x4, tau788)
        )
    
        tau825 = (
            einsum("iq,pqi->pq", a.y4, tau824)
        )
    
        tau826 = (
            einsum("pr,rq,rq,rq->pq", tau569, tau667, tau82, tau88)
        )
    
        tau827 = (
            einsum("pr,rq,rq->pq", tau273, tau665, tau88)
        )
    
        tau828 = (
            einsum("ip,pqi->pq", a.x3, tau765)
        )
    
        tau829 = (
            einsum("pr,rq,rq->pq", tau520, tau665, tau86)
        )
    
        tau830 = (
            einsum("rp,rq,rq,rq->pq", tau152, tau667, tau79, tau86)
        )
    
        tau831 = (
            einsum("iq,pqi->pq", a.y3, tau789)
        )
    
        tau832 = (
            einsum("jq,pqij->pqi", a.y4, tau688)
        )
    
        tau833 = (
            einsum("ip,pqi->pq", a.x3, tau832)
        )
    
        tau834 = (
            einsum("qi,pqi->pq", tau674, tau328)
        )
    
        tau835 = (
            einsum("pr,rq,rq->pq", tau525, tau665, tau82)
        )
    
        tau836 = (
            einsum("pr,rq,rq,rq->pq", tau208, tau665, tau82, tau88)
        )
    
        tau837 = (
            einsum("qi,pqi->pq", tau674, tau437)
        )
    
        tau838 = (
            einsum("qi,pqi->pq", tau674, tau433)
        )
    
        tau839 = (
            einsum("qi,pqi->pq", tau674, tau505)
        )
    
        tau840 = (
            einsum("jp,pqij->pqi", a.x3, tau804)
        )
    
        tau841 = (
            einsum("iq,pqi->pq", a.y4, tau840)
        )
    
        tau842 = (
            einsum("qi,pqi->pq", tau674, tau99)
        )
    
        tau843 = (
            einsum("iq,pqi->pq", a.y3, tau840)
        )
    
        tau844 = (
            einsum("rp,rq->pq", tau245, tau667)
        )
    
        tau845 = (
            einsum("jp,pija->pia", a.y4, tau791)
        )
    
        tau846 = (
            einsum("pr,rq,rq->pq", tau215, tau665, tau86)
        )
    
        tau847 = (
            einsum("qa,ap->pq", tau755, a.x1)
        )
    
        tau848 = (
            einsum("pr,rq,rq->pq", tau130, tau665, tau88)
        )
    
        tau849 = (
            einsum("pa,aq->pq", tau223, a.y1)
        )
    
        tau850 = (
            einsum("kp,pijk->pij", a.y3, tau726)
        )
    
        tau851 = (
            einsum("jp,pij->pi", a.y4, tau850)
        )
    
        tau852 = (
            einsum("qi,pqi->pq", tau674, tau90)
        )
    
        tau853 = (
            einsum("rp,rq,rq->pq", tau316, tau667, tau88)
        )
    
        tau854 = (
            einsum("rq,rq,rq,pr->pq", tau665, tau82, tau88, tau94)
        )
    
        tau855 = (
            einsum("qi,pqi->pq", tau674, tau158)
        )
    
        tau856 = (
            einsum("qi,pqi->pq", tau674, tau263)
        )
    
        tau857 = (
            einsum("pr,rq,rq->pq", tau377, tau665, tau79)
        )
    
        tau858 = (
            einsum("rp,rq,rq->pq", tau134, tau667, tau79)
        )
    
        tau859 = (
            einsum("pr,rq,rq->pq", tau502, tau667, tau79)
        )
    
        tau860 = (
            einsum("ip,pqi->pq", a.x4, tau794)
        )
    
        tau861 = (
            einsum("jp,pija->pia", a.y4, tau759)
        )
    
        tau862 = (
            einsum("jq,pqij->pqi", a.y4, tau793)
        )
    
        tau863 = (
            einsum("ip,pqi->pq", a.x4, tau862)
        )
    
        tau864 = (
            einsum("pi,qi->pq", tau65, tau674)
        )
    
        tau865 = (
            einsum("ip,pqi->pq", a.x4, tau832)
        )
    
        tau866 = (
            einsum("pr,rq,rq,rq->pq", tau245, tau667, tau82, tau88)
        )
    
        tau867 = (
            einsum("kp,pijk->pij", a.y3, tau671)
        )
    
        tau868 = (
            einsum("jp,pji->pi", a.y4, tau867)
        )
    
        tau869 = (
            einsum("pr,rq,rq,rq->pq", tau295, tau665, tau79, tau86)
        )
    
        tau870 = (
            einsum("pj,pij->pi", tau674, tau595)
        )
    
        tau871 = (
            einsum("qi,pqi->pq", tau674, tau122)
        )
    
        tau872 = (
            einsum("jp,pija->pia", a.y3, tau730)
        )
    
        tau873 = (
            einsum("rp,rq,rq->pq", tau243, tau667, tau86)
        )
    
        tau874 = (
            einsum("rp,rq,rq->pq", tau525, tau667, tau86)
        )
    
        tau875 = (
            einsum("pr,rq,rq,rq->pq", tau295, tau665, tau82, tau88)
        )
    
        tau876 = (
            einsum("ip,pqi->pq", a.x3, tau862)
        )
    
        tau877 = (
            einsum("pr,rq,rq,rq->pq", tau270, tau665, tau79, tau86)
        )
    
        tau878 = (
            einsum("pr,rq,rq->pq", tau316, tau665, tau82)
        )
    
        tau879 = (
            einsum("rp,rq,rq->pq", tau451, tau665, tau82)
        )
    
        tau880 = (
            einsum("pr,rq,rq->pq", tau230, tau665, tau88)
        )
    
        tau881 = (
            einsum("pa,aq->pq", tau379, a.y1)
        )
    
        tau882 = (
            einsum("pr,rq,rq,rq->pq", tau569, tau667, tau79, tau86)
        )
    
        tau883 = (
            einsum("iq,pqi->pq", a.y3, tau697)
        )
    
        tau884 = (
            einsum("iq,pqi->pq", a.y3, tau805)
        )
    
        tau885 = (
            einsum("pr,rq,rq->pq", tau525, tau665, tau79)
        )
    
        tau886 = (
            einsum("pr,rq,rq->pq", tau443, tau665, tau79)
        )
    
        tau887 = (
            einsum("pr,rq,rq,rq->pq", tau270, tau665, tau82, tau88)
        )
    
        tau888 = (
            einsum("pr,rq,rq->pq", tau108, tau665, tau82)
        )
    
        tau889 = (
            einsum("rp,rq->pq", tau295, tau665)
        )
    
        tau890 = (
            einsum("pj,pij->pi", tau674, tau261)
        )
    
        tau891 = (
            einsum("rp,rq,rq->pq", tau525, tau667, tau88)
        )
    
        tau892 = (
            einsum("qi,pqi->pq", tau674, tau199)
        )
    
        tau893 = (
            einsum("cp,pacb->pab", a.x2, tau406)
        )
    
        tau894 = (
            einsum("bq,pba->pqa", a.y1, tau893)
        )
    
        tau895 = (
            einsum("qi,pqi->pq", tau674, tau83)
        )
    
        tau896 = (
            einsum("pr,rq,rq->pq", tau459, tau667, tau79)
        )
    
        tau897 = (
            einsum("qi,pqi->pq", tau674, tau104)
        )
    
        tau898 = (
            einsum("iq,pqi->pq", a.y3, tau711)
        )
    
        tau899 = (
            einsum("ip,pqi->pq", a.x4, tau724)
        )
    
        tau900 = (
            einsum("pj,pji->pi", tau674, tau261)
        )
    
        tau901 = (
            einsum("ip,pqi->pq", a.x3, tau689)
        )
    
        tau902 = (
            einsum("rp,rq,rq->pq", tau130, tau667, tau82)
        )
    
        tau903 = (
            einsum("qi,pqi->pq", tau674, tau221)
        )
    
        tau904 = (
            einsum("qi,pqi->pq", tau674, tau356)
        )
    
        tau905 = (
            einsum("rp,rq,rq->pq", tau520, tau665, tau88)
        )
    
        tau906 = (
            einsum("pj,pij->pi", tau674, tau164)
        )
    
        tau907 = (
            einsum("rp,rq,rq,rq->pq", tau397, tau667, tau82, tau88)
        )
    
        tau908 = (
            einsum("pr,rq,rq->pq", tau459, tau667, tau82)
        )
    
        tau909 = (
            einsum("pr,rq->pq", tau152, tau665)
        )
    
        tau910 = (
            einsum("qi,pqi->pq", tau674, tau399)
        )
    
        tau911 = (
            einsum("rp,rq,rq,rq->pq", tau152, tau667, tau82, tau88)
        )
    
        tau912 = (
            einsum("iq,pqi->pq", a.y3, tau824)
        )
    
        tau913 = (
            einsum("rp,rq,rq->pq", tau484, tau667, tau79)
        )
    
        tau914 = (
            einsum("pr,rq,rq->pq", tau134, tau665, tau79)
        )
    
        tau915 = (
            einsum("jq,pqij->pqi", a.y3, tau764)
        )
    
        tau916 = (
            einsum("ip,pqi->pq", a.x4, tau915)
        )
    
        tau917 = (
            einsum("rp,rq,rq,rq->pq", tau397, tau667, tau79, tau86)
        )
    
        tau918 = (
            einsum("rp,rq,rq->pq", tau280, tau667, tau88)
        )
    
        tau919 = (
            einsum("ip,pqi->pq", a.x3, tau915)
        )
    
        tau920 = (
            einsum("rp,rq,rq->pq", tau134, tau667, tau82)
        )
    
        tau921 = (
            einsum("rp,rq,rq->pq", tau387, tau665, tau79)
        )
    
        tau922 = (
            einsum("qi,pqi->pq", tau674, tau332)
        )
    
        tau923 = (
            einsum("qj,pij->pqi", tau73, tau121)
        )
    
        tau924 = (
            einsum("pk,pijk->pij", tau73, tau383)
        )
    
        tau925 = (
            einsum("kp,pkij->pij", a.x3, tau36)
        )
    
        tau926 = (
            einsum("qj,pij->pqi", tau73, tau925)
        )
    
        tau927 = (
            einsum("qj,pij->pqi", tau674, tau68)
        )
    
        tau928 = (
            einsum("kp,pkij->pij", a.y4, tau682)
        )
    
        tau929 = (
            einsum("jp,pqji->pqi", a.x3, tau147)
        )
    
        tau930 = (
            einsum("ji,jp->pi", h.f.oo, a.x4)
        )
    
        tau931 = (
            einsum("qp,iq->pi", tau585, a.x4)
        )
    
        tau932 = (
            einsum("pk,pijk->pij", tau674, tau260)
        )
    
        tau933 = (
            einsum("kp,pkij->pij", a.y4, tau671)
        )
    
        tau934 = (
            einsum("cp,pcab->pab", a.y1, tau283)
        )
    
        tau935 = (
            einsum("bi,pab->pia", a.t1, tau934)
        )
    
        tau936 = (
            einsum("ai,pja->pij", a.t1, tau935)
        )
    
        tau937 = (
            einsum("bp,piba->pia", a.y1, tau532)
        )
    
        tau938 = (
            einsum("ai,pja->pij", a.t1, tau937)
        )
    
        tau939 = (
            einsum("pr,rq,rq,rq->pq", tau102, tau665, tau78, tau82)
        )
    
        tau940 = (
            einsum("pk,pijk->pij", tau73, tau594)
        )
    
        tau941 = (
            einsum("qi,pqi->pq", tau674, tau412)
        )
    
        tau942 = (
            einsum("kp,pkij->pij", a.x4, tau36)
        )
    
        tau943 = (
            einsum("qj,pij->pqi", tau73, tau942)
        )
    
        tau944 = (
            einsum("pr,rq,rq,ir->pqi", tau465, tau665, tau78, a.x4)
        )
    
        tau945 = (
            einsum("rp,rq,rq,ir->pqi", tau152, tau667, tau82, a.x4)
        )
    
        tau946 = (
            einsum("pr,rq,rq,rq->pq", tau351, tau665, tau78, tau82)
        )
    
        tau947 = (
            einsum("qi,pqi->pq", tau73, tau746)
        )
    
        tau948 = (
            einsum("qj,pij->pqi", tau674, tau49)
        )
    
        tau949 = (
            einsum("jp,pqji->pqi", a.x3, tau764)
        )
    
        tau950 = (
            einsum("qj,pij->pqi", tau674, tau226)
        )
    
        tau951 = (
            einsum("qj,pji->pqi", tau73, tau47)
        )
    
        tau952 = (
            einsum("aq,pqa->pq", a.y1, tau580)
        )
    
        tau953 = (
            einsum("lp,ijlk->pijk", a.x4, tau259)
        )
    
        tau954 = (
            einsum("kq,pikj->pqij", a.y4, tau953)
        )
    
        tau955 = (
            einsum("jp,pqji->pqi", a.x3, tau954)
        )
    
        tau956 = (
            einsum("pr,rq,rq->pq", tau265, tau665, tau78)
        )
    
        tau957 = (
            einsum("qj,pij->pqi", tau73, tau49)
        )
    
        tau958 = (
            einsum("bp,piba->pia", a.y2, tau729)
        )
    
        tau959 = (
            einsum("ai,pja->pij", a.t1, tau958)
        )
    
        tau960 = (
            einsum("kp,pikj->pij", a.x3, tau113)
        )
    
        tau961 = (
            einsum("jq,pji->pqi", a.y4, tau960)
        )
    
        tau962 = (
            einsum("kp,pkij->pij", a.y4, tau341)
        )
    
        tau963 = (
            einsum("pk,pijk->pij", tau73, tau260)
        )
    
        tau964 = (
            einsum("qj,pij->pqi", tau674, tau3)
        )
    
        tau965 = (
            einsum("lp,lijk->pijk", a.y4, tau112)
        )
    
        tau966 = (
            einsum("pk,pijk->pij", tau73, tau965)
        )
    
        tau967 = (
            einsum("qi,pqi->pq", tau73, tau718)
        )
    
        tau968 = (
            einsum("pr,rq,rq,rq->pq", tau318, tau667, tau86, tau87)
        )
    
        tau969 = (
            einsum("ap,pija->pij", a.y1, tau257)
        )
    
        tau970 = (
            einsum("rp,rq,rq,ir->pqi", tau397, tau667, tau82, a.x4)
        )
    
        tau971 = (
            einsum("jp,pqji->pqi", a.x3, tau537)
        )
    
        tau972 = (
            einsum("qj,pji->pqi", tau674, tau10)
        )
    
        tau973 = (
            einsum("kp,pkij->pij", a.y4, tau253)
        )
    
        tau974 = (
            einsum("jp,pqji->pqi", a.x4, tau236)
        )
    
        tau975 = (
            einsum("qj,pij->pqi", tau674, tau311)
        )
    
        tau976 = (
            einsum("pr,rq,rq->pq", tau479, tau667, tau87)
        )
    
        tau977 = (
            einsum("qj,pij->pqi", tau674, tau427)
        )
    
        tau978 = (
            einsum("ji,jp->pi", tau12, a.x4)
        )
    
        tau979 = (
            einsum("qj,pij->pqi", tau73, tau34)
        )
    
        tau980 = (
            einsum("qj,pji->pqi", tau674, tau39)
        )
    
        tau981 = (
            einsum("pr,rq,rq,ir->pqi", tau270, tau665, tau82, a.x4)
        )
    
        tau982 = (
            einsum("kp,pikj->pij", a.y4, tau726)
        )
    
        tau983 = (
            einsum("pr,rq,rq,rq->pq", tau239, tau665, tau78, tau82)
        )
    
        tau984 = (
            einsum("ji,jp->pi", tau28, a.x3)
        )
    
        tau985 = (
            einsum("qj,pij->pqi", tau73, tau311)
        )
    
        tau986 = (
            einsum("ji,jp->pi", h.f.oo, a.x3)
        )
    
        tau987 = (
            einsum("qj,pij->pqi", tau73, tau3)
        )
    
        tau988 = (
            einsum("pr,rq,rq,ir->pqi", tau208, tau82, tau87, a.x4)
        )
    
        tau989 = (
            einsum("qi,pqi->pq", tau73, tau776)
        )
    
        tau990 = (
            einsum("pr,rq,rq,rq->pq", tau144, tau665, tau78, tau86)
        )
    
        tau991 = (
            einsum("jp,pqji->pqi", a.x3, tau793)
        )
    
        tau992 = (
            einsum("qj,pij->pqi", tau674, tau942)
        )
    
        tau993 = (
            einsum("jp,pqji->pqi", a.x4, tau764)
        )
    
        tau994 = (
            einsum("jp,pqji->pqi", a.x4, tau723)
        )
    
        tau995 = (
            einsum("jp,pqji->pqi", a.x3, tau236)
        )
    
        tau996 = (
            einsum("pr,rq,rq,ir->pqi", tau270, tau82, tau87, a.x4)
        )
    
        tau997 = (
            einsum("pr,rq,rq,rq->pq", tau239, tau667, tau82, tau87)
        )
    
        tau998 = (
            einsum("qj,pji->pqi", tau73, tau10)
        )
    
        tau999 = (
            einsum("pr,rq,rq,ir->pqi", tau245, tau78, tau82, a.x4)
        )
    
        tau1000 = (
            einsum("pr,rq,rq,rq->pq", tau351, tau667, tau82, tau87)
        )
    
        tau1001 = (
            einsum("rp,rq,ir->pqi", tau265, tau82, a.x4)
        )
    
        tau1002 = (
            einsum("kp,pkij->pij", a.x4, tau9)
        )
    
        tau1003 = (
            einsum("qj,pij->pqi", tau73, tau1002)
        )
    
        tau1004 = (
            einsum("jp,pqji->pqi", a.x4, tau688)
        )
    
        tau1005 = (
            einsum("qj,pji->pqi", tau73, tau39)
        )
    
        tau1006 = (
            einsum("qj,pij->pqi", tau674, tau925)
        )
    
        tau1007 = (
            einsum("kp,pikj->pij", a.y4, tau334)
        )
    
        tau1008 = (
            einsum("rp,rq,rq,ir->pqi", tau397, tau78, tau82, a.x4)
        )
    
        tau1009 = (
            einsum("qj,pji->pqi", tau73, tau37)
        )
    
        tau1010 = (
            einsum("pr,rq,rq,rq->pq", tau465, tau667, tau86, tau87)
        )
    
        tau1011 = (
            einsum("jp,pqji->pqi", a.x3, tau688)
        )
    
        tau1012 = (
            einsum("qi,pqi->pq", tau674, tau390)
        )
    
        tau1013 = (
            einsum("jp,pqji->pqi", a.x4, tau346)
        )
    
        tau1014 = (
            einsum("pr,rq,ir->pqi", tau309, tau665, a.x4)
        )
    
        tau1015 = (
            einsum("rp,rq,rq,ir->pqi", tau152, tau78, tau82, a.x4)
        )
    
        tau1016 = (
            einsum("aq,pqa->pq", a.y1, tau408)
        )
    
        tau1017 = (
            einsum("ji,jp->pi", tau28, a.x4)
        )
    
        tau1018 = (
            einsum("pr,rq,rq,ir->pqi", tau295, tau665, tau82, a.x4)
        )
    
        tau1019 = (
            einsum("rp,rq,ir->pqi", tau230, tau78, a.x4)
        )
    
        tau1020 = (
            einsum("jp,pqji->pqi", a.x4, tau373)
        )
    
        tau1021 = (
            einsum("rp,rq,ir->pqi", tau479, tau82, a.x4)
        )
    
        tau1022 = (
            einsum("jp,pqji->pqi", a.x3, tau346)
        )
    
        tau1023 = (
            einsum("pk,pijk->pij", tau674, tau965)
        )
    
        tau1024 = (
            einsum("ji,jp->pi", tau12, a.x3)
        )
    
        tau1025 = (
            einsum("qj,pji->pqi", tau674, tau47)
        )
    
        tau1026 = (
            einsum("pk,pijk->pij", tau674, tau594)
        )
    
        tau1027 = (
            einsum("rq,rq,pr,ir->pqi", tau82, tau87, tau94, a.x4)
        )
    
        tau1028 = (
            einsum("jp,pqji->pqi", a.x3, tau373)
        )
    
        tau1029 = (
            einsum("pr,rq,rq,ir->pqi", tau569, tau78, tau82, a.x4)
        )
    
        tau1030 = (
            einsum("qj,pij->pqi", tau674, tau1002)
        )
    
        tau1031 = (
            einsum("pr,rq,rq->pq", tau265, tau667, tau87)
        )
    
        tau1032 = (
            einsum("kp,pikj->pij", a.x4, tau140)
        )
    
        tau1033 = (
            einsum("jq,pji->pqi", a.y4, tau1032)
        )
    
        tau1034 = (
            einsum("pk,pijk->pij", tau674, tau383)
        )
    
        tau1035 = (
            einsum("pr,rq,rq,ir->pqi", tau208, tau665, tau82, a.x4)
        )
    
        tau1036 = (
            einsum("qj,pij->pqi", tau674, tau121)
        )
    
        tau1037 = (
            einsum("pr,rq,rq,rq->pq", tau102, tau667, tau82, tau87)
        )
    
        tau1038 = (
            einsum("qj,pij->pqi", tau73, tau427)
        )
    
        tau1039 = (
            einsum("kp,pkij->pij", a.x3, tau9)
        )
    
        tau1040 = (
            einsum("qj,pij->pqi", tau674, tau1039)
        )
    
        tau1041 = (
            einsum("pr,rq,rq,ir->pqi", tau465, tau667, tau87, a.x4)
        )
    
        tau1042 = (
            einsum("qj,pij->pqi", tau674, tau34)
        )
    
        tau1043 = (
            einsum("qj,pij->pqi", tau73, tau1039)
        )
    
        tau1044 = (
            einsum("pr,rq,rq,ir->pqi", tau245, tau667, tau82, a.x4)
        )
    
        tau1045 = (
            einsum("pr,rq,ir->pqi", tau309, tau87, a.x4)
        )
    
        tau1046 = (
            einsum("jp,pqji->pqi", a.x4, tau793)
        )
    
        tau1047 = (
            einsum("qj,pij->pqi", tau73, tau68)
        )
    
        tau1048 = (
            einsum("jp,pqji->pqi", a.x3, tau723)
        )
    
        tau1049 = (
            einsum("pr,rq,rq->pq", tau479, tau665, tau78)
        )
    
        tau1050 = (
            einsum("qj,pji->pqi", tau674, tau37)
        )
    
        tau1051 = (
            einsum("qj,pij->pqi", tau73, tau226)
        )
    
        tau1052 = (
            einsum("pr,rq,rq,rq->pq", tau371, tau667, tau82, tau87)
        )
    
        tau1053 = (
            einsum("pr,rq,rq,ir->pqi", tau295, tau82, tau87, a.x4)
        )
    
        tau1054 = (
            einsum("pr,rq,rq,rq->pq", tau144, tau667, tau86, tau87)
        )
    
        tau1055 = (
            einsum("rp,rq,ir->pqi", tau230, tau667, a.x4)
        )
    
        tau1056 = (
            einsum("rq,rq,pr,ir->pqi", tau665, tau82, tau94, a.x4)
        )
    
        tau1057 = (
            einsum("pr,rq,rq,rq->pq", tau465, tau665, tau78, tau86)
        )
    
        tau1058 = (
            einsum("pr,rq,rq,ir->pqi", tau569, tau667, tau82, a.x4)
        )
    
        tau1059 = (
            einsum("pr,rq,rq,rq->pq", tau318, tau665, tau78, tau86)
        )
    
        tau1060 = (
            einsum("jp,pqji->pqi", a.x4, tau147)
        )
    
        tau1061 = (
            einsum("pr,rq,rq,rq->pq", tau371, tau665, tau78, tau82)
        )
    
        tau1062 = (
            einsum("lp,lijk->pijk", a.y3, tau167)
        )
    
        tau1063 = (
            einsum("pk,pijk->pij", tau674, tau1062)
        )
    
        tau1064 = (
            einsum("rp,rq,ir->pqi", tau566, tau78, a.x4)
        )
    
        tau1065 = (
            einsum("jp,pqji->pqi", a.x3, tau422)
        )
    
        tau1066 = (
            einsum("pr,rq,rq,ir->pqi", tau245, tau78, tau79, a.x4)
        )
    
        tau1067 = (
            einsum("kp,pikj->pij", a.y3, tau334)
        )
    
        tau1068 = (
            einsum("rp,rq,ir->pqi", tau479, tau79, a.x4)
        )
    
        tau1069 = (
            einsum("kp,pkij->pij", a.y3, tau740)
        )
    
        tau1070 = (
            einsum("kp,pkij->pij", a.y3, tau671)
        )
    
        tau1071 = (
            einsum("pr,rq,rq,ir->pqi", tau295, tau665, tau79, a.x4)
        )
    
        tau1072 = (
            einsum("pk,pijk->pij", tau73, tau1062)
        )
    
        tau1073 = (
            einsum("lp,iljk->pijk", a.y3, tau259)
        )
    
        tau1074 = (
            einsum("pk,pikj->pij", tau674, tau1073)
        )
    
        tau1075 = (
            einsum("jq,pij->pqi", a.y3, tau169)
        )
    
        tau1076 = (
            einsum("rp,rq,rq,ir->pqi", tau152, tau667, tau79, a.x4)
        )
    
        tau1077 = (
            einsum("pr,rq,rq,rq->pq", tau351, tau667, tau79, tau87)
        )
    
        tau1078 = (
            einsum("pr,rq,ir->pqi", tau280, tau78, a.x4)
        )
    
        tau1079 = (
            einsum("jq,pji->pqi", a.y3, tau1032)
        )
    
        tau1080 = (
            einsum("jq,pji->pqi", a.y3, tau169)
        )
    
        tau1081 = (
            einsum("pr,rq,rq,rq->pq", tau239, tau667, tau79, tau87)
        )
    
        tau1082 = (
            einsum("kp,pikj->pij", a.y3, tau473)
        )
    
        tau1083 = (
            einsum("rp,rq,ir->pqi", tau273, tau87, a.x4)
        )
    
        tau1084 = (
            einsum("pr,rq,rq,rq->pq", tau318, tau667, tau87, tau88)
        )
    
        tau1085 = (
            einsum("kq,pikj->pqij", a.y3, tau953)
        )
    
        tau1086 = (
            einsum("jp,pqji->pqi", a.x3, tau1085)
        )
    
        tau1087 = (
            einsum("pr,rq,rq,ir->pqi", tau245, tau667, tau79, a.x4)
        )
    
        tau1088 = (
            einsum("pr,rq,ir->pqi", tau520, tau665, a.x4)
        )
    
        tau1089 = (
            einsum("kp,pkij->pij", a.y3, tau682)
        )
    
        tau1090 = (
            einsum("pr,rq,rq,rq->pq", tau351, tau665, tau78, tau79)
        )
    
        tau1091 = (
            einsum("rq,rq,pr,ir->pqi", tau665, tau79, tau94, a.x4)
        )
    
        tau1092 = (
            einsum("pk,pijk->pij", tau73, tau163)
        )
    
        tau1093 = (
            einsum("pr,rq,ir->pqi", tau280, tau667, a.x4)
        )
    
        tau1094 = (
            einsum("pr,rq,rq,ir->pqi", tau569, tau667, tau79, a.x4)
        )
    
        tau1095 = (
            einsum("jq,pji->pqi", a.y3, tau960)
        )
    
        tau1096 = (
            einsum("pk,pikj->pij", tau73, tau1073)
        )
    
        tau1097 = (
            einsum("rp,rq,ir->pqi", tau566, tau667, a.x4)
        )
    
        tau1098 = (
            einsum("pr,rq,rq,rq->pq", tau239, tau665, tau78, tau79)
        )
    
        tau1099 = (
            einsum("kp,pikj->pij", a.y3, tau726)
        )
    
        tau1100 = (
            einsum("pr,rq,rq,rq->pq", tau144, tau667, tau87, tau88)
        )
    
        tau1101 = (
            einsum("pr,rq,rq,ir->pqi", tau569, tau78, tau79, a.x4)
        )
    
        tau1102 = (
            einsum("rp,rq,ir->pqi", tau265, tau79, a.x4)
        )
    
        tau1103 = (
            einsum("pr,rq,rq,rq->pq", tau102, tau665, tau78, tau79)
        )
    
        tau1104 = (
            einsum("pr,rq,ir->pqi", tau520, tau87, a.x4)
        )
    
        tau1105 = (
            einsum("pr,rq,rq,ir->pqi", tau270, tau79, tau87, a.x4)
        )
    
        tau1106 = (
            einsum("pr,rq,rq,rq->pq", tau318, tau665, tau78, tau88)
        )
    
        tau1107 = (
            einsum("pr,rq,rq,rq->pq", tau371, tau667, tau79, tau87)
        )
    
        tau1108 = (
            einsum("pr,rq,rq,rq->pq", tau465, tau667, tau87, tau88)
        )
    
        tau1109 = (
            einsum("rp,rq,rq,ir->pqi", tau152, tau78, tau79, a.x4)
        )
    
        tau1110 = (
            einsum("lp,lijk->pijk", a.y3, tau112)
        )
    
        tau1111 = (
            einsum("pk,pijk->pij", tau674, tau1110)
        )
    
        tau1112 = (
            einsum("pr,rq,rq,ir->pqi", tau295, tau79, tau87, a.x4)
        )
    
        tau1113 = (
            einsum("pk,pijk->pij", tau73, tau1110)
        )
    
        tau1114 = (
            einsum("pr,rq,rq,rq->pq", tau144, tau665, tau78, tau88)
        )
    
        tau1115 = (
            einsum("pr,rq,rq,rq->pq", tau465, tau665, tau78, tau88)
        )
    
        tau1116 = (
            einsum("kp,pkij->pij", a.y3, tau341)
        )
    
        tau1117 = (
            einsum("pr,rq,rq,ir->pqi", tau208, tau665, tau79, a.x4)
        )
    
        tau1118 = (
            einsum("kp,pkij->pij", a.y3, tau473)
        )
    
        tau1119 = (
            einsum("pr,rq,rq,ir->pqi", tau270, tau665, tau79, a.x4)
        )
    
        tau1120 = (
            einsum("kp,pkij->pij", a.y3, tau253)
        )
    
        tau1121 = (
            einsum("pr,rq,rq,rq->pq", tau371, tau665, tau78, tau79)
        )
    
        tau1122 = (
            einsum("rq,rq,pr,ir->pqi", tau79, tau87, tau94, a.x4)
        )
    
        tau1123 = (
            einsum("rp,rq,ir->pqi", tau273, tau665, a.x4)
        )
    
        tau1124 = (
            einsum("pk,pijk->pij", tau674, tau163)
        )
    
        tau1125 = (
            einsum("pr,rq,rq,rq->pq", tau102, tau667, tau79, tau87)
        )
    
        tau1126 = (
            einsum("rp,rq,rq,ir->pqi", tau397, tau78, tau79, a.x4)
        )
    
        tau1127 = (
            einsum("pr,rq,rq,ir->pqi", tau208, tau79, tau87, a.x4)
        )
    
        tau1128 = (
            einsum("rp,rq,rq,ir->pqi", tau397, tau667, tau79, a.x4)
        )
    
        tau1129 = (
            einsum("kp,pikj->pij", a.y3, tau740)
        )
    
        rt1 = (
            - 2 * einsum("pi,ap->ai", tau4, a.x2)
            - einsum("p,ap,ip->ai", tau6, a.x1, a.x4)
            + 4 * einsum("o,oai->ai", tau7, h.l.pvo)
            - 2 * einsum("pi,ap->ai", tau11, a.x2)
            - 2 * einsum("aj,ji->ai", a.t1, tau12)
            - 4 * einsum("aj,ij->ai", a.t1, tau14)
            + 4 * einsum("p,ap,ip->ai", tau16, a.x2, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau19, a.x1, a.x3)
            + einsum("pa,ip->ai", tau22, a.x3)
            - einsum("pa,ip->ai", tau26, a.x3)
            - 4 * einsum("aj,ji->ai", a.t1, tau28)
            + einsum("p,ap,ip->ai", tau29, a.x1, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau31, a.x2, a.x4)
            + 2 * einsum("ia->ai", h.f.ov.conj())
            + einsum("pi,ap->ai", tau35, a.x1)
            + einsum("pi,ap->ai", tau38, a.x1)
            - 2 * einsum("pi,ap->ai", tau40, a.x1)
            + 2 * einsum("aj,ij->ai", a.t1, tau41)
            - 2 * einsum("pa,ip->ai", tau44, a.x3)
            - einsum("p,ap,ip->ai", tau46, a.x2, a.x3)
            + einsum("pi,ap->ai", tau48, a.x2)
            - 2 * einsum("pi,ap->ai", tau50, a.x1)
            - 2 * einsum("bj,jiab->ai", a.t1, tau51)
            + 4 * einsum("bi,ab->ai", a.t1, tau52)
            + einsum("pa,ip->ai", tau54, a.x4)
            + 2 * einsum("pa,ip->ai", tau55, a.x4)
            - 2 * einsum("pa,ip->ai", tau57, a.x4)
            + 2 * einsum("p,ap,ip->ai", tau58, a.x2, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau59, a.x2, a.x3)
            + 2 * einsum("aj,ij->ai", a.t1, tau60)
            + 2 * einsum("pa,ip->ai", tau63, a.x3)
            - einsum("pa,ip->ai", tau64, a.x4)
            + 2 * einsum("ab,bi->ai", h.f.vv, a.t1)
            + 4 * einsum("p,ap,ip->ai", tau66, a.x1, a.x3)
            - 2 * einsum("bi,ba->ai", a.t1, tau67)
            - 2 * einsum("ji,aj->ai", h.f.oo, a.t1)
            + einsum("pi,ap->ai", tau69, a.x2)
            + 2 * einsum("p,ap,ip->ai", tau70, a.x1, a.x3)
            + einsum("p,ap,ip->ai", tau71, a.x2, a.x3)
            - 2 * einsum("p,ap,ip->ai", tau72, a.x1, a.x4)
        )
    
        rx1 = (
            - 2 * einsum("pi,pia->ap", tau73, tau77)
            + einsum("qp,qp,qp,aq->ap", tau78, tau79, tau81, a.x1)
            - einsum("qp,qp,aq->ap", tau82, tau84, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau79, tau85, tau86, a.x1)
            + einsum("qp,qp,qpa->ap", tau87, tau88, tau91)
            + einsum("qp,aq->ap", tau95, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau78, tau79, tau97, a.x1)
            - einsum("qp,qp,aq->ap", tau100, tau86, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau103, tau78, tau82, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau78, tau88, tau105)
            - einsum("qp,qp,aq->ap", tau109, tau88, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau111, tau79, tau87, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau116, tau78, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau117, tau82, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau118, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau123, tau82, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau124, tau78, tau82, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau126, tau78, tau82, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau129, tau79, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau131, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau135, tau79, a.x1) / 2
            + einsum("pi,pia->ap", tau73, tau139)
            - einsum("qp,qp,aq->ap", tau143, tau87, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau145, tau78, tau86, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau149, tau82, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau150, tau79, tau86, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau153, tau82, tau88, a.x2)
            + einsum("qp,qp,aq->ap", tau154, tau88, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau157, tau78, a.x1)
            + einsum("qp,qp,qpa->ap", tau78, tau82, tau159)
            - einsum("qp,qp,qpa->ap", tau78, tau79, tau162)
            + 2 * einsum("ai,pi->ap", a.t1, tau165)
            - einsum("qp,qp,aq->ap", tau171, tau87, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau79, tau87, tau175)
            - einsum("qp,qp,aq->ap", tau179, tau86, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau180) / 2
            - einsum("qp,qp,aq->ap", tau185, tau88, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau187, tau78, tau82, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau191, tau82, a.x2) / 2
            + 2 * einsum("qp,qp,qpa->ap", tau78, tau86, tau194)
            + 2 * einsum("qp,qp,qp,aq->ap", tau196, tau82, tau88, a.x2)
            + einsum("qa,qp,qp,qp->ap", tau197, tau78, tau82, tau88) / 2
            - einsum("qp,qp,aq->ap", tau200, tau82, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau204, tau88, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau205, tau79, tau87, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau207, tau78, tau88, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau209, tau79, tau86, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau211, tau82, tau88, a.x2)
            + einsum("qa,qp,qp,qp->ap", tau212, tau79, tau86, tau87)
            + einsum("qp,qp,qpa->ap", tau78, tau88, tau214)
            + einsum("qp,qp,aq->ap", tau216, tau86, a.x1) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau218, tau78, tau79, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau220)
            - einsum("qp,qp,qpa->ap", tau87, tau88, tau222) / 2
            - einsum("qa,qp,qp,qp->ap", tau223, tau78, tau82, tau88) / 2
            + einsum("qp,qp,qp,aq->ap", tau79, tau87, tau97, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau86, tau87, tau228) / 2
            - einsum("qp,qp,aq->ap", tau231, tau86, a.x1) / 4
            - einsum("qp,qp,qpa->ap", tau78, tau79, tau233) / 2
            + einsum("qp,qp,qp,aq->ap", tau234, tau78, tau82, a.x1)
            + einsum("qp,qp,aq->ap", tau238, tau86, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau240, tau78, tau82, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau240, tau82, tau87, a.x2) / 2
            + einsum("qa,qp,qp,qp->ap", tau223, tau78, tau79, tau86)
            + einsum("qp,qp,aq->ap", tau244, tau82, a.x2) / 2
            - einsum("qp,aq->ap", tau246, a.x2) / 4
            + 2 * einsum("qp,qp,qpa->ap", tau78, tau86, tau250)
            - einsum("qp,qp,qpa->ap", tau78, tau82, tau252) / 2
            - 2 * einsum("qa,qp,qp,qp->ap", tau212, tau82, tau87, tau88)
            + einsum("ai,pi->ap", a.t1, tau255)
            + 2 * einsum("ip,pia->ap", a.y3, tau258)
            + 2 * einsum("ai,pi->ap", a.t1, tau262)
            - 2 * einsum("qp,qp,qpa->ap", tau82, tau87, tau264)
            - einsum("qp,qp,aq->ap", tau266, tau78, a.x1) / 4
            + einsum("qp,qp,qpa->ap", tau78, tau86, tau267)
            - einsum("qp,qp,qp,aq->ap", tau79, tau81, tau87, a.x2) / 2
            - einsum("qp,aq->ap", tau268, a.x2)
            + einsum("qp,qp,aq->ap", tau116, tau87, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau271, tau82, tau88, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau274, tau86, a.x1) / 2
            - 2 * einsum("qp,qp,aq->ap", tau275, tau88, a.x2)
            - 2 * einsum("qp,qp,qp,aq->ap", tau276, tau79, tau86, a.x1)
            - einsum("qp,qp,aq->ap", tau278, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau281, tau86, a.x2)
            + 2 * einsum("ip,pia->ap", a.y3, tau286)
            - einsum("qp,qp,aq->ap", tau287, tau88, a.x1) / 4
            + einsum("pi,pia->ap", tau73, tau290)
            - einsum("qp,qp,aq->ap", tau291, tau79, a.x2)
            + einsum("qp,aq->ap", tau292, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau293, tau86, tau87, a.x2) / 2
            + 2 * einsum("qp,qp,qpa->ap", tau82, tau87, tau294)
            + einsum("qp,aq->ap", tau296, a.x1) / 2
            + 2 * einsum("qa,qp,qp,qp->ap", tau297, tau82, tau87, tau88)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau298)
            - einsum("qp,qp,aq->ap", tau300, tau78, a.x1) / 2
            - einsum("ai,pi->ap", a.t1, tau301)
            - einsum("qp,qp,qpa->ap", tau86, tau87, tau304)
            - einsum("qp,qp,qpa->ap", tau87, tau88, tau307)
            + einsum("qp,qp,aq->ap", tau310, tau86, a.x2) / 2
            - einsum("qa,qp,qp,qp->ap", tau197, tau78, tau79, tau86)
            + einsum("qp,qp,aq->ap", tau313, tau79, a.x1)
            - einsum("qp,qp,qpa->ap", tau78, tau82, tau314) / 2
            + einsum("qp,qp,qp,aq->ap", tau124, tau82, tau87, a.x2)
            - einsum("qp,qp,aq->ap", tau315, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau317, tau86, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau319, tau86, tau87, a.x2) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau321, tau87, tau88, a.x2)
            + einsum("qp,qp,aq->ap", tau323, tau82, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau325, tau88, a.x1)
            + einsum("qp,qp,qpa->ap", tau78, tau79, tau327) / 2
            - 2 * einsum("qp,qp,qpa->ap", tau78, tau86, tau329)
            + einsum("qp,qp,aq->ap", tau330, tau78, a.x1) / 2
            - einsum("qp,aq->ap", tau331, a.x1)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau333) / 2
            - 2 * einsum("ai,pi->ap", a.t1, tau336)
            + einsum("qp,qp,qp,aq->ap", tau337, tau78, tau79, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau339, tau87, tau88, a.x2)
            - 2 * einsum("ai,pi->ap", a.t1, tau343)
            + einsum("ai,pi->ap", a.t1, tau345)
            + 2 * einsum("qp,qp,aq->ap", tau348, tau79, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau349, tau79, tau87, a.x2) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau207, tau87, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau352, tau79, tau87, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau355, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau357, tau86, a.x2)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau298) / 2
            - einsum("qp,qp,aq->ap", tau359, tau82, a.x1)
            - 2 * einsum("qp,qp,qp,aq->ap", tau360, tau78, tau79, a.x1)
            + einsum("qa,qp,qp,qp->ap", tau361, tau82, tau87, tau88)
            - einsum("qp,qp,qpa->ap", tau86, tau87, tau363)
            + 2 * einsum("qp,qp,qpa->ap", tau82, tau87, tau364)
            + einsum("qp,qp,qpa->ap", tau86, tau87, tau367) / 2
            - einsum("ip,pia->ap", a.y3, tau368)
            - einsum("qa,qp,qp,qp->ap", tau369, tau79, tau86, tau87) / 2
            - einsum("qp,qp,qpa->ap", tau82, tau87, tau370)
            - einsum("qp,qp,qp,aq->ap", tau372, tau78, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau375, tau88, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau376, tau82, a.x2)
            - einsum("qp,qp,aq->ap", tau378, tau86, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau143, tau78, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau380, tau82, tau88, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau78, tau88, tau381)
            + einsum("qp,aq->ap", tau382, a.x2) / 2
            - einsum("ai,pi->ap", a.t1, tau385)
            - einsum("qp,qp,qp,aq->ap", tau386, tau82, tau88, a.x1) / 2
            + 2 * einsum("qp,qp,aq->ap", tau388, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau389, tau82, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau391) / 2
            + einsum("qp,qp,qpa->ap", tau87, tau88, tau392) / 2
            - einsum("qp,qp,aq->ap", tau393, tau88, a.x1)
            + einsum("qa,qp,qp,qp->ap", tau394, tau79, tau86, tau87) / 2
            + 2 * einsum("qp,qp,aq->ap", tau395, tau88, a.x2)
            + einsum("qp,aq->ap", tau398, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau78, tau79, tau400)
            - einsum("qp,qp,qp,aq->ap", tau401, tau79, tau86, a.x2) / 2
            - einsum("qp,aq->ap", tau402, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau339, tau78, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau403, tau79, a.x2)
            - einsum("qp,qp,aq->ap", tau404, tau82, a.x2) / 4
            - einsum("qa,qp,qp,qp->ap", tau405, tau78, tau82, tau88)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau408)
            - einsum("qp,qp,qp,aq->ap", tau211, tau79, tau86, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau372, tau79, tau87, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau409, tau88, a.x2) / 2
            + einsum("qa,qp,qp,qp->ap", tau410, tau78, tau82, tau88) / 2
            - einsum("qp,qp,qp,aq->ap", tau187, tau82, tau87, a.x2)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau220) / 2
            + einsum("qp,qp,qp,aq->ap", tau411, tau82, tau88, a.x2) / 2
            - einsum("qa,qp,qp,qp->ap", tau297, tau79, tau86, tau87)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau413) / 2
            + einsum("qp,qp,qpa->ap", tau86, tau87, tau415)
            + einsum("qp,qp,aq->ap", tau416, tau79, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau418, tau78, tau88, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau171, tau78, a.x1)
            + einsum("qp,qp,aq->ap", tau419, tau88, a.x1)
            - einsum("qp,qp,aq->ap", tau330, tau87, a.x2) / 4
            - 2 * einsum("qp,qp,qpa->ap", tau78, tau86, tau420)
            + einsum("qp,qp,aq->ap", tau424, tau87, a.x2)
            - einsum("qp,qp,aq->ap", tau426, tau82, a.x2)
            - einsum("qp,qp,aq->ap", tau429, tau86, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau430, tau78, tau88, a.x1) / 2
            - 2 * einsum("ai,pi->ap", a.t1, tau432)
            + einsum("qp,qp,aq->ap", tau434, tau86, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau87, tau436)
            + einsum("qp,qp,qpa->ap", tau79, tau87, tau438)
            + 2 * einsum("qp,qp,aq->ap", tau440, tau79, a.x1)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau408) / 2
            + 2 * einsum("qp,qp,aq->ap", tau441, tau79, a.x1)
            - einsum("qp,qp,aq->ap", tau442, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau444, tau82, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau78, tau88, tau446) / 2
            + einsum("qp,qp,aq->ap", tau447, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau448, tau88, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau450, tau78, tau86, a.x1)
            - einsum("qp,qp,aq->ap", tau452, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau453, tau86, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau78, tau88, tau455) / 2
            - einsum("qp,qp,qp,aq->ap", tau456, tau79, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau457, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau460, tau86, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau461, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau462, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau463, tau82, a.x2) / 4
            - einsum("qa,qp,qp,qp->ap", tau361, tau79, tau86, tau87) / 2
            - einsum("qp,qp,qp,aq->ap", tau466, tau87, tau88, a.x2)
            + 2 * einsum("qa,qp,qp,qp->ap", tau405, tau78, tau79, tau86)
            - einsum("qp,qp,qpa->ap", tau78, tau79, tau468)
            + einsum("qp,qp,qp,aq->ap", tau469, tau78, tau82, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau87, tau88, tau470) / 2
            - einsum("qp,aq->ap", tau471, a.x2) / 4
            - 2 * einsum("ai,pi->ap", a.t1, tau475)
            + einsum("qp,qp,qp,aq->ap", tau476, tau87, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau380, tau79, tau86, a.x1)
            - einsum("qp,qp,aq->ap", tau477, tau82, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau478, tau82, a.x1)
            + einsum("qp,qp,aq->ap", tau480, tau87, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau481, tau82, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau82, tau87, tau483)
            - einsum("qp,qp,aq->ap", tau485, tau82, a.x2) / 4
            + einsum("qp,qp,qpa->ap", tau78, tau79, tau487) / 2
            - einsum("qp,qp,qp,aq->ap", tau411, tau79, tau86, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau300, tau87, a.x2)
            - einsum("qp,qp,qpa->ap", tau87, tau88, tau489)
            - einsum("qp,qp,qpa->ap", tau79, tau87, tau491) / 2
            + einsum("qp,qp,qp,aq->ap", tau492, tau78, tau88, a.x1) / 2
            + einsum("qa,qp,qp,qp->ap", tau379, tau79, tau86, tau87) / 2
            - einsum("qp,qp,aq->ap", tau493, tau79, a.x2)
            + einsum("qp,qp,aq->ap", tau494, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau495, tau82, a.x2)
            + einsum("qa,qp,qp,qp->ap", tau369, tau82, tau87, tau88)
            + einsum("qp,qp,aq->ap", tau496, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau497, tau88, a.x1)
            - einsum("qp,qp,aq->ap", tau157, tau87, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau500, tau78, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau111, tau78, tau79, a.x1)
            - 2 * einsum("qp,qp,qp,aq->ap", tau501, tau78, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau503, tau82, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau504, tau82, tau88, a.x2)
            + einsum("qp,qp,qpa->ap", tau79, tau87, tau506)
            + einsum("qp,qp,qp,aq->ap", tau508, tau86, tau87, a.x2) / 2
            + einsum("qp,qp,qpa->ap", tau87, tau88, tau510)
            - einsum("qp,qp,qp,aq->ap", tau103, tau82, tau87, a.x2)
            + einsum("qp,qp,aq->ap", tau511, tau79, a.x2) / 2
            + einsum("qp,qp,qpa->ap", tau86, tau87, tau512) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau513, tau78, tau86, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau514, tau86, tau87, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau196, tau79, tau86, a.x2)
            + einsum("qp,qp,aq->ap", tau515, tau78, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau513, tau86, tau87, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau150, tau82, tau88, a.x1) / 4
            + einsum("qp,qp,qpa->ap", tau87, tau88, tau517) / 2
            + einsum("qp,qp,qp,aq->ap", tau360, tau79, tau87, a.x2)
            + einsum("qp,qp,qpa->ap", tau78, tau82, tau518) / 2
            - einsum("qp,qp,aq->ap", tau515, tau87, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau521, tau86, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau522, tau82, a.x1)
            + 2 * einsum("qp,qp,aq->ap", tau524, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau526, tau79, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau209, tau82, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau527, tau79, a.x1)
            - einsum("qp,qp,aq->ap", tau528, tau82, a.x1) / 4
            - einsum("qp,qp,qp,aq->ap", tau319, tau78, tau86, a.x1)
            - 2 * einsum("qp,qp,qp,aq->ap", tau529, tau82, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau530, tau82, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau352, tau78, tau79, a.x1) / 2
            - einsum("qa,qp,qp,qp->ap", tau410, tau78, tau79, tau86)
            - einsum("qa,qp,qp,qp->ap", tau394, tau82, tau87, tau88)
            + 2 * einsum("ip,pia->ap", a.y4, tau534)
            + einsum("qp,qp,aq->ap", tau535, tau88, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau536, tau82, tau87, a.x2)
            + einsum("qp,qp,aq->ap", tau539, tau78, a.x1)
            + einsum("qp,qp,aq->ap", tau541, tau88, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau542, tau86, tau87, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau544, tau86, a.x1) / 2
            - einsum("qa,qp,qp,qp->ap", tau545, tau78, tau82, tau88) / 2
            - einsum("qp,qp,qp,aq->ap", tau508, tau78, tau86, a.x1)
            + einsum("qp,aq->ap", tau546, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau205, tau78, tau79, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau542, tau78, tau86, a.x1)
            - einsum("qp,qp,qpa->ap", tau78, tau86, tau548)
            + einsum("qp,qp,aq->ap", tau550, tau82, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau293, tau78, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau551, tau82, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau78, tau86, tau552)
            + einsum("qp,qp,aq->ap", tau554, tau79, a.x2) / 2
            + 2 * einsum("ip,pia->ap", a.y3, tau557)
            + einsum("ai,pi->ap", a.t1, tau559)
            + einsum("qp,qp,qp,aq->ap", tau560, tau82, tau87, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau561, tau82, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau562, tau82, tau87, a.x2)
            - 2 * einsum("pi,pia->ap", tau73, tau564)
            + einsum("qp,qp,qp,aq->ap", tau321, tau78, tau88, a.x1)
            + einsum("qp,qp,aq->ap", tau567, tau86, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau568, tau79, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau514, tau78, tau86, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau570, tau79, tau86, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau180)
            + einsum("qp,qp,qp,aq->ap", tau571, tau82, tau88, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau337, tau79, tau87, a.x2) / 4
            - einsum("ai,pi->ap", a.t1, tau572)
            + einsum("pi,pia->ap", tau73, tau574)
            - einsum("ip,pia->ap", a.y4, tau575)
            - einsum("qp,qp,qp,aq->ap", tau560, tau78, tau82, a.x1) / 4
            - 2 * einsum("qp,qp,qp,aq->ap", tau234, tau82, tau87, a.x2)
            - einsum("qp,qp,aq->ap", tau576, tau79, a.x1)
            + einsum("qp,qp,qpa->ap", tau78, tau79, tau577)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau580)
            - einsum("qp,qp,aq->ap", tau581, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau582, tau82, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau386, tau79, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau266, tau87, a.x2) / 2
            - 2 * einsum("qa,qp,qp,qp->ap", tau583, tau78, tau79, tau86)
            - einsum("qp,qp,qpa->ap", tau86, tau87, tau584) / 2
            - einsum("qp,qp,qp,aq->ap", tau586, tau86, tau87, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau587, tau79, a.x2) / 2
            - 2 * einsum("pi,pia->ap", tau73, tau589)
            + einsum("qp,qp,aq->ap", tau590, tau79, a.x2)
            + einsum("qp,aq->ap", tau591, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau492, tau87, tau88, a.x2)
            + einsum("qp,aq->ap", tau592, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau424, tau78, a.x1) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau126, tau82, tau87, a.x2)
            - einsum("qp,qp,qpa->ap", tau79, tau87, tau593)
            - einsum("pi,pia->ap", tau73, tau596)
            - einsum("qp,qp,aq->ap", tau597, tau88, a.x2)
            + einsum("qp,qp,qpa->ap", tau79, tau87, tau598) / 2
            - einsum("qp,qp,aq->ap", tau480, tau78, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau599, tau86, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau600, tau78, tau88, a.x1) / 4
            + einsum("qp,qp,qpa->ap", tau78, tau82, tau601) / 2
            - einsum("qp,qp,qp,aq->ap", tau476, tau78, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau602, tau79, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau218, tau79, tau87, a.x2)
            + einsum("ai,pi->ap", a.t1, tau604)
            - einsum("qp,qp,aq->ap", tau539, tau87, a.x2) / 2
            - 2 * einsum("qp,qp,aq->ap", tau605, tau79, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau418, tau87, tau88, a.x2)
            - 2 * einsum("pi,pia->ap", tau73, tau608)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau580) / 2
            - einsum("ip,pia->ap", a.y4, tau609)
            - einsum("qp,aq->ap", tau610, a.x1) / 4
            - einsum("qp,qp,qpa->ap", tau78, tau79, tau612) / 2
            + einsum("qp,qp,qp,aq->ap", tau117, tau79, tau86, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau613, tau79, a.x2)
            + 2 * einsum("qp,qp,aq->ap", tau614, tau88, a.x2)
            + einsum("qp,qp,qpa->ap", tau78, tau82, tau615)
            + einsum("qp,qp,aq->ap", tau500, tau87, a.x2)
            - 2 * einsum("qp,qp,aq->ap", tau616, tau88, a.x2)
            + einsum("qp,qp,aq->ap", tau617, tau86, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau618, tau82, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau619, tau88, a.x2)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau391)
            - einsum("qp,qp,qpa->ap", tau78, tau82, tau620)
            - einsum("qp,qp,qpa->ap", tau78, tau88, tau621)
            - einsum("qp,qp,aq->ap", tau622, tau79, a.x2) / 4
            - einsum("qp,qp,qp,aq->ap", tau562, tau78, tau82, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau153, tau79, tau86, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau570, tau82, tau88, a.x2)
            + einsum("qp,qp,qpa->ap", tau79, tau87, tau623) / 2
            - einsum("qp,qp,qpa->ap", tau79, tau87, tau624) / 2
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau413)
            + einsum("qa,qp,qp,qp->ap", tau545, tau78, tau79, tau86)
            - einsum("qp,qp,qpa->ap", tau78, tau82, tau625)
            + einsum("qp,qp,aq->ap", tau626, tau88, a.x1) / 2
            - einsum("qp,aq->ap", tau627, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau145, tau86, tau87, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau536, tau78, tau82, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau129, tau82, tau88, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau78, tau88, tau628) / 2
            + einsum("qp,qp,aq->ap", tau629, tau79, a.x1)
            - einsum("ip,pia->ap", a.y3, tau630)
            - einsum("qp,qp,aq->ap", tau631, tau86, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau632, tau86, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau600, tau87, tau88, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau633, tau82, tau88, a.x1)
            + einsum("qp,qp,aq->ap", tau634, tau79, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau456, tau82, tau88, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau86, tau87, tau635)
            - einsum("qp,qp,aq->ap", tau636, tau86, a.x1)
            + einsum("qa,qp,qp,qp->ap", tau583, tau78, tau82, tau88)
            - einsum("qp,qp,aq->ap", tau637, tau88, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau349, tau78, tau79, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau430, tau87, tau88, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau82, tau85, tau88, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau504, tau79, tau86, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau638, tau86, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau639, tau88, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau466, tau78, tau88, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau276, tau82, tau88, a.x1)
            + 2 * einsum("qp,qp,aq->ap", tau640, tau88, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau641, tau87, tau88, a.x2) / 2
            - 2 * einsum("qp,qp,qpa->ap", tau82, tau87, tau643)
            - einsum("qp,aq->ap", tau644, a.x1) / 4
            - einsum("qp,qp,qpa->ap", tau78, tau88, tau645) / 2
            + einsum("qp,qp,aq->ap", tau646, tau86, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau450, tau86, tau87, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau401, tau82, tau88, a.x2)
            + 2 * einsum("qp,qp,qp,aq->ap", tau633, tau79, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau647, tau82, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau571, tau79, tau86, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau648, tau88, a.x2)
            + 2 * einsum("ai,pi->ap", a.t1, tau649)
            - einsum("qp,qp,aq->ap", tau650, tau86, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau651, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau469, tau82, tau87, a.x2)
            - 2 * einsum("qp,qp,aq->ap", tau652, tau79, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau87, tau653)
            + einsum("qp,qp,qp,aq->ap", tau529, tau79, tau86, a.x2)
            + einsum("qp,qp,qpa->ap", tau78, tau86, tau654)
            + 2 * einsum("ai,pi->ap", a.t1, tau655)
            + einsum("qp,qp,qp,aq->ap", tau586, tau78, tau86, a.x1) / 2
            + einsum("pi,pia->ap", tau73, tau657)
            - einsum("qa,qp,qp,qp->ap", tau379, tau82, tau87, tau88)
            + einsum("qp,qp,aq->ap", tau658, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau659, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau660, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau641, tau78, tau88, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau661, tau86, a.x2)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau333)
            + einsum("qp,qp,aq->ap", tau662, tau82, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau271, tau79, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau663, tau82, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau501, tau86, tau87, a.x2)
            + einsum("qp,qp,aq->ap", tau664, tau88, a.x2) / 2
        )
    
        rx2 = (
            - einsum("qp,qp,aq->ap", tau666, tau88, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau542, tau667, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau668, tau79, a.x2) / 4
            - einsum("qp,qp,qpa->ap", tau665, tau82, tau653) / 2
            + einsum("qp,qp,qp,aq->ap", tau669, tau79, tau86, a.x1) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau513, tau665, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau171, tau667, a.x1) / 2
            + einsum("ai,pi->ap", a.t1, tau673)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau675)
            + einsum("qp,qp,qp,aq->ap", tau676, tau82, tau88, a.x1)
            + einsum("pi,pia->ap", tau674, tau608)
            - einsum("qp,qp,aq->ap", tau677, tau82, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau372, tau667, tau79, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau678, tau86, a.x1) / 2
            + einsum("qa,qp,qp,qp->ap", tau212, tau665, tau82, tau88)
            - 2 * einsum("pi,pia->ap", tau674, tau290)
            - einsum("qp,qp,qp,aq->ap", tau679, tau79, tau86, a.x2)
            + einsum("qp,qp,aq->ap", tau680, tau88, a.x2) / 2
            + einsum("ai,pi->ap", a.t1, tau684)
            + einsum("qp,qp,qp,aq->ap", tau476, tau667, tau88, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau685, tau82, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau686, tau82, a.x2) / 4
            - 2 * einsum("qp,qp,aq->ap", tau687, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau690, tau82, a.x2)
            - einsum("qa,qp,qp,qp->ap", tau197, tau667, tau82, tau88)
            - 2 * einsum("qp,qp,qp,aq->ap", tau691, tau79, tau86, a.x2)
            - einsum("qp,qp,qpa->ap", tau665, tau88, tau392)
            - einsum("qp,qp,qp,aq->ap", tau542, tau665, tau86, a.x2)
            + einsum("qp,qp,aq->ap", tau692, tau88, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau693, tau88, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau111, tau667, tau79, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau665, tau86, tau512)
            + einsum("qp,qp,aq->ap", tau698, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau480, tau667, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau699, tau88, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau700, tau82, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau701, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau702, tau79, a.x1)
            - einsum("qp,qp,aq->ap", tau703, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau300, tau667, a.x1)
            + einsum("qp,qp,aq->ap", tau704, tau79, a.x2)
            + einsum("qp,qp,qpa->ap", tau667, tau79, tau612)
            + einsum("qp,qp,aq->ap", tau500, tau667, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau450, tau665, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau708, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau709, tau82, a.x1)
            + einsum("qa,qp,qp,qp->ap", tau583, tau667, tau79, tau86)
            - einsum("qp,qp,aq->ap", tau710, tau79, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau492, tau665, tau88, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau79, tau400) / 2
            + einsum("qp,qp,qp,aq->ap", tau469, tau665, tau82, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau124, tau667, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau712, tau88, a.x1)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau714)
            - einsum("qp,qp,qp,aq->ap", tau514, tau667, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau480, tau665, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau715, tau79, tau86, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau685, tau79, tau86, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau716, tau88, a.x2) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau86, tau584)
            + einsum("qp,qp,qp,aq->ap", tau319, tau667, tau86, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau116, tau667, a.x1)
            - einsum("qp,qp,aq->ap", tau717, tau82, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau337, tau665, tau79, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau719)
            + einsum("qp,qp,qp,aq->ap", tau187, tau665, tau82, a.x2) / 2
            + 2 * einsum("qp,qp,aq->ap", tau721, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau722, tau86, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau725, tau88, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau667, tau82, tau625) / 2
            + einsum("ai,pi->ap", a.t1, tau728)
            - 2 * einsum("qp,qp,qpa->ap", tau667, tau88, tau214)
            + 2 * einsum("ip,pia->ap", a.y3, tau731)
            - einsum("qp,qp,qp,aq->ap", tau732, tau82, tau88, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau732, tau79, tau86, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau82, tau370) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau82, tau159) / 2
            - 2 * einsum("qa,qp,qp,qp->ap", tau212, tau665, tau79, tau86)
            - einsum("qp,qp,aq->ap", tau733, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau536, tau667, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau734, tau79, a.x2) / 4
            - einsum("ip,pia->ap", a.y3, tau737)
            + einsum("qp,qp,qpa->ap", tau665, tau88, tau470)
            + einsum("qp,qp,qpa->ap", tau667, tau82, tau314)
            - einsum("qp,qp,aq->ap", tau738, tau79, a.x2) / 2
            - 2 * einsum("pi,pia->ap", tau674, tau574)
            - 2 * einsum("ai,pi->ap", a.t1, tau742)
            + einsum("qp,qp,qp,aq->ap", tau514, tau665, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau743, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau744, tau88, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau745, tau86, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau126, tau665, tau82, a.x2)
            - einsum("qp,qp,qpa->ap", tau667, tau86, tau267) / 2
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau747) / 2
            + einsum("qp,qp,qp,aq->ap", tau691, tau82, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau748, tau86, a.x1)
            - einsum("qp,qp,aq->ap", tau749, tau82, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau750, tau82, tau88, a.x1) / 2
            - 2 * einsum("ai,pi->ap", a.t1, tau752)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau719) / 2
            - einsum("qp,qp,aq->ap", tau266, tau665, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau753, tau86, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau349, tau665, tau79, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau754, tau79, tau86, a.x2)
            + 2 * einsum("qp,qp,qp,aq->ap", tau756, tau82, tau88, a.x1)
            - einsum("ip,pia->ap", a.y3, tau760)
            + 2 * einsum("qp,qp,aq->ap", tau762, tau82, a.x1)
            + 2 * einsum("qp,qp,qpa->ap", tau665, tau79, tau175)
            - 2 * einsum("qp,qp,qpa->ap", tau667, tau88, tau105)
            - einsum("qp,qp,aq->ap", tau157, tau667, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau171, tau665, a.x2)
            - einsum("qp,qp,aq->ap", tau763, tau86, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau349, tau667, tau79, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau766, tau79, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau424, tau667, a.x1)
            + einsum("qa,qp,qp,qp->ap", tau545, tau667, tau82, tau88)
            + einsum("qp,qp,aq->ap", tau767, tau88, a.x2) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau82, tau643)
            - 2 * einsum("qa,qp,qp,qp->ap", tau583, tau667, tau82, tau88)
            + einsum("qp,qp,qpa->ap", tau665, tau86, tau228)
            + einsum("qa,qp,qp,qp->ap", tau369, tau665, tau79, tau86)
            + einsum("qa,qp,qp,qp->ap", tau361, tau665, tau79, tau86)
            + einsum("qp,qp,qp,aq->ap", tau205, tau667, tau79, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau768, tau79, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau769, tau86, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau770, tau82, tau88, a.x2) / 2
            - einsum("qa,qp,qp,qp->ap", tau297, tau665, tau82, tau88)
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau772)
            + einsum("qp,qp,aq->ap", tau773, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau424, tau665, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau79, tau327)
            + einsum("qp,qp,aq->ap", tau266, tau667, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau86, tau194)
            + einsum("qp,qp,aq->ap", tau774, tau82, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau777) / 2
            - einsum("ai,pi->ap", a.t1, tau778)
            - einsum("qp,qp,aq->ap", tau779, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau116, tau665, a.x2) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau234, tau667, tau82, a.x1)
            + einsum("qp,qp,qpa->ap", tau665, tau88, tau222)
            - einsum("qp,qp,qp,aq->ap", tau466, tau667, tau88, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau562, tau665, tau82, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau780, tau86, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau560, tau665, tau82, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau781, tau79, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau782, tau88, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau586, tau665, tau86, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau783, tau79, a.x2) / 4
            - einsum("qp,qp,aq->ap", tau784, tau88, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau785, tau79, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau786, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau790, tau88, a.x1) / 2
            - einsum("ip,pia->ap", a.y4, tau792)
            - einsum("qp,qp,aq->ap", tau795, tau86, a.x2)
            + einsum("qp,qp,qpa->ap", tau667, tau79, tau233)
            + 2 * einsum("ip,pia->ap", a.y4, tau796)
            + einsum("qp,qp,aq->ap", tau797, tau82, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau798, tau88, a.x1) / 2
            - 2 * einsum("pi,pia->ap", tau674, tau657)
            + einsum("qp,qp,qpa->ap", tau665, tau88, tau489) / 2
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau799) / 2
            + einsum("qp,qp,aq->ap", tau330, tau665, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau79, tau800, a.x2) / 2
            - einsum("qp,aq->ap", tau801, a.x2) / 4
            + 2 * einsum("qp,qp,qpa->ap", tau667, tau88, tau381)
            - einsum("qp,qp,qpa->ap", tau667, tau82, tau615) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau321, tau667, tau88, a.x1)
            + 2 * einsum("qp,qp,aq->ap", tau802, tau86, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau352, tau665, tau79, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau234, tau665, tau82, a.x2)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau777)
            + einsum("qp,qp,qp,aq->ap", tau679, tau82, tau88, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau562, tau667, tau82, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau124, tau665, tau82, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau79, tau803, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau665, tau79, tau97, a.x2)
            - einsum("qp,qp,aq->ap", tau79, tau806, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau667, tau79, tau97, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau103, tau665, tau82, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau807, tau86, a.x1)
            + 2 * einsum("ai,pi->ap", a.t1, tau808)
            - einsum("qp,qp,aq->ap", tau809, tau82, a.x1)
            + einsum("qp,qp,aq->ap", tau79, tau810, a.x1) / 2
            - einsum("qp,aq->ap", tau811, a.x2)
            + 2 * einsum("ai,pi->ap", a.t1, tau812)
            + einsum("qp,qp,aq->ap", tau813, tau86, a.x1) / 2
            - einsum("qa,qp,qp,qp->ap", tau405, tau667, tau79, tau86)
            + einsum("qp,qp,qp,aq->ap", tau111, tau665, tau79, a.x2)
            + einsum("qp,qp,qpa->ap", tau665, tau82, tau264)
            + einsum("ai,pi->ap", a.t1, tau815)
            - einsum("qp,qp,qp,aq->ap", tau372, tau665, tau79, a.x2)
            + einsum("qp,qp,aq->ap", tau79, tau816, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau754, tau82, tau88, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau600, tau665, tau88, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau560, tau667, tau82, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau79, tau817, tau86, a.x1) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau218, tau665, tau79, a.x2)
            + einsum("qp,qp,aq->ap", tau818, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau79, tau819, a.x1)
            - einsum("qp,qp,aq->ap", tau79, tau820, a.x2) / 4
            + einsum("qp,qp,qpa->ap", tau665, tau88, tau307) / 2
            + einsum("qp,qp,qp,aq->ap", tau82, tau821, tau88, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau822, tau88, a.x1)
            - 2 * einsum("qp,qp,qp,aq->ap", tau126, tau667, tau82, a.x1)
            + einsum("qp,aq->ap", tau823, a.x2) / 2
            + einsum("qa,qp,qp,qp->ap", tau394, tau665, tau82, tau88) / 2
            - einsum("qp,qp,qpa->ap", tau665, tau86, tau367)
            - einsum("qp,qp,qp,aq->ap", tau103, tau667, tau82, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau187, tau667, tau82, a.x1)
            - einsum("qp,qp,aq->ap", tau79, tau825, a.x1)
            - einsum("qp,aq->ap", tau826, a.x2)
            + 2 * einsum("qp,qp,qpa->ap", tau667, tau88, tau621)
            - einsum("qp,qp,aq->ap", tau827, tau86, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau539, tau665, a.x2)
            + einsum("qp,qp,qpa->ap", tau667, tau88, tau645)
            - einsum("qp,qp,aq->ap", tau828, tau88, a.x1)
            - 2 * einsum("qp,qp,qp,aq->ap", tau501, tau665, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau829, tau88, a.x1) / 4
            - einsum("qp,qp,qp,aq->ap", tau79, tau821, tau86, a.x2)
            + einsum("qp,aq->ap", tau830, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau831, tau86, a.x1)
            - einsum("qp,qp,aq->ap", tau833, tau88, a.x2)
            + einsum("qa,qp,qp,qp->ap", tau197, tau667, tau79, tau86) / 2
            - 2 * einsum("qp,qp,aq->ap", tau834, tau86, a.x2)
            - einsum("qp,qp,qpa->ap", tau665, tau82, tau294)
            + einsum("qp,qp,qpa->ap", tau667, tau82, tau620) / 2
            + einsum("qp,qp,aq->ap", tau835, tau88, a.x2) / 2
            + einsum("qp,aq->ap", tau836, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau143, tau667, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau79, tau837, a.x1)
            - einsum("qp,qp,aq->ap", tau838, tau86, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau79, tau839, a.x1)
            - einsum("qp,qp,qpa->ap", tau667, tau82, tau601)
            + einsum("qp,qp,qp,aq->ap", tau501, tau667, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau841, tau88, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau842, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau843, tau86, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau770, tau79, tau86, a.x2)
            - einsum("qp,qp,qpa->ap", tau665, tau79, tau623)
            + einsum("qp,qp,qp,aq->ap", tau817, tau82, tau88, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau79, tau844, tau86, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau469, tau667, tau82, a.x1)
            - einsum("qp,qp,qpa->ap", tau665, tau88, tau510) / 2
            + einsum("qp,qp,qp,aq->ap", tau508, tau667, tau86, a.x1) / 2
            + 2 * einsum("ip,pia->ap", a.y3, tau845)
            + einsum("qp,qp,aq->ap", tau79, tau846, a.x2) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau79, tau491)
            - einsum("qp,qp,qp,aq->ap", tau82, tau847, tau88, a.x2)
            + einsum("qp,qp,aq->ap", tau82, tau848, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau79, tau849, tau86, a.x2)
            - 2 * einsum("qp,qp,qpa->ap", tau665, tau79, tau506)
            + einsum("pi,pia->ap", tau674, tau589)
            - 2 * einsum("ai,pi->ap", a.t1, tau851)
            + einsum("qp,qp,aq->ap", tau852, tau88, a.x1)
            + einsum("qp,qp,qpa->ap", tau667, tau79, tau162) / 2
            - einsum("qp,qp,qp,aq->ap", tau145, tau667, tau86, a.x1) / 4
            + einsum("qp,qp,qpa->ap", tau667, tau86, tau420)
            - einsum("qp,qp,qp,aq->ap", tau756, tau79, tau86, a.x1)
            + einsum("qp,qp,aq->ap", tau157, tau665, a.x2)
            + 2 * einsum("qp,qp,aq->ap", tau82, tau853, a.x1)
            - einsum("qp,aq->ap", tau854, a.x2) / 4
            + einsum("qp,qp,aq->ap", tau82, tau855, a.x2)
            - 2 * einsum("qp,qp,aq->ap", tau82, tau856, a.x1)
            + einsum("qp,qp,aq->ap", tau857, tau86, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau82, tau858, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau82, tau859, a.x2) / 4
            - einsum("qp,qp,qp,aq->ap", tau586, tau667, tau86, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau82, tau860, a.x2) / 2
            + 2 * einsum("qa,qp,qp,qp->ap", tau405, tau667, tau82, tau88)
            + einsum("pi,pia->ap", tau674, tau77)
            + 2 * einsum("ip,pia->ap", a.y3, tau861)
            - einsum("qp,qp,qpa->ap", tau665, tau88, tau91) / 2
            - einsum("qp,qp,qp,aq->ap", tau450, tau667, tau86, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau79, tau863, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau145, tau665, tau86, a.x2) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau82, tau864, tau88, a.x1)
            + einsum("qp,qp,aq->ap", tau79, tau865, a.x2) / 2
            + einsum("qp,aq->ap", tau866, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau293, tau665, tau86, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau476, tau665, tau88, a.x2) / 2
            - 2 * einsum("ai,pi->ap", a.t1, tau868)
            - einsum("qp,qp,qp,aq->ap", tau418, tau665, tau88, a.x2) / 2
            - einsum("qp,aq->ap", tau869, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau293, tau667, tau86, a.x1) / 2
            - einsum("ai,pi->ap", a.t1, tau870)
            + einsum("qp,qp,qp,aq->ap", tau207, tau665, tau88, a.x2)
            - einsum("qp,qp,aq->ap", tau82, tau871, a.x2) / 2
            - einsum("ip,pia->ap", a.y4, tau872)
            + einsum("qp,qp,aq->ap", tau79, tau873, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau337, tau667, tau79, a.x1) / 4
            - einsum("qa,qp,qp,qp->ap", tau410, tau667, tau82, tau88)
            + einsum("qp,qp,qp,aq->ap", tau79, tau86, tau864, a.x1)
            + einsum("qp,qp,aq->ap", tau79, tau874, a.x1) / 2
            + einsum("qa,qp,qp,qp->ap", tau379, tau665, tau82, tau88) / 2
            + einsum("qp,aq->ap", tau875, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau876, tau88, a.x2) / 2
            + einsum("qp,aq->ap", tau877, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau667, tau86, tau329)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau714) / 2
            - einsum("qp,qp,aq->ap", tau878, tau88, a.x2)
            + einsum("qp,qp,aq->ap", tau79, tau879, a.x1) / 2
            + 2 * einsum("qp,qp,qpa->ap", tau665, tau79, tau593)
            - einsum("qa,qp,qp,qp->ap", tau394, tau665, tau79, tau86)
            - einsum("qp,qp,qpa->ap", tau665, tau82, tau364)
            + einsum("qp,qp,qpa->ap", tau667, tau86, tau548) / 2
            + einsum("qp,qp,aq->ap", tau86, tau880, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau82, tau88, tau881, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau665, tau79, tau81, a.x2)
            + einsum("qp,qp,qpa->ap", tau667, tau86, tau552) / 2
            + einsum("qp,aq->ap", tau882, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau665, tau79, tau598)
            - 2 * einsum("qp,qp,qp,aq->ap", tau207, tau667, tau88, a.x1)
            - einsum("qp,qp,aq->ap", tau82, tau883, a.x2)
            + einsum("qp,qp,aq->ap", tau82, tau884, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau321, tau665, tau88, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau319, tau665, tau86, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau205, tau665, tau79, a.x2)
            + einsum("qa,qp,qp,qp->ap", tau410, tau667, tau79, tau86) / 2
            - einsum("qa,qp,qp,qp->ap", tau361, tau665, tau82, tau88) / 2
            - einsum("qp,qp,aq->ap", tau86, tau885, a.x2)
            - einsum("qp,qp,aq->ap", tau82, tau886, a.x2)
            - einsum("qp,qp,qpa->ap", tau665, tau86, tau415) / 2
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau747)
            - einsum("qp,aq->ap", tau887, a.x1) / 4
            - einsum("qp,qp,qpa->ap", tau667, tau82, tau518)
            - einsum("qa,qp,qp,qp->ap", tau545, tau667, tau79, tau86) / 2
            - einsum("qa,qp,qp,qp->ap", tau369, tau665, tau82, tau88) / 2
            + einsum("qp,qp,qpa->ap", tau79, tau86, tau675) / 2
            - einsum("qa,qp,qp,qp->ap", tau379, tau665, tau79, tau86)
            + einsum("qp,qp,aq->ap", tau88, tau888, a.x1) / 2
            - einsum("qp,qp,aq->ap", tau539, tau667, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau240, tau665, tau82, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau339, tau665, tau88, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau79, tau86, tau889, a.x1) / 2
            - einsum("ai,pi->ap", a.t1, tau890)
            + einsum("qp,qp,qp,aq->ap", tau360, tau667, tau79, a.x1)
            + 2 * einsum("pi,pia->ap", tau674, tau596)
            + einsum("qp,qp,aq->ap", tau515, tau665, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau86, tau250)
            - einsum("qp,qp,aq->ap", tau500, tau665, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau79, tau86, tau881, a.x1) / 2
            - einsum("qp,qp,qpa->ap", tau665, tau86, tau635) / 2
            - einsum("qp,qp,aq->ap", tau82, tau891, a.x1)
            + einsum("qp,qp,qpa->ap", tau667, tau82, tau252)
            + einsum("qp,qp,aq->ap", tau82, tau892, a.x1)
            + einsum("pi,pia->ap", tau674, tau564)
            - einsum("qp,qp,qp,aq->ap", tau339, tau667, tau88, a.x1)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau894) / 2
            - 2 * einsum("qp,qp,qpa->ap", tau665, tau79, tau438)
            + einsum("qp,qp,aq->ap", tau82, tau895, a.x2)
            + einsum("qp,qp,aq->ap", tau86, tau896, a.x2) / 2
            - einsum("qp,qp,qp,aq->ap", tau82, tau849, tau88, a.x2) / 2
            + einsum("qp,qp,aq->ap", tau88, tau897, a.x2)
            + einsum("qp,qp,aq->ap", tau86, tau898, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau641, tau667, tau88, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau667, tau88, tau455)
            + einsum("qp,qp,qp,aq->ap", tau466, tau665, tau88, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau79, tau899, a.x1)
            + einsum("qp,qp,qp,aq->ap", tau513, tau667, tau86, a.x1)
            + einsum("qp,qp,qpa->ap", tau82, tau88, tau894)
            + 2 * einsum("qa,qp,qp,qp->ap", tau297, tau665, tau79, tau86)
            + einsum("qp,qp,qpa->ap", tau665, tau79, tau624)
            - einsum("qp,qp,qpa->ap", tau667, tau88, tau628)
            + 2 * einsum("ai,pi->ap", a.t1, tau900)
            + 2 * einsum("qp,qp,aq->ap", tau86, tau901, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau715, tau82, tau88, a.x2) / 4
            + einsum("qp,qp,qpa->ap", tau667, tau79, tau468) / 2
            - einsum("qp,qp,qp,aq->ap", tau430, tau667, tau88, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau508, tau665, tau86, a.x2)
            - einsum("qp,qp,aq->ap", tau330, tau667, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau300, tau665, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau515, tau667, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau88, tau902, a.x1) / 4
            - einsum("qp,qp,aq->ap", tau88, tau903, a.x1) / 2
            + einsum("qp,qp,qp,aq->ap", tau600, tau667, tau88, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau86, tau363) / 2
            + einsum("qa,qp,qp,qp->ap", tau223, tau667, tau82, tau88)
            - einsum("qp,qp,qpa->ap", tau667, tau79, tau577) / 2
            - 2 * einsum("qp,qp,aq->ap", tau86, tau904, a.x2)
            - einsum("qp,qp,qpa->ap", tau79, tau86, tau799)
            + einsum("qp,qp,aq->ap", tau86, tau905, a.x1) / 2
            - einsum("ai,pi->ap", a.t1, tau906)
            + einsum("qp,qp,qpa->ap", tau665, tau86, tau304) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau79, tau487)
            + einsum("qp,aq->ap", tau907, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau82, tau844, tau88, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau418, tau667, tau88, a.x1)
            - 2 * einsum("pi,pia->ap", tau674, tau139)
            - einsum("qp,qp,aq->ap", tau88, tau908, a.x2) / 4
            - einsum("qp,qp,qp,aq->ap", tau79, tau86, tau909, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau750, tau79, tau86, a.x1) / 4
            + einsum("qp,qp,aq->ap", tau79, tau910, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau676, tau79, tau86, a.x1) / 2
            - 2 * einsum("qp,qp,qp,aq->ap", tau360, tau665, tau79, a.x2)
            - einsum("qp,aq->ap", tau911, a.x1)
            + 2 * einsum("qp,qp,aq->ap", tau82, tau912, a.x1)
            - einsum("qp,qp,qpa->ap", tau665, tau82, tau436) / 2
            + einsum("qp,qp,aq->ap", tau82, tau913, a.x2) / 2
            - einsum("qp,qp,qpa->ap", tau667, tau88, tau446)
            + 2 * einsum("qp,qp,qp,aq->ap", tau79, tau847, tau86, a.x2)
            + einsum("qp,qp,aq->ap", tau82, tau914, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau82, tau916, a.x1)
            - einsum("qp,aq->ap", tau917, a.x1) / 4
            - einsum("qp,qp,qp,aq->ap", tau641, tau665, tau88, a.x2) / 4
            + einsum("qp,qp,qp,aq->ap", tau536, tau665, tau82, a.x2) / 2
            + einsum("qp,qp,qp,aq->ap", tau430, tau665, tau88, a.x2) / 2
            + 2 * einsum("qp,qp,aq->ap", tau86, tau918, a.x2)
            + einsum("qp,qp,qp,aq->ap", tau218, tau667, tau79, a.x1)
            + einsum("qp,qp,aq->ap", tau86, tau919, a.x1) / 2
            + einsum("qp,qp,qpa->ap", tau665, tau82, tau483) / 2
            - einsum("qp,qp,qpa->ap", tau665, tau88, tau517)
            + einsum("qp,qp,qp,aq->ap", tau82, tau88, tau909, a.x2) / 2
            - einsum("qp,qp,aq->ap", tau79, tau920, a.x1) / 4
            - einsum("qp,qp,qp,aq->ap", tau82, tau88, tau889, a.x1)
            - einsum("qp,qp,qp,aq->ap", tau352, tau667, tau79, a.x1) / 4
            + einsum("qp,qp,qp,aq->ap", tau240, tau667, tau82, a.x1) / 2
            - einsum("qp,qp,qp,aq->ap", tau669, tau82, tau88, a.x1)
            - einsum("qp,qp,qpa->ap", tau82, tau88, tau772) / 2
            - einsum("qp,qp,qp,aq->ap", tau667, tau79, tau81, a.x1) / 2
            + einsum("qp,qp,aq->ap", tau143, tau665, a.x2)
            - einsum("qp,qp,qp,aq->ap", tau492, tau667, tau88, a.x1)
            - einsum("qp,qp,qpa->ap", tau667, tau86, tau654) / 2
            + 2 * einsum("qp,qp,aq->ap", tau82, tau921, a.x1)
            - einsum("qa,qp,qp,qp->ap", tau223, tau667, tau79, tau86) / 2
        )
    
        rx3 = (
            - einsum("qp,qp,qi,qp->ip", tau665, tau78, tau80, tau82) / 2
            + einsum("qp,qp,iq->ip", tau816, tau87, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau665, tau85, tau86, a.x3)
            - einsum("qp,qp,iq->ip", tau666, tau87, a.x4) / 4
            + einsum("qp,qp,iq->ip", tau457, tau665, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau82, tau922, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau82, tau923)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau142)
            + einsum("qp,qp,iq->ip", tau660, tau667, a.x4)
            - einsum("pj,pij->ip", tau674, tau924)
            + einsum("qp,qp,qpi->ip", tau665, tau86, tau926)
            + einsum("qp,qp,qpi->ip", tau78, tau82, tau927)
            - einsum("qp,qp,qp,iq->ip", tau78, tau86, tau909, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau321, tau665, tau78, a.x4)
            + einsum("pj,pji->ip", tau73, tau928)
            - einsum("qp,qp,qp,iq->ip", tau430, tau667, tau87, a.x4)
            + 2 * einsum("qp,qp,iq->ip", tau524, tau667, a.x4)
            + einsum("pj,pij->ip", tau674, tau344)
            - einsum("qp,qp,qp,iq->ip", tau380, tau665, tau86, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau386, tau665, tau86, a.x3)
            + 2 * einsum("qp,qp,qp,iq->ip", tau196, tau667, tau82, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau78, tau821, tau86, a.x3)
            + einsum("qp,qp,iq->ip", tau78, tau897, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau86, tau929)
            - einsum("qp,qp,qp,iq->ip", tau209, tau665, tau86, a.x3)
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau82, tau930) / 2
            - einsum("qp,qp,qp,qi->ip", tau667, tau86, tau87, tau931) / 4
            - einsum("pj,pji->ip", tau73, tau932)
            - 2 * einsum("pj,pji->ip", tau73, tau933)
            - einsum("jp,pji->ip", a.y4, tau936)
            + einsum("qp,qp,qp,iq->ip", tau817, tau82, tau87, a.x4)
            + 2 * einsum("jp,pji->ip", a.y4, tau938)
            + einsum("qp,qp,iq->ip", tau629, tau665, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau411, tau667, tau86, a.x3) / 4
            + einsum("qp,iq->ip", tau939, a.x4) / 2
            + 2 * einsum("pj,pij->ip", tau674, tau940)
            + einsum("qp,qp,iq->ip", tau725, tau87, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau117, tau667, tau86, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau135, tau665, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau622, tau667, a.x3) / 4
            + einsum("qp,qp,qpi->ip", tau86, tau87, tau711) / 2
            - einsum("qp,qp,iq->ip", tau82, tau941, a.x4)
            - einsum("qp,qp,iq->ip", tau315, tau667, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau142) / 2
            - einsum("qp,qp,iq->ip", tau819, tau87, a.x3)
            - einsum("qp,qp,iq->ip", tau185, tau665, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau276, tau665, tau82, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau750, tau86, tau87, a.x3) / 4
            + einsum("qp,qp,qp,iq->ip", tau476, tau667, tau87, a.x4)
            + einsum("qp,qp,iq->ip", tau768, tau78, a.x3) / 2
            + 2 * einsum("qp,qp,iq->ip", tau614, tau667, a.x4)
            - einsum("qp,qp,qpi->ip", tau665, tau82, tau943) / 2
            + einsum("qp,qp,qp,iq->ip", tau401, tau667, tau82, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau732, tau86, tau87, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau86, tau941, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau118, tau665, a.x3)
            + einsum("qp,qp,iq->ip", tau541, tau667, a.x4)
            - einsum("qp,qp,iq->ip", tau602, tau667, a.x3) / 4
            + einsum("qp,qpi->ip", tau86, tau944) / 2
            + einsum("qp,qp,iq->ip", tau78, tau876, a.x4) / 2
            - einsum("qp,qpi->ip", tau87, tau945)
            - einsum("qp,qp,qpi->ip", tau665, tau86, tau178)
            - 2 * einsum("pj,pij->ip", tau674, tau431)
            - einsum("qp,iq->ip", tau946, a.x4) / 4
            + einsum("qp,qp,qp,iq->ip", tau86, tau87, tau881, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau86, tau947, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau78, tau820, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau738, tau78, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau704, tau78, a.x3)
            + 2 * einsum("qp,qp,qpi->ip", tau82, tau87, tau824)
            - einsum("qp,qp,qpi->ip", tau667, tau86, tau277)
            - 2 * einsum("qp,qp,qpi->ip", tau82, tau87, tau948)
            + einsum("qp,qp,iq->ip", tau535, tau665, a.x4) / 2
            + einsum("qp,qi,qp,qp->ip", tau667, tau80, tau82, tau87)
            - einsum("qp,qp,iq->ip", tau733, tau78, a.x4)
            + einsum("qp,qp,iq->ip", tau803, tau87, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau416, tau665, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau154, tau665, a.x4) / 2
            + 2 * einsum("qp,qp,iq->ip", tau348, tau665, a.x3)
            + einsum("qp,qp,qpi->ip", tau86, tau87, tau949) / 2
            - einsum("qp,qp,iq->ip", tau829, tau87, a.x4) / 4
            - 2 * einsum("qp,qp,qp,iq->ip", tau691, tau78, tau86, a.x3)
            - einsum("qp,qp,qpi->ip", tau86, tau87, tau950) / 2
            + 2 * einsum("qp,qp,iq->ip", tau441, tau665, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau150, tau665, tau86, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau299) / 2
            - einsum("qp,qp,iq->ip", tau452, tau665, a.x3)
            - einsum("qp,qp,qpi->ip", tau667, tau82, tau951) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau207, tau667, tau87, a.x4)
            - einsum("qp,qp,iq->ip", tau448, tau665, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau352, tau667, tau87, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau785, tau87, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau679, tau78, tau86, a.x3)
            - einsum("qp,qp,iq->ip", tau82, tau952, a.x4) / 2
            + 2 * einsum("qp,qp,qpi->ip", tau78, tau86, tau720)
            - einsum("qp,qp,qp,iq->ip", tau401, tau667, tau86, a.x3) / 2
            + 2 * einsum("qp,qp,iq->ip", tau388, tau665, a.x3)
            + einsum("qp,qp,iq->ip", tau375, tau665, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau218, tau667, tau87, a.x3)
            - einsum("qp,qp,qpi->ip", tau86, tau87, tau789)
            + einsum("qp,qp,qp,iq->ip", tau78, tau82, tau821, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau955)
            + 2 * einsum("qp,qp,qp,iq->ip", tau756, tau82, tau87, a.x4)
            + einsum("qp,qp,iq->ip", tau86, tau956, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau360, tau667, tau87, a.x3)
            - einsum("qp,qp,qpi->ip", tau78, tau86, tau840)
            - einsum("qp,qp,qp,iq->ip", tau78, tau82, tau847, a.x4)
            + einsum("qp,qp,iq->ip", tau87, tau874, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau211, tau667, tau82, a.x4)
            + einsum("qp,qp,qpi->ip", tau78, tau82, tau805) / 2
            + einsum("qp,qp,iq->ip", tau78, tau865, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau770, tau78, tau82, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau82, tau425)
            + einsum("qp,qp,qpi->ip", tau665, tau82, tau957)
            - einsum("qp,qp,iq->ip", tau781, tau87, a.x3) / 4
            - einsum("qp,qp,qp,iq->ip", tau685, tau86, tau87, a.x3) / 4
            + 2 * einsum("jp,pij->ip", a.y4, tau959)
            - 2 * einsum("qp,qp,qp,iq->ip", tau529, tau667, tau82, a.x4)
            + einsum("qp,qp,iq->ip", tau773, tau78, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau732, tau82, tau87, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau665, tau82, tau85, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau961)
            - 2 * einsum("pj,pji->ip", tau674, tau962)
            + 2 * einsum("pj,pji->ip", tau674, tau963)
            - 2 * einsum("qp,qp,qpi->ip", tau78, tau86, tau964)
            - einsum("qp,qp,iq->ip", tau568, tau667, a.x3) / 2
            - einsum("qi,qp,qp,qp->ip", tau110, tau665, tau78, tau82) / 2
            + 2 * einsum("pj,pji->ip", tau674, tau966)
            - einsum("qp,qp,iq->ip", tau87, tau903, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau82, tau967, a.x4)
            - einsum("qp,qp,iq->ip", tau782, tau87, a.x4) / 4
            + einsum("qp,iq->ip", tau968, a.x3) / 2
            + 2 * einsum("jp,pij->ip", a.y4, tau969)
            - einsum("qp,qp,qp,iq->ip", tau117, tau667, tau82, a.x4)
            + einsum("qp,qpi->ip", tau87, tau970) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau971) / 2
            - 2 * einsum("qp,qp,qpi->ip", tau78, tau86, tau972)
            + einsum("pj,pji->ip", tau674, tau973)
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau115)
            + einsum("qp,qp,iq->ip", tau692, tau78, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau82, tau974)
            + einsum("qp,qp,qpi->ip", tau82, tau87, tau975)
            + einsum("qp,qp,iq->ip", tau86, tau976, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau205, tau665, tau78, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau641, tau667, tau87, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau86, tau977)
            - einsum("qp,qp,qp,qi->ip", tau667, tau82, tau87, tau978)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau170)
            - einsum("qp,qp,iq->ip", tau493, tau667, a.x3)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau955) / 2
            + einsum("qp,qp,qp,iq->ip", tau78, tau849, tau86, a.x3)
            - einsum("qp,qp,qpi->ip", tau665, tau86, tau979) / 2
            + einsum("qp,qp,iq->ip", tau658, tau667, a.x3) / 2
            - 2 * einsum("pj,pij->ip", tau73, tau751)
            - 2 * einsum("qp,qp,qpi->ip", tau82, tau87, tau980)
            - einsum("qp,qpi->ip", tau87, tau981) / 4
            - 2 * einsum("pj,pji->ip", tau73, tau982)
            - einsum("qp,iq->ip", tau983, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau712, tau87, a.x4)
            - einsum("qp,qp,iq->ip", tau702, tau87, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau339, tau667, tau87, a.x4)
            + einsum("qp,qp,qp,qi->ip", tau667, tau86, tau87, tau984)
            - einsum("qp,qp,qpi->ip", tau665, tau82, tau985) / 2
            + einsum("qp,qp,qpi->ip", tau665, tau82, tau322) / 2
            - einsum("qp,qp,qp,qi->ip", tau665, tau78, tau86, tau986)
            + einsum("qp,qp,qpi->ip", tau667, tau86, tau987)
            + einsum("qp,qp,iq->ip", tau698, tau78, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau78, tau910, a.x3)
            + einsum("pj,pij->ip", tau674, tau558)
            + einsum("qp,qp,iq->ip", tau822, tau87, a.x4)
            - einsum("qp,qp,iq->ip", tau204, tau667, a.x4)
            - einsum("qp,qpi->ip", tau667, tau988)
            - einsum("qp,qp,iq->ip", tau619, tau667, a.x4)
            + einsum("qi,qp,qp,qp->ip", tau110, tau667, tau82, tau87)
            - einsum("qp,qp,qp,iq->ip", tau456, tau665, tau86, a.x3)
            + einsum("qp,qp,iq->ip", tau82, tau989, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau554, tau667, a.x3) / 2
            + einsum("qp,iq->ip", tau990, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau78, tau908, a.x4) / 4
            + 2 * einsum("qp,qp,iq->ip", tau395, tau667, a.x4)
            - einsum("qp,qp,iq->ip", tau648, tau667, a.x4)
            - einsum("qp,qp,qpi->ip", tau78, tau86, tau991)
            - 2 * einsum("qp,qp,iq->ip", tau275, tau667, a.x4)
            + einsum("qp,qp,qpi->ip", tau82, tau87, tau992)
            - einsum("qp,qp,qpi->ip", tau82, tau87, tau993)
            + 2 * einsum("qp,qp,qpi->ip", tau82, tau87, tau994)
            - einsum("qp,qp,iq->ip", tau779, tau78, a.x4)
            + einsum("qp,qp,qpi->ip", tau667, tau86, tau995) / 2
            + einsum("qp,qp,iq->ip", tau766, tau87, a.x3) / 2
            + einsum("qp,qpi->ip", tau665, tau996) / 2
            + einsum("qp,iq->ip", tau997, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau86, tau998)
            + 2 * einsum("qp,qp,qp,iq->ip", tau78, tau847, tau86, a.x3)
            + einsum("qp,qp,iq->ip", tau590, tau667, a.x3)
            - einsum("qp,qpi->ip", tau667, tau999) / 4
            + einsum("qp,iq->ip", tau1000, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1001) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau360, tau665, tau78, a.x3)
            + einsum("qp,qp,qpi->ip", tau667, tau82, tau1003)
            - einsum("qp,qp,qpi->ip", tau78, tau82, tau697)
            - einsum("qp,qp,qpi->ip", tau78, tau82, tau1004)
            + einsum("qp,qp,qpi->ip", tau665, tau82, tau1005)
            - einsum("qp,qp,iq->ip", tau109, tau665, a.x4)
            - einsum("qi,qp,qp,qp->ip", tau417, tau667, tau86, tau87) / 2
            + einsum("qp,qp,qp,iq->ip", tau600, tau667, tau87, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau211, tau667, tau86, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau337, tau667, tau87, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau287, tau665, a.x4) / 4
            + einsum("qp,qp,iq->ip", tau511, tau667, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau86, tau87, tau1006) / 2
            - einsum("qp,qp,iq->ip", tau78, tau783, a.x3) / 4
            + einsum("qp,qp,iq->ip", tau78, tau846, a.x3) / 2
            + einsum("pj,pji->ip", tau674, tau1007)
            + einsum("pj,pij->ip", tau73, tau672)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1001) / 4
            + einsum("qp,qp,qp,iq->ip", tau111, tau665, tau78, a.x3)
            + einsum("qp,qp,iq->ip", tau78, tau800, a.x3) / 2
            - einsum("qp,qpi->ip", tau665, tau1008) / 4
            + einsum("qp,qp,iq->ip", tau767, tau78, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau86, tau1009) / 2
            + einsum("qp,qp,qp,iq->ip", tau570, tau667, tau86, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau852, tau87, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau418, tau667, tau87, a.x4)
            + einsum("qp,iq->ip", tau1010, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau527, tau665, a.x3)
            - einsum("qp,qp,iq->ip", tau87, tau902, a.x4) / 4
            + einsum("qp,qp,qp,iq->ip", tau676, tau82, tau87, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau715, tau78, tau86, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau87, tau879, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau87, tau899, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau352, tau665, tau78, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau754, tau78, tau86, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau476, tau665, tau78, a.x4) / 2
            - 2 * einsum("pj,pij->ip", tau73, tau741)
            + 2 * einsum("qp,qp,qpi->ip", tau78, tau86, tau1011)
            - einsum("qp,qp,qp,iq->ip", tau386, tau665, tau82, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau86, tau989, a.x3)
            - einsum("qp,qp,iq->ip", tau637, tau665, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau349, tau667, tau87, a.x3) / 2
            - 2 * einsum("qp,qp,iq->ip", tau652, tau665, a.x3)
            + einsum("pj,pij->ip", tau73, tau814)
            + einsum("qi,qp,qp,qp->ip", tau417, tau665, tau78, tau86)
            + einsum("qp,qp,iq->ip", tau1012, tau82, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau82, tau1013)
            - einsum("qp,qp,qp,iq->ip", tau111, tau667, tau87, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau78, tau835, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau641, tau665, tau78, a.x4) / 4
            + einsum("qp,qp,qp,iq->ip", tau78, tau844, tau86, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau86, tau1014)
            + einsum("qp,qp,iq->ip", tau86, tau952, a.x3)
            - einsum("qp,qp,iq->ip", tau668, tau78, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau651, tau667, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau418, tau665, tau78, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau129, tau665, tau86, a.x3)
            + einsum("qp,qpi->ip", tau665, tau1015) / 2
            + 2 * einsum("qp,qp,iq->ip", tau640, tau667, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau676, tau86, tau87, a.x3) / 2
            + einsum("qi,qp,qp,qp->ip", tau449, tau665, tau78, tau86)
            + einsum("qp,qp,iq->ip", tau587, tau667, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau1016, tau86, a.x3) / 2
            - 2 * einsum("qp,qp,iq->ip", tau605, tau665, a.x3)
            - 2 * einsum("qp,qp,qp,iq->ip", tau321, tau667, tau87, a.x4)
            + einsum("qi,qp,qp,qp->ip", tau1017, tau665, tau78, tau82)
            - einsum("qp,qp,qp,iq->ip", tau600, tau665, tau78, a.x4) / 4
            - einsum("qp,qp,iq->ip", tau442, tau665, a.x3)
            + einsum("qp,qpi->ip", tau87, tau1018) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau86, tau1019) / 4
            + einsum("qp,qp,qp,iq->ip", tau665, tau78, tau81, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau82, tau1020) / 2
            - einsum("qp,qp,qp,iq->ip", tau492, tau667, tau87, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau1021) / 4
            + einsum("qp,qp,qpi->ip", tau665, tau86, tau1022) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau82, tau864, tau87, a.x4)
            - einsum("pj,pji->ip", tau73, tau1023)
            + einsum("qp,qp,iq->ip", tau664, tau667, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau504, tau667, tau86, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau529, tau667, tau86, a.x3)
            + einsum("pj,pji->ip", tau73, tau727)
            + einsum("qi,qp,qp,qp->ip", tau1024, tau667, tau86, tau87) / 2
            + einsum("qp,qp,iq->ip", tau798, tau87, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau597, tau667, a.x4)
            + einsum("qp,qp,iq->ip", tau313, tau665, a.x3)
            - einsum("qp,qp,iq->ip", tau716, tau78, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau291, tau667, a.x3)
            + einsum("qp,qp,iq->ip", tau1016, tau82, a.x4)
            - 2 * einsum("pj,pij->ip", tau674, tau474)
            - einsum("qp,qp,qp,qi->ip", tau667, tau82, tau87, tau930)
            + einsum("qp,qp,qp,iq->ip", tau339, tau665, tau78, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau82, tau1025)
            + einsum("qp,qp,iq->ip", tau837, tau87, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau633, tau665, tau82, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau271, tau665, tau86, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau403, tau667, a.x3)
            - einsum("pj,pij->ip", tau73, tau1026)
            + einsum("qp,qpi->ip", tau667, tau1027) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau961) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau86, tau1028)
            - einsum("qp,qp,iq->ip", tau639, tau665, a.x4) / 4
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau299)
            + einsum("qp,qp,iq->ip", tau82, tau947, a.x4)
            + einsum("qp,qpi->ip", tau667, tau1029) / 2
            + einsum("qp,qp,qp,iq->ip", tau207, tau665, tau78, a.x4)
            - einsum("jp,pij->ip", a.y4, tau938)
            - einsum("qp,qp,qp,iq->ip", tau349, tau665, tau78, a.x3)
            - einsum("qp,qp,qpi->ip", tau78, tau82, tau1030) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau115) / 2
            - 2 * einsum("pj,pji->ip", tau674, tau335)
            - einsum("qp,qp,iq->ip", tau1031, tau86, a.x3) / 4
            + einsum("qp,qp,qp,iq->ip", tau86, tau87, tau889, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau78, tau878, a.x4)
            - einsum("qp,qp,iq->ip", tau78, tau833, a.x4)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1033) / 2
            + einsum("qp,qp,iq->ip", tau680, tau78, a.x4) / 2
            - 2 * einsum("qp,qp,iq->ip", tau616, tau667, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau667, tau87, tau97, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau665, tau86, tau184) / 2
            + einsum("qp,qp,qp,iq->ip", tau153, tau667, tau86, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau78, tau873, a.x3) / 2
            + 2 * einsum("pj,pij->ip", tau73, tau1034)
            + einsum("qp,qpi->ip", tau78, tau1035) / 2
            + einsum("qp,qp,qp,iq->ip", tau430, tau665, tau78, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau703, tau78, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau78, tau841, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau744, tau78, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau828, tau87, a.x4)
            - einsum("qp,qp,qpi->ip", tau78, tau82, tau1036) / 2
            - einsum("qp,qp,qp,iq->ip", tau82, tau87, tau881, a.x4)
            - einsum("qp,iq->ip", tau1037, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau86, tau1038) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau86, tau1040)
            + einsum("qp,qp,iq->ip", tau634, tau667, a.x3)
            - einsum("qp,qp,iq->ip", tau734, tau78, a.x3) / 4
            - einsum("qp,qp,qp,iq->ip", tau817, tau86, tau87, a.x3) / 2
            - einsum("qp,qpi->ip", tau86, tau1041)
            + einsum("qp,qp,qp,iq->ip", tau372, tau667, tau87, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau784, tau87, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau86, tau87, tau1042)
            - einsum("qp,qp,qpi->ip", tau667, tau86, tau1043) / 2
            + einsum("qp,qpi->ip", tau78, tau1044) / 2
            + einsum("qp,qp,iq->ip", tau131, tau667, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau86, tau1045) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau170) / 2
            - einsum("qp,qp,qp,iq->ip", tau571, tau667, tau86, a.x3) / 4
            - einsum("qi,qp,qp,qp->ip", tau449, tau667, tau86, tau87) / 2
            - einsum("qp,qp,qp,iq->ip", tau78, tau82, tau849, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau380, tau665, tau82, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau667, tau81, tau87, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau626, tau665, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau82, tau1046) / 2
            + einsum("qp,qp,qp,iq->ip", tau691, tau78, tau82, a.x4)
            - einsum("qp,qp,iq->ip", tau87, tau920, a.x3) / 4
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau86, tau931) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau82, tau1047) / 2
            - einsum("qp,qp,qpi->ip", tau86, tau87, tau1048)
            + 2 * einsum("jp,pij->ip", a.y4, tau936)
            - 2 * einsum("qp,qp,qp,iq->ip", tau276, tau665, tau86, a.x3)
            + einsum("qp,qp,qpi->ip", tau667, tau82, tau190) / 2
            + einsum("qp,qp,qp,iq->ip", tau205, tau667, tau87, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau613, tau667, a.x3)
            + einsum("qp,qp,iq->ip", tau494, tau665, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau1049, tau86, a.x3) / 4
            + einsum("qp,qp,qpi->ip", tau86, tau87, tau1050)
            - einsum("qp,qp,qp,iq->ip", tau372, tau665, tau78, a.x3)
            - 2 * einsum("qi,qp,qp,qp->ip", tau206, tau665, tau78, tau86)
            - 2 * einsum("qp,qp,qp,qi->ip", tau665, tau78, tau86, tau984)
            + 2 * einsum("qp,qp,qp,iq->ip", tau633, tau665, tau86, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau86, tau1051)
            - einsum("qp,qp,iq->ip", tau78, tau863, a.x3)
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1033)
            + 2 * einsum("qp,qp,iq->ip", tau440, tau665, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau971)
            - einsum("qp,iq->ip", tau1052, a.x3)
            + einsum("qi,qp,qp,qp->ip", tau217, tau665, tau78, tau82)
            - einsum("qp,qp,iq->ip", tau78, tau806, a.x3)
            - einsum("qp,qp,iq->ip", tau1012, tau86, a.x3)
            - einsum("qp,qp,iq->ip", tau576, tau665, a.x3)
            - einsum("qp,qp,iq->ip", tau497, tau665, a.x4)
            + einsum("qi,qp,qp,qp->ip", tau206, tau667, tau86, tau87)
            - einsum("qp,qp,qpi->ip", tau665, tau82, tau439)
            + einsum("qp,qp,qp,iq->ip", tau86, tau864, tau87, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau196, tau667, tau86, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau129, tau665, tau82, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau86, tau967, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau669, tau86, tau87, a.x3) / 2
            - 2 * einsum("qi,qp,qp,qp->ip", tau1017, tau667, tau82, tau87)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau1021) / 2
            - 2 * einsum("qi,qp,qp,qp->ip", tau217, tau667, tau82, tau87)
            - einsum("qp,qp,iq->ip", tau526, tau665, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau504, tau667, tau82, a.x4)
            + einsum("qp,qp,qpi->ip", tau667, tau86, tau203) / 2
            + einsum("qp,qp,iq->ip", tau87, tau888, a.x4) / 2
            - einsum("qp,qpi->ip", tau665, tau1053)
            + einsum("qp,qp,iq->ip", tau810, tau87, a.x3) / 2
            - einsum("jp,pji->ip", a.y4, tau969)
            - einsum("qp,qp,qpi->ip", tau82, tau87, tau707)
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau82, tau978) / 2
            - einsum("qp,iq->ip", tau1054, a.x3) / 4
            + einsum("qp,qp,iq->ip", tau419, tau665, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau770, tau78, tau86, a.x3)
            + einsum("qp,qp,qpi->ip", tau86, tau87, tau1055) / 2
            + einsum("qp,qp,iq->ip", tau839, tau87, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau492, tau665, tau78, a.x4) / 2
            - einsum("qi,qp,qp,qp->ip", tau1024, tau665, tau78, tau86)
            - einsum("qp,qpi->ip", tau78, tau1056) / 4
            - einsum("qp,iq->ip", tau1057, a.x4)
            - einsum("qp,qpi->ip", tau78, tau1058)
            + einsum("qp,qp,iq->ip", tau659, tau667, a.x3) / 2
            - einsum("jp,pji->ip", a.y4, tau959)
            - einsum("qp,qp,iq->ip", tau825, tau87, a.x3)
            - einsum("qp,iq->ip", tau1059, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau665, tau78, tau97, a.x3)
            + einsum("qp,qp,qp,qi->ip", tau667, tau86, tau87, tau986) / 2
            - einsum("qp,qp,iq->ip", tau710, tau87, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau355, tau667, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau790, tau87, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau82, tau1060) / 2
            + einsum("qp,qp,iq->ip", tau86, tau922, a.x3)
            - 2 * einsum("qp,qp,qp,iq->ip", tau218, tau665, tau78, a.x3)
            + einsum("qp,qp,iq->ip", tau325, tau665, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau754, tau78, tau82, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau337, tau665, tau78, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau756, tau86, tau87, a.x3)
            + einsum("qp,iq->ip", tau1061, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau701, tau87, a.x4) / 2
        )
    
        rx4 = (
            - einsum("pj,pij->ip", tau73, tau1063)
            + einsum("qp,qp,qp,iq->ip", tau456, tau665, tau88, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau88, tau977) / 2
            - einsum("qp,qp,qp,iq->ip", tau504, tau667, tau88, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau1019) / 2
            + einsum("qi,qp,qp,qp->ip", tau417, tau667, tau87, tau88)
            + einsum("qp,qp,qp,iq->ip", tau276, tau665, tau88, a.x3)
            - einsum("qp,qp,qpi->ip", tau667, tau88, tau1064)
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1065)
            - einsum("qp,qp,iq->ip", tau88, tau967, a.x3)
            + einsum("qp,qpi->ip", tau667, tau1066) / 2
            + einsum("qp,qp,qp,iq->ip", tau536, tau665, tau78, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau88, tau956, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau88, tau952, a.x3) / 2
            - 2 * einsum("pj,pji->ip", tau674, tau1067)
            - einsum("qp,qp,qpi->ip", tau667, tau88, tau995)
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1068) / 2
            - einsum("qp,qp,qp,qi->ip", tau667, tau87, tau88, tau986)
            + einsum("pj,pij->ip", tau73, tau1069)
            + einsum("pj,pji->ip", tau73, tau1070)
            + 2 * einsum("qp,qp,qpi->ip", tau667, tau88, tau277)
            - einsum("qp,qp,qp,iq->ip", tau124, tau665, tau78, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau78, tau88, tau909, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau196, tau667, tau79, a.x4)
            + einsum("qp,qp,iq->ip", tau462, tau665, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau145, tau665, tau78, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau838, tau87, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau636, tau665, a.x4)
            - einsum("qp,qp,qpi->ip", tau665, tau79, tau322)
            - einsum("qp,qp,qp,iq->ip", tau665, tau79, tau85, a.x4)
            + einsum("qp,qp,iq->ip", tau88, tau947, a.x3)
            - einsum("qp,qpi->ip", tau87, tau1071)
            + einsum("qp,qp,iq->ip", tau1049, tau88, a.x3) / 2
            + 2 * einsum("pj,pij->ip", tau674, tau1072)
            + einsum("qp,qp,qpi->ip", tau79, tau87, tau707) / 2
            - einsum("qp,qp,iq->ip", tau79, tau947, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau631, tau665, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau562, tau665, tau78, a.x3) / 2
            - einsum("pj,pji->ip", tau73, tau1074)
            - einsum("qp,qp,iq->ip", tau200, tau665, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1075)
            + einsum("qp,qp,iq->ip", tau323, tau665, a.x3) / 2
            - einsum("qi,qp,qp,qp->ip", tau1024, tau667, tau87, tau88)
            - 2 * einsum("qp,qp,iq->ip", tau78, tau904, a.x4)
            - einsum("qp,qp,iq->ip", tau717, tau78, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau319, tau665, tau78, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau117, tau667, tau79, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau386, tau665, tau79, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau129, tau665, tau88, a.x3) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau513, tau665, tau78, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau79, tau1060)
            - einsum("qp,qp,qp,iq->ip", tau669, tau87, tau88, a.x3)
            - einsum("jp,pij->ip", a.y3, tau969)
            - einsum("qp,qp,qpi->ip", tau665, tau88, tau184)
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau1055) / 4
            + einsum("qp,qpi->ip", tau87, tau1076) / 2
            - einsum("qp,qp,iq->ip", tau582, tau667, a.x3) / 4
            + einsum("qp,qp,iq->ip", tau78, tau842, a.x4)
            + einsum("qp,qp,iq->ip", tau662, tau665, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau79, tau817, tau87, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau858, tau87, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau88, tau989, a.x3) / 2
            - einsum("qp,iq->ip", tau1077, a.x4) / 4
            + einsum("qp,qp,iq->ip", tau1016, tau88, a.x3)
            + einsum("qp,qp,iq->ip", tau78, tau895, a.x3)
            + einsum("qp,qp,iq->ip", tau78, tau884, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau809, tau87, a.x3)
            + 2 * einsum("qp,qp,qp,iq->ip", tau196, tau667, tau88, a.x3)
            + 2 * einsum("qp,qp,qpi->ip", tau667, tau88, tau1078)
            - einsum("qp,qp,iq->ip", tau463, tau667, a.x3) / 4
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau1079) / 2
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau178) / 2
            + 2 * einsum("jp,pij->ip", a.y3, tau938)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau1080)
            - einsum("qp,iq->ip", tau1081, a.x3) / 4
            + einsum("qp,qp,qpi->ip", tau78, tau79, tau697) / 2
            - einsum("qp,qp,iq->ip", tau708, tau87, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau380, tau665, tau88, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau88, tau1038)
            + einsum("qp,qp,qp,iq->ip", tau209, tau665, tau88, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau79, tau1036)
            - einsum("qp,qp,qp,iq->ip", tau560, tau665, tau78, a.x3) / 4
            + einsum("pj,pij->ip", tau674, tau1082)
            + einsum("qp,qp,qp,iq->ip", tau124, tau667, tau87, a.x3)
            - einsum("qp,qp,qpi->ip", tau665, tau88, tau1083) / 4
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau499) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau88, tau1045)
            + einsum("qp,qp,iq->ip", tau769, tau87, a.x4)
            - einsum("qp,iq->ip", tau1084, a.x3)
            + einsum("qp,qp,iq->ip", tau496, tau667, a.x4)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau1086)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau1079)
            + 2 * einsum("qp,qp,qpi->ip", tau665, tau79, tau439)
            - einsum("qp,qp,iq->ip", tau663, tau665, a.x3) / 2
            - 2 * einsum("qi,qp,qp,qp->ip", tau217, tau665, tau78, tau79)
            + einsum("qp,qp,iq->ip", tau444, tau665, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau661, tau667, a.x4)
            - einsum("qp,qp,qpi->ip", tau79, tau87, tau994)
            - einsum("qp,qp,iq->ip", tau786, tau87, a.x4)
            - einsum("qi,qp,qp,qp->ip", tau110, tau667, tau79, tau87) / 2
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau711)
            + 2 * einsum("qp,qp,qp,iq->ip", tau78, tau79, tau847, a.x4)
            - einsum("qp,qp,qpi->ip", tau78, tau79, tau1025) / 2
            - einsum("qp,qpi->ip", tau78, tau1087) / 4
            + einsum("qp,qp,qpi->ip", tau87, tau88, tau950)
            - einsum("qp,qp,iq->ip", tau78, tau859, a.x3) / 4
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau1088) / 4
            + einsum("qp,qp,iq->ip", tau550, tau665, a.x3) / 2
            - 2 * einsum("pj,pji->ip", tau73, tau850)
            + einsum("qp,qp,iq->ip", tau503, tau667, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau79, tau952, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau103, tau667, tau87, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau508, tau665, tau78, a.x4)
            - einsum("jp,pij->ip", a.y3, tau959)
            - einsum("qp,qp,qp,iq->ip", tau570, tau667, tau88, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau293, tau667, tau87, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau818, tau87, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau536, tau667, tau87, a.x3)
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau499)
            - einsum("qp,qp,iq->ip", tau88, tau922, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau87, tau898, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau357, tau667, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau633, tau665, tau88, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau78, tau821, tau88, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau103, tau665, tau78, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau685, tau87, tau88, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau238, tau667, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau700, tau87, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau678, tau87, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau79, tau87, tau881, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau770, tau78, tau79, a.x4)
            + 2 * einsum("qp,qp,qp,iq->ip", tau633, tau665, tau79, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau79, tau190)
            - einsum("qp,qp,iq->ip", tau426, tau667, a.x3)
            - einsum("qp,qp,iq->ip", tau231, tau665, a.x4) / 4
            + einsum("qp,qp,qp,iq->ip", tau514, tau665, tau78, a.x4)
            - einsum("qp,qp,qpi->ip", tau79, tau87, tau975) / 2
            - 2 * einsum("pj,pji->ip", tau73, tau1089)
            - 2 * einsum("qi,qp,qp,qp->ip", tau1017, tau665, tau78, tau79)
            - einsum("qp,qp,qpi->ip", tau78, tau79, tau1046)
            - einsum("qp,qp,qpi->ip", tau79, tau87, tau992) / 2
            + 2 * einsum("jp,pji->ip", a.y3, tau936)
            + einsum("qp,qp,iq->ip", tau78, tau914, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau78, tau883, a.x3)
            - einsum("qp,qp,iq->ip", tau1012, tau79, a.x4)
            - einsum("qp,qp,iq->ip", tau763, tau78, a.x4)
            - einsum("qp,qp,iq->ip", tau667, tau84, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau79, tau87, tau948)
            - einsum("qp,qp,iq->ip", tau359, tau665, a.x3)
            + einsum("qp,iq->ip", tau1090, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau460, tau667, a.x4) / 4
            - einsum("qp,qp,qpi->ip", tau665, tau88, tau926) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau1014) / 2
            - einsum("qp,qp,qp,iq->ip", tau87, tau88, tau881, a.x3)
            + einsum("qp,qpi->ip", tau78, tau1091) / 2
            - einsum("qp,qp,qp,qi->ip", tau665, tau78, tau79, tau978)
            + einsum("qp,qp,qpi->ip", tau665, tau79, tau985)
            + einsum("qp,qp,iq->ip", tau78, tau860, a.x3) / 2
            + 2 * einsum("pj,pij->ip", tau674, tau1092)
            - einsum("qp,qp,qp,iq->ip", tau514, tau667, tau87, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau544, tau665, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau88, tau1093)
            + einsum("qp,qp,qpi->ip", tau79, tau87, tau993) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau88, tau203)
            - einsum("qp,qp,iq->ip", tau78, tau780, a.x4)
            + einsum("qp,qp,iq->ip", tau78, tau896, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau529, tau667, tau79, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau78, tau79, tau821, a.x4)
            - einsum("qp,qp,iq->ip", tau495, tau667, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau319, tau667, tau87, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau88, tau1040) / 2
            + einsum("qp,qp,iq->ip", tau447, tau665, a.x4) / 2
            - 2 * einsum("qi,qp,qp,qp->ip", tau206, tau667, tau87, tau88)
            + 2 * einsum("qp,qp,iq->ip", tau87, tau921, a.x3)
            + 2 * einsum("jp,pji->ip", a.y3, tau969)
            - einsum("qp,qp,qp,iq->ip", tau469, tau667, tau87, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau562, tau667, tau87, a.x3)
            + einsum("qp,qp,qp,qi->ip", tau667, tau79, tau87, tau930) / 2
            + einsum("qp,qp,qp,iq->ip", tau770, tau78, tau88, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau234, tau665, tau78, a.x3)
            + einsum("qp,qp,iq->ip", tau647, tau665, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau632, tau667, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau78, tau797, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau840) / 2
            - einsum("qp,qp,qp,iq->ip", tau293, tau665, tau78, a.x4)
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau88, tau986) / 2
            - einsum("qp,qp,iq->ip", tau485, tau667, a.x3) / 4
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau156)
            + einsum("qp,qpi->ip", tau78, tau1094) / 2
            - 2 * einsum("qp,qp,qpi->ip", tau665, tau79, tau1005)
            + einsum("qp,qp,qp,iq->ip", tau560, tau667, tau87, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau504, tau667, tau79, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau87, tau88, tau1006)
            + einsum("qp,qp,qpi->ip", tau667, tau79, tau974) / 2
            - einsum("qp,qp,iq->ip", tau317, tau667, a.x4)
            - 2 * einsum("qp,qp,qp,iq->ip", tau691, tau78, tau79, a.x4)
            - einsum("qp,qp,iq->ip", tau709, tau87, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau145, tau667, tau87, a.x4) / 4
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau1050) / 2
            + 2 * einsum("jp,pji->ip", a.y3, tau959)
            - einsum("qp,qp,iq->ip", tau404, tau667, a.x3) / 4
            + einsum("qp,qp,qp,iq->ip", tau586, tau665, tau78, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau501, tau667, tau87, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau756, tau79, tau87, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau1080) / 2
            - einsum("qp,qp,qp,qi->ip", tau665, tau78, tau79, tau930)
            - einsum("qp,qp,qp,iq->ip", tau386, tau665, tau88, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau87, tau1095)
            + einsum("qp,qp,iq->ip", tau618, tau667, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau646, tau667, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau450, tau667, tau87, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau88, tau1051) / 2
            + einsum("qp,qp,qpi->ip", tau79, tau87, tau980)
            - einsum("qp,qp,iq->ip", tau722, tau87, a.x4) / 4
            + 2 * einsum("pj,pji->ip", tau674, tau1096)
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau1097) / 2
            - 2 * einsum("qp,qp,qp,qi->ip", tau667, tau87, tau88, tau984)
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau88, tau984)
            - einsum("qp,qp,qp,iq->ip", tau240, tau665, tau78, a.x3) / 4
            + einsum("pj,pij->ip", tau674, tau254)
            - einsum("qp,qp,qp,iq->ip", tau715, tau78, tau88, a.x3) / 4
            + einsum("pj,pij->ip", tau73, tau683)
            - einsum("qp,qp,qp,iq->ip", tau754, tau78, tau88, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau817, tau87, tau88, a.x3)
            - einsum("qp,qp,qp,iq->ip", tau211, tau667, tau79, a.x4) / 2
            - einsum("qp,qp,qp,iq->ip", tau676, tau79, tau87, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau461, tau665, a.x4)
            + einsum("qp,iq->ip", tau1098, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau508, tau667, tau87, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau401, tau667, tau88, a.x3)
            + einsum("pj,pji->ip", tau73, tau1099)
            + einsum("qp,qp,qp,iq->ip", tau732, tau79, tau87, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau78, tau871, a.x3) / 2
            + einsum("qi,qp,qp,qp->ip", tau110, tau665, tau78, tau79)
            + einsum("qp,qp,qp,iq->ip", tau129, tau665, tau79, a.x4)
            + einsum("qi,qp,qp,qp->ip", tau217, tau667, tau79, tau87)
            - einsum("qp,qp,iq->ip", tau78, tau885, a.x4)
            + einsum("qp,qp,iq->ip", tau1031, tau88, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1065) / 2
            + einsum("qp,iq->ip", tau1100, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau87, tau88, tau789) / 2
            + einsum("qp,qp,iq->ip", tau599, tau665, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau528, tau665, a.x3) / 4
            - einsum("qp,qp,qp,iq->ip", tau87, tau88, tau889, a.x3)
            + 2 * einsum("qp,qp,qpi->ip", tau665, tau79, tau1013)
            + einsum("qp,qp,iq->ip", tau1012, tau88, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau679, tau78, tau88, a.x3) / 2
            + einsum("pj,pji->ip", tau674, tau603)
            - einsum("qp,qpi->ip", tau667, tau1101)
            + einsum("qp,qp,iq->ip", tau216, tau665, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau774, tau87, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau78, tau1102) / 2
            - einsum("qp,qp,iq->ip", tau530, tau665, a.x3) / 4
            + 2 * einsum("qp,qp,iq->ip", tau721, tau78, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau78, tau79, tau849, a.x4)
            - einsum("qp,iq->ip", tau1103, a.x4)
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau1104) / 2
            + einsum("qp,qp,iq->ip", tau857, tau87, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau244, tau667, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau450, tau665, tau78, a.x4)
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau979)
            - einsum("qp,qpi->ip", tau665, tau1105) / 4
            - einsum("qp,qp,iq->ip", tau429, tau667, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau149, tau667, a.x3) / 2
            - einsum("qp,qp,iq->ip", tau78, tau886, a.x3)
            - 2 * einsum("pj,pij->ip", tau674, tau342)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1075) / 2
            + einsum("qp,qp,qp,iq->ip", tau691, tau78, tau88, a.x3)
            + einsum("qi,qp,qp,qp->ip", tau206, tau665, tau78, tau88)
            + einsum("qp,iq->ip", tau1106, a.x3) / 2
            + einsum("qp,qp,qp,qi->ip", tau667, tau79, tau87, tau978) / 2
            - einsum("qp,qp,qp,iq->ip", tau117, tau667, tau88, a.x3)
            + einsum("qp,qp,iq->ip", tau87, tau892, a.x3)
            + 2 * einsum("qp,qp,iq->ip", tau87, tau912, a.x3)
            + einsum("qp,qp,iq->ip", tau481, tau667, a.x3) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau864, tau87, tau88, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau79, tau943)
            - einsum("qp,qp,qp,iq->ip", tau586, tau667, tau87, a.x4) / 4
            - 2 * einsum("pj,pij->ip", tau73, tau867)
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau949)
            + einsum("qp,qp,iq->ip", tau79, tau941, a.x4) / 2
            + einsum("qp,iq->ip", tau1107, a.x3) / 2
            + einsum("qp,qpi->ip", tau88, tau1041) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau79, tau927) / 2
            - 2 * einsum("qp,qp,iq->ip", tau78, tau834, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau187, tau665, tau78, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau78, tau849, tau88, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau78, tau844, tau88, a.x3) / 4
            - einsum("qp,qp,qpi->ip", tau78, tau79, tau805)
            + einsum("qp,qp,qpi->ip", tau667, tau88, tau1043)
            - einsum("qp,iq->ip", tau1108, a.x4)
            - einsum("qp,qpi->ip", tau665, tau1109)
            - 2 * einsum("qp,qp,qp,iq->ip", tau276, tau665, tau79, a.x4)
            + einsum("qp,qp,iq->ip", tau551, tau665, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau434, tau665, a.x4)
            + einsum("qp,qp,iq->ip", tau78, tau848, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau191, tau667, a.x3) / 2
            + einsum("qp,qp,qp,iq->ip", tau411, tau667, tau88, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau964)
            + einsum("qi,qp,qp,qp->ip", tau449, tau667, tau87, tau88)
            - einsum("qp,qp,qi,qp->ip", tau667, tau79, tau80, tau87) / 2
            + 2 * einsum("pj,pji->ip", tau73, tau1111)
            + einsum("qp,qp,iq->ip", tau376, tau667, a.x3)
            + 2 * einsum("qp,qp,iq->ip", tau853, tau87, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau1028) / 2
            + einsum("qp,qp,iq->ip", tau123, tau667, a.x3)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1068) / 4
            - einsum("qp,qp,iq->ip", tau1016, tau79, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau79, tau967, a.x4) / 2
            + einsum("qp,qp,qpi->ip", tau78, tau79, tau1004) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau79, tau1047)
            + einsum("qp,qp,iq->ip", tau78, tau913, a.x3) / 2
            - einsum("qp,qp,qp,iq->ip", tau380, tau665, tau79, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau1102) / 4
            + einsum("qp,qpi->ip", tau665, tau1112) / 2
            - einsum("qp,qpi->ip", tau88, tau944)
            - einsum("pj,pji->ip", tau674, tau1113)
            - einsum("qp,qp,iq->ip", tau686, tau78, a.x3) / 4
            - einsum("qp,qp,iq->ip", tau748, tau87, a.x4)
            - einsum("qp,qp,qp,iq->ip", tau78, tau847, tau88, a.x3)
            - einsum("qp,qp,iq->ip", tau477, tau667, a.x3) / 2
            + einsum("qp,qp,qpi->ip", tau667, tau79, tau951)
            + einsum("qp,qp,iq->ip", tau745, tau78, a.x4)
            - einsum("qp,iq->ip", tau1114, a.x3) / 4
            + einsum("qp,qp,qp,iq->ip", tau211, tau667, tau88, a.x3)
            - einsum("qp,qp,qp,qi->ip", tau665, tau78, tau88, tau931) / 4
            - einsum("qp,qp,iq->ip", tau78, tau843, a.x4)
            + einsum("qp,iq->ip", tau1115, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau310, tau667, a.x4) / 2
            - einsum("qp,qp,iq->ip", tau179, tau665, a.x4)
            - einsum("qp,qp,qpi->ip", tau87, tau88, tau1042) / 2
            + einsum("qp,qp,qp,iq->ip", tau469, tau665, tau78, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau665, tau79, tau1020)
            - einsum("qp,qp,qp,iq->ip", tau271, tau665, tau88, a.x3) / 4
            + einsum("qp,qp,qp,iq->ip", tau750, tau87, tau88, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau88, tau720)
            - einsum("qp,qp,iq->ip", tau100, tau667, a.x4) / 2
            - einsum("qi,qp,qp,qp->ip", tau417, tau665, tau78, tau88) / 2
            + einsum("pj,pji->ip", tau674, tau1116)
            - einsum("qp,qp,iq->ip", tau581, tau665, a.x3)
            - einsum("qp,qp,iq->ip", tau78, tau795, a.x4)
            - 2 * einsum("qp,qp,iq->ip", tau687, tau87, a.x3)
            - einsum("qp,qpi->ip", tau78, tau1117)
            + 2 * einsum("qp,qp,qp,iq->ip", tau756, tau87, tau88, a.x3)
            - einsum("qp,qp,qpi->ip", tau665, tau88, tau1022)
            - 2 * einsum("pj,pij->ip", tau674, tau1118)
            + einsum("qp,qp,qp,qi->ip", tau667, tau87, tau88, tau931) / 2
            - einsum("qp,qp,qp,iq->ip", tau187, tau667, tau87, a.x3)
            + einsum("qi,qp,qp,qp->ip", tau1017, tau667, tau79, tau87)
            - einsum("qp,qp,iq->ip", tau638, tau665, a.x4) / 4
            - einsum("qp,qp,iq->ip", tau478, tau665, a.x3)
            + einsum("qp,qpi->ip", tau87, tau1119) / 2
            - 2 * einsum("qp,qp,qpi->ip", tau667, tau88, tau998)
            - 2 * einsum("qp,qp,iq->ip", tau856, tau87, a.x3)
            - 2 * einsum("pj,pji->ip", tau674, tau1120)
            + einsum("qi,qp,qp,qp->ip", tau1024, tau665, tau78, tau88) / 2
            - 2 * einsum("qp,qp,qpi->ip", tau667, tau88, tau987)
            + einsum("qp,qp,qp,iq->ip", tau754, tau78, tau79, a.x4)
            + 2 * einsum("qp,qp,iq->ip", tau78, tau802, a.x4)
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau156) / 2
            - einsum("qp,qp,iq->ip", tau831, tau87, a.x4)
            - einsum("qp,iq->ip", tau1121, a.x3)
            - einsum("qp,qp,iq->ip", tau278, tau667, a.x4)
            + einsum("qp,qp,iq->ip", tau79, tau922, a.x4)
            + einsum("qp,qp,iq->ip", tau78, tau855, a.x3)
            - einsum("qp,qp,iq->ip", tau88, tau976, a.x3) / 4
            - einsum("qp,qpi->ip", tau667, tau1122) / 4
            + 2 * einsum("qp,qp,iq->ip", tau78, tau901, a.x4)
            + einsum("qp,qp,qp,iq->ip", tau126, tau665, tau78, a.x3)
            + einsum("qp,qp,qpi->ip", tau667, tau79, tau425) / 2
            + einsum("qp,qp,qp,iq->ip", tau513, tau667, tau87, a.x4)
            + einsum("qp,qp,qpi->ip", tau87, tau88, tau1123) / 2
            - einsum("qp,qp,qp,iq->ip", tau153, tau667, tau88, a.x3)
            - einsum("qp,qp,iq->ip", tau749, tau87, a.x3)
            - einsum("qp,qp,iq->ip", tau88, tau941, a.x3)
            - einsum("qp,qp,qpi->ip", tau667, tau79, tau1003) / 2
            - einsum("qi,qp,qp,qp->ip", tau449, tau665, tau78, tau88) / 2
            + 2 * einsum("qp,qp,iq->ip", tau762, tau87, a.x3)
            + einsum("qp,qp,iq->ip", tau87, tau919, a.x4) / 2
            + einsum("qp,qp,qp,qi->ip", tau665, tau78, tau79, tau80)
            - einsum("qp,qp,qpi->ip", tau665, tau78, tau1095) / 2
            - einsum("qp,qp,iq->ip", tau87, tau916, a.x3)
            - einsum("pj,pij->ip", tau73, tau1124)
            + einsum("qp,iq->ip", tau1125, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau561, tau665, a.x3) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau501, tau665, tau78, a.x4)
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau972)
            - 2 * einsum("qp,qp,qpi->ip", tau665, tau79, tau957)
            - 2 * einsum("qp,qp,qp,iq->ip", tau529, tau667, tau88, a.x3)
            - einsum("qp,qp,iq->ip", tau87, tau891, a.x3)
            + einsum("qp,qp,iq->ip", tau753, tau87, a.x4)
            - einsum("jp,pji->ip", a.y3, tau938)
            - einsum("jp,pij->ip", a.y3, tau936)
            - einsum("qp,qp,iq->ip", tau690, tau78, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau665, tau85, tau88, a.x3) / 2
            + 2 * einsum("qp,qp,qpi->ip", tau667, tau88, tau929)
            - einsum("qp,qp,iq->ip", tau378, tau665, a.x4) / 4
            + einsum("qp,qp,qpi->ip", tau78, tau88, tau991) / 2
            + einsum("qp,qpi->ip", tau665, tau1126) / 2
            + einsum("qp,qp,qp,iq->ip", tau240, tau667, tau87, a.x3) / 2
            - einsum("qp,qp,qpi->ip", tau667, tau87, tau1086) / 2
            - einsum("qp,qp,qp,iq->ip", tau401, tau667, tau79, a.x4) / 2
            + einsum("qp,qp,qp,iq->ip", tau676, tau87, tau88, a.x3)
            + einsum("qp,qp,qp,iq->ip", tau571, tau667, tau88, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau813, tau87, a.x4) / 2
            + einsum("qp,qp,iq->ip", tau617, tau667, a.x4) / 2
            - einsum("qp,qp,qpi->ip", tau78, tau88, tau1011)
            - einsum("qp,qp,iq->ip", tau677, tau78, a.x3) / 2
            + einsum("qp,qp,iq->ip", tau389, tau665, a.x3)
            - einsum("qp,qp,qpi->ip", tau667, tau79, tau923) / 2
            - einsum("qp,qp,qpi->ip", tau79, tau87, tau824)
            + einsum("qp,qpi->ip", tau667, tau1127) / 2
            - 2 * einsum("qp,qp,qp,iq->ip", tau234, tau667, tau87, a.x3)
            - einsum("qp,qp,iq->ip", tau79, tau989, a.x4)
            - 2 * einsum("qp,qp,qp,iq->ip", tau126, tau667, tau87, a.x3)
            + einsum("qp,qp,qpi->ip", tau78, tau79, tau1030)
            - einsum("qp,qp,qp,iq->ip", tau150, tau665, tau88, a.x3) / 4
            - einsum("qp,qp,qp,iq->ip", tau732, tau87, tau88, a.x3)
            + einsum("qp,qp,qpi->ip", tau665, tau88, tau1009)
            + einsum("qp,qp,qpi->ip", tau87, tau88, tau1048) / 2
            + einsum("qp,qp,qp,iq->ip", tau79, tau864, tau87, a.x4)
            - einsum("qp,qp,iq->ip", tau522, tau665, a.x3)
            - einsum("qp,qpi->ip", tau87, tau1128) / 4
            - 2 * einsum("pj,pij->ip", tau73, tau1129)
        )
        tau0 = (
            einsum("oia,ojb->ijab", h.l.pov, h.l.pov)
        )
    
        tau1 = (
            einsum("bp,ijab->pija", a.x1, tau0)
        )
    
        tau2 = (
            einsum("ai,pjka->pijk", a.t1, tau1)
        )
    
        tau3 = (
            einsum("kp,pijk->pij", a.x3, tau2)
        )
    
        tau4 = (
            einsum("jp,pij->pi", a.x4, tau3)
        )
    
        tau5 = (
            einsum("ia,ap->pi", h.f.ov, a.x2)
        )
    
        tau6 = (
            einsum("pi,ip->p", tau5, a.x3)
        )
    
        tau7 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )
    
        tau8 = (
            einsum("oij,oka->ijka", h.l.poo, h.l.pov)
        )
    
        tau9 = (
            einsum("ap,ijka->pijk", a.x1, tau8)
        )
    
        tau10 = (
            einsum("kp,pijk->pij", a.x3, tau9)
        )
    
        tau11 = (
            einsum("jp,pji->pi", a.x4, tau10)
        )
    
        tau12 = (
            einsum("ia,aj->ij", h.f.ov, a.t1)
        )
    
        tau13 = (
            einsum("o,oia->ia", tau7, h.l.pov)
        )
    
        tau14 = (
            einsum("ai,ja->ij", a.t1, tau13)
        )
    
        tau15 = (
            einsum("ia,ap->pi", tau13, a.x1)
        )
    
        tau16 = (
            einsum("pi,ip->p", tau15, a.x3)
        )
    
        tau17 = (
            einsum("bj,jiab->ia", a.t1, tau0)
        )
    
        tau18 = (
            einsum("ia,ap->pi", tau17, a.x2)
        )
    
        tau19 = (
            einsum("pi,ip->p", tau18, a.x4)
        )
    
        tau20 = (
            einsum("ap,pija->pij", a.x2, tau1)
        )
    
        tau21 = (
            einsum("jp,pij->pi", a.x4, tau20)
        )
    
        tau22 = (
            einsum("ai,pi->pa", a.t1, tau21)
        )
    
        tau23 = (
            einsum("oia,obc->iabc", h.l.pov, h.l.pvv)
        )
    
        tau24 = (
            einsum("cp,iabc->piab", a.x2, tau23)
        )
    
        tau25 = (
            einsum("bp,piba->pia", a.x1, tau24)
        )
    
        tau26 = (
            einsum("ip,pia->pa", a.x4, tau25)
        )
    
        tau27 = (
            einsum("ai,oia->o", a.t1, h.l.pov)
        )
    
        tau28 = (
            einsum("o,oij->ij", tau27, h.l.poo)
        )
    
        tau29 = (
            einsum("pi,ip->p", tau18, a.x3)
        )
    
        tau30 = (
            einsum("ia,ap->pi", tau17, a.x1)
        )
    
        tau31 = (
            einsum("pi,ip->p", tau30, a.x3)
        )
    
        tau32 = (
            einsum("bp,ijab->pija", a.x2, tau0)
        )
    
        tau33 = (
            einsum("ai,pjka->pijk", a.t1, tau32)
        )
    
        tau34 = (
            einsum("kp,pijk->pij", a.x3, tau33)
        )
    
        tau35 = (
            einsum("jp,pij->pi", a.x4, tau34)
        )
    
        tau36 = (
            einsum("ap,ijka->pijk", a.x2, tau8)
        )
    
        tau37 = (
            einsum("kp,pijk->pij", a.x3, tau36)
        )
    
        tau38 = (
            einsum("jp,pji->pi", a.x4, tau37)
        )
    
        tau39 = (
            einsum("kp,pijk->pij", a.x4, tau36)
        )
    
        tau40 = (
            einsum("jp,pji->pi", a.x3, tau39)
        )
    
        tau41 = (
            einsum("ak,kija->ij", a.t1, tau8)
        )
    
        tau42 = (
            einsum("ap,pija->pij", a.x1, tau32)
        )
    
        tau43 = (
            einsum("jp,pij->pi", a.x4, tau42)
        )
    
        tau44 = (
            einsum("ai,pi->pa", a.t1, tau43)
        )
    
        tau45 = (
            einsum("ia,ap->pi", h.f.ov, a.x1)
        )
    
        tau46 = (
            einsum("pi,ip->p", tau45, a.x4)
        )
    
        tau47 = (
            einsum("kp,pijk->pij", a.x4, tau9)
        )
    
        tau48 = (
            einsum("jp,pji->pi", a.x3, tau47)
        )
    
        tau49 = (
            einsum("kp,pijk->pij", a.x4, tau33)
        )
    
        tau50 = (
            einsum("jp,pij->pi", a.x3, tau49)
        )
    
        tau51 = (
            einsum("oij,oab->ijab", h.l.poo, h.l.pvv)
        )
    
        tau52 = (
            einsum("o,oab->ab", tau7, h.l.pvv)
        )
    
        tau53 = (
            einsum("jp,pij->pi", a.x3, tau42)
        )
    
        tau54 = (
            einsum("ai,pi->pa", a.t1, tau53)
        )
    
        tau55 = (
            einsum("ip,pia->pa", a.x3, tau25)
        )
    
        tau56 = (
            einsum("jp,pij->pi", a.x3, tau20)
        )
    
        tau57 = (
            einsum("ai,pi->pa", a.t1, tau56)
        )
    
        tau58 = (
            einsum("pi,ip->p", tau45, a.x3)
        )
    
        tau59 = (
            einsum("pi,ip->p", tau15, a.x4)
        )
    
        tau60 = (
            einsum("ai,ja->ij", a.t1, tau17)
        )
    
        tau61 = (
            einsum("cp,iabc->piab", a.x1, tau23)
        )
    
        tau62 = (
            einsum("bp,piba->pia", a.x2, tau61)
        )
    
        tau63 = (
            einsum("ip,pia->pa", a.x4, tau62)
        )
    
        tau64 = (
            einsum("ip,pia->pa", a.x3, tau62)
        )
    
        tau65 = (
            einsum("ia,ap->pi", tau13, a.x2)
        )
    
        tau66 = (
            einsum("pi,ip->p", tau65, a.x4)
        )
    
        tau67 = (
            einsum("ci,iabc->ab", a.t1, tau23)
        )
    
        tau68 = (
            einsum("kp,pijk->pij", a.x4, tau2)
        )
    
        tau69 = (
            einsum("jp,pij->pi", a.x3, tau68)
        )
    
        tau70 = (
            einsum("pi,ip->p", tau5, a.x4)
        )
    
        tau71 = (
            einsum("pi,ip->p", tau30, a.x4)
        )
    
        tau72 = (
            einsum("pi,ip->p", tau65, a.x3)
        )
    
        tau73 = (
            einsum("ap,aq->pq", a.x2, a.y1)
        )
    
        tau74 = (
            einsum("ip,iq->pq", a.x3, a.y4)
        )
    
        tau75 = (
            einsum("bp,ijba->pija", a.y2, tau51)
        )
    
        tau76 = (
            einsum("jp,pija->pia", a.y3, tau75)
        )
    
        tau77 = (
            einsum("ip,qia->pqa", a.x4, tau76)
        )
    
        tau78 = (
            einsum("ap,aq->pq", a.x2, a.y2)
        )
    
        tau79 = (
            einsum("ip,iq->pq", a.x3, a.y3)
        )
    
        tau80 = (
            einsum("ip,iq->pq", a.x4, a.y4)
        )
    
        tau81 = (
            einsum("qr,pr,pr,pr->pq", tau73, tau78, tau79, tau80)
        )
    
        tau82 = (
            einsum("jq,pija->pqia", a.x4, tau1)
        )
    
        tau83 = (
            einsum("iq,pqia->pqa", a.x3, tau82)
        )
    
        tau84 = (
            einsum("ap,aq->pq", a.x1, a.y1)
        )
    
        tau85 = (
            einsum("ip,iq->pq", a.x4, a.y3)
        )
    
        tau86 = (
            einsum("pr,qr,pr,qr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau87 = (
            einsum("jp,pija->pia", a.x4, tau32)
        )
    
        tau88 = (
            einsum("iq,pia->pqa", a.x3, tau87)
        )
    
        tau89 = (
            einsum("jq,pij->pqi", a.x4, tau42)
        )
    
        tau90 = (
            einsum("iq,pqi->pq", a.x3, tau89)
        )
    
        tau91 = (
            einsum("pr,qr,qr,ar->pqa", tau73, tau79, tau80, a.y2)
        )
    
        tau92 = (
            einsum("ba,bp->pa", tau67, a.x2)
        )
    
        tau93 = (
            einsum("pa,aq->pq", tau92, a.y2)
        )
    
        tau94 = (
            einsum("pr,pr,qr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau95 = (
            einsum("jq,pija->pqia", a.x4, tau32)
        )
    
        tau96 = (
            einsum("ip,pqia->pqa", a.x3, tau95)
        )
    
        tau97 = (
            einsum("pq,pq,pq,aq->pa", tau73, tau74, tau85, a.y2)
        )
    
        tau98 = (
            einsum("aq,pija->pqij", a.x2, tau1)
        )
    
        tau99 = (
            einsum("jq,pqij->pqi", a.x4, tau98)
        )
    
        tau100 = (
            einsum("iq,pqi->pq", a.x3, tau99)
        )
    
        tau101 = (
            einsum("ap,aq->pq", a.x1, a.y2)
        )
    
        tau102 = (
            einsum("pr,qr,pr,qr->pq", tau101, tau73, tau74, tau79)
        )
    
        tau103 = (
            einsum("ip,pqia->pqa", a.x4, tau95)
        )
    
        tau104 = (
            einsum("pa,aq->pq", tau92, a.y1)
        )
    
        tau105 = (
            einsum("oia,obj->ijab", h.l.pov, h.l.pvo)
        )
    
        tau106 = (
            einsum("bp,ijab->pija", a.y2, tau105)
        )
    
        tau107 = (
            einsum("ap,qija->pqij", a.x2, tau106)
        )
    
        tau108 = (
            einsum("jq,pqij->pqi", a.y4, tau107)
        )
    
        tau109 = (
            einsum("ip,pqi->pq", a.x3, tau108)
        )
    
        tau110 = (
            einsum("ap,qija->pqij", a.x1, tau32)
        )
    
        tau111 = (
            einsum("jp,pqij->pqi", a.x4, tau110)
        )
    
        tau112 = (
            einsum("iq,pqi->pq", a.x4, tau111)
        )
    
        tau113 = (
            einsum("pr,pr,qr,ar->pqa", tau73, tau74, tau79, a.y2)
        )
    
        tau114 = (
            einsum("jq,pqij->pqi", a.x3, tau110)
        )
    
        tau115 = (
            einsum("ip,pqi->pq", a.x3, tau114)
        )
    
        tau116 = (
            einsum("pr,pr,qr,ar->pqa", tau73, tau80, tau85, a.y2)
        )
    
        tau117 = (
            einsum("pi,iq->pq", tau43, a.x4)
        )
    
        tau118 = (
            einsum("bp,ijab->pija", a.x2, tau51)
        )
    
        tau119 = (
            einsum("aq,pija->pqij", a.y1, tau118)
        )
    
        tau120 = (
            einsum("jq,pqij->pqi", a.y4, tau119)
        )
    
        tau121 = (
            einsum("ip,pqi->pq", a.x4, tau120)
        )
    
        tau122 = (
            einsum("qr,pr,pr,qr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau123 = (
            einsum("bp,ijba->pija", a.x2, tau0)
        )
    
        tau124 = (
            einsum("jq,pija->pqia", a.x4, tau123)
        )
    
        tau125 = (
            einsum("iq,pqia->pqa", a.x3, tau124)
        )
    
        tau126 = (
            einsum("ai,ap->pi", a.t1, a.y1)
        )
    
        tau127 = (
            einsum("bi,jkab->ijka", a.t1, tau0)
        )
    
        tau128 = (
            einsum("kp,ijka->pija", a.x4, tau127)
        )
    
        tau129 = (
            einsum("jq,pjia->pqia", a.y3, tau128)
        )
    
        tau130 = (
            einsum("qi,pqia->pqa", tau126, tau129)
        )
    
        tau131 = (
            einsum("pr,pr,pr,qr->pq", tau101, tau73, tau80, tau85)
        )
    
        tau132 = (
            einsum("jq,pija->pqia", a.x3, tau32)
        )
    
        tau133 = (
            einsum("ip,pqia->pqa", a.x3, tau132)
        )
    
        tau134 = (
            einsum("ap,qija->pqij", a.x2, tau32)
        )
    
        tau135 = (
            einsum("jp,qpij->pqi", a.x4, tau134)
        )
    
        tau136 = (
            einsum("ip,qpi->pq", a.x4, tau135)
        )
    
        tau137 = (
            einsum("pr,pr,qr,ar->pqa", tau101, tau74, tau79, a.y1)
        )
    
        tau138 = (
            einsum("ij,jp->pi", tau41, a.x3)
        )
    
        tau139 = (
            einsum("pi,iq->pq", tau138, a.y3)
        )
    
        tau140 = (
            einsum("cp,iacb->piab", a.y2, tau23)
        )
    
        tau141 = (
            einsum("bp,qiba->pqia", a.x2, tau140)
        )
    
        tau142 = (
            einsum("ai,pqja->pqij", a.t1, tau141)
        )
    
        tau143 = (
            einsum("jp,pqij->pqi", a.x3, tau142)
        )
    
        tau144 = (
            einsum("iq,pqi->pq", a.y4, tau143)
        )
    
        tau145 = (
            einsum("pr,pr,qr,qr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau146 = (
            einsum("iq,pqia->pqa", a.x3, tau95)
        )
    
        tau147 = (
            einsum("jq,pji->pqi", a.y4, tau49)
        )
    
        tau148 = (
            einsum("qi,pqi->pq", tau126, tau147)
        )
    
        tau149 = (
            einsum("qr,pr,pr,ar->pqa", tau74, tau78, tau79, a.y1)
        )
    
        tau150 = (
            einsum("pq,pq,pq,aq->pa", tau101, tau79, tau80, a.y1)
        )
    
        tau151 = (
            einsum("jp,pqij->pqi", a.x4, tau134)
        )
    
        tau152 = (
            einsum("ip,pqi->pq", a.x3, tau151)
        )
    
        tau153 = (
            einsum("pr,qr,pr,pr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau154 = (
            einsum("ai,ap->pi", a.t1, a.y2)
        )
    
        tau155 = (
            einsum("kp,ijka->pija", a.x3, tau8)
        )
    
        tau156 = (
            einsum("jq,pija->pqia", a.y4, tau155)
        )
    
        tau157 = (
            einsum("qi,pqia->pqa", tau154, tau156)
        )
    
        tau158 = (
            einsum("ap,ai->pi", a.x2, a.z1)
        )
    
        tau159 = (
            einsum("kp,ijka->pija", a.x4, tau8)
        )
    
        tau160 = (
            einsum("pj,pija->pia", tau158, tau159)
        )
    
        tau161 = (
            einsum("ip,pqia->pqa", a.x3, tau124)
        )
    
        tau162 = (
            einsum("pr,pr,pr,qr->pq", tau78, tau80, tau84, tau85)
        )
    
        tau163 = (
            einsum("jp,pija->pia", a.x3, tau32)
        )
    
        tau164 = (
            einsum("iq,pia->pqa", a.x3, tau163)
        )
    
        tau165 = (
            einsum("jp,pqij->pqi", a.x3, tau110)
        )
    
        tau166 = (
            einsum("iq,pqi->pq", a.x3, tau165)
        )
    
        tau167 = (
            einsum("pr,qr,pr,ar->pqa", tau73, tau80, tau85, a.y2)
        )
    
        tau168 = (
            einsum("jq,pqij->pqi", a.x4, tau110)
        )
    
        tau169 = (
            einsum("ip,pqi->pq", a.x4, tau168)
        )
    
        tau170 = (
            einsum("pr,qr,pr,ar->pqa", tau73, tau74, tau79, a.y2)
        )
    
        tau171 = (
            einsum("pr,qr,pr,pr->pq", tau73, tau78, tau79, tau80)
        )
    
        tau172 = (
            einsum("bp,ijba->pija", a.x1, tau0)
        )
    
        tau173 = (
            einsum("jq,pija->pqia", a.x4, tau172)
        )
    
        tau174 = (
            einsum("iq,pqia->pqa", a.x3, tau173)
        )
    
        tau175 = (
            einsum("kq,pikj->pqij", a.y3, tau36)
        )
    
        tau176 = (
            einsum("jp,pqji->pqi", a.x3, tau175)
        )
    
        tau177 = (
            einsum("qi,pqi->pq", tau126, tau176)
        )
    
        tau178 = (
            einsum("aj,pij->pia", a.z1, tau37)
        )
    
        tau179 = (
            einsum("kq,pikj->pqij", a.y4, tau36)
        )
    
        tau180 = (
            einsum("jp,pqji->pqi", a.x3, tau179)
        )
    
        tau181 = (
            einsum("qi,pqi->pq", tau126, tau180)
        )
    
        tau182 = (
            einsum("pr,pr,qr,qr->pq", tau73, tau74, tau78, tau85)
        )
    
        tau183 = (
            einsum("ip,pqia->pqa", a.x3, tau82)
        )
    
        tau184 = (
            einsum("jq,pija->pqia", a.y3, tau155)
        )
    
        tau185 = (
            einsum("qi,pqia->pqa", tau126, tau184)
        )
    
        tau186 = (
            einsum("qr,pr,pr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau187 = (
            einsum("pr,qr,pr,qr->pq", tau101, tau73, tau80, tau85)
        )
    
        tau188 = (
            einsum("pr,pr,qr,pr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau189 = (
            einsum("jq,pji->pqi", a.y4, tau34)
        )
    
        tau190 = (
            einsum("qi,pqi->pq", tau154, tau189)
        )
    
        tau191 = (
            einsum("ij,jp->pi", h.f.oo, a.y4)
        )
    
        tau192 = (
            einsum("qi,ip->pq", tau191, a.x4)
        )
    
        tau193 = (
            einsum("bi,pjab->pija", a.t1, tau140)
        )
    
        tau194 = (
            einsum("jp,qija->pqia", a.x3, tau193)
        )
    
        tau195 = (
            einsum("iq,pqia->pqa", a.y4, tau194)
        )
    
        tau196 = (
            einsum("pr,pr,qr,ar->pqa", tau78, tau80, tau85, a.y1)
        )
    
        tau197 = (
            einsum("pi,iq->pq", tau21, a.x3)
        )
    
        tau198 = (
            einsum("pr,qr,pr,ar->pqa", tau73, tau79, tau80, a.y2)
        )
    
        tau199 = (
            einsum("ai,jkla->ijkl", a.t1, tau127)
        )
    
        tau200 = (
            einsum("lp,ijkl->pijk", a.x4, tau199)
        )
    
        tau201 = (
            einsum("kp,pijk->pij", a.x3, tau200)
        )
    
        tau202 = (
            einsum("jq,pij->pqi", a.y4, tau201)
        )
    
        tau203 = (
            einsum("iq,pqi->pq", a.y3, tau202)
        )
    
        tau204 = (
            einsum("ij,jp->pi", tau12, a.y3)
        )
    
        tau205 = (
            einsum("qi,ip->pq", tau204, a.x3)
        )
    
        tau206 = (
            einsum("jp,qija->pqia", a.x4, tau193)
        )
    
        tau207 = (
            einsum("iq,pqia->pqa", a.y4, tau206)
        )
    
        tau208 = (
            einsum("ba,bp->pa", tau52, a.y1)
        )
    
        tau209 = (
            einsum("cp,iacb->piab", a.y1, tau23)
        )
    
        tau210 = (
            einsum("bi,pjba->pija", a.t1, tau209)
        )
    
        tau211 = (
            einsum("jp,qija->pqia", a.x4, tau210)
        )
    
        tau212 = (
            einsum("iq,pqia->pqa", a.y4, tau211)
        )
    
        tau213 = (
            einsum("ip,ai->pa", a.x3, a.z1)
        )
    
        tau214 = (
            einsum("ip,pqi->pq", a.x4, tau108)
        )
    
        tau215 = (
            einsum("jq,pij->pqi", a.x4, tau20)
        )
    
        tau216 = (
            einsum("iq,pqi->pq", a.x3, tau215)
        )
    
        tau217 = (
            einsum("pr,qr,qr,ar->pqa", tau78, tau79, tau80, a.y1)
        )
    
        tau218 = (
            einsum("ia,pi->pa", h.f.ov, tau154)
        )
    
        tau219 = (
            einsum("bq,piab->pqia", a.y1, tau24)
        )
    
        tau220 = (
            einsum("ai,pqja->pqij", a.t1, tau219)
        )
    
        tau221 = (
            einsum("jp,pqij->pqi", a.x3, tau220)
        )
    
        tau222 = (
            einsum("iq,pqi->pq", a.y4, tau221)
        )
    
        tau223 = (
            einsum("iq,pqi->pq", a.x3, tau111)
        )
    
        tau224 = (
            einsum("pr,pr,qr,ar->pqa", tau74, tau78, tau85, a.y1)
        )
    
        tau225 = (
            einsum("qi,ip->pq", tau191, a.x3)
        )
    
        tau226 = (
            einsum("pr,qr,pr,pr->pq", tau78, tau80, tau84, tau85)
        )
    
        tau227 = (
            einsum("pr,qr,pr,qr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau228 = (
            einsum("jp,pija->pia", a.x4, tau123)
        )
    
        tau229 = (
            einsum("iq,pia->pqa", a.x3, tau228)
        )
    
        tau230 = (
            einsum("pi,ia->pa", tau126, tau13)
        )
    
        tau231 = (
            einsum("ip,qpi->pq", a.x3, tau151)
        )
    
        tau232 = (
            einsum("qr,pr,pr,ar->pqa", tau74, tau84, tau85, a.y2)
        )
    
        tau233 = (
            einsum("ij,jp->pi", tau14, a.x4)
        )
    
        tau234 = (
            einsum("pi,iq->pq", tau233, a.y3)
        )
    
        tau235 = (
            einsum("ij,jp->pi", tau60, a.x3)
        )
    
        tau236 = (
            einsum("pi,iq->pq", tau235, a.y3)
        )
    
        tau237 = (
            einsum("ai,pjka->pijk", a.t1, tau123)
        )
    
        tau238 = (
            einsum("kp,pijk->pij", a.x3, tau237)
        )
    
        tau239 = (
            einsum("jq,pji->pqi", a.y3, tau238)
        )
    
        tau240 = (
            einsum("qi,pqi->pq", tau154, tau239)
        )
    
        tau241 = (
            einsum("jq,pij->pqi", a.y3, tau39)
        )
    
        tau242 = (
            einsum("qi,pqi->pq", tau154, tau241)
        )
    
        tau243 = (
            einsum("qi,pi->pq", tau126, tau65)
        )
    
        tau244 = (
            einsum("qi,ip->pq", tau204, a.x4)
        )
    
        tau245 = (
            einsum("pr,qr,qr,ar->pqa", tau73, tau74, tau85, a.y2)
        )
    
        tau246 = (
            einsum("qr,pr,qr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau247 = (
            einsum("qr,qr,pr,pr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau248 = (
            einsum("bp,ijab->pija", a.y1, tau105)
        )
    
        tau249 = (
            einsum("jp,pija->pia", a.y4, tau248)
        )
    
        tau250 = (
            einsum("ip,qia->pqa", a.x3, tau249)
        )
    
        tau251 = (
            einsum("qj,pija->pqia", tau126, tau32)
        )
    
        tau252 = (
            einsum("qi,pqia->pqa", tau154, tau251)
        )
    
        tau253 = (
            einsum("bj,ijba->ia", a.t1, tau0)
        )
    
        tau254 = (
            einsum("ia,ap->pi", tau253, a.x2)
        )
    
        tau255 = (
            einsum("qi,pi->pq", tau154, tau254)
        )
    
        tau256 = (
            einsum("pr,qr,qr,pr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau257 = (
            einsum("pr,qr,qr,pr->pq", tau73, tau78, tau80, tau85)
        )
    
        tau258 = (
            einsum("jp,pija->pia", a.x3, tau1)
        )
    
        tau259 = (
            einsum("iq,pia->pqa", a.x3, tau258)
        )
    
        tau260 = (
            einsum("ij,jp->pi", tau28, a.y3)
        )
    
        tau261 = (
            einsum("qi,ip->pq", tau260, a.x4)
        )
    
        tau262 = (
            einsum("pi,iq->pq", tau235, a.y4)
        )
    
        tau263 = (
            einsum("pi,iq->pq", tau43, a.x3)
        )
    
        tau264 = (
            einsum("qr,pr,pr,ar->pqa", tau74, tau78, tau85, a.y1)
        )
    
        tau265 = (
            einsum("ip,pia->pa", a.x3, tau228)
        )
    
        tau266 = (
            einsum("pr,qr,qr,qr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau267 = (
            einsum("jp,pqji->pqi", a.x4, tau175)
        )
    
        tau268 = (
            einsum("qi,pqi->pq", tau154, tau267)
        )
    
        tau269 = (
            einsum("pq,pq,pq,aq->pa", tau78, tau79, tau80, a.y1)
        )
    
        tau270 = (
            einsum("qr,pr,pr,qr->pq", tau78, tau80, tau84, tau85)
        )
    
        tau271 = (
            einsum("pq,pq,pq,aq->pa", tau74, tau84, tau85, a.y2)
        )
    
        tau272 = (
            einsum("ip,pqi->pq", a.x3, tau135)
        )
    
        tau273 = (
            einsum("qi,pi->pq", tau126, tau5)
        )
    
        tau274 = (
            einsum("ba,bp->pa", tau52, a.y2)
        )
    
        tau275 = (
            einsum("qa,ap->pq", tau274, a.x2)
        )
    
        tau276 = (
            einsum("ip,pqi->pq", a.x3, tau168)
        )
    
        tau277 = (
            einsum("pr,qr,pr,ar->pqa", tau73, tau74, tau85, a.y2)
        )
    
        tau278 = (
            einsum("ba,bp->pa", h.f.vv, a.y1)
        )
    
        tau279 = (
            einsum("qi,pqia->pqa", tau154, tau129)
        )
    
        tau280 = (
            einsum("pr,qr,pr,qr->pq", tau73, tau78, tau79, tau80)
        )
    
        tau281 = (
            einsum("jp,pija->pia", a.x4, tau1)
        )
    
        tau282 = (
            einsum("iq,pia->pqa", a.x3, tau281)
        )
    
        tau283 = (
            einsum("ip,ai->pa", a.x4, a.z1)
        )
    
        tau284 = (
            einsum("bi,jkba->ijka", a.t1, tau0)
        )
    
        tau285 = (
            einsum("kp,ijka->pija", a.x3, tau284)
        )
    
        tau286 = (
            einsum("jq,pjia->pqia", a.y4, tau285)
        )
    
        tau287 = (
            einsum("qi,pqia->pqa", tau154, tau286)
        )
    
        tau288 = (
            einsum("iq,pqi->pq", a.x4, tau151)
        )
    
        tau289 = (
            einsum("qr,pr,pr,ar->pqa", tau74, tau79, tau84, a.y2)
        )
    
        tau290 = (
            einsum("ip,pia->pa", a.x3, tau87)
        )
    
        tau291 = (
            einsum("pr,qr,qr,qr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau292 = (
            einsum("qr,pr,pr,pr->pq", tau73, tau74, tau78, tau85)
        )
    
        tau293 = (
            einsum("pr,qr,qr,pr->pq", tau101, tau73, tau80, tau85)
        )
    
        tau294 = (
            einsum("jp,pqij->pqi", a.x4, tau220)
        )
    
        tau295 = (
            einsum("iq,pqi->pq", a.y3, tau294)
        )
    
        tau296 = (
            einsum("pr,qr,pr,ar->pqa", tau101, tau74, tau85, a.y1)
        )
    
        tau297 = (
            einsum("qi,ip->pq", tau260, a.x3)
        )
    
        tau298 = (
            einsum("ia,pi->pa", tau13, tau154)
        )
    
        tau299 = (
            einsum("qi,pi->pq", tau154, tau65)
        )
    
        tau300 = (
            einsum("ab,bp->pa", h.f.vv, a.x2)
        )
    
        tau301 = (
            einsum("pa,aq->pq", tau300, a.y2)
        )
    
        tau302 = (
            einsum("pr,qr,pr,pr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau303 = (
            einsum("qr,pr,pr,qr->pq", tau73, tau74, tau78, tau85)
        )
    
        tau304 = (
            einsum("ip,pqia->pqa", a.x3, tau173)
        )
    
        tau305 = (
            einsum("ai,aj->ij", a.t1, a.z1)
        )
    
        tau306 = (
            einsum("ij,jp->pi", tau305, a.x3)
        )
    
        tau307 = (
            einsum("ip,qpi->pq", a.x3, tau135)
        )
    
        tau308 = (
            einsum("pr,pr,qr,ar->pqa", tau74, tau84, tau85, a.y2)
        )
    
        tau309 = (
            einsum("kp,pijk->pij", a.x4, tau237)
        )
    
        tau310 = (
            einsum("jq,pji->pqi", a.y4, tau309)
        )
    
        tau311 = (
            einsum("qi,pqi->pq", tau126, tau310)
        )
    
        tau312 = (
            einsum("pr,qr,qr,pr->pq", tau101, tau73, tau74, tau79)
        )
    
        tau313 = (
            einsum("bi,pjab->pija", a.t1, tau209)
        )
    
        tau314 = (
            einsum("jp,qija->pqia", a.x3, tau313)
        )
    
        tau315 = (
            einsum("iq,pqia->pqa", a.y3, tau314)
        )
    
        tau316 = (
            einsum("aq,pija->pqij", a.y2, tau118)
        )
    
        tau317 = (
            einsum("jq,pqij->pqi", a.y3, tau316)
        )
    
        tau318 = (
            einsum("ip,pqi->pq", a.x4, tau317)
        )
    
        tau319 = (
            einsum("jq,pqij->pqi", a.y4, tau316)
        )
    
        tau320 = (
            einsum("ip,pqi->pq", a.x3, tau319)
        )
    
        tau321 = (
            einsum("ij,jp->pi", tau12, a.y4)
        )
    
        tau322 = (
            einsum("qi,ip->pq", tau321, a.x3)
        )
    
        tau323 = (
            einsum("jp,pija->pia", a.x4, tau172)
        )
    
        tau324 = (
            einsum("iq,pia->pqa", a.x3, tau323)
        )
    
        tau325 = (
            einsum("kp,ijka->pija", a.x4, tau284)
        )
    
        tau326 = (
            einsum("jq,pjia->pqia", a.y3, tau325)
        )
    
        tau327 = (
            einsum("qi,pqia->pqa", tau126, tau326)
        )
    
        tau328 = (
            einsum("qi,pqia->pqa", tau126, tau141)
        )
    
        tau329 = (
            einsum("ia,pi->pa", h.f.ov, tau126)
        )
    
        tau330 = (
            einsum("ip,pqi->pq", a.x3, tau89)
        )
    
        tau331 = (
            einsum("pr,pr,qr,ar->pqa", tau73, tau79, tau80, a.y2)
        )
    
        tau332 = (
            einsum("ij,jp->pi", tau14, a.x3)
        )
    
        tau333 = (
            einsum("pi,iq->pq", tau332, a.y3)
        )
    
        tau334 = (
            einsum("pr,qr,pr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau335 = (
            einsum("jq,pji->pqi", a.y3, tau34)
        )
    
        tau336 = (
            einsum("qi,pqi->pq", tau154, tau335)
        )
    
        tau337 = (
            einsum("ap,qija->pqij", a.x2, tau248)
        )
    
        tau338 = (
            einsum("jq,pqij->pqi", a.y4, tau337)
        )
    
        tau339 = (
            einsum("ip,pqi->pq", a.x4, tau338)
        )
    
        tau340 = (
            einsum("qr,pr,qr,ar->pqa", tau74, tau84, tau85, a.y2)
        )
    
        tau341 = (
            einsum("qr,pr,qr,ar->pqa", tau74, tau78, tau85, a.y1)
        )
    
        tau342 = (
            einsum("ip,pqi->pq", a.x4, tau89)
        )
    
        tau343 = (
            einsum("bp,qiba->pqia", a.x2, tau209)
        )
    
        tau344 = (
            einsum("ai,pqja->pqij", a.t1, tau343)
        )
    
        tau345 = (
            einsum("jp,pqij->pqi", a.x4, tau344)
        )
    
        tau346 = (
            einsum("iq,pqi->pq", a.y4, tau345)
        )
    
        tau347 = (
            einsum("pr,qr,qr,ar->pqa", tau101, tau79, tau80, a.y1)
        )
    
        tau348 = (
            einsum("pr,qr,pr,ar->pqa", tau78, tau79, tau80, a.y1)
        )
    
        tau349 = (
            einsum("kp,ikja->pija", a.y4, tau8)
        )
    
        tau350 = (
            einsum("jp,qjia->pqia", a.x4, tau349)
        )
    
        tau351 = (
            einsum("qi,pqia->pqa", tau154, tau350)
        )
    
        tau352 = (
            einsum("pi,ia->pa", tau126, tau253)
        )
    
        tau353 = (
            einsum("jp,qija->pqia", a.x3, tau210)
        )
    
        tau354 = (
            einsum("iq,pqia->pqa", a.y3, tau353)
        )
    
        tau355 = (
            einsum("qi,pqi->pq", tau126, tau241)
        )
    
        tau356 = (
            einsum("ij,jp->pi", tau60, a.x4)
        )
    
        tau357 = (
            einsum("pi,iq->pq", tau356, a.y3)
        )
    
        tau358 = (
            einsum("pr,pr,qr,ar->pqa", tau78, tau79, tau80, a.y1)
        )
    
        tau359 = (
            einsum("pr,pr,qr,pr->pq", tau73, tau74, tau78, tau85)
        )
    
        tau360 = (
            einsum("pi,iq->pq", tau233, a.y4)
        )
    
        tau361 = (
            einsum("jp,qpij->pqi", a.x3, tau134)
        )
    
        tau362 = (
            einsum("ip,qpi->pq", a.x3, tau361)
        )
    
        tau363 = (
            einsum("pr,pr,qr,ar->pqa", tau80, tau84, tau85, a.y2)
        )
    
        tau364 = (
            einsum("ip,pqi->pq", a.x3, tau111)
        )
    
        tau365 = (
            einsum("bj,piab->pija", a.z1, tau24)
        )
    
        tau366 = (
            einsum("jp,pija->pia", a.x3, tau365)
        )
    
        tau367 = (
            einsum("iq,pqia->pqa", a.y3, tau206)
        )
    
        tau368 = (
            einsum("jq,pija->pqia", a.x3, tau1)
        )
    
        tau369 = (
            einsum("ip,pqia->pqa", a.x3, tau368)
        )
    
        tau370 = (
            einsum("pr,pr,qr,pr->pq", tau74, tau78, tau79, tau84)
        )
    
        tau371 = (
            einsum("bp,ijba->pija", a.y1, tau51)
        )
    
        tau372 = (
            einsum("jp,pija->pia", a.y4, tau371)
        )
    
        tau373 = (
            einsum("ip,qia->pqa", a.x3, tau372)
        )
    
        tau374 = (
            einsum("ip,qia->pqa", a.x3, tau76)
        )
    
        tau375 = (
            einsum("pi,iq->pq", tau332, a.y4)
        )
    
        tau376 = (
            einsum("jp,pija->pia", a.y3, tau106)
        )
    
        tau377 = (
            einsum("ip,qia->pqa", a.x4, tau376)
        )
    
        tau378 = (
            einsum("bq,piab->pqia", a.y2, tau24)
        )
    
        tau379 = (
            einsum("qi,pqia->pqa", tau126, tau378)
        )
    
        tau380 = (
            einsum("pq,pq,pq,aq->pa", tau73, tau79, tau80, a.y2)
        )
    
        tau381 = (
            einsum("kp,ikja->pija", a.y3, tau8)
        )
    
        tau382 = (
            einsum("jp,qjia->pqia", a.x3, tau381)
        )
    
        tau383 = (
            einsum("qi,pqia->pqa", tau154, tau382)
        )
    
        tau384 = (
            einsum("pq,pq,pq,aq->pa", tau79, tau80, tau84, a.y2)
        )
    
        tau385 = (
            einsum("pr,qr,pr,pr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau386 = (
            einsum("jp,pija->pia", a.y4, tau106)
        )
    
        tau387 = (
            einsum("ip,qia->pqa", a.x4, tau386)
        )
    
        tau388 = (
            einsum("jq,pji->pqi", a.y4, tau201)
        )
    
        tau389 = (
            einsum("iq,pqi->pq", a.y3, tau388)
        )
    
        tau390 = (
            einsum("pr,pr,qr,qr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau391 = (
            einsum("oij,okl->ijkl", h.l.poo, h.l.poo)
        )
    
        tau392 = (
            einsum("lp,ijkl->pijk", a.y3, tau391)
        )
    
        tau393 = (
            einsum("kp,qijk->pqij", a.x4, tau392)
        )
    
        tau394 = (
            einsum("jq,pqij->pqi", a.y4, tau393)
        )
    
        tau395 = (
            einsum("ip,pqi->pq", a.x3, tau394)
        )
    
        tau396 = (
            einsum("jp,pqij->pqi", a.x4, tau98)
        )
    
        tau397 = (
            einsum("ip,pqi->pq", a.x3, tau396)
        )
    
        tau398 = (
            einsum("qr,pr,pr,qr->pq", tau73, tau78, tau79, tau80)
        )
    
        tau399 = (
            einsum("iq,pia->pqa", a.x4, tau87)
        )
    
        tau400 = (
            einsum("jp,pqij->pqi", a.x3, tau134)
        )
    
        tau401 = (
            einsum("iq,pqi->pq", a.x3, tau400)
        )
    
        tau402 = (
            einsum("pr,qr,pr,ar->pqa", tau101, tau80, tau85, a.y1)
        )
    
        tau403 = (
            einsum("qi,pqia->pqa", tau154, tau343)
        )
    
        tau404 = (
            einsum("ip,qia->pqa", a.x4, tau372)
        )
    
        tau405 = (
            einsum("jp,pqij->pqi", a.x4, tau142)
        )
    
        tau406 = (
            einsum("iq,pqi->pq", a.y4, tau405)
        )
    
        tau407 = (
            einsum("qr,qr,pr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau408 = (
            einsum("pr,pr,qr,ar->pqa", tau74, tau78, tau79, a.y1)
        )
    
        tau409 = (
            einsum("bi,pjba->pija", a.t1, tau140)
        )
    
        tau410 = (
            einsum("jp,qija->pqia", a.x3, tau409)
        )
    
        tau411 = (
            einsum("iq,pqia->pqa", a.y4, tau410)
        )
    
        tau412 = (
            einsum("pr,pr,qr,pr->pq", tau101, tau73, tau80, tau85)
        )
    
        tau413 = (
            einsum("pr,qr,qr,pr->pq", tau73, tau74, tau78, tau79)
        )
    
        tau414 = (
            einsum("iq,pia->pqa", a.x4, tau323)
        )
    
        tau415 = (
            einsum("pi,iq->pq", tau356, a.y4)
        )
    
        tau416 = (
            einsum("ip,qia->pqa", a.x3, tau376)
        )
    
        tau417 = (
            einsum("lp,ijkl->pijk", a.y4, tau391)
        )
    
        tau418 = (
            einsum("kp,qijk->pqij", a.x4, tau417)
        )
    
        tau419 = (
            einsum("jq,pqij->pqi", a.y3, tau418)
        )
    
        tau420 = (
            einsum("ip,pqi->pq", a.x3, tau419)
        )
    
        tau421 = (
            einsum("pr,pr,qr,ar->pqa", tau101, tau80, tau85, a.y1)
        )
    
        tau422 = (
            einsum("pr,qr,qr,pr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau423 = (
            einsum("jq,pija->pqia", a.y3, tau159)
        )
    
        tau424 = (
            einsum("qi,pqia->pqa", tau126, tau423)
        )
    
        tau425 = (
            einsum("iq,pqi->pq", a.y3, tau221)
        )
    
        tau426 = (
            einsum("qr,pr,pr,ar->pqa", tau79, tau80, tau84, a.y2)
        )
    
        tau427 = (
            einsum("ij,jp->pi", h.f.oo, a.y3)
        )
    
        tau428 = (
            einsum("qi,ip->pq", tau427, a.x3)
        )
    
        tau429 = (
            einsum("pi,ia->pa", tau154, tau253)
        )
    
        tau430 = (
            einsum("qi,pi->pq", tau154, tau5)
        )
    
        tau431 = (
            einsum("pr,qr,pr,qr->pq", tau73, tau78, tau80, tau85)
        )
    
        tau432 = (
            einsum("jp,pija->pia", a.x3, tau172)
        )
    
        tau433 = (
            einsum("iq,pia->pqa", a.x3, tau432)
        )
    
        tau434 = (
            einsum("qi,pqia->pqa", tau126, tau382)
        )
    
        tau435 = (
            einsum("ai,jkla->ijkl", a.t1, tau8)
        )
    
        tau436 = (
            einsum("lp,ijkl->pijk", a.x3, tau435)
        )
    
        tau437 = (
            einsum("kq,pijk->pqij", a.y3, tau436)
        )
    
        tau438 = (
            einsum("jp,pqij->pqi", a.x4, tau437)
        )
    
        tau439 = (
            einsum("iq,pqi->pq", a.y4, tau438)
        )
    
        tau440 = (
            einsum("ai,pqja->pqij", a.t1, tau378)
        )
    
        tau441 = (
            einsum("jp,pqij->pqi", a.x3, tau440)
        )
    
        tau442 = (
            einsum("iq,pqi->pq", a.y4, tau441)
        )
    
        tau443 = (
            einsum("qi,pqi->pq", tau154, tau147)
        )
    
        tau444 = (
            einsum("jp,pqij->pqi", a.x3, tau344)
        )
    
        tau445 = (
            einsum("iq,pqi->pq", a.y3, tau444)
        )
    
        tau446 = (
            einsum("pr,pr,pr,qr->pq", tau101, tau73, tau74, tau85)
        )
    
        tau447 = (
            einsum("jp,pija->pia", a.y3, tau248)
        )
    
        tau448 = (
            einsum("ip,qia->pqa", a.x3, tau447)
        )
    
        tau449 = (
            einsum("pr,pr,pr,qr->pq", tau101, tau73, tau74, tau79)
        )
    
        tau450 = (
            einsum("jp,pija->pia", a.y3, tau371)
        )
    
        tau451 = (
            einsum("ip,qia->pqa", a.x3, tau450)
        )
    
        tau452 = (
            einsum("qi,pqi->pq", tau126, tau335)
        )
    
        tau453 = (
            einsum("lp,ijkl->pijk", a.x4, tau435)
        )
    
        tau454 = (
            einsum("kq,pijk->pqij", a.y3, tau453)
        )
    
        tau455 = (
            einsum("jp,pqij->pqi", a.x3, tau454)
        )
    
        tau456 = (
            einsum("iq,pqi->pq", a.y4, tau455)
        )
    
        tau457 = (
            einsum("oab,ocd->abcd", h.l.pvv, h.l.pvv)
        )
    
        tau458 = (
            einsum("dp,abcd->pabc", a.x2, tau457)
        )
    
        tau459 = (
            einsum("cq,pabc->pqab", a.y2, tau458)
        )
    
        tau460 = (
            einsum("bq,pqba->pqa", a.y1, tau459)
        )
    
        tau461 = (
            einsum("qi,pqi->pq", tau154, tau180)
        )
    
        tau462 = (
            einsum("jq,pji->pqi", a.y3, tau49)
        )
    
        tau463 = (
            einsum("qi,pqi->pq", tau154, tau462)
        )
    
        tau464 = (
            einsum("pi,ip->p", tau158, a.x4)
        )
    
        tau465 = (
            einsum("ia,ip->pa", h.f.ov, a.x3)
        )
    
        tau466 = (
            einsum("jq,pij->pqi", a.x3, tau42)
        )
    
        tau467 = (
            einsum("ip,pqi->pq", a.x3, tau466)
        )
    
        tau468 = (
            einsum("iq,pqi->pq", a.x3, tau396)
        )
    
        tau469 = (
            einsum("ip,pqi->pq", a.x3, tau99)
        )
    
        tau470 = (
            einsum("pq,pq,pq,aq->pa", tau101, tau74, tau85, a.y1)
        )
    
        tau471 = (
            einsum("pr,qr,qr,ar->pqa", tau101, tau74, tau85, a.y1)
        )
    
        tau472 = (
            einsum("ip,pqi->pq", a.x3, tau215)
        )
    
        tau473 = (
            einsum("iq,pqi->pq", a.y3, tau345)
        )
    
        tau474 = (
            einsum("qr,qr,pr,qr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau475 = (
            einsum("aj,pij->pia", a.z1, tau39)
        )
    
        tau476 = (
            einsum("qa,ap->pq", tau208, a.x2)
        )
    
        tau477 = (
            einsum("pq,pq,pq,aq->pa", tau74, tau78, tau85, a.y1)
        )
    
        tau478 = (
            einsum("qr,pr,pr,pr->pq", tau74, tau78, tau79, tau84)
        )
    
        tau479 = (
            einsum("jq,pjia->pqia", a.y4, tau128)
        )
    
        tau480 = (
            einsum("qi,pqia->pqa", tau154, tau479)
        )
    
        tau481 = (
            einsum("qi,pqi->pq", tau126, tau462)
        )
    
        tau482 = (
            einsum("ij,jp->pi", tau305, a.x4)
        )
    
        tau483 = (
            einsum("pr,pr,qr,ar->pqa", tau101, tau74, tau85, a.y1)
        )
    
        tau484 = (
            einsum("qj,pija->pqia", tau154, tau32)
        )
    
        tau485 = (
            einsum("qi,pqia->pqa", tau126, tau484)
        )
    
        tau486 = (
            einsum("ij,jp->pi", tau28, a.y4)
        )
    
        tau487 = (
            einsum("qi,ip->pq", tau486, a.x4)
        )
    
        tau488 = (
            einsum("jq,pqij->pqi", a.y3, tau107)
        )
    
        tau489 = (
            einsum("ip,pqi->pq", a.x3, tau488)
        )
    
        tau490 = (
            einsum("pr,qr,pr,ar->pqa", tau74, tau79, tau84, a.y2)
        )
    
        tau491 = (
            einsum("qi,pqia->pqa", tau126, tau479)
        )
    
        tau492 = (
            einsum("pr,pr,qr,pr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau493 = (
            einsum("jq,pij->pqi", a.y4, tau39)
        )
    
        tau494 = (
            einsum("qi,pqi->pq", tau126, tau493)
        )
    
        tau495 = (
            einsum("qi,pqia->pqa", tau126, tau350)
        )
    
        tau496 = (
            einsum("pr,pr,pr,qr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau497 = (
            einsum("ia,ip->pa", tau13, a.x3)
        )
    
        tau498 = (
            einsum("pj,pija->pia", tau158, tau155)
        )
    
        tau499 = (
            einsum("jq,pji->pqi", a.y3, tau309)
        )
    
        tau500 = (
            einsum("qi,pqi->pq", tau126, tau499)
        )
    
        tau501 = (
            einsum("jp,qija->pqia", a.x4, tau313)
        )
    
        tau502 = (
            einsum("iq,pqia->pqa", a.y3, tau501)
        )
    
        tau503 = (
            einsum("ip,pia->pa", a.x3, tau323)
        )
    
        tau504 = (
            einsum("pr,qr,pr,ar->pqa", tau78, tau80, tau85, a.y1)
        )
    
        tau505 = (
            einsum("qr,pr,pr,ar->pqa", tau80, tau84, tau85, a.y2)
        )
    
        tau506 = (
            einsum("ip,pqi->pq", a.x3, tau120)
        )
    
        tau507 = (
            einsum("ip,pqi->pq", a.x3, tau317)
        )
    
        tau508 = (
            einsum("pr,pr,qr,qr->pq", tau73, tau74, tau78, tau79)
        )
    
        tau509 = (
            einsum("iq,pia->pqa", a.x4, tau281)
        )
    
        tau510 = (
            einsum("qr,qr,pr,ar->pqa", tau79, tau80, tau84, a.y2)
        )
    
        tau511 = (
            einsum("jq,pjia->pqia", a.y4, tau325)
        )
    
        tau512 = (
            einsum("qi,pqia->pqa", tau154, tau511)
        )
    
        tau513 = (
            einsum("pr,pr,qr,ar->pqa", tau73, tau74, tau85, a.y2)
        )
    
        tau514 = (
            einsum("iq,pqi->pq", a.y3, tau405)
        )
    
        tau515 = (
            einsum("pr,qr,qr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau516 = (
            einsum("qi,pqi->pq", tau154, tau493)
        )
    
        tau517 = (
            einsum("pi,iq->pq", tau53, a.x3)
        )
    
        tau518 = (
            einsum("jq,pjia->pqia", a.y3, tau285)
        )
    
        tau519 = (
            einsum("qi,pqia->pqa", tau154, tau518)
        )
    
        tau520 = (
            einsum("jp,pija->pia", a.x3, tau325)
        )
    
        tau521 = (
            einsum("pr,pr,qr,ar->pqa", tau101, tau79, tau80, a.y1)
        )
    
        tau522 = (
            einsum("qr,qr,pr,pr->pq", tau78, tau80, tau84, tau85)
        )
    
        tau523 = (
            einsum("iq,pqia->pqa", a.y4, tau353)
        )
    
        tau524 = (
            einsum("pr,qr,pr,qr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau525 = (
            einsum("iq,pqia->pqa", a.y3, tau194)
        )
    
        tau526 = (
            einsum("ip,qia->pqa", a.x4, tau447)
        )
    
        tau527 = (
            einsum("jp,qjia->pqia", a.x4, tau381)
        )
    
        tau528 = (
            einsum("qi,pqia->pqa", tau154, tau527)
        )
    
        tau529 = (
            einsum("jq,pija->pqia", a.y4, tau159)
        )
    
        tau530 = (
            einsum("qi,pqia->pqa", tau126, tau529)
        )
    
        tau531 = (
            einsum("pr,qr,pr,ar->pqa", tau79, tau80, tau84, a.y2)
        )
    
        tau532 = (
            einsum("kq,pijk->pqij", a.y4, tau453)
        )
    
        tau533 = (
            einsum("jp,pqij->pqi", a.x3, tau532)
        )
    
        tau534 = (
            einsum("iq,pqi->pq", a.y3, tau533)
        )
    
        tau535 = (
            einsum("qi,pqia->pqa", tau154, tau219)
        )
    
        tau536 = (
            einsum("qi,pqi->pq", tau154, tau499)
        )
    
        tau537 = (
            einsum("iq,pqi->pq", a.x3, tau168)
        )
    
        tau538 = (
            einsum("jq,pqij->pqi", a.y3, tau119)
        )
    
        tau539 = (
            einsum("ip,pqi->pq", a.x3, tau538)
        )
    
        tau540 = (
            einsum("jp,qija->pqia", a.x4, tau409)
        )
    
        tau541 = (
            einsum("iq,pqia->pqa", a.y3, tau540)
        )
    
        tau542 = (
            einsum("pr,pr,qr,pr->pq", tau101, tau73, tau74, tau79)
        )
    
        tau543 = (
            einsum("jp,pqji->pqi", a.x4, tau179)
        )
    
        tau544 = (
            einsum("qi,pqi->pq", tau126, tau543)
        )
    
        tau545 = (
            einsum("ip,pqi->pq", a.x3, tau338)
        )
    
        tau546 = (
            einsum("qr,qr,qr,pr->pq", tau78, tau79, tau80, tau84)
        )
    
        tau547 = (
            einsum("iq,pqia->pqa", a.y4, tau314)
        )
    
        tau548 = (
            einsum("jp,pqij->pqi", a.x4, tau440)
        )
    
        tau549 = (
            einsum("iq,pqi->pq", a.y3, tau548)
        )
    
        tau550 = (
            einsum("iq,pqi->pq", a.y4, tau294)
        )
    
        tau551 = (
            einsum("jp,pija->pia", a.x4, tau365)
        )
    
        tau552 = (
            einsum("ab,bp->pa", tau67, a.y2)
        )
    
        tau553 = (
            einsum("iq,pqia->pqa", a.y4, tau540)
        )
    
        tau554 = (
            einsum("dp,abdc->pabc", a.y2, tau457)
        )
    
        tau555 = (
            einsum("cp,qacb->pqab", a.x2, tau554)
        )
    
        tau556 = (
            einsum("bq,pqba->pqa", a.y1, tau555)
        )
    
        tau557 = (
            einsum("jq,pij->pqi", a.y4, tau37)
        )
    
        tau558 = (
            einsum("qi,pqi->pq", tau154, tau557)
        )
    
        tau559 = (
            einsum("ip,pqia->pqa", a.x4, tau82)
        )
    
        tau560 = (
            einsum("qi,pqia->pqa", tau126, tau156)
        )
    
        tau561 = (
            einsum("iq,pqia->pqa", a.y3, tau211)
        )
    
        tau562 = (
            einsum("ip,pqi->pq", a.x4, tau319)
        )
    
        tau563 = (
            einsum("ij,jp->pi", tau41, a.x4)
        )
    
        tau564 = (
            einsum("pi,iq->pq", tau563, a.y3)
        )
    
        tau565 = (
            einsum("cj,iacb->ijab", a.z1, tau23)
        )
    
        tau566 = (
            einsum("bp,ijba->pija", a.x2, tau565)
        )
    
        tau567 = (
            einsum("jp,pija->pia", a.x3, tau566)
        )
    
        tau568 = (
            einsum("ba,bp->pa", h.f.vv, a.y2)
        )
    
        tau569 = (
            einsum("kp,ijka->pija", a.x3, tau127)
        )
    
        tau570 = (
            einsum("jq,pjia->pqia", a.y3, tau569)
        )
    
        tau571 = (
            einsum("qi,pqia->pqa", tau154, tau570)
        )
    
        tau572 = (
            einsum("jq,pija->pqia", a.x3, tau172)
        )
    
        tau573 = (
            einsum("ip,pqia->pqa", a.x3, tau572)
        )
    
        tau574 = (
            einsum("jp,pija->pia", a.x3, tau123)
        )
    
        tau575 = (
            einsum("qr,qr,pr,pr->pq", tau74, tau78, tau79, tau84)
        )
    
        tau576 = (
            einsum("jq,pqij->pqi", a.y3, tau337)
        )
    
        tau577 = (
            einsum("ip,pqi->pq", a.x3, tau576)
        )
    
        tau578 = (
            einsum("qi,pqi->pq", tau154, tau543)
        )
    
        tau579 = (
            einsum("qi,ip->pq", tau427, a.x4)
        )
    
        tau580 = (
            einsum("pi,ip->p", tau158, a.x3)
        )
    
        tau581 = (
            einsum("ia,ip->pa", tau13, a.x4)
        )
    
        tau582 = (
            einsum("qi,pqi->pq", tau126, tau267)
        )
    
        tau583 = (
            einsum("jq,pji->pqi", a.y4, tau238)
        )
    
        tau584 = (
            einsum("qi,pqi->pq", tau126, tau583)
        )
    
        tau585 = (
            einsum("pi,iq->pq", tau563, a.y4)
        )
    
        tau586 = (
            einsum("pa,aq->pq", tau300, a.y1)
        )
    
        tau587 = (
            einsum("qi,pqia->pqa", tau126, tau570)
        )
    
        tau588 = (
            einsum("ia,ip->pa", tau17, a.x3)
        )
    
        tau589 = (
            einsum("qi,ip->pq", tau486, a.x3)
        )
    
        tau590 = (
            einsum("qi,pqia->pqa", tau154, tau423)
        )
    
        tau591 = (
            einsum("ip,pqia->pqa", a.x4, tau173)
        )
    
        tau592 = (
            einsum("pi,iq->pq", tau138, a.y4)
        )
    
        tau593 = (
            einsum("ip,pia->pa", a.x3, tau281)
        )
    
        tau594 = (
            einsum("jq,pij->pqi", a.y3, tau37)
        )
    
        tau595 = (
            einsum("qi,pqi->pq", tau126, tau594)
        )
    
        tau596 = (
            einsum("pr,qr,pr,ar->pqa", tau101, tau74, tau79, a.y1)
        )
    
        tau597 = (
            einsum("jp,qjia->pqia", a.x3, tau349)
        )
    
        tau598 = (
            einsum("qi,pqia->pqa", tau154, tau597)
        )
    
        tau599 = (
            einsum("qi,pqi->pq", tau154, tau594)
        )
    
        tau600 = (
            einsum("pr,qr,pr,ar->pqa", tau101, tau79, tau80, a.y1)
        )
    
        tau601 = (
            einsum("jq,pjia->pqia", a.y4, tau569)
        )
    
        tau602 = (
            einsum("qi,pqia->pqa", tau126, tau601)
        )
    
        tau603 = (
            einsum("qi,pqia->pqa", tau126, tau597)
        )
    
        tau604 = (
            einsum("pr,qr,qr,pr->pq", tau74, tau78, tau79, tau84)
        )
    
        tau605 = (
            einsum("ia,ip->pa", h.f.ov, a.x4)
        )
    
        tau606 = (
            einsum("qi,pqi->pq", tau154, tau583)
        )
    
        tau607 = (
            einsum("jp,pija->pia", a.y4, tau75)
        )
    
        tau608 = (
            einsum("ip,qia->pqa", a.x3, tau607)
        )
    
        tau609 = (
            einsum("pr,pr,pr,qr->pq", tau101, tau73, tau79, tau80)
        )
    
        tau610 = (
            einsum("qr,pr,pr,pr->pq", tau74, tau78, tau84, tau85)
        )
    
        tau611 = (
            einsum("qi,pqi->pq", tau154, tau176)
        )
    
        tau612 = (
            einsum("iq,pqi->pq", a.y4, tau548)
        )
    
        tau613 = (
            einsum("qi,pqia->pqa", tau154, tau326)
        )
    
        tau614 = (
            einsum("ip,pqi->pq", a.x4, tau488)
        )
    
        tau615 = (
            einsum("qi,pqi->pq", tau126, tau239)
        )
    
        tau616 = (
            einsum("ip,qia->pqa", a.x4, tau249)
        )
    
        tau617 = (
            einsum("kq,pijk->pqij", a.y4, tau436)
        )
    
        tau618 = (
            einsum("jp,pqij->pqi", a.x4, tau617)
        )
    
        tau619 = (
            einsum("iq,pqi->pq", a.y3, tau618)
        )
    
        tau620 = (
            einsum("qi,pqia->pqa", tau154, tau184)
        )
    
        tau621 = (
            einsum("jp,pija->pia", a.x4, tau285)
        )
    
        tau622 = (
            einsum("jp,pija->pia", a.x4, tau566)
        )
    
        tau623 = (
            einsum("ip,pqi->pq", a.x4, tau538)
        )
    
        tau624 = (
            einsum("ab,bp->pa", tau67, a.y1)
        )
    
        tau625 = (
            einsum("ip,qia->pqa", a.x4, tau607)
        )
    
        tau626 = (
            einsum("qi,pqia->pqa", tau154, tau529)
        )
    
        tau627 = (
            einsum("qi,pqi->pq", tau126, tau189)
        )
    
        tau628 = (
            einsum("ip,pqi->pq", a.x4, tau576)
        )
    
        tau629 = (
            einsum("iq,pqi->pq", a.y4, tau444)
        )
    
        tau630 = (
            einsum("ip,qia->pqa", a.x3, tau386)
        )
    
        tau631 = (
            einsum("iq,pqia->pqa", a.y4, tau501)
        )
    
        tau632 = (
            einsum("iq,pqia->pqa", a.y3, tau410)
        )
    
        tau633 = (
            einsum("iq,pqi->pq", a.y3, tau441)
        )
    
        tau634 = (
            einsum("qi,pqia->pqa", tau126, tau286)
        )
    
        tau635 = (
            einsum("qi,pqi->pq", tau126, tau557)
        )
    
        tau636 = (
            einsum("qi,ip->pq", tau321, a.x4)
        )
    
        tau637 = (
            einsum("qi,pi->pq", tau126, tau254)
        )
    
        tau638 = (
            einsum("qi,pqia->pqa", tau126, tau527)
        )
    
        tau639 = (
            einsum("qi,pqia->pqa", tau126, tau511)
        )
    
        tau640 = (
            einsum("ia,ip->pa", tau17, a.x4)
        )
    
        tau641 = (
            einsum("qi,pqia->pqa", tau126, tau518)
        )
    
        tau642 = (
            einsum("qi,pqia->pqa", tau154, tau601)
        )
    
        tau643 = (
            einsum("ip,qia->pqa", a.x4, tau450)
        )
    
        tau644 = (
            einsum("qi,pqi->pq", tau154, tau310)
        )
    
        tau645 = (
            einsum("iq,pqi->pq", a.y3, tau143)
        )
    
        tau646 = (
            einsum("bp,qiba->pqia", a.x1, tau140)
        )
    
        tau647 = (
            einsum("ai,pqja->pqij", a.t1, tau646)
        )
    
        tau648 = (
            einsum("jp,pqij->pqi", a.x4, tau647)
        )
    
        tau649 = (
            einsum("iq,pqi->pq", a.y4, tau648)
        )
    
        tau650 = (
            einsum("ai,pjka->pijk", a.t1, tau172)
        )
    
        tau651 = (
            einsum("kp,pijk->pij", a.x4, tau650)
        )
    
        tau652 = (
            einsum("jq,pji->pqi", a.y4, tau651)
        )
    
        tau653 = (
            einsum("qi,pqi->pq", tau154, tau652)
        )
    
        tau654 = (
            einsum("kq,pikj->pqij", a.y4, tau9)
        )
    
        tau655 = (
            einsum("jp,pqji->pqi", a.x4, tau654)
        )
    
        tau656 = (
            einsum("qi,pqi->pq", tau154, tau655)
        )
    
        tau657 = (
            einsum("jp,pqij->pqi", a.x3, tau647)
        )
    
        tau658 = (
            einsum("iq,pqi->pq", a.y3, tau657)
        )
    
        tau659 = (
            einsum("bp,qiba->pqia", a.x1, tau209)
        )
    
        tau660 = (
            einsum("ai,pqja->pqij", a.t1, tau659)
        )
    
        tau661 = (
            einsum("jp,pqij->pqi", a.x4, tau660)
        )
    
        tau662 = (
            einsum("iq,pqi->pq", a.y4, tau661)
        )
    
        tau663 = (
            einsum("ap,qija->pqij", a.x1, tau1)
        )
    
        tau664 = (
            einsum("jp,pqij->pqi", a.x4, tau663)
        )
    
        tau665 = (
            einsum("iq,pqi->pq", a.x4, tau664)
        )
    
        tau666 = (
            einsum("qr,pr,qr,pr->pq", tau101, tau79, tau80, tau84)
        )
    
        tau667 = (
            einsum("jp,pqij->pqi", a.x3, tau663)
        )
    
        tau668 = (
            einsum("iq,pqi->pq", a.x3, tau667)
        )
    
        tau669 = (
            einsum("jq,pij->pqi", a.y4, tau47)
        )
    
        tau670 = (
            einsum("qi,pqi->pq", tau154, tau669)
        )
    
        tau671 = (
            einsum("jp,qpij->pqi", a.x4, tau663)
        )
    
        tau672 = (
            einsum("ip,qpi->pq", a.x3, tau671)
        )
    
        tau673 = (
            einsum("ba,bp->pa", tau67, a.x1)
        )
    
        tau674 = (
            einsum("pa,aq->pq", tau673, a.y1)
        )
    
        tau675 = (
            einsum("ab,bp->pa", h.f.vv, a.x1)
        )
    
        tau676 = (
            einsum("pa,aq->pq", tau675, a.y2)
        )
    
        tau677 = (
            einsum("jp,pqij->pqi", a.x3, tau660)
        )
    
        tau678 = (
            einsum("iq,pqi->pq", a.y3, tau677)
        )
    
        tau679 = (
            einsum("bq,piab->pqia", a.y1, tau61)
        )
    
        tau680 = (
            einsum("ai,pqja->pqij", a.t1, tau679)
        )
    
        tau681 = (
            einsum("jp,pqij->pqi", a.x3, tau680)
        )
    
        tau682 = (
            einsum("iq,pqi->pq", a.y4, tau681)
        )
    
        tau683 = (
            einsum("qa,ap->pq", tau274, a.x1)
        )
    
        tau684 = (
            einsum("qr,qr,pr,pr->pq", tau101, tau74, tau79, tau84)
        )
    
        tau685 = (
            einsum("iq,pia->pqa", a.x4, tau228)
        )
    
        tau686 = (
            einsum("bp,ijab->pija", a.x1, tau51)
        )
    
        tau687 = (
            einsum("aq,pija->pqij", a.y2, tau686)
        )
    
        tau688 = (
            einsum("jq,pqij->pqi", a.y4, tau687)
        )
    
        tau689 = (
            einsum("ip,pqi->pq", a.x4, tau688)
        )
    
        tau690 = (
            einsum("jp,qpij->pqi", a.x3, tau663)
        )
    
        tau691 = (
            einsum("ip,qpi->pq", a.x3, tau690)
        )
    
        tau692 = (
            einsum("qj,pija->pqia", tau154, tau172)
        )
    
        tau693 = (
            einsum("qi,pqia->pqa", tau126, tau692)
        )
    
        tau694 = (
            einsum("ip,qpi->pq", a.x4, tau671)
        )
    
        tau695 = (
            einsum("bq,piab->pqia", a.y2, tau61)
        )
    
        tau696 = (
            einsum("ai,pqja->pqij", a.t1, tau695)
        )
    
        tau697 = (
            einsum("jp,pqij->pqi", a.x3, tau696)
        )
    
        tau698 = (
            einsum("iq,pqi->pq", a.y3, tau697)
        )
    
        tau699 = (
            einsum("jq,pji->pqi", a.y3, tau651)
        )
    
        tau700 = (
            einsum("qi,pqi->pq", tau126, tau699)
        )
    
        tau701 = (
            einsum("qr,pr,pr,qr->pq", tau101, tau80, tau84, tau85)
        )
    
        tau702 = (
            einsum("qi,pqia->pqa", tau154, tau659)
        )
    
        tau703 = (
            einsum("jq,pija->pqia", a.x3, tau123)
        )
    
        tau704 = (
            einsum("ip,pqia->pqa", a.x3, tau703)
        )
    
        tau705 = (
            einsum("jp,pqji->pqi", a.x3, tau654)
        )
    
        tau706 = (
            einsum("qi,pqi->pq", tau154, tau705)
        )
    
        tau707 = (
            einsum("qr,pr,pr,pr->pq", tau101, tau79, tau80, tau84)
        )
    
        tau708 = (
            einsum("ia,ap->pi", tau253, a.x1)
        )
    
        tau709 = (
            einsum("qi,pi->pq", tau154, tau708)
        )
    
        tau710 = (
            einsum("qi,pqi->pq", tau126, tau655)
        )
    
        tau711 = (
            einsum("qr,pr,qr,pr->pq", tau101, tau74, tau79, tau84)
        )
    
        tau712 = (
            einsum("ap,ai->pi", a.x1, a.z1)
        )
    
        tau713 = (
            einsum("pi,ip->p", tau712, a.x4)
        )
    
        tau714 = (
            einsum("cp,qacb->pqab", a.x1, tau554)
        )
    
        tau715 = (
            einsum("bq,pqba->pqa", a.y1, tau714)
        )
    
        tau716 = (
            einsum("ip,qpi->pq", a.x3, tau664)
        )
    
        tau717 = (
            einsum("kp,pijk->pij", a.x3, tau650)
        )
    
        tau718 = (
            einsum("jq,pji->pqi", a.y4, tau717)
        )
    
        tau719 = (
            einsum("qi,pqi->pq", tau126, tau718)
        )
    
        tau720 = (
            einsum("jq,pji->pqi", a.y4, tau68)
        )
    
        tau721 = (
            einsum("qi,pqi->pq", tau154, tau720)
        )
    
        tau722 = (
            einsum("pi,ip->p", tau712, a.x3)
        )
    
        tau723 = (
            einsum("ip,pqia->pqa", a.x4, tau124)
        )
    
        tau724 = (
            einsum("bp,ijba->pija", a.x1, tau565)
        )
    
        tau725 = (
            einsum("jp,pija->pia", a.x3, tau724)
        )
    
        tau726 = (
            einsum("qi,pi->pq", tau126, tau708)
        )
    
        tau727 = (
            einsum("qr,pr,pr,pr->pq", tau101, tau74, tau84, tau85)
        )
    
        tau728 = (
            einsum("qj,pija->pqia", tau126, tau172)
        )
    
        tau729 = (
            einsum("qi,pqia->pqa", tau154, tau728)
        )
    
        tau730 = (
            einsum("qr,pr,pr,qr->pq", tau101, tau74, tau84, tau85)
        )
    
        tau731 = (
            einsum("pr,pr,pr,qr->pq", tau101, tau79, tau80, tau84)
        )
    
        tau732 = (
            einsum("ap,qija->pqij", a.x1, tau106)
        )
    
        tau733 = (
            einsum("jq,pqij->pqi", a.y4, tau732)
        )
    
        tau734 = (
            einsum("ip,pqi->pq", a.x3, tau733)
        )
    
        tau735 = (
            einsum("bj,piab->pija", a.z1, tau61)
        )
    
        tau736 = (
            einsum("jp,pija->pia", a.x3, tau735)
        )
    
        tau737 = (
            einsum("ip,pqi->pq", a.x3, tau671)
        )
    
        tau738 = (
            einsum("ip,pqi->pq", a.x3, tau664)
        )
    
        tau739 = (
            einsum("jq,pij->pqi", a.y3, tau47)
        )
    
        tau740 = (
            einsum("qi,pqi->pq", tau154, tau739)
        )
    
        tau741 = (
            einsum("aq,pija->pqij", a.y1, tau686)
        )
    
        tau742 = (
            einsum("jq,pqij->pqi", a.y4, tau741)
        )
    
        tau743 = (
            einsum("ip,pqi->pq", a.x3, tau742)
        )
    
        tau744 = (
            einsum("iq,pqi->pq", a.y4, tau677)
        )
    
        tau745 = (
            einsum("qi,pqia->pqa", tau126, tau695)
        )
    
        tau746 = (
            einsum("jq,pij->pqi", a.y3, tau10)
        )
    
        tau747 = (
            einsum("qi,pqi->pq", tau126, tau746)
        )
    
        tau748 = (
            einsum("qi,pi->pq", tau126, tau15)
        )
    
        tau749 = (
            einsum("qi,pqi->pq", tau154, tau699)
        )
    
        tau750 = (
            einsum("iq,pqi->pq", a.y3, tau648)
        )
    
        tau751 = (
            einsum("qr,qr,pr,pr->pq", tau101, tau80, tau84, tau85)
        )
    
        tau752 = (
            einsum("jq,pqij->pqi", a.y3, tau732)
        )
    
        tau753 = (
            einsum("ip,pqi->pq", a.x4, tau752)
        )
    
        tau754 = (
            einsum("jq,pij->pqi", a.y4, tau10)
        )
    
        tau755 = (
            einsum("qi,pqi->pq", tau126, tau754)
        )
    
        tau756 = (
            einsum("iq,pia->pqa", a.x3, tau574)
        )
    
        tau757 = (
            einsum("jp,pqij->pqi", a.x4, tau696)
        )
    
        tau758 = (
            einsum("iq,pqi->pq", a.y4, tau757)
        )
    
        tau759 = (
            einsum("pr,pr,qr,pr->pq", tau101, tau74, tau84, tau85)
        )
    
        tau760 = (
            einsum("qi,pi->pq", tau126, tau45)
        )
    
        tau761 = (
            einsum("pr,pr,qr,qr->pq", tau101, tau79, tau80, tau84)
        )
    
        tau762 = (
            einsum("qi,pqia->pqa", tau126, tau646)
        )
    
        tau763 = (
            einsum("pr,pr,qr,qr->pq", tau101, tau74, tau84, tau85)
        )
    
        tau764 = (
            einsum("iq,pqi->pq", a.y3, tau681)
        )
    
        tau765 = (
            einsum("qi,pqi->pq", tau126, tau720)
        )
    
        tau766 = (
            einsum("jq,pji->pqi", a.y3, tau68)
        )
    
        tau767 = (
            einsum("qi,pqi->pq", tau126, tau766)
        )
    
        tau768 = (
            einsum("jq,pji->pqi", a.y3, tau3)
        )
    
        tau769 = (
            einsum("qi,pqi->pq", tau154, tau768)
        )
    
        tau770 = (
            einsum("kq,pikj->pqij", a.y3, tau9)
        )
    
        tau771 = (
            einsum("jp,pqji->pqi", a.x3, tau770)
        )
    
        tau772 = (
            einsum("qi,pqi->pq", tau126, tau771)
        )
    
        tau773 = (
            einsum("qi,pqi->pq", tau126, tau669)
        )
    
        tau774 = (
            einsum("jq,pji->pqi", a.y3, tau717)
        )
    
        tau775 = (
            einsum("qi,pqi->pq", tau154, tau774)
        )
    
        tau776 = (
            einsum("jp,pqij->pqi", a.x4, tau680)
        )
    
        tau777 = (
            einsum("iq,pqi->pq", a.y3, tau776)
        )
    
        tau778 = (
            einsum("pj,pija->pia", tau712, tau159)
        )
    
        tau779 = (
            einsum("aj,pij->pia", a.z1, tau47)
        )
    
        tau780 = (
            einsum("qi,pqi->pq", tau154, tau746)
        )
    
        tau781 = (
            einsum("qa,ap->pq", tau208, a.x1)
        )
    
        tau782 = (
            einsum("aj,pij->pia", a.z1, tau10)
        )
    
        tau783 = (
            einsum("qi,pqi->pq", tau154, tau766)
        )
    
        tau784 = (
            einsum("jq,pqij->pqi", a.y3, tau741)
        )
    
        tau785 = (
            einsum("ip,pqi->pq", a.x3, tau784)
        )
    
        tau786 = (
            einsum("ip,pqi->pq", a.x4, tau733)
        )
    
        tau787 = (
            einsum("jp,pqji->pqi", a.x4, tau770)
        )
    
        tau788 = (
            einsum("qi,pqi->pq", tau154, tau787)
        )
    
        tau789 = (
            einsum("iq,pqi->pq", a.y4, tau776)
        )
    
        tau790 = (
            einsum("dp,abcd->pabc", a.x1, tau457)
        )
    
        tau791 = (
            einsum("cq,pabc->pqab", a.y2, tau790)
        )
    
        tau792 = (
            einsum("bq,pqba->pqa", a.y1, tau791)
        )
    
        tau793 = (
            einsum("pi,qi->pq", tau15, tau154)
        )
    
        tau794 = (
            einsum("jp,pija->pia", a.x4, tau735)
        )
    
        tau795 = (
            einsum("qi,pqi->pq", tau126, tau739)
        )
    
        tau796 = (
            einsum("qi,pqi->pq", tau126, tau768)
        )
    
        tau797 = (
            einsum("iq,pqi->pq", a.y3, tau757)
        )
    
        tau798 = (
            einsum("ip,pqi->pq", a.x4, tau742)
        )
    
        tau799 = (
            einsum("iq,pqi->pq", a.y4, tau657)
        )
    
        tau800 = (
            einsum("qi,pi->pq", tau154, tau45)
        )
    
        tau801 = (
            einsum("ap,qija->pqij", a.x1, tau248)
        )
    
        tau802 = (
            einsum("jq,pqij->pqi", a.y3, tau801)
        )
    
        tau803 = (
            einsum("ip,pqi->pq", a.x4, tau802)
        )
    
        tau804 = (
            einsum("qi,pqi->pq", tau154, tau754)
        )
    
        tau805 = (
            einsum("iq,pqi->pq", a.y3, tau661)
        )
    
        tau806 = (
            einsum("ip,pqi->pq", a.x3, tau752)
        )
    
        tau807 = (
            einsum("qi,pqi->pq", tau126, tau652)
        )
    
        tau808 = (
            einsum("jp,pija->pia", a.x4, tau724)
        )
    
        tau809 = (
            einsum("pa,aq->pq", tau673, a.y2)
        )
    
        tau810 = (
            einsum("qi,pqi->pq", tau126, tau774)
        )
    
        tau811 = (
            einsum("pj,pija->pia", tau712, tau155)
        )
    
        tau812 = (
            einsum("jq,pqij->pqi", a.y3, tau687)
        )
    
        tau813 = (
            einsum("ip,pqi->pq", a.x3, tau812)
        )
    
        tau814 = (
            einsum("ip,pqi->pq", a.x4, tau784)
        )
    
        tau815 = (
            einsum("jq,pqij->pqi", a.y4, tau801)
        )
    
        tau816 = (
            einsum("ip,pqi->pq", a.x4, tau815)
        )
    
        tau817 = (
            einsum("qi,pqia->pqa", tau154, tau679)
        )
    
        tau818 = (
            einsum("ip,pqi->pq", a.x3, tau815)
        )
    
        tau819 = (
            einsum("ip,pqi->pq", a.x3, tau688)
        )
    
        tau820 = (
            einsum("pa,aq->pq", tau675, a.y1)
        )
    
        tau821 = (
            einsum("ip,pqi->pq", a.x3, tau802)
        )
    
        tau822 = (
            einsum("jq,pji->pqi", a.y4, tau3)
        )
    
        tau823 = (
            einsum("qi,pqi->pq", tau154, tau822)
        )
    
        tau824 = (
            einsum("qi,pqi->pq", tau126, tau787)
        )
    
        tau825 = (
            einsum("qi,pqi->pq", tau126, tau822)
        )
    
        tau826 = (
            einsum("iq,pqi->pq", a.y4, tau697)
        )
    
        tau827 = (
            einsum("qi,pqi->pq", tau154, tau771)
        )
    
        tau828 = (
            einsum("ip,pqi->pq", a.x4, tau812)
        )
    
        tau829 = (
            einsum("qi,pqi->pq", tau154, tau718)
        )
    
        tau830 = (
            einsum("qi,pqi->pq", tau126, tau705)
        )
    
        tau831 = (
            einsum("pq,pq,pq,iq->pi", tau74, tau78, tau84, a.y3)
        )
    
        tau832 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau73, tau79, a.y4)
        )
    
        tau833 = (
            einsum("kq,pkij->pqij", a.y4, tau33)
        )
    
        tau834 = (
            einsum("qj,pqji->pqi", tau154, tau833)
        )
    
        tau835 = (
            einsum("qj,pqji->pqi", tau154, tau654)
        )
    
        tau836 = (
            einsum("jq,pqji->pqi", a.y3, tau660)
        )
    
        tau837 = (
            einsum("qr,pr,pr,ir->pqi", tau101, tau80, tau84, a.y3)
        )
    
        tau838 = (
            einsum("qj,pqji->pqi", tau154, tau770)
        )
    
        tau839 = (
            einsum("pr,pr,qr,ir->pqi", tau73, tau74, tau78, a.y3)
        )
    
        tau840 = (
            einsum("qr,qr,pr,ir->pqi", tau78, tau80, tau84, a.y3)
        )
    
        tau841 = (
            einsum("pr,qr,qr,ir->pqi", tau101, tau73, tau79, a.y4)
        )
    
        tau842 = (
            einsum("qr,pr,pr,ir->pqi", tau101, tau84, tau85, a.y4)
        )
    
        tau843 = (
            einsum("qj,pqij->pqi", tau126, tau654)
        )
    
        tau844 = (
            einsum("jp,pji->pi", a.x3, tau42)
        )
    
        tau845 = (
            einsum("jq,pji->pqi", a.x3, tau42)
        )
    
        tau846 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau73, tau80, a.y3)
        )
    
        tau847 = (
            einsum("pr,qr,pr,ir->pqi", tau101, tau73, tau74, a.y3)
        )
    
        tau848 = (
            einsum("pr,qr,qr,ir->pqi", tau101, tau73, tau80, a.y3)
        )
    
        tau849 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau73, tau85, a.y4)
        )
    
        tau850 = (
            einsum("pr,qr,pr,ir->pqi", tau101, tau73, tau80, a.y3)
        )
    
        tau851 = (
            einsum("qr,pr,pr,ir->pqi", tau73, tau78, tau80, a.y3)
        )
    
        tau852 = (
            einsum("qj,pqij->pqi", tau126, tau179)
        )
    
        tau853 = (
            einsum("kq,pkij->pqij", a.y4, tau2)
        )
    
        tau854 = (
            einsum("qj,pqji->pqi", tau126, tau853)
        )
    
        tau855 = (
            einsum("qj,pqij->pqi", tau126, tau770)
        )
    
        tau856 = (
            einsum("pr,pr,qr,ir->pqi", tau78, tau84, tau85, a.y4)
        )
    
        tau857 = (
            einsum("kq,pkij->pqij", a.y3, tau33)
        )
    
        tau858 = (
            einsum("qj,pqji->pqi", tau126, tau857)
        )
    
        tau859 = (
            einsum("pr,qr,pr,ir->pqi", tau73, tau78, tau80, a.y3)
        )
    
        tau860 = (
            einsum("pq,pq,pq,iq->pi", tau78, tau84, tau85, a.y4)
        )
    
        tau861 = (
            einsum("ji,jp->pi", tau14, a.y3)
        )
    
        tau862 = (
            einsum("jq,pqji->pqi", a.y4, tau220)
        )
    
        tau863 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau80, tau84, a.y3)
        )
    
        tau864 = (
            einsum("qr,pr,qr,ir->pqi", tau78, tau84, tau85, a.y4)
        )
    
        tau865 = (
            einsum("ji,jp->pi", tau41, a.y3)
        )
    
        tau866 = (
            einsum("jq,pqji->pqi", a.y4, tau696)
        )
    
        tau867 = (
            einsum("qj,pij->pqi", tau154, tau42)
        )
    
        tau868 = (
            einsum("qi,pqi->pq", tau126, tau867)
        )
    
        tau869 = (
            einsum("pr,qr,pr,ir->pqi", tau73, tau78, tau85, a.y4)
        )
    
        tau870 = (
            einsum("jq,pqji->pqi", a.y4, tau660)
        )
    
        tau871 = (
            einsum("qj,pqji->pqi", tau126, tau179)
        )
    
        tau872 = (
            einsum("ap,pqia->pqi", a.x1, tau378)
        )
    
        tau873 = (
            einsum("qi,pqi->pq", tau126, tau872)
        )
    
        tau874 = (
            einsum("ap,pqia->pqi", a.x2, tau695)
        )
    
        tau875 = (
            einsum("qi,pqi->pq", tau126, tau874)
        )
    
        tau876 = (
            einsum("qj,pqij->pqi", tau154, tau179)
        )
    
        tau877 = (
            einsum("kq,pikj->pqij", a.y4, tau200)
        )
    
        tau878 = (
            einsum("jq,pqji->pqi", a.y3, tau877)
        )
    
        tau879 = (
            einsum("jq,pqji->pqi", a.y4, tau680)
        )
    
        tau880 = (
            einsum("qr,pr,pr,ir->pqi", tau73, tau74, tau78, a.y3)
        )
    
        tau881 = (
            einsum("qr,pr,pr,ir->pqi", tau78, tau80, tau84, a.y3)
        )
    
        tau882 = (
            einsum("pr,qr,pr,ir->pqi", tau78, tau80, tau84, a.y3)
        )
    
        tau883 = (
            einsum("qr,pr,pr,ir->pqi", tau101, tau74, tau84, a.y3)
        )
    
        tau884 = (
            einsum("jp,pqji->pqi", a.x3, tau110)
        )
    
        tau885 = (
            einsum("jq,pqji->pqi", a.y3, tau440)
        )
    
        tau886 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau74, tau84, a.y3)
        )
    
        tau887 = (
            einsum("pk,pikj->pij", tau158, tau9)
        )
    
        tau888 = (
            einsum("qj,pqij->pqi", tau154, tau654)
        )
    
        tau889 = (
            einsum("ap,pija->pij", a.x2, tau735)
        )
    
        tau890 = (
            einsum("qj,pqji->pqi", tau154, tau179)
        )
    
        tau891 = (
            einsum("jq,pqji->pqi", a.y4, tau454)
        )
    
        tau892 = (
            einsum("jp,qpji->pqi", a.x3, tau134)
        )
    
        tau893 = (
            einsum("qr,pr,pr,ir->pqi", tau101, tau79, tau84, a.y4)
        )
    
        tau894 = (
            einsum("qj,pqij->pqi", tau154, tau175)
        )
    
        tau895 = (
            einsum("kq,pkij->pqij", a.y3, tau237)
        )
    
        tau896 = (
            einsum("qj,pqji->pqi", tau126, tau895)
        )
    
        tau897 = (
            einsum("pr,qr,pr,ir->pqi", tau74, tau78, tau84, a.y3)
        )
    
        tau898 = (
            einsum("kq,pkij->pqij", a.y4, tau237)
        )
    
        tau899 = (
            einsum("qj,pqji->pqi", tau126, tau898)
        )
    
        tau900 = (
            einsum("jq,pqji->pqi", a.y4, tau142)
        )
    
        tau901 = (
            einsum("qj,pqij->pqi", tau126, tau175)
        )
    
        tau902 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau79, tau84, a.y4)
        )
    
        tau903 = (
            einsum("pr,qr,qr,ir->pqi", tau101, tau73, tau85, a.y4)
        )
    
        tau904 = (
            einsum("pr,qr,pr,ir->pqi", tau78, tau79, tau84, a.y4)
        )
    
        tau905 = (
            einsum("jp,pqji->pqi", a.x3, tau663)
        )
    
        tau906 = (
            einsum("jq,pqji->pqi", a.y4, tau647)
        )
    
        tau907 = (
            einsum("qr,qr,pr,ir->pqi", tau78, tau79, tau84, a.y4)
        )
    
        tau908 = (
            einsum("qr,pr,pr,ir->pqi", tau74, tau78, tau84, a.y3)
        )
    
        tau909 = (
            einsum("jp,pqji->pqi", a.x3, tau134)
        )
    
        tau910 = (
            einsum("qj,pqij->pqi", tau154, tau770)
        )
    
        tau911 = (
            einsum("pr,qr,qr,ir->pqi", tau101, tau73, tau74, a.y3)
        )
    
        tau912 = (
            einsum("pq,pq,pq,iq->pi", tau101, tau73, tau85, a.y4)
        )
    
        tau913 = (
            einsum("pr,qr,pr,ir->pqi", tau101, tau84, tau85, a.y4)
        )
    
        tau914 = (
            einsum("qr,pr,pr,ir->pqi", tau78, tau84, tau85, a.y4)
        )
    
        tau915 = (
            einsum("lp,ijlk->pijk", a.y3, tau435)
        )
    
        tau916 = (
            einsum("kp,qikj->pqij", a.x4, tau915)
        )
    
        tau917 = (
            einsum("jq,pqji->pqi", a.y4, tau916)
        )
    
        tau918 = (
            einsum("pq,pq,pq,iq->pi", tau78, tau79, tau84, a.y4)
        )
    
        tau919 = (
            einsum("jq,pqji->pqi", a.y4, tau344)
        )
    
        tau920 = (
            einsum("pq,pq,pq,iq->pi", tau101, tau73, tau79, a.y4)
        )
    
        tau921 = (
            einsum("kq,pkij->pqij", a.y4, tau200)
        )
    
        tau922 = (
            einsum("jq,pqji->pqi", a.y3, tau921)
        )
    
        tau923 = (
            einsum("ap,pqia->pqi", a.x1, tau219)
        )
    
        tau924 = (
            einsum("qi,pqi->pq", tau154, tau923)
        )
    
        tau925 = (
            einsum("jq,pqji->pqi", a.y4, tau440)
        )
    
        tau926 = (
            einsum("pr,qr,pr,ir->pqi", tau101, tau73, tau85, a.y4)
        )
    
        tau927 = (
            einsum("kq,pkij->pqij", a.y4, tau650)
        )
    
        tau928 = (
            einsum("qj,pqji->pqi", tau126, tau927)
        )
    
        tau929 = (
            einsum("qr,pr,pr,ir->pqi", tau78, tau79, tau84, a.y4)
        )
    
        tau930 = (
            einsum("qj,pij->pqi", tau126, tau42)
        )
    
        tau931 = (
            einsum("qi,pqi->pq", tau154, tau930)
        )
    
        tau932 = (
            einsum("jq,pqji->pqi", a.y3, tau220)
        )
    
        tau933 = (
            einsum("bp,pqab->pqa", a.x1, tau459)
        )
    
        tau934 = (
            einsum("aq,pqa->pq", a.y1, tau933)
        )
    
        tau935 = (
            einsum("bp,pqab->pqa", a.x2, tau791)
        )
    
        tau936 = (
            einsum("aq,pqa->pq", a.y1, tau935)
        )
    
        tau937 = (
            einsum("lp,ijlk->pijk", a.y4, tau435)
        )
    
        tau938 = (
            einsum("kp,qikj->pqij", a.x4, tau937)
        )
    
        tau939 = (
            einsum("jq,pqji->pqi", a.y3, tau938)
        )
    
        tau940 = (
            einsum("ji,jp->pi", tau60, a.y3)
        )
    
        tau941 = (
            einsum("qj,pqji->pqi", tau154, tau895)
        )
    
        tau942 = (
            einsum("kq,pkij->pqij", a.y3, tau2)
        )
    
        tau943 = (
            einsum("qj,pqji->pqi", tau126, tau942)
        )
    
        tau944 = (
            einsum("qj,pqji->pqi", tau154, tau175)
        )
    
        tau945 = (
            einsum("ji,jp->pi", tau41, a.y4)
        )
    
        tau946 = (
            einsum("qj,pqji->pqi", tau126, tau175)
        )
    
        tau947 = (
            einsum("ji,jp->pi", tau60, a.y4)
        )
    
        tau948 = (
            einsum("jp,qpji->pqi", a.x3, tau663)
        )
    
        tau949 = (
            einsum("jq,pqji->pqi", a.y3, tau344)
        )
    
        tau950 = (
            einsum("jq,pqji->pqi", a.y3, tau647)
        )
    
        tau951 = (
            einsum("pr,qr,pr,ir->pqi", tau101, tau73, tau79, a.y4)
        )
    
        tau952 = (
            einsum("ap,pqia->pqi", a.x2, tau679)
        )
    
        tau953 = (
            einsum("qi,pqi->pq", tau154, tau952)
        )
    
        tau954 = (
            einsum("ji,jp->pi", tau14, a.y4)
        )
    
        tau955 = (
            einsum("kq,pkij->pqij", a.y3, tau650)
        )
    
        tau956 = (
            einsum("qj,pqji->pqi", tau126, tau955)
        )
    
        tau957 = (
            einsum("qr,pr,pr,ir->pqi", tau73, tau78, tau79, a.y4)
        )
    
        tau958 = (
            einsum("qj,pqji->pqi", tau154, tau927)
        )
    
        tau959 = (
            einsum("pr,qr,pr,ir->pqi", tau73, tau78, tau79, a.y4)
        )
    
        tau960 = (
            einsum("qj,pqji->pqi", tau154, tau853)
        )
    
        tau961 = (
            einsum("jq,pqji->pqi", a.y3, tau142)
        )
    
        tau962 = (
            einsum("qj,pqji->pqi", tau126, tau654)
        )
    
        tau963 = (
            einsum("qr,pr,pr,ir->pqi", tau73, tau78, tau85, a.y4)
        )
    
        tau964 = (
            einsum("jq,pqji->pqi", a.x3, tau110)
        )
    
        tau965 = (
            einsum("pk,pikj->pij", tau712, tau36)
        )
    
        tau966 = (
            einsum("qj,pqji->pqi", tau126, tau833)
        )
    
        tau967 = (
            einsum("kp,pikj->pij", a.x4, tau33)
        )
    
        tau968 = (
            einsum("pq,pq,pq,iq->pi", tau101, tau73, tau80, a.y3)
        )
    
        tau969 = (
            einsum("ap,pija->pij", a.x1, tau365)
        )
    
        tau970 = (
            einsum("kp,pikj->pij", a.x4, tau2)
        )
    
        tau971 = (
            einsum("jq,pqji->pqi", a.y3, tau680)
        )
    
        tau972 = (
            einsum("pq,pq,pq,iq->pi", tau101, tau73, tau74, a.y3)
        )
    
        tau973 = (
            einsum("qr,qr,pr,ir->pqi", tau74, tau78, tau84, a.y3)
        )
    
        tau974 = (
            einsum("pr,pr,qr,ir->pqi", tau101, tau73, tau74, a.y3)
        )
    
        tau975 = (
            einsum("pq,pq,pq,iq->pi", tau78, tau80, tau84, a.y3)
        )
    
        tau976 = (
            einsum("jq,pqji->pqi", a.y3, tau532)
        )
    
        tau977 = (
            einsum("qj,pqji->pqi", tau154, tau942)
        )
    
        tau978 = (
            einsum("qj,pqji->pqi", tau154, tau955)
        )
    
        tau979 = (
            einsum("qj,pqji->pqi", tau154, tau857)
        )
    
        tau980 = (
            einsum("jq,pqji->pqi", a.y3, tau696)
        )
    
        tau981 = (
            einsum("qj,pqji->pqi", tau154, tau898)
        )
    
        tau982 = (
            einsum("qj,pqji->pqi", tau126, tau770)
        )
    
        tau983 = (
            einsum("jp,pqji->pqi", a.x4, tau110)
        )
    
        tau984 = (
            einsum("jp,qpji->pqi", a.x4, tau134)
        )
    
        tau985 = (
            einsum("kp,pikj->pij", a.x3, tau33)
        )
    
        tau986 = (
            einsum("jp,pji->pi", a.x3, tau20)
        )
    
        tau987 = (
            einsum("lp,ijlk->pijk", a.x3, tau199)
        )
    
        tau988 = (
            einsum("kq,pikj->pqij", a.y4, tau987)
        )
    
        tau989 = (
            einsum("jq,pqji->pqi", a.y3, tau988)
        )
    
        tau990 = (
            einsum("kp,qikj->pqij", a.x3, tau937)
        )
    
        tau991 = (
            einsum("jq,pqji->pqi", a.y3, tau990)
        )
    
        tau992 = (
            einsum("kp,pikj->pij", a.y4, tau392)
        )
    
        tau993 = (
            einsum("jp,qji->pqi", a.x3, tau992)
        )
    
        tau994 = (
            einsum("jq,pji->pqi", a.x3, tau20)
        )
    
        tau995 = (
            einsum("kp,qikj->pqij", a.x3, tau915)
        )
    
        tau996 = (
            einsum("jq,pqji->pqi", a.y4, tau995)
        )
    
        tau997 = (
            einsum("kq,pkij->pqij", a.y4, tau987)
        )
    
        tau998 = (
            einsum("jq,pqji->pqi", a.y3, tau997)
        )
    
        tau999 = (
            einsum("jp,pqji->pqi", a.x3, tau98)
        )
    
        tau1000 = (
            einsum("jq,pqji->pqi", a.x4, tau110)
        )
    
        tau1001 = (
            einsum("jp,pqji->pqi", a.x4, tau134)
        )
    
        tau1002 = (
            einsum("kp,pikj->pij", a.x3, tau2)
        )
    
        tau1003 = (
            einsum("jq,pqji->pqi", a.y4, tau437)
        )
    
        tau1004 = (
            einsum("jq,pji->pqi", a.x4, tau42)
        )
    
        tau1005 = (
            einsum("jp,qpji->pqi", a.x4, tau663)
        )
    
        tau1006 = (
            einsum("jp,pqji->pqi", a.x4, tau663)
        )
    
        tau1007 = (
            einsum("kp,pikj->pij", a.y3, tau417)
        )
    
        tau1008 = (
            einsum("jp,qji->pqi", a.x3, tau1007)
        )
    
        tau1009 = (
            einsum("jq,pqji->pqi", a.x3, tau98)
        )
    
        tau1010 = (
            einsum("jp,pji->pi", a.x4, tau42)
        )
    
        tau1011 = (
            einsum("jq,pqji->pqi", a.y3, tau617)
        )
    
        rz1 = (
            - 2 * einsum("pi,ap->ai", tau4, a.x2)
            - einsum("p,ap,ip->ai", tau6, a.x1, a.x4)
            + 4 * einsum("o,oai->ai", tau7, h.l.pvo)
            - 2 * einsum("pi,ap->ai", tau11, a.x2)
            - 2 * einsum("aj,ji->ai", a.t1, tau12)
            - 4 * einsum("aj,ij->ai", a.t1, tau14)
            + 4 * einsum("p,ap,ip->ai", tau16, a.x2, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau19, a.x1, a.x3)
            + einsum("pa,ip->ai", tau22, a.x3)
            - einsum("pa,ip->ai", tau26, a.x3)
            - 4 * einsum("aj,ji->ai", a.t1, tau28)
            + einsum("p,ap,ip->ai", tau29, a.x1, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau31, a.x2, a.x4)
            + 2 * einsum("ia->ai", h.f.ov.conj())
            + einsum("pi,ap->ai", tau35, a.x1)
            + einsum("pi,ap->ai", tau38, a.x1)
            - 2 * einsum("pi,ap->ai", tau40, a.x1)
            + 2 * einsum("aj,ij->ai", a.t1, tau41)
            - 2 * einsum("pa,ip->ai", tau44, a.x3)
            - einsum("p,ap,ip->ai", tau46, a.x2, a.x3)
            + einsum("pi,ap->ai", tau48, a.x2)
            - 2 * einsum("pi,ap->ai", tau50, a.x1)
            - 2 * einsum("bj,jiab->ai", a.t1, tau51)
            + 4 * einsum("bi,ab->ai", a.t1, tau52)
            + einsum("pa,ip->ai", tau54, a.x4)
            + 2 * einsum("pa,ip->ai", tau55, a.x4)
            - 2 * einsum("pa,ip->ai", tau57, a.x4)
            + 2 * einsum("p,ap,ip->ai", tau58, a.x2, a.x4)
            - 2 * einsum("p,ap,ip->ai", tau59, a.x2, a.x3)
            + 2 * einsum("aj,ij->ai", a.t1, tau60)
            + 2 * einsum("pa,ip->ai", tau63, a.x3)
            - einsum("pa,ip->ai", tau64, a.x4)
            + 2 * einsum("ab,bi->ai", h.f.vv, a.t1)
            + 4 * einsum("p,ap,ip->ai", tau66, a.x1, a.x3)
            - 2 * einsum("bi,ba->ai", a.t1, tau67)
            - 2 * einsum("ji,aj->ai", h.f.oo, a.t1)
            + einsum("pi,ap->ai", tau69, a.x2)
            + 2 * einsum("p,ap,ip->ai", tau70, a.x1, a.x3)
            + einsum("p,ap,ip->ai", tau71, a.x2, a.x3)
            - 2 * einsum("p,ap,ip->ai", tau72, a.x1, a.x4)
        )
    
        ry1 = (
            - einsum("pq,pq,pqa->ap", tau73, tau74, tau77)
            - einsum("qp,qpa->ap", tau81, tau83)
            - einsum("qp,qpa->ap", tau86, tau88)
            - einsum("qp,pqa->ap", tau90, tau91) / 4
            - einsum("pq,pq,pq,aq->ap", tau79, tau80, tau93, a.y1)
            - einsum("qp,pqa->ap", tau94, tau96)
            + einsum("qp,qa->ap", tau100, tau97) / 2
            + einsum("qp,qpa->ap", tau102, tau103) / 2
            + einsum("pq,pq,pq,aq->ap", tau104, tau79, tau80, a.y2) / 2
            - einsum("pq,pq,aq->ap", tau109, tau85, a.y1)
            - einsum("qp,qpa->ap", tau112, tau113) / 4
            + einsum("qp,qpa->ap", tau115, tau116) / 2
            - einsum("qp,pqa->ap", tau117, tau113)
            + einsum("pq,pq,aq->ap", tau121, tau79, a.y2) / 2
            + einsum("qp,pqa->ap", tau122, tau125) / 2
            - einsum("pq,pq,pqa->ap", tau74, tau78, tau130) / 2
            + einsum("qp,pqa->ap", tau131, tau133) / 2
            + 2 * einsum("pq,qpa->ap", tau136, tau137)
            - einsum("pq,pq,pq,aq->ap", tau139, tau73, tau80, a.y2) / 2
            - einsum("pq,pq,aq->ap", tau144, tau85, a.y1)
            + einsum("qp,pqa->ap", tau145, tau146) / 2
            + einsum("pq,pq,aq->ap", tau148, tau79, a.y2)
            - einsum("qp,qpa->ap", tau112, tau149) / 4
            - einsum("qa,pq->ap", tau150, tau152) / 4
            - einsum("qp,qpa->ap", tau153, tau146)
            - 2 * einsum("pq,pq,pqa->ap", tau73, tau85, tau157)
            + einsum("ip,pia->ap", a.x3, tau160)
            + einsum("qp,pqa->ap", tau94, tau161) / 2
            + einsum("qp,pqa->ap", tau162, tau164) / 2
            + einsum("qp,qpa->ap", tau166, tau167) / 2
            - einsum("qp,qpa->ap", tau169, tau170)
            - einsum("qp,qpa->ap", tau171, tau174) / 4
            - einsum("pq,pq,aq->ap", tau177, tau80, a.y2) / 2
            + einsum("ip,pia->ap", a.x4, tau178)
            + einsum("pq,pq,aq->ap", tau181, tau85, a.y2)
            - einsum("pq,qpa->ap", tau182, tau183) / 4
            - 2 * einsum("pq,pq,pqa->ap", tau78, tau80, tau185)
            + einsum("qp,qpa->ap", tau186, tau125) / 2
            + einsum("qp,qpa->ap", tau112, tau170) / 2
            + einsum("qp,qpa->ap", tau187, tau133) / 2
            - einsum("qp,pqa->ap", tau188, tau88)
            + einsum("pq,pq,aq->ap", tau190, tau85, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau192, tau73, tau79, a.y2) / 2
            + 2 * einsum("pq,pq,pqa->ap", tau73, tau85, tau195)
            - einsum("qp,qpa->ap", tau115, tau196)
            - einsum("qp,pqa->ap", tau197, tau198) / 4
            - einsum("pq,pq,aq->ap", tau203, tau73, a.y2) / 2
            + einsum("pq,pq,pq,aq->ap", tau205, tau73, tau80, a.y2) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau79, tau207)
            - einsum("qa,pq,pq,pq->ap", tau208, tau74, tau78, tau85)
            - einsum("pq,pq,pqa->ap", tau78, tau79, tau212)
            + 2 * einsum("p,pa->ap", tau70, tau213)
            + 2 * einsum("pq,pq,aq->ap", tau214, tau79, a.y1)
            - einsum("qp,pqa->ap", tau216, tau217) / 4
            - einsum("qa,pq,pq,pq->ap", tau218, tau73, tau74, tau85)
            - einsum("pq,pq,aq->ap", tau222, tau85, a.y2)
            - einsum("qp,qpa->ap", tau223, tau224) / 4
            - einsum("pq,pq,pq,aq->ap", tau225, tau73, tau85, a.y2)
            + einsum("qp,pqa->ap", tau226, tau133) / 2
            - einsum("qp,qpa->ap", tau227, tau229)
            + einsum("qa,pq,pq,pq->ap", tau230, tau74, tau78, tau85)
            + einsum("qp,qpa->ap", tau231, tau232) / 2
            + einsum("pq,pq,pq,aq->ap", tau234, tau74, tau78, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau236, tau78, tau80, a.y1)
            + einsum("pq,pq,aq->ap", tau240, tau80, a.y1)
            + einsum("pq,pq,aq->ap", tau242, tau74, a.y1)
            - 2 * einsum("pq,pq,pq,aq->ap", tau243, tau74, tau85, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau244, tau73, tau74, a.y2)
            + einsum("qp,pqa->ap", tau90, tau245) / 2
            + 2 * einsum("qp,qpa->ap", tau246, tau88)
            - einsum("qp,qpa->ap", tau247, tau96) / 4
            + einsum("pq,pq,pq,aq->ap", tau74, tau85, tau93, a.y1) / 2
            - einsum("pq,pq,pqa->ap", tau78, tau85, tau250)
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau252) / 2
            - einsum("pq,pq,pq,aq->ap", tau255, tau74, tau85, a.y1) / 2
            + einsum("qp,qpa->ap", tau256, tau161) / 2
            + 2 * einsum("pq,qpa->ap", tau257, tau259)
            + einsum("pq,pq,pq,aq->ap", tau261, tau74, tau78, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau262, tau73, tau85, a.y2)
            + einsum("qp,pqa->ap", tau263, tau264) / 2
            + einsum("qa,qp->ap", tau265, tau266) / 2
            - einsum("pq,pq,aq->ap", tau268, tau74, a.y1) / 2
            + einsum("qp,qa->ap", tau100, tau269) / 2
            - einsum("qp,qpa->ap", tau270, tau133) / 4
            + einsum("qp,pqa->ap", tau117, tau170) / 2
            + einsum("qa,pq->ap", tau271, tau272) / 2
            + einsum("pq,pq,pq,aq->ap", tau273, tau79, tau80, a.y2) / 2
            + 2 * einsum("pq,pq,pq,aq->ap", tau275, tau79, tau80, a.y1)
            + 2 * einsum("qp,qpa->ap", tau276, tau277)
            + einsum("qa,pq,pq,pq->ap", tau278, tau78, tau79, tau80)
            - einsum("pq,pq,pq,aq->ap", tau275, tau74, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau73, tau74, tau279)
            - einsum("qp,qpa->ap", tau280, tau282)
            - einsum("p,pa->ap", tau6, tau283)
            - 2 * einsum("pq,pq,pqa->ap", tau73, tau85, tau287)
            - einsum("pq,qpa->ap", tau288, tau289)
            + einsum("qa,qp->ap", tau290, tau291) / 2
            - einsum("qp,pqa->ap", tau117, tau149)
            - einsum("qp,qpa->ap", tau292, tau174) / 4
            + einsum("qp,qpa->ap", tau293, tau164) / 2
            - einsum("pq,pq,aq->ap", tau295, tau74, a.y2)
            - einsum("qp,qpa->ap", tau231, tau296) / 4
            + einsum("pq,pq,pq,aq->ap", tau297, tau73, tau80, a.y2)
            - 2 * einsum("pi,ai->ap", tau50, a.z1)
            - 2 * einsum("qa,pq,pq,pq->ap", tau298, tau73, tau74, tau85)
            - 2 * einsum("pq,pq,pq,aq->ap", tau299, tau79, tau80, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau301, tau79, tau80, a.y1)
            + einsum("qp,qpa->ap", tau302, tau146) / 2
            - einsum("pq,qpa->ap", tau303, tau304)
            - 2 * einsum("pi,pia->ap", tau306, tau87)
            + einsum("pq,qpa->ap", tau307, tau308) / 2
            - einsum("pq,pq,aq->ap", tau311, tau79, a.y2) / 2
            - einsum("pq,qa->ap", tau152, tau271) / 4
            - einsum("pq,pq,pq,aq->ap", tau273, tau74, tau85, a.y2)
            - einsum("qp,qpa->ap", tau312, tau103) / 4
            + 2 * einsum("pq,pq,pqa->ap", tau78, tau80, tau315)
            + einsum("pq,pq,aq->ap", tau318, tau74, a.y1) / 2
            + einsum("pq,pq,aq->ap", tau320, tau85, a.y1) / 2
            - einsum("pq,pq,pq,aq->ap", tau322, tau73, tau85, a.y2)
            + einsum("qp,qpa->ap", tau280, tau324) / 2
            - einsum("pq,pq,pq,aq->ap", tau205, tau78, tau80, a.y1)
            + einsum("pq,pq,pqa->ap", tau74, tau78, tau327)
            + 2 * einsum("qa,pq,pq,pq->ap", tau208, tau78, tau79, tau80)
            - einsum("pq,pq,pq,aq->ap", tau236, tau73, tau80, a.y2) / 2
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau328)
            - einsum("qp,pqa->ap", tau122, tau146) / 4
            + einsum("qa,pq,pq,pq->ap", tau329, tau74, tau78, tau85) / 2
            + einsum("qp,pqa->ap", tau330, tau331) / 2
            + 2 * einsum("qa,pq,pq,pq->ap", tau274, tau73, tau74, tau85)
            + einsum("pq,pq,pq,aq->ap", tau333, tau73, tau80, a.y2)
            + einsum("qp,pqa->ap", tau334, tau229) / 2
            - einsum("pq,pq,aq->ap", tau336, tau80, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau339, tau79, a.y2)
            - einsum("qp,qpa->ap", tau272, tau340)
            + einsum("qp,qpa->ap", tau270, tau164) / 2
            - einsum("qp,qpa->ap", tau166, tau116) / 4
            + einsum("qp,pqa->ap", tau216, tau341) / 2
            + einsum("qp,pqa->ap", tau342, tau113) / 2
            - einsum("qp,qpa->ap", tau307, tau232)
            + einsum("qp,qpa->ap", tau171, tau83) / 2
            - einsum("pq,pq,aq->ap", tau346, tau79, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau139, tau78, tau80, a.y1)
            + einsum("qp,qpa->ap", tau152, tau347) / 2
            + einsum("qp,pqa->ap", tau197, tau348) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau79, tau351) / 2
            + einsum("qa,pq,pq,pq->ap", tau352, tau78, tau79, tau80)
            - einsum("pq,pq,pqa->ap", tau78, tau80, tau354)
            - 2 * einsum("pq,pq,aq->ap", tau355, tau74, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau357, tau73, tau74, a.y2)
            + einsum("qp,qpa->ap", tau223, tau358) / 2
            + einsum("pi,ai->ap", tau35, a.z1)
            - einsum("qp,qpa->ap", tau359, tau83)
            - 2 * einsum("pq,pq,pq,aq->ap", tau360, tau78, tau79, a.y1)
            - einsum("pq,qpa->ap", tau362, tau363) / 4
            + einsum("qp,qpa->ap", tau364, tau91) / 2
            - einsum("ip,pia->ap", a.x4, tau366)
            + einsum("pq,pq,pqa->ap", tau73, tau74, tau367) / 2
            - einsum("qp,qpa->ap", tau257, tau369)
            - einsum("qp,pqa->ap", tau263, tau348)
            - einsum("qp,pqa->ap", tau370, tau103) / 4
            + einsum("pq,pq,pqa->ap", tau78, tau85, tau373) / 2
            + einsum("pq,pq,pqa->ap", tau73, tau80, tau374) / 2
            - 2 * einsum("pq,pq,pq,aq->ap", tau375, tau73, tau85, a.y2)
            - einsum("qp,qpa->ap", tau115, tau167)
            - einsum("qp,pqa->ap", tau90, tau341) / 4
            + einsum("pq,pq,pqa->ap", tau73, tau74, tau377) / 2
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau379)
            - einsum("qp,qa->ap", tau100, tau380) / 4
            - 2 * einsum("p,pa->ap", tau72, tau283)
            - einsum("pq,pq,pqa->ap", tau73, tau80, tau383) / 2
            - einsum("pq,qa->ap", tau272, tau384)
            - einsum("qp,qpa->ap", tau293, tau133) / 4
            - einsum("pa->ap", tau265)
            - einsum("qp,qpa->ap", tau385, tau125) / 4
            - einsum("pq,pq,pqa->ap", tau73, tau79, tau387)
            - einsum("qp,pqa->ap", tau330, tau358)
            - einsum("pq,pq,aq->ap", tau389, tau78, a.y1) / 2
            - einsum("qp,pqa->ap", tau390, tau146) / 4
            + einsum("pq,pq,aq->ap", tau395, tau73, a.y2)
            + einsum("qp,qpa->ap", tau397, tau217) / 2
            - einsum("qp,qpa->ap", tau398, tau324) / 4
            - einsum("qp,qpa->ap", tau102, tau399)
            + einsum("pq,qpa->ap", tau401, tau402) / 2
            - einsum("qp,qpa->ap", tau186, tau146)
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau403)
            - einsum("pq,pq,pqa->ap", tau78, tau79, tau404)
            + 2 * einsum("pq,pq,aq->ap", tau406, tau79, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau225, tau78, tau85, a.y1) / 2
            + einsum("pq,qpa->ap", tau280, tau183) / 2
            - einsum("qp,pqa->ap", tau216, tau245) / 4
            - einsum("qp,qpa->ap", tau256, tau96) / 4
            + einsum("qp,qpa->ap", tau407, tau96) / 2
            - einsum("qp,qpa->ap", tau246, tau229)
            - einsum("qp,qpa->ap", tau169, tau408)
            - einsum("pq,pq,pqa->ap", tau73, tau85, tau411)
            + einsum("qp,pqa->ap", tau412, tau164) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau379) / 2
            - einsum("pq,qpa->ap", tau413, tau414) / 4
            + einsum("pq,pq,pq,aq->ap", tau415, tau78, tau79, a.y1)
            - einsum("pq,pq,pqa->ap", tau73, tau80, tau416)
            + einsum("qp,qpa->ap", tau112, tau408) / 2
            + einsum("pq,pq,pq,aq->ap", tau299, tau74, tau85, a.y1)
            - einsum("pq,pq,aq->ap", tau420, tau73, a.y2) / 2
            - einsum("qp,qpa->ap", tau401, tau421) / 4
            - einsum("qp,qpa->ap", tau422, tau161) / 4
            + einsum("pq,pq,pqa->ap", tau74, tau78, tau424)
            + einsum("qp,pqa->ap", tau390, tau125) / 2
            + einsum("pq,pq,aq->ap", tau425, tau80, a.y2) / 2
            + einsum("qp,qpa->ap", tau153, tau125) / 2
            - einsum("qp,qpa->ap", tau231, tau426) / 4
            + einsum("qa,pq->ap", tau150, tau272) / 2
            + einsum("pq,pq,pq,aq->ap", tau428, tau73, tau80, a.y2) / 2
            - einsum("qp,qpa->ap", tau276, tau198)
            - einsum("qa,pq,pq,pq->ap", tau429, tau73, tau79, tau80) / 2
            + einsum("pq,pq,pq,aq->ap", tau430, tau74, tau85, a.y1) / 2
            + einsum("qp,qpa->ap", tau431, tau433) / 2
            + einsum("pq,pq,pqa->ap", tau78, tau80, tau434)
            + einsum("pq,qpa->ap", tau182, tau304) / 2
            - einsum("pq,pq,aq->ap", tau439, tau78, a.y1) / 2
            + einsum("pq,pq,aq->ap", tau442, tau85, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau403) / 2
            - 2 * einsum("pq,pq,aq->ap", tau443, tau79, a.y1)
            + einsum("qp,pqa->ap", tau330, tau224) / 2
            + 2 * einsum("qp,qpa->ap", tau136, tau289)
            + 2 * einsum("pa->ap", tau290)
            - einsum("pq,pq,aq->ap", tau445, tau80, a.y2)
            - einsum("qp,pqa->ap", tau446, tau96)
            + 2 * einsum("pq,pq,pqa->ap", tau78, tau80, tau448)
            + einsum("qp,qpa->ap", tau81, tau174) / 2
            + einsum("qp,qpa->ap", tau169, tau113) / 2
            + einsum("pq,pq,pq,aq->ap", tau243, tau79, tau80, a.y2)
            - einsum("qp,pqa->ap", tau449, tau399)
            - einsum("pq,pq,pqa->ap", tau78, tau80, tau451)
            + einsum("qp,qpa->ap", tau307, tau426) / 2
            + einsum("pq,pq,aq->ap", tau452, tau80, a.y2)
            - einsum("pq,qpa->ap", tau280, tau304)
            + einsum("pq,pq,aq->ap", tau456, tau78, a.y1)
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau460) / 2
            - einsum("pq,pq,aq->ap", tau461, tau85, a.y1) / 2
            - einsum("pq,pq,pq,aq->ap", tau192, tau78, tau79, a.y1)
            - einsum("qp,qpa->ap", tau362, tau402) / 4
            + einsum("pq,pq,aq->ap", tau463, tau74, a.y1)
            + 2 * einsum("p,pa->ap", tau464, tau465)
            - einsum("qp,pqa->ap", tau467, tau196)
            + einsum("qp,qpa->ap", tau468, tau224) / 2
            - einsum("qp,qpa->ap", tau469, tau277)
            + einsum("pq,qa->ap", tau152, tau470) / 2
            + einsum("qp,qpa->ap", tau272, tau471) / 2
            + einsum("pq,pq,pq,aq->ap", tau244, tau74, tau78, a.y1) / 2
            - einsum("qp,qpa->ap", tau364, tau245)
            + einsum("qp,pqa->ap", tau472, tau358) / 2
            + 2 * einsum("pq,pq,aq->ap", tau473, tau74, a.y2)
            + einsum("qa,qp->ap", tau290, tau474) / 2
            - 2 * einsum("ip,pia->ap", a.x3, tau475)
            - einsum("pq,pq,pq,aq->ap", tau476, tau79, tau80, a.y2)
            - einsum("qp,qa->ap", tau100, tau477) / 4
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau252)
            + einsum("qp,pqa->ap", tau478, tau103) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau79, tau480) / 2
            - 2 * einsum("pq,pq,aq->ap", tau481, tau74, a.y2)
            + einsum("qp,qpa->ap", tau397, tau245) / 2
            + einsum("pi,pia->ap", tau482, tau163)
            + einsum("pq,pq,aq->ap", tau420, tau78, a.y1)
            + einsum("qp,pqa->ap", tau216, tau91) / 2
            + einsum("pq,qpa->ap", tau231, tau483) / 2
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau485)
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau460)
            + einsum("pq,pq,pq,aq->ap", tau487, tau73, tau79, a.y2)
            + einsum("pq,pq,aq->ap", tau489, tau80, a.y1) / 2
            + einsum("qp,pqa->ap", tau467, tau116) / 2
            - einsum("qp,qpa->ap", tau182, tau324) / 4
            + einsum("qp,qpa->ap", tau288, tau490) / 2
            - einsum("qp,pqa->ap", tau472, tau331) / 4
            + einsum("qp,pqa->ap", tau197, tau277) / 2
            - einsum("qa,pq,pq,pq->ap", tau329, tau78, tau79, tau80)
            + einsum("pq,pq,pqa->ap", tau78, tau79, tau491)
            - einsum("qp,pqa->ap", tau492, tau229) / 4
            + einsum("pq,pq,aq->ap", tau494, tau79, a.y2)
            + einsum("qp,qpa->ap", tau364, tau341) / 2
            - einsum("pq,pq,pq,aq->ap", tau262, tau78, tau85, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau78, tau79, tau495)
            - einsum("qp,pqa->ap", tau496, tau161) / 4
            + 4 * einsum("p,pa->ap", tau464, tau497)
            - 2 * einsum("ip,pia->ap", a.x4, tau498)
            + einsum("pq,pq,aq->ap", tau500, tau74, a.y2)
            - einsum("pq,pq,pqa->ap", tau74, tau78, tau502)
            - einsum("pq,pq,pq,aq->ap", tau357, tau74, tau78, a.y1) / 2
            + einsum("qp,pqa->ap", tau449, tau103) / 2
            + einsum("qp,qpa->ap", tau468, tau331) / 2
            + einsum("pq,qa->ap", tau171, tau503) / 2
            + einsum("pq,pq,pq,aq->ap", tau322, tau78, tau85, a.y1) / 2
            + einsum("qp,qpa->ap", tau115, tau504) / 2
            + einsum("qp,qpa->ap", tau362, tau505) / 2
            - einsum("pq,pq,aq->ap", tau506, tau85, a.y2)
            - einsum("pq,pq,aq->ap", tau507, tau80, a.y1)
            - einsum("qp,qpa->ap", tau508, tau509) / 4
            + einsum("qp,qpa->ap", tau272, tau510) / 2
            + einsum("pq,pq,pqa->ap", tau73, tau79, tau512)
            - einsum("qp,qpa->ap", tau397, tau91) / 4
            + 2 * einsum("qp,qpa->ap", tau227, tau88)
            + einsum("qp,qpa->ap", tau223, tau513) / 2
            - einsum("qa,pq,pq,pq->ap", tau274, tau73, tau79, tau80)
            + einsum("qp,qpa->ap", tau385, tau146) / 2
            - einsum("pq,pq,aq->ap", tau514, tau74, a.y1)
            - einsum("qp,pqa->ap", tau515, tau125) / 4
            - 2 * einsum("pq,pq,aq->ap", tau516, tau79, a.y1)
            + 4 * einsum("p,pa->ap", tau66, tau213)
            - einsum("pq,qpa->ap", tau136, tau490)
            + einsum("qp,qpa->ap", tau359, tau174) / 2
            - einsum("qp,pqa->ap", tau517, tau504) / 4
            - einsum("qp,qpa->ap", tau223, tau331) / 4
            + einsum("pq,pq,pqa->ap", tau73, tau80, tau519)
            + einsum("pi,pia->ap", tau158, tau520)
            - einsum("pq,qpa->ap", tau231, tau521) / 4
            - einsum("pq,qa->ap", tau272, tau470)
            + einsum("pq,qpa->ap", tau303, tau183) / 2
            + 2 * einsum("pq,pq,pq,aq->ap", tau476, tau74, tau85, a.y2)
            + einsum("qp,qpa->ap", tau522, tau133) / 2
            - einsum("qp,pqa->ap", tau334, tau88)
            + einsum("pq,pq,pqa->ap", tau78, tau85, tau523) / 2
            + einsum("pq,pq,pq,aq->ap", tau375, tau78, tau85, a.y1)
            + einsum("qp,qpa->ap", tau524, tau229) / 2
            + einsum("pq,qpa->ap", tau431, tau369) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau80, tau525)
            - einsum("pq,pq,pqa->ap", tau74, tau78, tau526)
            - 2 * einsum("pq,pq,pq,aq->ap", tau333, tau78, tau80, a.y1)
            - einsum("qp,pqa->ap", tau517, tau116) / 4
            + einsum("qp,qpa->ap", tau182, tau282) / 2
            - einsum("pq,pq,pq,aq->ap", tau428, tau78, tau80, a.y1)
            + einsum("qp,pqa->ap", tau517, tau196) / 2
            + einsum("qa,pq,pq,pq->ap", tau429, tau73, tau74, tau85)
            + einsum("pq,pq,pqa->ap", tau73, tau74, tau528)
            - einsum("pq,pq,pqa->ap", tau78, tau79, tau530) / 2
            - einsum("pq,qpa->ap", tau307, tau531)
            + einsum("pq,pq,aq->ap", tau534, tau73, a.y2)
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau535)
            - einsum("pq,pq,aq->ap", tau536, tau74, a.y1) / 2
            - einsum("qp,qa->ap", tau537, tau97)
            + einsum("pq,pq,aq->ap", tau539, tau80, a.y2) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau74, tau541)
            - einsum("qp,pqa->ap", tau542, tau103) / 4
            + einsum("qp,pqa->ap", tau515, tau146) / 2
            - einsum("qa,qp->ap", tau265, tau474) / 4
            + einsum("qp,qpa->ap", tau166, tau196) / 2
            - einsum("pq,pq,aq->ap", tau544, tau79, a.y2) / 2
            - einsum("pq,qpa->ap", tau307, tau483)
            + einsum("pq,pq,aq->ap", tau545, tau85, a.y2) / 2
            + einsum("qa,qp->ap", tau265, tau546) / 2
            + einsum("qp,qpa->ap", tau469, tau198) / 2
            + 2 * einsum("qp,qpa->ap", tau276, tau348)
            - 2 * einsum("qa,pq,pq,pq->ap", tau230, tau78, tau79, tau80)
            + einsum("pq,qpa->ap", tau307, tau521) / 2
            - einsum("pq,pq,pqa->ap", tau78, tau85, tau547)
            + einsum("qp,pqa->ap", tau517, tau167) / 2
            + einsum("pq,pq,aq->ap", tau549, tau74, a.y1) / 2
            + einsum("pq,pq,aq->ap", tau550, tau79, a.y2) / 2
            + 2 * einsum("ip,pia->ap", a.x3, tau551)
            - einsum("qa,pq,pq,pq->ap", tau552, tau73, tau74, tau85)
            + einsum("pq,pq,pqa->ap", tau73, tau79, tau553) / 2
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau556) / 2
            + einsum("pq,pq,aq->ap", tau558, tau85, a.y1)
            + einsum("qp,pqa->ap", tau90, tau217) / 2
            - einsum("qp,qpa->ap", tau524, tau88)
            + einsum("pq,qpa->ap", tau508, tau559) / 2
            - 2 * einsum("pq,pq,pq,aq->ap", tau234, tau73, tau74, a.y2)
            + einsum("pq,pq,pqa->ap", tau78, tau85, tau560)
            + einsum("pq,pq,pqa->ap", tau74, tau78, tau561) / 2
            - einsum("pq,pq,pq,aq->ap", tau415, tau73, tau79, a.y2) / 2
            + einsum("qp,pqa->ap", tau188, tau229) / 2
            + einsum("qp,pqa->ap", tau492, tau88) / 2
            - einsum("qp,pqa->ap", tau226, tau164) / 4
            - einsum("pq,qpa->ap", tau398, tau183) / 4
            - einsum("pq,qpa->ap", tau231, tau308) / 4
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau328) / 2
            + einsum("qp,pqa->ap", tau263, tau198) / 2
            - einsum("pq,pq,aq->ap", tau562, tau79, a.y1)
            - einsum("qp,pqa->ap", tau342, tau408) / 4
            - einsum("pq,pq,pq,aq->ap", tau564, tau74, tau78, a.y1) / 2
            + 2 * einsum("ip,pia->ap", a.x4, tau567)
            - einsum("qp,qpa->ap", tau431, tau259)
            + einsum("qp,pqa->ap", tau446, tau161) / 2
            + einsum("qa,pq,pq,pq->ap", tau568, tau73, tau74, tau85)
            + einsum("qp,qpa->ap", tau169, tau149) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau80, tau571) / 2
            - einsum("pq,qpa->ap", tau431, tau573)
            - 2 * einsum("pi,pia->ap", tau482, tau574)
            - einsum("qp,pqa->ap", tau478, tau399)
            + einsum("qp,qpa->ap", tau575, tau103) / 2
            - einsum("pq,pq,aq->ap", tau577, tau80, a.y2)
            - einsum("qp,qpa->ap", tau522, tau164)
            + einsum("pq,qpa->ap", tau413, tau509) / 2
            + einsum("pq,pq,aq->ap", tau578, tau79, a.y1)
            - einsum("pq,pq,pq,aq->ap", tau579, tau73, tau74, a.y2)
            + einsum("qp,pqa->ap", tau496, tau96) / 2
            - 2 * einsum("p,pa->ap", tau580, tau581)
            - einsum("qp,pqa->ap", tau342, tau170) / 4
            - einsum("qp,qpa->ap", tau302, tau125) / 4
            - einsum("pq,qpa->ap", tau257, tau433)
            + einsum("pq,pq,aq->ap", tau582, tau74, a.y2)
            + einsum("pq,pq,aq->ap", tau584, tau85, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau585, tau78, tau79, a.y1)
            - einsum("pq,pq,pq,aq->ap", tau586, tau79, tau80, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau78, tau80, tau587)
            - 2 * einsum("p,pa->ap", tau464, tau588)
            - einsum("qp,pqa->ap", tau162, tau133)
            - einsum("qp,qpa->ap", tau276, tau264)
            + einsum("qp,qpa->ap", tau469, tau264) / 2
            - 2 * einsum("pq,pq,pq,aq->ap", tau589, tau73, tau85, a.y2)
            - einsum("qp,pqa->ap", tau472, tau224) / 4
            - einsum("pq,pq,pqa->ap", tau73, tau74, tau590) / 2
            + einsum("qp,qpa->ap", tau413, tau591) / 2
            + einsum("pq,pq,pq,aq->ap", tau360, tau73, tau79, a.y2)
            + 2 * einsum("qp,qpa->ap", tau257, tau573)
            + einsum("qp,qpa->ap", tau307, tau296) / 2
            - einsum("pq,pq,pq,aq->ap", tau592, tau78, tau85, a.y1) / 2
            + einsum("pq,qa->ap", tau359, tau593) / 2
            - einsum("qa,pq,pq,pq->ap", tau568, tau73, tau79, tau80) / 2
            + einsum("pq,pq,pq,aq->ap", tau564, tau73, tau74, a.y2)
            + einsum("pq,pq,aq->ap", tau595, tau80, a.y2)
            - einsum("qp,qpa->ap", tau272, tau347)
            + einsum("pq,pq,aq->ap", tau439, tau73, a.y2)
            + einsum("qp,pqa->ap", tau117, tau408) / 2
            + einsum("p,pa->ap", tau29, tau283)
            + einsum("qp,qpa->ap", tau86, tau229) / 2
            + einsum("qp,qpa->ap", tau303, tau324) / 2
            - einsum("pq,pq,pq,aq->ap", tau301, tau74, tau85, a.y1) / 2
            - einsum("qa,pq->ap", tau503, tau81)
            - einsum("qp,qpa->ap", tau136, tau596)
            - einsum("qp,pqa->ap", tau145, tau125) / 4
            - 2 * einsum("pq,pq,pq,aq->ap", tau487, tau78, tau79, a.y1)
            + einsum("qp,pqa->ap", tau542, tau399) / 2
            - einsum("qp,qa->ap", tau266, tau290)
            + einsum("pq,pq,pqa->ap", tau73, tau85, tau598)
            - einsum("pq,pq,aq->ap", tau599, tau80, a.y1) / 2
            + einsum("qp,qpa->ap", tau231, tau600) / 2
            - einsum("pq,pq,pq,aq->ap", tau104, tau74, tau85, a.y2)
            + einsum("qa,qp->ap", tau477, tau537) / 2
            - einsum("pq,qa->ap", tau171, tau593) / 4
            + einsum("pi,pia->ap", tau306, tau228)
            - einsum("pq,pq,pqa->ap", tau78, tau85, tau602) / 2
            + einsum("qp,qpa->ap", tau401, tau363) / 2
            - einsum("pq,pq,pqa->ap", tau78, tau85, tau603) / 2
            + einsum("qa,pq,pq,pq->ap", tau298, tau73, tau79, tau80)
            - einsum("qp,qpa->ap", tau604, tau103) / 4
            - einsum("pq,qa->ap", tau359, tau503)
            - 2 * einsum("pq,pq,pq,aq->ap", tau297, tau78, tau80, a.y1)
            - einsum("pq,pq,aq->ap", tau395, tau78, a.y1) / 2
            - einsum("qp,qpa->ap", tau575, tau399)
            - einsum("qp,pqa->ap", tau467, tau167)
            - einsum("qp,qpa->ap", tau364, tau217)
            - einsum("p,pa->ap", tau580, tau605)
            + einsum("pq,qa->ap", tau152, tau384) / 2
            - einsum("pq,pq,aq->ap", tau606, tau85, a.y1) / 2
            + einsum("qa,pq,pq,pq->ap", tau552, tau73, tau79, tau80) / 2
            - 2 * einsum("p,pa->ap", tau19, tau213)
            - einsum("pq,pq,pqa->ap", tau73, tau85, tau608)
            - einsum("qp,pqa->ap", tau609, tau161) / 4
            + einsum("qp,pqa->ap", tau472, tau513) / 2
            - einsum("qp,pqa->ap", tau412, tau133)
            + einsum("qp,pqa->ap", tau610, tau88) / 2
            + einsum("pq,pq,pq,aq->ap", tau586, tau74, tau85, a.y2)
            - einsum("qp,qpa->ap", tau303, tau282)
            - einsum("pq,qpa->ap", tau401, tau505) / 4
            + einsum("qa,pq->ap", tau593, tau81) / 2
            + einsum("pq,pq,aq->ap", tau611, tau80, a.y1)
            - einsum("pq,pq,aq->ap", tau612, tau79, a.y1)
            - einsum("qp,pqa->ap", tau610, tau229) / 4
            - einsum("qp,pqa->ap", tau131, tau164) / 4
            + einsum("pq,qpa->ap", tau231, tau531) / 2
            - einsum("pq,pq,pqa->ap", tau73, tau74, tau613) / 2
            - einsum("qp,qpa->ap", tau413, tau559) / 4
            + einsum("pq,qa->ap", tau292, tau503) / 2
            - einsum("pq,pq,aq->ap", tau614, tau74, a.y1)
            - einsum("pq,qpa->ap", tau508, tau591) / 4
            - einsum("pq,pq,aq->ap", tau615, tau80, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau78, tau79, tau616) / 2
            + einsum("qp,qpa->ap", tau152, tau340) / 2
            - einsum("qp,qpa->ap", tau307, tau600)
            - einsum("pq,pq,aq->ap", tau619, tau73, a.y2) / 2
            + einsum("pq,pq,pq,aq->ap", tau579, tau74, tau78, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau73, tau80, tau620)
            + einsum("qp,pqa->ap", tau370, tau399) / 2
            - 2 * einsum("pq,pq,pq,aq->ap", tau261, tau73, tau74, a.y2)
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau535) / 2
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau485) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau556)
            - 2 * einsum("pi,pia->ap", tau158, tau621)
            - einsum("qp,qpa->ap", tau187, tau164)
            - einsum("ip,pia->ap", a.x3, tau622)
            - einsum("pq,pq,aq->ap", tau623, tau74, a.y2)
            - einsum("qp,qpa->ap", tau166, tau504) / 4
            - einsum("qa,pq,pq,pq->ap", tau624, tau78, tau79, tau80)
            + einsum("qp,pqa->ap", tau342, tau149) / 2
            + einsum("pq,pq,pqa->ap", tau73, tau79, tau625) / 2
            + einsum("pq,pq,pq,aq->ap", tau592, tau73, tau85, a.y2)
            + einsum("qp,qpa->ap", tau292, tau83) / 2
            + einsum("pq,pq,aq->ap", tau203, tau78, a.y1)
            - einsum("qp,qpa->ap", tau468, tau358) / 4
            + einsum("pq,pq,pq,aq->ap", tau255, tau79, tau80, a.y1)
            + einsum("pq,pq,pqa->ap", tau73, tau79, tau626)
            - einsum("pq,pq,aq->ap", tau627, tau85, a.y2) / 2
            + 2 * einsum("pq,pq,aq->ap", tau628, tau74, a.y2)
            - einsum("qp,qpa->ap", tau288, tau137)
            + einsum("pq,pq,aq->ap", tau629, tau85, a.y2) / 2
            + 2 * einsum("pq,pq,pqa->ap", tau73, tau85, tau630)
            + einsum("pq,pq,pqa->ap", tau78, tau79, tau631) / 2
            + einsum("pq,pq,pq,aq->ap", tau589, tau78, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau73, tau80, tau632) / 2
            - einsum("pq,pq,aq->ap", tau633, tau80, a.y1)
            + einsum("qp,qpa->ap", tau312, tau399) / 2
            + einsum("qp,qpa->ap", tau422, tau96) / 2
            + einsum("qa,pq,pq,pq->ap", tau624, tau74, tau78, tau85) / 2
            - einsum("pq,pq,pq,aq->ap", tau585, tau73, tau79, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau78, tau85, tau634)
            + einsum("qp,qpa->ap", tau247, tau161) / 2
            + einsum("qp,pqa->ap", tau609, tau96) / 2
            - einsum("qa,pq,pq,pq->ap", tau278, tau74, tau78, tau85) / 2
            + einsum("qp,qpa->ap", tau604, tau399) / 2
            - einsum("pq,pq,aq->ap", tau635, tau85, a.y2) / 2
            + einsum("qp,pqa->ap", tau467, tau504) / 2
            - einsum("pq,pq,pq,aq->ap", tau636, tau78, tau79, a.y1)
            - einsum("qp,qpa->ap", tau407, tau161) / 4
            + einsum("pq,pq,pq,aq->ap", tau636, tau73, tau79, a.y2) / 2
            - einsum("pq,pq,pq,aq->ap", tau637, tau79, tau80, a.y2) / 2
            - einsum("qa,qp->ap", tau269, tau537)
            - einsum("pq,pq,pqa->ap", tau74, tau78, tau638) / 2
            - einsum("qp,qpa->ap", tau397, tau341) / 4
            - einsum("qp,pqa->ap", tau197, tau264) / 4
            - einsum("qp,qpa->ap", tau152, tau510) / 4
            - einsum("pq,qa->ap", tau292, tau593) / 4
            + einsum("pq,qpa->ap", tau288, tau596) / 2
            - einsum("pq,pq,pqa->ap", tau78, tau79, tau639) / 2
            + einsum("p,pa->ap", tau580, tau640)
            - einsum("qp,qpa->ap", tau152, tau471) / 4
            - 2 * einsum("pq,pq,pqa->ap", tau78, tau80, tau641)
            - einsum("qp,qpa->ap", tau469, tau348)
            + einsum("pq,pq,pqa->ap", tau73, tau85, tau642)
            - einsum("pq,pq,pq,aq->ap", tau430, tau79, tau80, a.y1)
            - einsum("qa,qp->ap", tau265, tau291) / 4
            + einsum("pq,pq,aq->ap", tau619, tau78, a.y1)
            + einsum("qa,qp->ap", tau380, tau537) / 2
            - einsum("pq,pq,aq->ap", tau534, tau78, a.y1) / 2
            + einsum("qa,pq,pq,pq->ap", tau218, tau73, tau79, tau80) / 2
            + einsum("pq,pq,pq,aq->ap", tau637, tau74, tau85, a.y2)
            + einsum("pq,pq,pqa->ap", tau74, tau78, tau643) / 2
            + einsum("pq,qpa->ap", tau398, tau304) / 2
            - einsum("qp,pqa->ap", tau263, tau277)
            + einsum("pq,pq,aq->ap", tau644, tau79, a.y1)
            - einsum("qp,pqa->ap", tau330, tau513)
            + einsum("pq,qpa->ap", tau362, tau421) / 2
            - einsum("qp,qpa->ap", tau468, tau513) / 4
            - einsum("pq,pq,aq->ap", tau456, tau73, a.y2) / 2
            + einsum("pq,pq,aq->ap", tau389, tau73, a.y2)
            + einsum("qp,qpa->ap", tau398, tau282) / 2
            - einsum("qa,pq,pq,pq->ap", tau352, tau74, tau78, tau85) / 2
            - einsum("qa,qp->ap", tau290, tau546)
            + einsum("pq,pq,aq->ap", tau645, tau80, a.y1) / 2
            + einsum("qp,qpa->ap", tau508, tau414) / 2
        )
    
        ry2 = (
            - einsum("pq,pq,aq->ap", tau101, tau203, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau649, tau79, a.y1)
            - einsum("qp,pqa->ap", tau263, tau296)
            - einsum("pq,pq,aq->ap", tau653, tau79, a.y1) / 2
            + einsum("pq,qpa->ap", tau86, tau183) / 2
            - einsum("pq,pq,aq->ap", tau656, tau79, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau658, tau80, a.y1)
            - einsum("pq,pq,pqa->ap", tau79, tau84, tau625)
            - einsum("qa,pq->ap", tau150, tau397) / 4
            + einsum("pq,pq,aq->ap", tau662, tau79, a.y2) / 2
            + einsum("qp,pqa->ap", tau517, tau402) / 2
            - einsum("pq,qpa->ap", tau665, tau170) / 4
            + einsum("pq,pq,pq,aq->ap", tau101, tau192, tau79, a.y1) / 2
            - einsum("qp,qpa->ap", tau666, tau88)
            + einsum("pq,qpa->ap", tau668, tau504) / 2
            + einsum("pq,pq,aq->ap", tau670, tau79, a.y1)
            + 2 * einsum("pq,qpa->ap", tau246, tau304)
            + einsum("pq,qpa->ap", tau102, tau414) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau85, tau603)
            - einsum("pq,qpa->ap", tau672, tau224)
            + einsum("pq,pq,pq,aq->ap", tau674, tau74, tau85, a.y2) / 2
            + einsum("qa,pq->ap", tau271, tau364) / 2
            - einsum("qp,pqa->ap", tau90, tau340) / 4
            - einsum("pq,pq,pq,aq->ap", tau676, tau79, tau80, a.y1) / 2
            + 4 * einsum("p,pa->ap", tau16, tau283)
            + 2 * einsum("pq,qa,pq,pq->ap", tau101, tau208, tau74, tau85)
            + einsum("qp,pqa->ap", tau446, tau183) / 2
            - einsum("pq,pq,pqa->ap", tau74, tau84, tau367)
            - einsum("pq,pq,pqa->ap", tau101, tau85, tau634) / 2
            - einsum("pq,qa->ap", tau364, tau470)
            + einsum("pq,qpa->ap", tau469, tau308) / 2
            + einsum("qa,pq,pq,pq->ap", tau552, tau74, tau84, tau85) / 2
            + einsum("pq,qa,pq,pq->ap", tau101, tau329, tau79, tau80) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau357, tau74, a.y1)
            + 2 * einsum("pq,pq,aq->ap", tau678, tau80, a.y2)
            + einsum("pq,pq,pqa->ap", tau74, tau84, tau541) / 2
            - einsum("pq,qpa->ap", tau187, tau573)
            + einsum("pq,pq,aq->ap", tau682, tau85, a.y2) / 2
            + 2 * einsum("pq,pq,pq,aq->ap", tau683, tau74, tau85, a.y1)
            + einsum("qp,pqa->ap", tau145, tau174) / 2
            - einsum("pq,pq,aq->ap", tau389, tau84, a.y2) / 2
            - einsum("pq,qa,pq,pq->ap", tau101, tau352, tau79, tau80) / 2
            + einsum("pq,qa->ap", tau397, tau470) / 2
            - einsum("qp,pqa->ap", tau117, tau137)
            - einsum("pq,qpa->ap", tau668, tau167)
            - einsum("qa,pq,pq,pq->ap", tau552, tau79, tau80, tau84)
            + 2 * einsum("pq,qpa->ap", tau684, tau399)
            - einsum("pq,qa->ap", tau364, tau384)
            - einsum("pq,qpa->ap", tau684, tau685)
            - einsum("pq,pq,pq,aq->ap", tau101, tau139, tau80, a.y1) / 2
            - einsum("qa,pq,pq,pq->ap", tau274, tau74, tau84, tau85)
            + einsum("pq,pq,aq->ap", tau689, tau79, a.y1) / 2
            - einsum("pq,qpa->ap", tau691, tau116)
            - einsum("qp,pqa->ap", tau472, tau308) / 4
            - 2 * einsum("p,pa->ap", tau31, tau283)
            - einsum("pq,pq,pqa->ap", tau101, tau79, tau616)
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau693)
            + einsum("qp,qpa->ap", tau694, tau170) / 2
            + einsum("pq,qpa->ap", tau522, tau433) / 2
            - 2 * einsum("pq,pq,pqa->ap", tau79, tau84, tau626)
            - einsum("qp,pqa->ap", tau122, tau174) / 4
            + einsum("pq,qpa->ap", tau166, tau505) / 2
            + einsum("pq,pq,aq->ap", tau698, tau80, a.y1) / 2
            - einsum("pq,qpa->ap", tau166, tau363) / 4
            - einsum("pq,pq,pqa->ap", tau84, tau85, tau195)
            + einsum("pq,qa->ap", tau385, tau503) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau205, tau80, a.y1) / 2
            - einsum("pq,qpa->ap", tau223, tau600) / 4
            + einsum("pq,pq,pq,aq->ap", tau244, tau74, tau84, a.y2) / 2
            - einsum("pq,pq,pq,aq->ap", tau101, tau585, tau79, a.y1) / 2
            + einsum("pq,qpa->ap", tau537, tau347) / 2
            - einsum("pq,qpa->ap", tau86, tau304)
            - einsum("pq,pq,pq,aq->ap", tau357, tau74, tau84, a.y2) / 2
            + einsum("pq,qa->ap", tau186, tau593) / 2
            - einsum("pq,pq,aq->ap", tau700, tau74, a.y2) / 2
            + einsum("pq,qpa->ap", tau701, tau133) / 2
            + einsum("pq,pq,aq->ap", tau456, tau84, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau101, tau236, tau80, a.y1) / 2
            - 2 * einsum("qa,pq,pq,pq->ap", tau298, tau79, tau80, tau84)
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau702)
            + einsum("pq,qpa->ap", tau115, tau402) / 2
            - einsum("pq,qa,pq,pq->ap", tau101, tau278, tau79, tau80) / 2
            - einsum("pq,qpa->ap", tau524, tau304)
            - einsum("pq,qpa->ap", tau701, tau704) / 4
            + einsum("pq,qpa->ap", tau169, tau490) / 2
            + einsum("pq,pq,pqa->ap", tau79, tau84, tau480)
            - einsum("qp,pqa->ap", tau94, tau304)
            - einsum("pq,pq,pqa->ap", tau74, tau84, tau377)
            + einsum("qp,pqa->ap", tau609, tau304) / 2
            - einsum("qa,pq->ap", tau271, tau397) / 4
            - einsum("pq,qpa->ap", tau468, tau296) / 4
            + einsum("pq,pq,aq->ap", tau706, tau85, a.y1)
            - einsum("pq,qpa->ap", tau291, tau83) / 4
            + einsum("qa,pq->ap", tau265, tau707) / 2
            + einsum("pq,pq,pq,aq->ap", tau709, tau74, tau85, a.y1)
            + einsum("pq,pq,aq->ap", tau710, tau79, a.y2)
            + einsum("pq,qpa->ap", tau711, tau103) / 2
            + einsum("pq,qpa->ap", tau270, tau573) / 2
            + einsum("qp,pqa->ap", tau117, tau490) / 2
            - einsum("p,pa->ap", tau713, tau465)
            + einsum("pq,pq,aq->ap", tau101, tau395, a.y1)
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau715)
            - einsum("qa,pq,pq,pq->ap", tau429, tau74, tau84, tau85) / 2
            - einsum("pq,qpa->ap", tau716, tau358) / 4
            - einsum("pq,pq,aq->ap", tau719, tau85, a.y2) / 2
            + einsum("pq,pq,aq->ap", tau721, tau79, a.y1)
            + einsum("pq,qpa->ap", tau716, tau224) / 2
            - einsum("pq,qpa->ap", tau166, tau402) / 4
            + 4 * einsum("p,pa->ap", tau722, tau581)
            + einsum("qp,pqa->ap", tau226, tau573) / 2
            + einsum("pq,qpa->ap", tau672, tau513) / 2
            + einsum("qp,pqa->ap", tau496, tau304) / 2
            + einsum("pq,qpa->ap", tau112, tau289) / 2
            + einsum("pq,pq,aq->ap", tau203, tau84, a.y2)
            + einsum("pq,qa->ap", tau153, tau593) / 2
            + einsum("pq,qpa->ap", tau187, tau433) / 2
            - einsum("qp,pqa->ap", tau478, tau414)
            + 2 * einsum("qp,qpa->ap", tau684, tau723)
            - einsum("pq,qpa->ap", tau276, tau521)
            - einsum("ip,pia->ap", a.x4, tau725)
            + einsum("pq,pq,pq,aq->ap", tau726, tau79, tau80, a.y2)
            + einsum("pq,qa,pq,pq->ap", tau101, tau352, tau74, tau85)
            - einsum("pq,pq,pqa->ap", tau74, tau84, tau279) / 2
            - einsum("qa,pq->ap", tau265, tau727) / 4
            + einsum("pq,pq,pqa->ap", tau101, tau80, tau641)
            - einsum("qp,qpa->ap", tau716, tau198) / 4
            + einsum("qp,pqa->ap", tau412, tau433) / 2
            - einsum("qp,pqa->ap", tau412, tau573)
            - einsum("qp,pqa->ap", tau263, tau426)
            - einsum("pq,pq,pqa->ap", tau84, tau85, tau642) / 2
            + einsum("qa,pq->ap", tau150, tau364) / 2
            + einsum("pq,qpa->ap", tau469, tau521) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau80, tau451) / 2
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau729) / 2
            + einsum("pq,qpa->ap", tau730, tau161) / 2
            + einsum("qp,qpa->ap", tau716, tau277) / 2
            + einsum("qp,qpa->ap", tau731, tau125) / 2
            - einsum("pq,qpa->ap", tau604, tau414) / 4
            + einsum("pq,qpa->ap", tau474, tau174) / 2
            + einsum("pq,qpa->ap", tau524, tau183) / 2
            - einsum("pq,pq,aq->ap", tau101, tau619, a.y1) / 2
            + einsum("pq,pq,pq,aq->ap", tau236, tau80, tau84, a.y2)
            + einsum("pq,qpa->ap", tau256, tau282) / 2
            + einsum("qp,pqa->ap", tau90, tau471) / 2
            - einsum("qp,pqa->ap", tau197, tau232) / 4
            - einsum("pq,qpa->ap", tau246, tau183)
            + 2 * einsum("pq,pq,aq->ap", tau734, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau84, tau85, tau411) / 2
            - einsum("pq,qpa->ap", tau694, tau113) / 4
            - einsum("qp,qpa->ap", tau711, tau399)
            - einsum("pq,qpa->ap", tau247, tau324) / 4
            - einsum("qp,pqa->ap", tau542, tau591) / 4
            - einsum("pq,qpa->ap", tau102, tau591)
            - einsum("qp,qpa->ap", tau672, tau277)
            + einsum("pq,pq,pqa->ap", tau80, tau84, tau383)
            + 2 * einsum("ip,pia->ap", a.x4, tau736)
            - einsum("qp,pqa->ap", tau446, tau304)
            - einsum("qa,pq->ap", tau477, tau737) / 4
            - einsum("pq,qpa->ap", tau468, tau426) / 4
            + einsum("qa,pq->ap", tau477, tau738) / 2
            - einsum("qp,pqa->ap", tau517, tau505) / 4
            - einsum("pq,qpa->ap", tau666, tau161)
            - einsum("pq,pq,aq->ap", tau74, tau740, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau79, tau84, tau351)
            + einsum("qp,pqa->ap", tau122, tau83) / 2
            + einsum("pq,qpa->ap", tau223, tau426) / 2
            + einsum("pq,pq,aq->ap", tau743, tau85, a.y2) / 2
            - einsum("pq,pq,aq->ap", tau744, tau85, a.y2)
            - einsum("qp,pqa->ap", tau330, tau531)
            + einsum("pq,qpa->ap", tau468, tau600) / 2
            + einsum("pq,qpa->ap", tau100, tau471) / 2
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau745)
            - 2 * einsum("pq,pq,aq->ap", tau747, tau80, a.y2)
            - einsum("qp,pqa->ap", tau467, tau402)
            + einsum("pq,pq,pq,aq->ap", tau74, tau748, tau85, a.y2)
            - 2 * einsum("pq,pq,pq,aq->ap", tau101, tau589, tau85, a.y1)
            - einsum("qp,pqa->ap", tau342, tau490) / 4
            - einsum("pq,qa->ap", tau738, tau97)
            - einsum("pq,pq,aq->ap", tau101, tau420, a.y1) / 2
            + einsum("qp,pqa->ap", tau449, tau591) / 2
            - einsum("qp,qpa->ap", tau727, tau146) / 4
            + einsum("pq,pq,aq->ap", tau101, tau389, a.y1)
            + einsum("pi,pia->ap", tau712, tau621)
            + einsum("pq,pq,aq->ap", tau74, tau749, a.y1)
            - 2 * einsum("pq,pq,pq,aq->ap", tau101, tau375, tau85, a.y1)
            + einsum("pq,qa,pq,pq->ap", tau101, tau230, tau79, tau80)
            - einsum("pq,pq,pqa->ap", tau101, tau79, tau491) / 2
            - einsum("pq,pq,pq,aq->ap", tau709, tau79, tau80, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau84, tau613)
            + einsum("pq,pq,pqa->ap", tau101, tau80, tau185)
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau693) / 2
            - einsum("qp,qpa->ap", tau684, tau103)
            - einsum("pq,qpa->ap", tau227, tau183)
            + einsum("pq,pq,aq->ap", tau74, tau750, a.y1) / 2
            - einsum("pq,qpa->ap", tau711, tau723)
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau745) / 2
            - einsum("qp,pqa->ap", tau467, tau363)
            + einsum("qp,qpa->ap", tau751, tau704) / 2
            - einsum("pq,qpa->ap", tau115, tau421)
            + einsum("pi,ai->ap", tau69, a.z1)
            + einsum("qp,qpa->ap", tau665, tau113) / 2
            - einsum("qp,pqa->ap", tau342, tau596) / 4
            + einsum("pq,pq,aq->ap", tau74, tau753, a.y1) / 2
            - einsum("pq,qpa->ap", tau672, tau331)
            - 2 * einsum("pq,pq,pq,aq->ap", tau487, tau79, tau84, a.y2)
            - einsum("pq,qpa->ap", tau546, tau174)
            + einsum("pq,qpa->ap", tau694, tau408) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau80, tau315)
            + einsum("qp,qpa->ap", tau727, tau125) / 2
            - einsum("pq,qa->ap", tau186, tau503)
            + einsum("pq,pq,aq->ap", tau755, tau85, a.y2)
            + einsum("pq,pq,aq->ap", tau101, tau439, a.y1)
            + einsum("qp,pqa->ap", tau610, tau324) / 2
            + einsum("qp,pqa->ap", tau117, tau596) / 2
            - einsum("pq,qpa->ap", tau716, tau513) / 4
            + einsum("qp,qpa->ap", tau701, tau756) / 2
            + einsum("qa,pq,pq,pq->ap", tau218, tau74, tau84, tau85) / 2
            - einsum("qp,qpa->ap", tau737, tau245) / 4
            - einsum("pa->ap", tau593)
            - 2 * einsum("pi,pia->ap", tau712, tau520)
            + 2 * einsum("pq,qpa->ap", tau276, tau483)
            + einsum("qp,pqa->ap", tau342, tau289) / 2
            + einsum("pq,pq,aq->ap", tau758, tau79, a.y1) / 2
            + einsum("pq,qpa->ap", tau291, tau174) / 2
            + einsum("pq,pq,pq,aq->ap", tau234, tau74, tau84, a.y2)
            - einsum("pq,qpa->ap", tau469, tau531)
            - einsum("pq,pq,pq,aq->ap", tau636, tau79, tau84, a.y2)
            + einsum("qp,pqa->ap", tau197, tau426) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau74, tau643)
            - einsum("pq,pq,pq,aq->ap", tau192, tau79, tau84, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau101, tau579, tau74, a.y1)
            + einsum("pq,qpa->ap", tau312, tau591) / 2
            - einsum("qp,pqa->ap", tau226, tau433) / 4
            - einsum("qp,pqa->ap", tau515, tau83) / 4
            - einsum("qp,qpa->ap", tau751, tau133) / 4
            + einsum("qp,qpa->ap", tau759, tau146) / 2
            - einsum("qa,pq,pq,pq->ap", tau568, tau74, tau84, tau85) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau636, tau79, a.y1) / 2
            - einsum("pq,pq,pq,aq->ap", tau760, tau79, tau80, a.y2)
            + einsum("qp,qpa->ap", tau761, tau88) / 2
            + einsum("p,pa->ap", tau71, tau213)
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau762)
            + einsum("pq,pq,pq,aq->ap", tau375, tau84, tau85, a.y2)
            + einsum("qp,qpa->ap", tau763, tau229) / 2
            - einsum("pq,pq,aq->ap", tau764, tau80, a.y2)
            - einsum("pq,qa->ap", tau385, tau593) / 4
            + einsum("pq,qpa->ap", tau100, tau510) / 2
            + einsum("pq,qpa->ap", tau575, tau414) / 2
            + einsum("qp,qpa->ap", tau668, tau116) / 2
            - 2 * einsum("p,pa->ap", tau722, tau640)
            + einsum("pq,qpa->ap", tau672, tau358) / 2
            - einsum("pq,pq,aq->ap", tau765, tau79, a.y2) / 2
            + einsum("pq,pq,aq->ap", tau74, tau767, a.y2)
            - einsum("pq,pq,pqa->ap", tau101, tau79, tau631)
            + einsum("pq,pq,pqa->ap", tau101, tau80, tau354) / 2
            - einsum("qp,pqa->ap", tau472, tau521) / 4
            - einsum("pq,pq,pqa->ap", tau80, tau84, tau519) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau85, tau523)
            + einsum("pq,qpa->ap", tau666, tau96) / 2
            + einsum("qa,pq->ap", tau269, tau737) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau592, tau85, a.y1)
            + einsum("pq,pq,aq->ap", tau769, tau80, a.y1)
            - 2 * einsum("pq,pq,pq,aq->ap", tau748, tau79, tau80, a.y2)
            + einsum("pq,pq,aq->ap", tau772, tau80, a.y2)
            - einsum("qp,pqa->ap", tau609, tau183) / 4
            + einsum("pq,qpa->ap", tau266, tau83) / 2
            + einsum("pq,pq,pq,aq->ap", tau74, tau760, tau85, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau79, tau404) / 2
            - einsum("qa,pq->ap", tau380, tau737) / 4
            - einsum("pq,qpa->ap", tau575, tau591)
            + einsum("pi,pia->ap", tau482, tau432)
            + einsum("pq,pq,pqa->ap", tau101, tau79, tau530)
            + einsum("pq,pq,pqa->ap", tau101, tau74, tau638)
            + einsum("qp,pqa->ap", tau216, tau347) / 2
            - einsum("pq,qpa->ap", tau100, tau347) / 4
            + einsum("pq,pq,pq,aq->ap", tau101, tau333, tau80, a.y1)
            - einsum("pq,pq,aq->ap", tau773, tau79, a.y2) / 2
            + einsum("qp,pqa->ap", tau197, tau296) / 2
            - einsum("pq,pq,aq->ap", tau775, tau80, a.y1) / 2
            + einsum("qp,qpa->ap", tau737, tau341) / 2
            + einsum("qp,qpa->ap", tau737, tau91) / 2
            - einsum("pq,pq,aq->ap", tau439, tau84, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau84, tau77) / 2
            - einsum("pq,qpa->ap", tau169, tau137)
            + einsum("pq,qpa->ap", tau407, tau324) / 2
            - einsum("qa,pq->ap", tau269, tau738)
            - einsum("qp,pqa->ap", tau145, tau83) / 4
            - einsum("pq,pq,pqa->ap", tau80, tau84, tau374)
            - 2 * einsum("pq,pq,pq,aq->ap", tau360, tau79, tau84, a.y2)
            + einsum("pq,qa->ap", tau302, tau503) / 2
            + einsum("pq,pq,aq->ap", tau74, tau777, a.y2) / 2
            - einsum("pq,pq,pqa->ap", tau84, tau85, tau630)
            - einsum("pq,pq,pqa->ap", tau74, tau84, tau528) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau85, tau373)
            + einsum("pq,pq,pq,aq->ap", tau261, tau74, tau84, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau139, tau80, tau84, a.y2)
            + einsum("pq,pq,pqa->ap", tau84, tau85, tau608) / 2
            - einsum("qp,pqa->ap", tau162, tau573)
            - einsum("qp,qpa->ap", tau665, tau408) / 4
            - einsum("pq,qpa->ap", tau270, tau433) / 4
            + einsum("qp,pqa->ap", tau263, tau232) / 2
            - 2 * einsum("ip,pia->ap", a.x3, tau778)
            - einsum("pq,pq,pq,aq->ap", tau564, tau74, tau84, a.y2) / 2
            + einsum("ip,pia->ap", a.x3, tau779)
            + einsum("qp,qpa->ap", tau738, tau245) / 2
            - einsum("qp,qpa->ap", tau672, tau348)
            + einsum("qp,qpa->ap", tau672, tau198) / 2
            - einsum("qp,pqa->ap", tau390, tau174) / 4
            - einsum("qp,pqa->ap", tau216, tau471) / 4
            - einsum("qp,qpa->ap", tau759, tau125)
            + einsum("pq,qpa->ap", tau115, tau363) / 2
            - einsum("pq,qpa->ap", tau537, tau510)
            - einsum("qp,pqa->ap", tau197, tau600) / 4
            - einsum("pq,qpa->ap", tau522, tau573)
            - einsum("pq,qa->ap", tau302, tau593) / 4
            + einsum("pq,qpa->ap", tau537, tau340) / 2
            + einsum("pq,pq,aq->ap", tau780, tau80, a.y1)
            - 2 * einsum("p,pa->ap", tau713, tau497)
            + einsum("qp,pqa->ap", tau472, tau483) / 2
            - einsum("pq,qpa->ap", tau100, tau340) / 4
            + 2 * einsum("pq,pq,pq,aq->ap", tau781, tau79, tau80, a.y2)
            - 2 * einsum("ip,pia->ap", a.x4, tau782)
            + einsum("pq,pq,pq,aq->ap", tau101, tau262, tau85, a.y1)
            + 2 * einsum("p,pa->ap", tau58, tau283)
            + einsum("pq,qpa->ap", tau761, tau161) / 2
            + 2 * einsum("pq,pq,pqa->ap", tau79, tau84, tau207)
            - einsum("pq,qa,pq,pq->ap", tau101, tau329, tau74, tau85)
            - einsum("qp,pqa->ap", tau334, tau324)
            - einsum("qa,pq,pq,pq->ap", tau218, tau79, tau80, tau84)
            + einsum("qp,pqa->ap", tau342, tau137) / 2
            + einsum("pq,qa->ap", tau737, tau97) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau85, tau560) / 2
            - einsum("pq,pq,aq->ap", tau74, tau783, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau785, tau80, a.y2)
            + einsum("qp,qpa->ap", tau711, tau685) / 2
            - einsum("pq,qpa->ap", tau169, tau289)
            + einsum("qp,pqa->ap", tau542, tau414) / 2
            + 2 * einsum("pq,qpa->ap", tau691, tau196)
            - 2 * einsum("pq,pq,pq,aq->ap", tau101, tau234, tau74, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau101, tau487, tau79, a.y1)
            - einsum("pq,pq,aq->ap", tau786, tau79, a.y1)
            - einsum("qp,pqa->ap", tau449, tau414)
            - 2 * einsum("pq,pq,pq,aq->ap", tau101, tau261, tau74, a.y1)
            + einsum("pq,qpa->ap", tau169, tau596) / 2
            + einsum("pq,pq,aq->ap", tau74, tau788, a.y1)
            + einsum("pq,pq,pqa->ap", tau101, tau74, tau130)
            + einsum("qa,pq->ap", tau290, tau727) / 2
            + einsum("qa,pq,pq,pq->ap", tau298, tau74, tau84, tau85)
            - einsum("pq,pq,aq->ap", tau789, tau79, a.y2)
            + einsum("qp,qpa->ap", tau707, tau146) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau297, tau80, a.y1)
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau792)
            - 2 * einsum("pq,pq,pq,aq->ap", tau74, tau793, tau85, a.y1)
            + einsum("qp,pqa->ap", tau330, tau521) / 2
            - einsum("qp,qpa->ap", tau691, tau504)
            - einsum("pq,qpa->ap", tau761, tau96) / 4
            + einsum("qp,pqa->ap", tau94, tau183) / 2
            + einsum("qp,pqa->ap", tau492, tau324) / 2
            + einsum("pq,qpa->ap", tau293, tau573) / 2
            - einsum("pq,qpa->ap", tau422, tau282) / 4
            - einsum("qp,qpa->ap", tau763, tau88)
            + einsum("pq,qpa->ap", tau751, tau164) / 2
            - einsum("pq,qpa->ap", tau112, tau490) / 4
            - einsum("qp,pqa->ap", tau610, tau282) / 4
            - einsum("qp,pqa->ap", tau216, tau510) / 4
            - einsum("ip,pia->ap", a.x3, tau794)
            - einsum("pq,pq,pqa->ap", tau101, tau79, tau495) / 2
            + einsum("qp,pqa->ap", tau90, tau510) / 2
            - einsum("pq,qpa->ap", tau266, tau174)
            - einsum("pq,qpa->ap", tau112, tau596) / 4
            - einsum("qp,qpa->ap", tau701, tau164) / 4
            + einsum("qp,qpa->ap", tau716, tau348) / 2
            - einsum("qp,pqa->ap", tau188, tau324)
            - einsum("qp,qpa->ap", tau707, tau125)
            + einsum("pq,pq,pq,aq->ap", tau579, tau74, tau84, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau85, tau602)
            + einsum("pq,pq,pqa->ap", tau84, tau85, tau157)
            + einsum("pq,pq,aq->ap", tau74, tau795, a.y2)
            + einsum("qa,pq->ap", tau290, tau731) / 2
            + einsum("pq,pq,pq,aq->ap", tau101, tau564, tau74, a.y1)
            + einsum("qp,pqa->ap", tau390, tau83) / 2
            - einsum("p,pa->ap", tau46, tau213)
            - 2 * einsum("pi,ai->ap", tau4, a.z1)
            + einsum("qa,pq->ap", tau384, tau397) / 2
            - 2 * einsum("pq,pq,aq->ap", tau796, tau80, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau74, tau781, tau85, a.y2)
            + einsum("pi,pia->ap", tau306, tau281)
            - einsum("pq,pq,aq->ap", tau74, tau797, a.y1)
            - einsum("pq,pq,pq,aq->ap", tau428, tau80, tau84, a.y2)
            + einsum("qp,pqa->ap", tau467, tau505) / 2
            - einsum("pq,qa,pq,pq->ap", tau101, tau624, tau74, tau85)
            + einsum("pq,pq,pqa->ap", tau74, tau84, tau590)
            - einsum("pq,pq,pqa->ap", tau84, tau85, tau598) / 2
            - 2 * einsum("pi,pia->ap", tau306, tau323)
            + 2 * einsum("pq,qpa->ap", tau276, tau531)
            + einsum("qp,qpa->ap", tau730, tau88) / 2
            + einsum("pq,pq,pqa->ap", tau80, tau84, tau571)
            - einsum("pq,pq,pqa->ap", tau79, tau84, tau553)
            + einsum("qp,qpa->ap", tau666, tau229) / 2
            - einsum("pq,pq,aq->ap", tau79, tau798, a.y2)
            + einsum("qp,pqa->ap", tau263, tau600) / 2
            - einsum("qp,qpa->ap", tau737, tau217) / 4
            + 2 * einsum("pq,pq,pqa->ap", tau101, tau74, tau502)
            - einsum("qa,pq->ap", tau290, tau759)
            - einsum("qp,qpa->ap", tau738, tau341)
            - einsum("pq,pq,pq,aq->ap", tau592, tau84, tau85, a.y2) / 2
            - 2 * einsum("pq,pq,pqa->ap", tau101, tau74, tau424)
            - einsum("qp,qpa->ap", tau668, tau196)
            + 2 * einsum("pq,pq,aq->ap", tau799, tau85, a.y1)
            + einsum("pq,pq,pq,aq->ap", tau79, tau80, tau800, a.y1) / 2
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau792) / 2
            + einsum("qp,pqa->ap", tau370, tau414) / 2
            - einsum("pq,pq,aq->ap", tau74, tau803, a.y2)
            - 2 * einsum("pq,pq,aq->ap", tau804, tau85, a.y1)
            - einsum("pq,pq,aq->ap", tau74, tau805, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau101, tau360, tau79, a.y1)
            - einsum("pq,pq,pq,aq->ap", tau101, tau322, tau85, a.y1)
            - einsum("pq,pq,aq->ap", tau80, tau806, a.y1)
            + einsum("qp,pqa->ap", tau472, tau531) / 2
            + einsum("pq,pq,aq->ap", tau79, tau807, a.y2)
            + einsum("pq,pq,pqa->ap", tau84, tau85, tau287)
            - einsum("pq,qpa->ap", tau763, tau161)
            + 2 * einsum("pq,pq,pqa->ap", tau101, tau74, tau526)
            + einsum("qp,pqa->ap", tau188, tau282) / 2
            - 2 * einsum("pq,pq,pqa->ap", tau79, tau84, tau512)
            + 2 * einsum("ip,pia->ap", a.x3, tau808)
            + einsum("pq,pq,pq,aq->ap", tau79, tau80, tau809, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau101, tau456, a.y1) / 2
            + einsum("pq,pq,pqa->ap", tau80, tau84, tau525) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau80, tau434) / 2
            - einsum("pq,pq,pqa->ap", tau80, tau84, tau632)
            - einsum("pq,pq,pqa->ap", tau101, tau80, tau448)
            + einsum("qp,pqa->ap", tau515, tau174) / 2
            - 2 * einsum("p,pa->ap", tau59, tau213)
            - einsum("pq,qpa->ap", tau469, tau483)
            - einsum("qp,pqa->ap", tau517, tau421) / 4
            - einsum("qp,pqa->ap", tau496, tau183) / 4
            - einsum("pq,pq,pq,aq->ap", tau683, tau79, tau80, a.y1)
            + einsum("pq,qpa->ap", tau223, tau296) / 2
            - einsum("pq,pq,pqa->ap", tau101, tau80, tau587) / 2
            + einsum("qa,pq->ap", tau265, tau759) / 2
            - einsum("pq,pq,pq,aq->ap", tau205, tau80, tau84, a.y2)
            - einsum("qp,pqa->ap", tau370, tau591) / 4
            + einsum("pq,qpa->ap", tau604, tau591) / 2
            + einsum("pq,pq,pq,aq->ap", tau589, tau84, tau85, a.y2)
            + einsum("qp,pqa->ap", tau517, tau363) / 2
            - einsum("pq,qpa->ap", tau115, tau505)
            - einsum("pq,qpa->ap", tau537, tau471)
            - einsum("qa,pq->ap", tau265, tau731) / 4
            + einsum("pq,qa,pq,pq->ap", tau101, tau278, tau74, tau85)
            - einsum("qp,qpa->ap", tau731, tau146) / 4
            + einsum("pq,pq,aq->ap", tau80, tau810, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau262, tau84, tau85, a.y2) / 2
            + einsum("ip,pia->ap", a.x4, tau811)
            + einsum("pq,qpa->ap", tau166, tau421) / 2
            + einsum("qp,pqa->ap", tau162, tau433) / 2
            - einsum("pq,qpa->ap", tau312, tau414) / 4
            - einsum("qp,qpa->ap", tau694, tau149) / 4
            - einsum("pq,qpa->ap", tau276, tau308)
            + einsum("pq,pq,aq->ap", tau80, tau813, a.y1) / 2
            - einsum("pq,qpa->ap", tau407, tau282) / 4
            + einsum("qp,pqa->ap", tau131, tau573) / 2
            - einsum("qp,pqa->ap", tau131, tau433) / 4
            - einsum("pq,pq,aq->ap", tau395, tau84, a.y2) / 2
            - einsum("qp,pqa->ap", tau492, tau282) / 4
            + einsum("pq,qpa->ap", tau112, tau137) / 2
            + einsum("pq,pq,aq->ap", tau74, tau814, a.y2) / 2
            - einsum("qp,pqa->ap", tau90, tau347) / 4
            + einsum("pq,pq,aq->ap", tau420, tau84, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau101, tau428, tau80, a.y1) / 2
            - einsum("pq,qpa->ap", tau256, tau324) / 4
            - einsum("pq,qa,pq,pq->ap", tau101, tau208, tau79, tau80)
            - einsum("pq,pq,pqa->ap", tau74, tau85, tau715) / 2
            - einsum("pq,qpa->ap", tau730, tau96) / 4
            + einsum("pq,pq,pq,aq->ap", tau322, tau84, tau85, a.y2) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau85, tau250) / 2
            - einsum("pq,qpa->ap", tau474, tau83) / 4
            - 2 * einsum("pq,pq,pq,aq->ap", tau297, tau80, tau84, a.y2)
            + einsum("pq,qpa->ap", tau468, tau232) / 2
            + einsum("qa,pq->ap", tau380, tau738) / 2
            + einsum("pq,pq,pq,aq->ap", tau225, tau84, tau85, a.y2) / 2
            - einsum("pq,pq,pq,aq->ap", tau101, tau415, tau79, a.y1) / 2
            + 2 * einsum("qa,pq,pq,pq->ap", tau274, tau79, tau80, tau84)
            + 2 * einsum("pq,pq,pqa->ap", tau79, tau84, tau387)
            + einsum("pq,pq,aq->ap", tau101, tau534, a.y1)
            + einsum("qp,pqa->ap", tau467, tau421) / 2
            + einsum("qp,pqa->ap", tau478, tau591) / 2
            + einsum("pq,pq,aq->ap", tau79, tau816, a.y2) / 2
            + einsum("pq,qa,pq,pq->ap", tau101, tau624, tau79, tau80) / 2
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau817) / 2
            - 2 * einsum("pq,qa,pq,pq->ap", tau101, tau230, tau74, tau85)
            + einsum("qp,qpa->ap", tau738, tau217) / 2
            - einsum("pq,pq,pq,aq->ap", tau674, tau79, tau80, a.y2)
            + 2 * einsum("pq,qpa->ap", tau227, tau304)
            + einsum("pq,qpa->ap", tau422, tau324) / 2
            + einsum("pq,qpa->ap", tau247, tau282) / 2
            - einsum("pq,qa->ap", tau153, tau503)
            - 2 * einsum("pi,pia->ap", tau482, tau258)
            - einsum("pq,pq,pqa->ap", tau101, tau74, tau561)
            + 2 * einsum("qp,qpa->ap", tau691, tau167)
            + einsum("qp,pqa->ap", tau330, tau308) / 2
            + einsum("p,pa->ap", tau713, tau588)
            - einsum("qp,qpa->ap", tau730, tau229) / 4
            - einsum("pq,pq,aq->ap", tau534, tau84, a.y2) / 2
            - einsum("pq,pq,pq,aq->ap", tau726, tau74, tau85, a.y2) / 2
            + einsum("qp,pqa->ap", tau334, tau282) / 2
            - einsum("pq,pq,aq->ap", tau818, tau85, a.y2)
            + einsum("qa,pq,pq,pq->ap", tau429, tau79, tau80, tau84)
            + einsum("qp,pqa->ap", tau216, tau340) / 2
            + einsum("pq,pq,pq,aq->ap", tau585, tau79, tau84, a.y2)
            - 2 * einsum("pq,pq,pq,aq->ap", tau333, tau80, tau84, a.y2)
            + einsum("pq,qpa->ap", tau546, tau83) / 2
            + einsum("pq,pq,pqa->ap", tau79, tau80, tau702) / 2
            + einsum("pq,qpa->ap", tau716, tau331) / 2
            - einsum("pq,pq,pq,aq->ap", tau74, tau800, tau85, a.y1)
            - einsum("pq,pq,pqa->ap", tau79, tau80, tau817)
            - einsum("qp,pqa->ap", tau330, tau483)
            - einsum("pq,pq,pq,aq->ap", tau74, tau809, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau762) / 2
            + 2 * einsum("pa->ap", tau503)
            - einsum("pq,pq,pq,aq->ap", tau101, tau225, tau85, a.y1)
            - einsum("qp,pqa->ap", tau117, tau289)
            - einsum("qp,qpa->ap", tau738, tau91)
            - einsum("pq,pq,aq->ap", tau819, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau74, tau85, tau729)
            + einsum("pq,pq,aq->ap", tau619, tau84, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau79, tau80, tau820, a.y2)
            - einsum("pq,pq,pqa->ap", tau80, tau84, tau620) / 2
            + einsum("pq,pq,pq,aq->ap", tau415, tau79, tau84, a.y2)
            - einsum("pq,pq,pq,aq->ap", tau74, tau820, tau85, a.y2) / 2
            + einsum("qp,qpa->ap", tau672, tau264) / 2
            + 2 * einsum("pq,pq,aq->ap", tau80, tau821, a.y2)
            + einsum("pq,pq,pq,aq->ap", tau676, tau74, tau85, a.y1)
            - einsum("qa,pq->ap", tau290, tau707)
            + 2 * einsum("p,pa->ap", tau722, tau605)
            + einsum("pq,qpa->ap", tau665, tau149) / 2
            + einsum("pq,pq,pq,aq->ap", tau79, tau793, tau80, a.y1)
            - einsum("pq,pq,pq,aq->ap", tau101, tau244, tau74, a.y1)
            - 2 * einsum("pq,pq,aq->ap", tau823, tau85, a.y1)
            - einsum("pq,pq,aq->ap", tau74, tau824, a.y2) / 2
            + einsum("pq,pq,aq->ap", tau825, tau85, a.y2)
            - 2 * einsum("pq,pq,pqa->ap", tau101, tau74, tau327)
            - einsum("pq,pq,aq->ap", tau826, tau85, a.y1)
            - einsum("pq,pq,aq->ap", tau80, tau827, a.y1) / 2
            - einsum("pq,pq,aq->ap", tau74, tau828, a.y1)
            - einsum("qp,qpa->ap", tau761, tau229) / 4
            + einsum("pq,pq,aq->ap", tau829, tau85, a.y1)
            + einsum("pq,pq,pqa->ap", tau101, tau79, tau639)
            + einsum("pq,qpa->ap", tau763, tau96) / 2
            + einsum("pq,pq,pqa->ap", tau101, tau85, tau547) / 2
            - einsum("qp,qpa->ap", tau716, tau264) / 4
            - einsum("pq,pq,aq->ap", tau830, tau85, a.y2) / 2
            - einsum("pq,qpa->ap", tau293, tau433) / 4
            - einsum("pq,qpa->ap", tau223, tau232) / 4
            + einsum("qa,pq,pq,pq->ap", tau568, tau79, tau80, tau84)
            + einsum("pq,pq,pqa->ap", tau101, tau79, tau212) / 2
            + einsum("pq,pq,pqa->ap", tau80, tau84, tau416) / 2
            - einsum("pq,qpa->ap", tau751, tau756) / 4
        )
    
        ry3 = (
            - einsum("pq,pq,pqi->ip", tau101, tau80, tau576)
            - einsum("pq,qi->ip", tau342, tau831) / 4
            + einsum("qp,pqi->ip", tau359, tau671) / 2
            + 2 * einsum("pq,pq,pq,iq->ip", tau78, tau781, tau80, a.y3)
            - einsum("qp,pqi->ip", tau117, tau832)
            + einsum("qp,pqi->ip", tau707, tau151) / 2
            - 2 * einsum("pj,pji->ip", tau712, tau49)
            - einsum("qp,qpi->ip", tau398, tau664) / 4
            + einsum("pq,pq,pqi->ip", tau84, tau85, tau834)
            - 2 * einsum("pq,pq,pqi->ip", tau73, tau85, tau835)
            + 2 * einsum("pq,pq,pqi->ip", tau78, tau80, tau836)
            + einsum("qp,pqi->ip", tau186, tau396) / 2
            + einsum("pq,qi,pq,pq->ip", tau101, tau260, tau73, tau80)
            - einsum("qp,qpi->ip", tau231, tau837) / 4
            + einsum("pq,pq,pq,iq->ip", tau261, tau78, tau84, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau78, tau820, tau85, a.y4) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau80, tau838)
            + einsum("qp,qpi->ip", tau665, tau839) / 2
            + einsum("pq,qpi->ip", tau100, tau840) / 2
            - einsum("qp,pqi->ip", tau169, tau841)
            + einsum("pq,pq,pq,iq->ip", tau255, tau80, tau84, a.y3)
            - einsum("qp,pqi->ip", tau152, tau842) / 4
            - einsum("pq,pq,pqi->ip", tau78, tau85, tau843) / 2
            - einsum("pq,pq,pqi->ip", tau80, tau84, tau317)
            - einsum("qp,pqi->ip", tau446, tau89)
            + einsum("pq,pq,pqi->ip", tau78, tau85, tau742) / 2
            - einsum("pq,qi->ip", tau412, tau844)
            + einsum("qp,pqi->ip", tau162, tau845) / 2
            - einsum("pq,pq,iq->ip", tau78, tau798, a.y3)
            - einsum("pq,qpi->ip", tau90, tau846) / 4
            - einsum("pq,pq,iq->ip", tau765, tau78, a.y3) / 2
            - einsum("pq,qpi->ip", tau169, tau847)
            + einsum("qp,pqi->ip", tau469, tau848) / 2
            - 2 * einsum("ji,pj->ip", tau305, tau43)
            - einsum("pq,pq,pq,iq->ip", tau275, tau84, tau85, a.y4)
            - 2 * einsum("qi,pq,pq,pq->ip", tau260, tau78, tau80, tau84)
            + einsum("pq,pq,pq,iq->ip", tau244, tau78, tau84, a.y4) / 2
            + einsum("pq,qpi->ip", tau90, tau849) / 2
            + einsum("pq,pq,pq,iq->ip", tau726, tau78, tau80, a.y3)
            - einsum("qp,pqi->ip", tau397, tau850) / 4
            + 2 * einsum("qp,pqi->ip", tau276, tau840)
            - einsum("pq,pq,pqi->ip", tau84, tau85, tau108)
            - einsum("pq,pq,iq->ip", tau649, tau73, a.y3)
            + einsum("pq,qpi->ip", tau422, tau111) / 2
            - einsum("qp,pqi->ip", tau738, tau851)
            - einsum("pq,pq,iq->ip", tau101, tau339, a.y3)
            + einsum("pq,qi->ip", tau188, tau21) / 2
            - einsum("pq,pq,iq->ip", tau656, tau73, a.y3) / 2
            - einsum("qp,pqi->ip", tau81, tau664)
            - einsum("pq,pq,pq,iq->ip", tau101, tau104, tau85, a.y4)
            + einsum("pq,pq,pqi->ip", tau101, tau85, tau852)
            - einsum("qp,qpi->ip", tau280, tau671)
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau419) / 2
            - einsum("pq,qi->ip", tau131, tau53) / 4
            - einsum("pq,pq,pq,iq->ip", tau101, tau273, tau85, a.y4)
            + einsum("pq,pq,pqi->ip", tau78, tau85, tau854)
            + einsum("pq,qpi->ip", tau537, tau848) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau80, tau812) / 2
            + einsum("pq,pq,iq->ip", tau578, tau84, a.y3)
            - einsum("pq,pq,pq,iq->ip", tau101, tau586, tau80, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau80, tau855)
            + einsum("qp,pqi->ip", tau330, tau856) / 2
            + einsum("pq,qi,pq,pq->ip", tau101, tau204, tau73, tau80) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau80, tau858)
            + einsum("qp,qpi->ip", tau672, tau859) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau357, tau73, a.y4)
            + einsum("pq,pq,pqi->ip", tau80, tau84, tau488) / 2
            - einsum("pq,qi->ip", tau197, tau860) / 4
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau80, tau861)
            + einsum("pq,pq,pq,iq->ip", tau415, tau78, tau84, a.y3)
            - einsum("pq,pq,iq->ip", tau78, tau824, a.y4) / 2
            - 2 * einsum("pq,pq,iq->ip", tau516, tau84, a.y3)
            - einsum("pq,pq,pqi->ip", tau101, tau85, tau862)
            - einsum("qp,pqi->ip", tau152, tau863) / 4
            - einsum("qp,pqi->ip", tau246, tau99)
            - einsum("pq,qpi->ip", tau515, tau215) / 4
            + einsum("qp,qpi->ip", tau231, tau842) / 2
            + einsum("pq,qpi->ip", tau537, tau864) / 2
            + einsum("pq,pq,pq,qi->ip", tau78, tau80, tau84, tau865)
            - einsum("pq,pq,pqi->ip", tau73, tau85, tau866)
            - einsum("pq,pq,pq,iq->ip", tau676, tau73, tau80, a.y3) / 2
            - einsum("qi,pq,pq,pq->ip", tau427, tau78, tau80, tau84)
            - einsum("p,pi->ip", tau713, tau5)
            + einsum("pq,pq,iq->ip", tau80, tau868, a.y3)
            + einsum("qp,pqi->ip", tau226, tau466) / 2
            + einsum("qp,qpi->ip", tau716, tau869) / 2
            - einsum("qp,pqi->ip", tau302, tau396) / 4
            - einsum("pq,pq,pq,iq->ip", tau101, tau637, tau80, a.y3) / 2
            - einsum("pq,qpi->ip", tau422, tau396) / 4
            - einsum("pq,pq,pq,iq->ip", tau192, tau78, tau84, a.y3)
            - einsum("qp,pqi->ip", tau86, tau168)
            - einsum("pq,pq,pqi->ip", tau78, tau85, tau870)
            + einsum("pj,pij->ip", tau158, tau47)
            - einsum("pq,pq,pqi->ip", tau101, tau85, tau871) / 2
            - einsum("pq,pq,pq,iq->ip", tau255, tau84, tau85, a.y4) / 2
            + einsum("pq,pq,iq->ip", tau101, tau121, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau80, tau873, a.y3)
            - einsum("pq,pq,pq,iq->ip", tau726, tau78, tau85, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau85, tau875, a.y4)
            - einsum("pq,pq,pqi->ip", tau84, tau85, tau876) / 2
            + einsum("pq,pq,iq->ip", tau767, tau78, a.y4)
            - einsum("qp,qpi->ip", tau666, tau135)
            + einsum("qp,pqi->ip", tau737, tau851) / 2
            + einsum("pq,pq,pq,iq->ip", tau73, tau80, tau809, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau773, tau78, a.y3) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau878) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau85, tau879) / 2
            - einsum("qp,pqi->ip", tau412, tau466)
            - 2 * einsum("p,pi->ip", tau464, tau30)
            + einsum("qp,pqi->ip", tau270, tau114) / 2
            - einsum("pq,pq,pq,iq->ip", tau709, tau73, tau80, a.y3) / 2
            - einsum("qp,qpi->ip", tau665, tau880) / 4
            + einsum("pq,qpi->ip", tau223, tau881) / 2
            - 2 * einsum("pq,qi,pq,pq->ip", tau101, tau486, tau73, tau85)
            - 2 * einsum("p,pi->ip", tau59, tau158)
            + einsum("pq,pq,pq,iq->ip", tau709, tau73, tau85, a.y4)
            - einsum("pq,pq,iq->ip", tau73, tau786, a.y3)
            - einsum("pq,qpi->ip", tau216, tau882) / 4
            - einsum("pq,pq,iq->ip", tau73, tau783, a.y4) / 2
            + einsum("qp,pqi->ip", tau302, tau111) / 2
            - einsum("pq,qpi->ip", tau136, tau883)
            + einsum("pq,qpi->ip", tau216, tau846) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau419)
            + einsum("qp,pqi->ip", tau524, tau99) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau394)
            + einsum("qp,pqi->ip", tau153, tau396) / 2
            + einsum("pq,qpi->ip", tau270, tau884) / 2
            - einsum("pq,pq,pqi->ip", tau80, tau84, tau885)
            - einsum("qp,pqi->ip", tau186, tau111)
            - einsum("qi,pq->ip", tau21, tau492) / 4
            - einsum("qp,pqi->ip", tau469, tau840)
            - einsum("qi,pq->ip", tau21, tau610) / 4
            + 2 * einsum("pq,qpi->ip", tau136, tau886)
            - 2 * einsum("jp,pji->ip", a.x4, tau887)
            - einsum("qp,qpi->ip", tau672, tau851)
            + einsum("pq,pq,pqi->ip", tau73, tau85, tau888)
            - einsum("pq,pq,iq->ip", tau78, tau803, a.y4)
            - 2 * einsum("p,pi->ip", tau713, tau65)
            - einsum("jp,pij->ip", a.x4, tau889)
            + einsum("pj,pji->ip", tau158, tau68)
            + einsum("pq,pq,pqi->ip", tau84, tau85, tau890)
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau891) / 2
            - einsum("pq,qpi->ip", tau122, tau89) / 4
            - einsum("pq,pq,iq->ip", tau536, tau84, a.y4) / 2
            + einsum("pq,qpi->ip", tau701, tau892) / 2
            - einsum("pq,qpi->ip", tau288, tau893)
            + einsum("pq,pq,pqi->ip", tau80, tau84, tau894)
            + einsum("qp,pqi->ip", tau472, tau849) / 2
            + einsum("pq,qpi->ip", tau187, tau165) / 2
            - einsum("pq,pq,pq,iq->ip", tau80, tau84, tau93, a.y3)
            + einsum("qp,qpi->ip", tau730, tau135) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau80, tau896) / 2
            - einsum("pq,pq,pq,iq->ip", tau73, tau809, tau85, a.y4)
            - einsum("qp,pqi->ip", tau759, tau135)
            + einsum("p,pi->ip", tau71, tau158)
            - einsum("pq,qpi->ip", tau112, tau897) / 4
            + einsum("pq,qpi->ip", tau145, tau89) / 2
            - einsum("qp,pqi->ip", tau131, tau845) / 4
            + einsum("qp,pqi->ip", tau330, tau846) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau360, tau78, tau84, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau760, tau78, tau85, a.y4) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau85, tau899)
            + einsum("pq,pq,pqi->ip", tau101, tau80, tau538) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau261, tau73, a.y4)
            - einsum("pq,pq,pqi->ip", tau84, tau85, tau900)
            + 2 * einsum("pq,pq,pq,iq->ip", tau275, tau80, tau84, a.y3)
            + einsum("pq,pq,iq->ip", tau463, tau84, a.y4)
            - einsum("pq,pq,pqi->ip", tau101, tau80, tau901) / 2
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau394) / 2
            + einsum("p,pi->ip", tau713, tau18)
            + einsum("qp,pqi->ip", tau759, tau151) / 2
            + einsum("qp,pqi->ip", tau364, tau850) / 2
            + einsum("pq,qpi->ip", tau288, tau902) / 2
            + einsum("qp,pqi->ip", tau292, tau664) / 2
            + einsum("pq,qpi->ip", tau100, tau903) / 2
            + einsum("qp,pqi->ip", tau117, tau904) / 2
            - einsum("pq,qpi->ip", tau431, tau905)
            - einsum("pq,pq,iq->ip", tau653, tau73, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau878)
            - einsum("pq,qi->ip", tau188, tau43)
            + 2 * einsum("pq,pq,pqi->ip", tau73, tau85, tau906)
            + 2 * einsum("pq,pq,pq,iq->ip", tau683, tau73, tau85, a.y4)
            - einsum("qp,pqi->ip", tau112, tau907) / 4
            + einsum("pj,pji->ip", tau482, tau42)
            - einsum("pq,pq,pqi->ip", tau78, tau80, tau784)
            + einsum("pq,pq,iq->ip", tau78, tau807, a.y3)
            + einsum("qp,qpi->ip", tau398, tau671) / 2
            + 4 * einsum("p,pi->ip", tau66, tau712)
            - einsum("pq,qpi->ip", tau522, tau884)
            + 2 * einsum("qp,pqi->ip", tau276, tau903)
            - einsum("qp,pqi->ip", tau226, tau845) / 4
            + einsum("qp,pqi->ip", tau342, tau908) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau273, tau80, a.y3) / 2
            - einsum("qp,qpi->ip", tau307, tau863)
            + einsum("qp,pqi->ip", tau112, tau841) / 2
            + einsum("pq,pq,iq->ip", tau242, tau84, a.y4)
            - einsum("pq,pq,iq->ip", tau101, tau346, a.y3)
            - einsum("pq,qpi->ip", tau701, tau909) / 4
            + einsum("pq,qpi->ip", tau694, tau880) / 2
            + einsum("pq,qpi->ip", tau390, tau215) / 2
            + einsum("pq,qpi->ip", tau515, tau89) / 2
            - einsum("pq,pq,iq->ip", tau700, tau78, a.y4) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau80, tau910) / 2
            - einsum("qp,pqi->ip", tau187, tau114)
            + einsum("pq,qi->ip", tau226, tau844) / 2
            - einsum("pq,qpi->ip", tau390, tau89) / 4
            + einsum("qp,pqi->ip", tau169, tau911) / 2
            + einsum("pq,qpi->ip", tau247, tau396) / 2
            + einsum("pq,qi->ip", tau197, tau912) / 2
            + einsum("pq,pq,pq,iq->ip", tau299, tau84, tau85, a.y4)
            - einsum("qp,qpi->ip", tau182, tau664) / 4
            - einsum("qp,pqi->ip", tau272, tau913)
            + einsum("pq,pq,iq->ip", tau78, tau795, a.y4)
            + einsum("pq,qpi->ip", tau90, tau882) / 2
            + einsum("pq,pq,iq->ip", tau101, tau494, a.y3)
            + 2 * einsum("pq,pq,iq->ip", tau101, tau473, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau73, tau800, tau85, a.y4)
            + 2 * einsum("qp,pqi->ip", tau246, tau168)
            - einsum("pq,qpi->ip", tau537, tau840)
            - einsum("pq,qpi->ip", tau223, tau914) / 4
            + einsum("qp,pqi->ip", tau385, tau111) / 2
            - einsum("qp,pqi->ip", tau472, tau846) / 4
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau917) / 2
            - einsum("qp,pqi->ip", tau609, tau215) / 4
            + einsum("pq,qi->ip", tau342, tau918) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau85, tau919) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau73, tau793, tau85, a.y4)
            + einsum("pq,qi->ip", tau263, tau860) / 2
            - einsum("pq,qi->ip", tau342, tau920) / 4
            + einsum("qp,pqi->ip", tau342, tau832) / 2
            - einsum("pq,pq,iq->ip", tau73, tau797, a.y4)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau922)
            + einsum("pq,qpi->ip", tau751, tau361) / 2
            - einsum("pq,pq,iq->ip", tau85, tau924, a.y4)
            + einsum("pq,pq,pqi->ip", tau84, tau85, tau925) / 2
            + einsum("qp,pqi->ip", tau397, tau926) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau104, tau80, a.y3) / 2
            + einsum("pq,pq,iq->ip", tau73, tau753, a.y4) / 2
            + einsum("qp,pqi->ip", tau272, tau863) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau243, tau85, a.y4)
            + einsum("qp,pqi->ip", tau472, tau882) / 2
            - einsum("qp,pqi->ip", tau272, tau837)
            + einsum("pq,qi->ip", tau131, tau844) / 2
            - einsum("pq,qi->ip", tau263, tau912)
            - einsum("pq,pq,pq,iq->ip", tau760, tau78, tau80, a.y3)
            - einsum("pq,pq,pq,iq->ip", tau78, tau781, tau85, a.y4)
            - einsum("pq,pq,pqi->ip", tau78, tau85, tau928) / 2
            - einsum("pq,qpi->ip", tau169, tau929)
            + einsum("qi,pq->ip", tau43, tau492) / 2
            - einsum("qp,pqi->ip", tau112, tau911) / 4
            + einsum("pq,qpi->ip", tau112, tau847) / 2
            + einsum("pq,pq,iq->ip", tau85, tau931, a.y4)
            - einsum("pq,pq,pqi->ip", tau78, tau85, tau815)
            + einsum("pq,pq,iq->ip", tau73, tau788, a.y4)
            - einsum("pq,pq,iq->ip", tau268, tau84, a.y4) / 2
            - einsum("qp,pqi->ip", tau731, tau151) / 4
            + 2 * einsum("qp,qpi->ip", tau136, tau893)
            - einsum("pq,qpi->ip", tau751, tau400) / 4
            + 2 * einsum("pq,qpi->ip", tau257, tau690)
            + einsum("qp,qpi->ip", tau231, tau863) / 2
            + einsum("qp,qpi->ip", tau303, tau664) / 2
            - einsum("pq,pq,pq,iq->ip", tau636, tau78, tau84, a.y3)
            + einsum("pq,pq,pqi->ip", tau101, tau80, tau932) / 2
            - einsum("pq,pq,iq->ip", tau85, tau934, a.y4) / 2
            - einsum("pq,qi,pq,pq->ip", tau101, tau321, tau73, tau85)
            + einsum("pq,qpi->ip", tau291, tau168) / 2
            + einsum("pq,pq,iq->ip", tau85, tau936, a.y4)
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau891)
            - einsum("qp,pqi->ip", tau162, tau466)
            - einsum("pq,qi->ip", tau162, tau844)
            - einsum("pq,pq,pqi->ip", tau73, tau80, tau752)
            + einsum("qp,pqi->ip", tau169, tau907) / 2
            + einsum("pq,qpi->ip", tau522, tau165) / 2
            + 2 * einsum("pi->ip", tau43)
            + einsum("qp,pqi->ip", tau469, tau864) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau939) / 2
            + einsum("qp,qpi->ip", tau307, tau837) / 2
            + einsum("pq,pq,pq,iq->ip", tau84, tau85, tau93, a.y4) / 2
            + einsum("qi,pq,pq,pq->ip", tau486, tau78, tau84, tau85)
            - 2 * einsum("pq,pq,pq,iq->ip", tau299, tau80, tau84, a.y3)
            + einsum("qi,pq,pq,pq->ip", tau191, tau78, tau84, tau85) / 2
            + einsum("pq,pq,pq,qi->ip", tau78, tau80, tau84, tau940)
            + einsum("pq,pq,pqi->ip", tau80, tau84, tau941)
            - einsum("pq,pq,iq->ip", tau101, tau623, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau101, tau579, tau73, a.y4)
            - einsum("pq,pq,iq->ip", tau562, tau84, a.y3)
            + einsum("pq,qpi->ip", tau474, tau168) / 2
            - 2 * einsum("pq,pq,pqi->ip", tau78, tau80, tau943)
            - einsum("pq,pq,pqi->ip", tau80, tau84, tau944) / 2
            - einsum("pi->ip", tau21)
            + einsum("pq,qi->ip", tau162, tau53) / 2
            - einsum("pq,qi,pq,pq->ip", tau101, tau191, tau73, tau85)
            + 2 * einsum("jp,pji->ip", a.x4, tau889)
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau85, tau945)
            + einsum("pq,pq,pq,iq->ip", tau101, tau487, tau73, a.y3)
            - einsum("pq,pq,pqi->ip", tau73, tau85, tau688)
            + einsum("pq,pq,iq->ip", tau670, tau73, a.y3)
            + einsum("pq,qpi->ip", tau546, tau99) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau636, tau73, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau80, tau946)
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau85, tau947)
            + einsum("pq,qpi->ip", tau293, tau884) / 2
            + einsum("pq,pq,iq->ip", tau721, tau73, a.y3)
            + einsum("pq,qpi->ip", tau431, tau948) / 2
            + 4 * einsum("p,pi->ip", tau464, tau15)
            + einsum("qp,pqi->ip", tau293, tau114) / 2
            + einsum("qp,pqi->ip", tau171, tau664) / 2
            - einsum("pq,qpi->ip", tau100, tau864) / 4
            - einsum("qp,pqi->ip", tau364, tau926)
            - 2 * einsum("pq,pq,iq->ip", tau101, tau355, a.y4)
            + 2 * einsum("qp,qpi->ip", tau257, tau905)
            - einsum("pq,pq,iq->ip", tau78, tau805, a.y4)
            - einsum("pq,pq,pqi->ip", tau101, tau80, tau949)
            + einsum("pq,pq,iq->ip", tau78, tau814, a.y4) / 2
            - einsum("qp,pqi->ip", tau385, tau396) / 4
            + einsum("pq,qpi->ip", tau256, tau396) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau80, tau950)
            + einsum("qi,pq->ip", tau43, tau610) / 2
            + einsum("pq,pq,iq->ip", tau78, tau816, a.y3) / 2
            + einsum("qp,pqi->ip", tau446, tau215) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau637, tau85, a.y4)
            + einsum("pj,ji->ip", tau21, tau305)
            - einsum("qp,pqi->ip", tau342, tau904) / 4
            - einsum("qp,qpi->ip", tau307, tau842)
            + einsum("pq,pq,pq,iq->ip", tau73, tau793, tau80, a.y3)
            + einsum("pq,qpi->ip", tau223, tau926) / 2
            - einsum("qp,pqi->ip", tau330, tau849)
            - 2 * einsum("pj,pij->ip", tau712, tau39)
            + einsum("pq,pq,iq->ip", tau777, tau78, a.y4) / 2
            + einsum("pq,pq,iq->ip", tau101, tau582, a.y4)
            - einsum("p,pi->ip", tau46, tau158)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau917)
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau922) / 2
            - einsum("pq,qpi->ip", tau293, tau165) / 4
            + einsum("qp,pqi->ip", tau152, tau913) / 2
            - 2 * einsum("pj,pji->ip", tau482, tau20)
            - einsum("pq,qpi->ip", tau112, tau951) / 4
            - einsum("qp,pqi->ip", tau227, tau99)
            + einsum("pq,pq,pq,iq->ip", tau73, tau80, tau800, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau101, tau295, a.y4)
            + einsum("pq,qi->ip", tau117, tau920) / 2
            + einsum("pq,qpi->ip", tau122, tau215) / 2
            + einsum("qp,qpi->ip", tau182, tau671) / 2
            + einsum("pq,pq,iq->ip", tau85, tau953, a.y4) / 2
            + einsum("pq,pq,pq,qi->ip", tau78, tau84, tau85, tau954)
            + einsum("pq,pq,iq->ip", tau318, tau84, a.y4) / 2
            - einsum("qp,qpi->ip", tau431, tau690)
            + einsum("pq,pq,iq->ip", tau101, tau550, a.y3) / 2
            + 2 * einsum("pq,pq,pqi->ip", tau78, tau80, tau802)
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau939)
            + einsum("pq,pq,pqi->ip", tau78, tau80, tau956)
            + einsum("pq,qpi->ip", tau665, tau957) / 2
            + einsum("qp,pqi->ip", tau727, tau135) / 2
            - einsum("qi,pq,pq,pq->ip", tau204, tau78, tau80, tau84)
            + 2 * einsum("p,pi->ip", tau464, tau45)
            + einsum("pq,pq,pqi->ip", tau73, tau85, tau958)
            - einsum("qp,pqi->ip", tau472, tau856) / 4
            - 2 * einsum("pq,pq,pq,iq->ip", tau487, tau78, tau84, a.y3)
            + einsum("qp,qpi->ip", tau694, tau959) / 2
            - einsum("qp,pqi->ip", tau171, tau671) / 4
            - 2 * einsum("pq,pq,pqi->ip", tau73, tau85, tau960)
            - 2 * einsum("pq,pq,iq->ip", tau101, tau481, a.y4)
            + einsum("qp,qpi->ip", tau761, tau135) / 2
            - einsum("qp,pqi->ip", tau737, tau859) / 4
            + einsum("pq,pq,pq,iq->ip", tau101, tau564, tau73, a.y4)
            + einsum("qp,qpi->ip", tau288, tau883) / 2
            + einsum("pq,qi,pq,pq->ip", tau101, tau427, tau73, tau80) / 2
            + 2 * einsum("pq,pq,pqi->ip", tau73, tau85, tau733)
            + einsum("qp,pqi->ip", tau397, tau881) / 2
            - einsum("pq,pq,iq->ip", tau80, tau936, a.y3) / 2
            - einsum("qp,pqi->ip", tau738, tau869)
            - einsum("pq,qpi->ip", tau266, tau168)
            + 2 * einsum("qp,pqi->ip", tau227, tau168)
            + einsum("pq,pq,iq->ip", tau101, tau148, a.y3)
            + einsum("qp,qpi->ip", tau431, tau667) / 2
            + einsum("pq,pq,pqi->ip", tau80, tau84, tau961) / 2
            + einsum("pq,pq,iq->ip", tau73, tau750, a.y4) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau192, tau73, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau85, tau962)
            - einsum("pq,qpi->ip", tau145, tau215) / 4
            + einsum("pq,pq,pqi->ip", tau84, tau85, tau319) / 2
            - einsum("qp,pqi->ip", tau707, tau135)
            + einsum("qp,pqi->ip", tau272, tau842) / 2
            - einsum("pq,qpi->ip", tau270, tau165) / 4
            - einsum("pq,qi->ip", tau226, tau53) / 4
            + einsum("qp,pqi->ip", tau738, tau963) / 2
            - einsum("qp,pqi->ip", tau293, tau964) / 4
            + einsum("qp,pqi->ip", tau152, tau837) / 2
            + einsum("jp,pji->ip", a.x4, tau965)
            - einsum("pq,pq,iq->ip", tau78, tau789, a.y3)
            - einsum("pq,pq,pq,qi->ip", tau101, tau73, tau80, tau940) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau85, tau966) / 2
            + einsum("pj,pji->ip", tau712, tau967)
            - einsum("pq,qpi->ip", tau257, tau667)
            + einsum("pq,qpi->ip", tau216, tau856) / 2
            + einsum("pq,pq,pq,iq->ip", tau234, tau78, tau84, a.y4)
            - einsum("qp,pqi->ip", tau524, tau168)
            + einsum("pq,pq,pq,iq->ip", tau430, tau84, tau85, a.y4) / 2
            + einsum("qp,qpi->ip", tau672, tau963) / 2
            - einsum("pq,pq,pq,qi->ip", tau101, tau73, tau80, tau865) / 2
            - einsum("pq,qi->ip", tau197, tau968) / 4
            - einsum("pq,qpi->ip", tau546, tau168)
            + einsum("pq,pq,pq,iq->ip", tau579, tau78, tau84, a.y4) / 2
            + einsum("qp,pqi->ip", tau94, tau215) / 2
            + einsum("qp,pqi->ip", tau496, tau89) / 2
            - einsum("jp,pji->ip", a.x4, tau969)
            - 2 * einsum("pj,pji->ip", tau158, tau970)
            - einsum("pq,pq,pq,iq->ip", tau301, tau84, tau85, a.y4) / 2
            - einsum("qp,qpi->ip", tau751, tau892) / 4
            - einsum("pq,pq,pqi->ip", tau78, tau80, tau971)
            + einsum("pq,qi->ip", tau412, tau53) / 2
            - einsum("qp,qpi->ip", tau730, tau151) / 4
            + einsum("qp,pqi->ip", tau737, tau869) / 2
            - einsum("qp,pqi->ip", tau469, tau903)
            - 2 * einsum("pq,pq,pq,qi->ip", tau78, tau80, tau84, tau861)
            - 2 * einsum("pq,pq,pq,qi->ip", tau101, tau73, tau85, tau954)
            - einsum("qp,qpi->ip", tau701, tau361) / 4
            - einsum("qp,qpi->ip", tau716, tau963) / 4
            - einsum("qp,pqi->ip", tau270, tau964) / 4
            - 2 * einsum("p,pi->ip", tau19, tau712)
            - einsum("pq,qpi->ip", tau468, tau926) / 4
            - einsum("pq,pq,iq->ip", tau73, tau740, a.y4) / 2
            - einsum("qp,pqi->ip", tau727, tau151) / 4
            + 2 * einsum("pq,pq,pq,iq->ip", tau101, tau476, tau85, a.y4)
            + einsum("pq,qpi->ip", tau407, tau111) / 2
            - einsum("pq,pq,iq->ip", tau85, tau868, a.y4) / 2
            + einsum("pq,qi->ip", tau342, tau972) / 2
            + einsum("qp,pqi->ip", tau112, tau973) / 2
            + einsum("qp,pqi->ip", tau81, tau671) / 2
            - einsum("pq,qpi->ip", tau90, tau856) / 4
            + einsum("qp,pqi->ip", tau609, tau89) / 2
            - einsum("pq,qi->ip", tau117, tau918)
            + einsum("pq,pq,pqi->ip", tau101, tau85, tau338) / 2
            - einsum("qp,pqi->ip", tau169, tau973)
            - einsum("qp,pqi->ip", tau117, tau908)
            + einsum("pq,pq,pq,iq->ip", tau301, tau80, tau84, a.y3)
            + einsum("qp,pqi->ip", tau364, tau914) / 2
            - einsum("qp,qpi->ip", tau288, tau886)
            - einsum("pq,pq,pq,iq->ip", tau357, tau78, tau84, a.y4) / 2
            - einsum("qp,pqi->ip", tau496, tau215) / 4
            - einsum("qp,qpi->ip", tau136, tau902)
            + einsum("pq,pq,iq->ip", tau73, tau758, a.y3) / 2
            - einsum("pq,qpi->ip", tau694, tau839) / 4
            - einsum("pq,qpi->ip", tau407, tau396) / 4
            - einsum("pq,qi->ip", tau117, tau972)
            - einsum("qp,qpi->ip", tau303, tau671)
            + einsum("pq,pq,iq->ip", tau689, tau73, a.y3) / 2
            - einsum("pq,qpi->ip", tau537, tau903)
            - einsum("pq,pq,pqi->ip", tau101, tau85, tau120)
            - einsum("qp,pqi->ip", tau342, tau974) / 4
            + einsum("pq,qpi->ip", tau468, tau850) / 2
            + einsum("qp,pqi->ip", tau738, tau859) / 2
            - einsum("pq,qpi->ip", tau216, tau849) / 4
            - einsum("qp,pqi->ip", tau292, tau671) / 4
            + einsum("qp,pqi->ip", tau86, tau99) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau360, tau73, a.y3)
            + einsum("pq,pq,iq->ip", tau101, tau500, a.y4)
            - einsum("qp,pqi->ip", tau330, tau882)
            + einsum("pq,pq,iq->ip", tau549, tau84, a.y4) / 2
            + einsum("qp,qpi->ip", tau751, tau909) / 2
            - einsum("pq,qpi->ip", tau187, tau884)
            + einsum("pq,pq,iq->ip", tau662, tau78, a.y3) / 2
            + einsum("pq,pq,iq->ip", tau80, tau934, a.y3)
            - einsum("pq,qpi->ip", tau291, tau99) / 4
            - einsum("qp,qpi->ip", tau672, tau869)
            - einsum("qp,pqi->ip", tau359, tau664)
            - einsum("pq,qi->ip", tau263, tau975)
            + 2 * einsum("pq,pq,iq->ip", tau101, tau628, a.y4)
            - einsum("pq,pq,iq->ip", tau614, tau84, a.y4)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau976)
            + einsum("qp,pqi->ip", tau187, tau964) / 2
            - einsum("qp,pqi->ip", tau276, tau848)
            - einsum("pq,qi->ip", tau334, tau43)
            + einsum("pq,qi->ip", tau117, tau831) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau80, tau977)
            - einsum("pq,pq,iq->ip", tau514, tau84, a.y4)
            - einsum("qp,qpi->ip", tau763, tau135)
            - einsum("qp,qpi->ip", tau694, tau957) / 4
            + 2 * einsum("pq,pq,iq->ip", tau406, tau84, a.y3)
            + einsum("pq,qi->ip", tau263, tau968) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau476, tau80, a.y3)
            - einsum("qp,pqi->ip", tau276, tau864)
            + einsum("qp,qpi->ip", tau763, tau151) / 2
            + einsum("pq,pq,pq,iq->ip", tau674, tau78, tau85, a.y4) / 2
            + einsum("qp,pqi->ip", tau131, tau466) / 2
            - einsum("qp,qpi->ip", tau761, tau151) / 4
            - einsum("pq,pq,pq,iq->ip", tau674, tau78, tau80, a.y3)
            + einsum("qi,pq->ip", tau21, tau334) / 2
            - einsum("pq,qpi->ip", tau223, tau850) / 4
            - einsum("pq,qpi->ip", tau247, tau111) / 4
            - einsum("pq,pq,iq->ip", tau73, tau828, a.y4)
            - einsum("pq,qpi->ip", tau474, tau99) / 4
            + einsum("pq,qpi->ip", tau468, tau914) / 2
            - einsum("qp,pqi->ip", tau153, tau111)
            + einsum("pq,pq,pq,iq->ip", tau676, tau73, tau85, a.y4)
            + einsum("qp,qpi->ip", tau280, tau664) / 2
            + einsum("pq,pq,iq->ip", tau710, tau78, a.y3)
            - einsum("pq,pq,iq->ip", tau101, tau311, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau80, tau931, a.y3) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau80, tau978) / 2
            - einsum("qp,pqi->ip", tau364, tau881)
            - einsum("qp,pqi->ip", tau522, tau114)
            - einsum("pq,pq,pqi->ip", tau80, tau84, tau979) / 2
            - einsum("pq,pq,pq,qi->ip", tau78, tau84, tau85, tau945) / 2
            + 2 * einsum("p,pi->ip", tau70, tau712)
            - einsum("pq,pq,pq,iq->ip", tau683, tau73, tau80, a.y3)
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau234, tau73, a.y4)
            + einsum("pq,pq,iq->ip", tau80, tau875, a.y3) / 2
            + einsum("pq,pq,pq,iq->ip", tau748, tau78, tau85, a.y4)
            + einsum("pq,qpi->ip", tau169, tau897) / 2
            + einsum("pq,qpi->ip", tau169, tau951) / 2
            - einsum("pq,qpi->ip", tau100, tau848) / 4
            + einsum("qp,qpi->ip", tau666, tau151) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau585, tau73, a.y3) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau244, tau73, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau101, tau415, tau73, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau101, tau544, a.y3) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau748, tau78, tau80, a.y3)
            + einsum("pq,qpi->ip", tau266, tau99) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau80, tau980) / 2
            - einsum("pq,pq,pq,iq->ip", tau430, tau80, tau84, a.y3)
            - einsum("qp,pqi->ip", tau737, tau963) / 4
            - einsum("pq,pq,pqi->ip", tau84, tau85, tau981) / 2
            + einsum("qp,qpi->ip", tau307, tau913) / 2
            + einsum("pq,pq,iq->ip", tau73, tau749, a.y4)
            - einsum("pq,qpi->ip", tau468, tau881) / 4
            + einsum("pq,pq,pq,iq->ip", tau585, tau78, tau84, a.y3)
            + einsum("qp,pqi->ip", tau522, tau964) / 2
            - einsum("qp,qpi->ip", tau716, tau859) / 4
            - einsum("pq,qpi->ip", tau665, tau959) / 4
            + 2 * einsum("pq,pq,iq->ip", tau214, tau84, a.y3)
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau976) / 2
            + einsum("pq,qi->ip", tau197, tau975) / 2
            - einsum("qp,pqi->ip", tau94, tau89)
            + einsum("qp,qpi->ip", tau701, tau400) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau243, tau80, a.y3)
            + einsum("qi,pq,pq,pq->ip", tau321, tau78, tau84, tau85) / 2
            - einsum("qp,pqi->ip", tau397, tau914) / 4
            - 2 * einsum("pq,pq,iq->ip", tau443, tau84, a.y3)
            - einsum("pq,pq,iq->ip", tau612, tau84, a.y3)
            - einsum("pq,pq,iq->ip", tau80, tau953, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau78, tau80, tau820, a.y3)
            + einsum("pq,qpi->ip", tau112, tau929) / 2
            + einsum("pq,pq,iq->ip", tau644, tau84, a.y3)
            - einsum("qp,qpi->ip", tau257, tau948)
            - einsum("pq,pq,pq,qi->ip", tau78, tau84, tau85, tau947) / 2
            + einsum("qp,qpi->ip", tau716, tau851) / 2
            + einsum("pq,pq,iq->ip", tau80, tau924, a.y3) / 2
            - 2 * einsum("pq,pq,pqi->ip", tau78, tau80, tau982)
            + einsum("qp,pqi->ip", tau731, tau135) / 2
            + einsum("qp,pqi->ip", tau117, tau974) / 2
            + einsum("qp,pqi->ip", tau412, tau845) / 2
            - einsum("pq,pq,pq,iq->ip", tau564, tau78, tau84, a.y4) / 2
            + einsum("pq,pq,iq->ip", tau85, tau873, a.y4) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau586, tau85, a.y4)
            - einsum("pq,qpi->ip", tau256, tau111) / 4
            - einsum("qp,qpi->ip", tau231, tau913) / 4
            + 2 * einsum("jp,pij->ip", a.x4, tau969)
        )
    
        ry4 = (
            einsum("pq,pq,iq->ip", tau682, tau78, a.y3) / 2
            + einsum("pq,qpi->ip", tau312, tau983) / 2
            + 2 * einsum("pq,qpi->ip", tau276, tau847)
            - einsum("qp,qpi->ip", tau684, tau984)
            + einsum("ji,pj->ip", tau305, tau53)
            + einsum("pq,pq,iq->ip", tau74, tau931, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau299, tau74, tau84, a.y3)
            - 2 * einsum("pq,pq,pq,iq->ip", tau333, tau78, tau84, a.y4)
            - einsum("pq,qi->ip", tau517, tau860) / 4
            + einsum("pq,pq,iq->ip", tau78, tau810, a.y4)
            + 2 * einsum("pq,pq,iq->ip", tau73, tau799, a.y3)
            - einsum("pq,qpi->ip", tau691, tau859)
            - einsum("qp,pqi->ip", tau152, tau902) / 4
            + einsum("qp,pqi->ip", tau263, tau908) / 2
            - 2 * einsum("jp,pji->ip", a.x3, tau965)
            + einsum("pq,pq,pqi->ip", tau73, tau74, tau978)
            + einsum("qp,pqi->ip", tau737, tau839) / 2
            - 2 * einsum("pj,pji->ip", tau712, tau985)
            - 2 * einsum("pq,pq,pqi->ip", tau101, tau74, tau858)
            + einsum("pq,qi->ip", tau496, tau844) / 2
            - einsum("pq,qi->ip", tau609, tau986) / 4
            - einsum("pq,pq,pq,iq->ip", tau101, tau322, tau73, a.y3)
            - einsum("pq,pq,pqi->ip", tau74, tau78, tau855) / 2
            + einsum("pq,pq,iq->ip", tau73, tau769, a.y4)
            + einsum("qi,pq,pq,pq->ip", tau260, tau74, tau78, tau84)
            + einsum("pq,qpi->ip", tau413, tau671) / 2
            - einsum("pq,pq,pqi->ip", tau78, tau79, tau962) / 2
            - einsum("pq,qi->ip", tau472, tau920) / 4
            - einsum("qp,pqi->ip", tau166, tau848) / 4
            + einsum("qi,pq,pq,pq->ip", tau427, tau74, tau78, tau84) / 2
            - einsum("qp,pqi->ip", tau197, tau908) / 4
            + einsum("pq,qpi->ip", tau537, tau841) / 2
            - einsum("qp,qpi->ip", tau691, tau963)
            - einsum("pq,pq,iq->ip", tau336, tau84, a.y4) / 2
            - 2 * einsum("pq,pq,iq->ip", tau73, tau823, a.y3)
            - einsum("pq,qpi->ip", tau86, tau884)
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau989)
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau989) / 2
            + einsum("pq,pq,iq->ip", tau489, tau84, a.y4) / 2
            - einsum("pq,qi->ip", tau446, tau844)
            - einsum("pq,qi->ip", tau330, tau972)
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau991) / 2
            + einsum("pq,qpi->ip", tau291, tau964) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau79, tau815) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau993)
            - einsum("pq,pq,pq,iq->ip", tau101, tau476, tau79, a.y4)
            - einsum("qp,pqi->ip", tau397, tau897) / 4
            - einsum("pq,pq,iq->ip", tau78, tau818, a.y3)
            - einsum("pq,qpi->ip", tau668, tau869)
            + einsum("pq,qpi->ip", tau716, tau959) / 2
            - einsum("pq,pq,pq,iq->ip", tau726, tau74, tau78, a.y3) / 2
            - einsum("qp,pqi->ip", tau223, tau841) / 4
            + einsum("pq,qpi->ip", tau390, tau994) / 2
            - einsum("pq,pq,iq->ip", tau606, tau84, a.y3) / 2
            + einsum("qp,pqi->ip", tau312, tau168) / 2
            + einsum("qp,qpi->ip", tau401, tau837) / 2
            - einsum("pq,qpi->ip", tau730, tau892) / 4
            + einsum("qp,pqi->ip", tau166, tau840) / 2
            + einsum("pq,qpi->ip", tau716, tau880) / 2
            - einsum("pq,qpi->ip", tau469, tau929)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau991)
            - einsum("pq,qpi->ip", tau604, tau111) / 4
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau996)
            - einsum("qp,pqi->ip", tau102, tau168)
            + einsum("pq,pq,pq,iq->ip", tau73, tau79, tau800, a.y4) / 2
            - einsum("qp,qpi->ip", tau711, tau135)
            + einsum("pq,qpi->ip", tau668, tau963) / 2
            - einsum("pq,pq,iq->ip", tau101, tau222, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau101, tau243, tau79, a.y4)
            + einsum("pq,pq,iq->ip", tau74, tau953, a.y3) / 2
            - einsum("pq,pq,iq->ip", tau101, tau615, a.y4) / 2
            - einsum("qp,qpi->ip", tau508, tau671) / 4
            - einsum("pq,pq,iq->ip", tau109, tau84, a.y3)
            - 2 * einsum("pq,qi,pq,pq->ip", tau101, tau260, tau73, tau74)
            - einsum("pq,pq,pq,qi->ip", tau101, tau73, tau79, tau947) / 2
            - einsum("pq,pq,iq->ip", tau74, tau924, a.y3)
            + einsum("qp,pqi->ip", tau422, tau964) / 2
            + einsum("pq,pq,iq->ip", tau101, tau539, a.y4) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau79, tau733)
            + einsum("qp,pqi->ip", tau449, tau89) / 2
            + einsum("qp,pqi->ip", tau468, tau973) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau104, tau74, a.y3)
            - einsum("pq,qpi->ip", tau166, tau926) / 4
            + einsum("pq,qpi->ip", tau100, tau907) / 2
            + 4 * einsum("p,pi->ip", tau722, tau65)
            + einsum("pq,pq,pqi->ip", tau101, tau79, tau120) / 2
            + einsum("pq,pq,iq->ip", tau73, tau780, a.y4)
            + einsum("qp,pqi->ip", tau468, tau841) / 2
            - einsum("pq,pq,pq,iq->ip", tau430, tau79, tau84, a.y4)
            + 2 * einsum("p,pi->ip", tau722, tau5)
            + einsum("pq,pq,pqi->ip", tau74, tau84, tau885) / 2
            - 2 * einsum("pq,pq,iq->ip", tau747, tau78, a.y4)
            - einsum("pq,qpi->ip", tau231, tau902) / 4
            + einsum("pq,pq,pq,iq->ip", tau430, tau74, tau84, a.y3) / 2
            - einsum("qp,pqi->ip", tau272, tau886)
            + einsum("pq,qpi->ip", tau303, tau948) / 2
            + 2 * einsum("pq,pq,iq->ip", tau78, tau821, a.y4)
            + einsum("pq,qpi->ip", tau90, tau974) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau637, tau74, a.y3)
            - 2 * einsum("pq,pq,pqi->ip", tau79, tau84, tau834)
            - einsum("pq,qi,pq,pq->ip", tau101, tau204, tau73, tau74)
            + einsum("pq,pq,pqi->ip", tau73, tau79, tau835)
            - einsum("pq,qpi->ip", tau231, tau883) / 4
            + einsum("pq,pq,iq->ip", tau240, tau84, a.y4)
            + einsum("qp,qpi->ip", tau668, tau859) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau74, tau538)
            - 2 * einsum("p,pi->ip", tau722, tau18)
            + einsum("pq,pq,iq->ip", tau645, tau84, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau764, tau78, a.y4)
            - 2 * einsum("pj,pji->ip", tau306, tau42)
            + einsum("qp,pqi->ip", tau407, tau964) / 2
            - einsum("pq,qpi->ip", tau182, tau948) / 4
            + einsum("pq,pq,iq->ip", tau101, tau545, a.y3) / 2
            + einsum("qp,pqi->ip", tau738, tau880) / 2
            - einsum("pq,pq,iq->ip", tau144, tau84, a.y3)
            + einsum("qp,pqi->ip", tau517, tau856) / 2
            - einsum("pq,pq,iq->ip", tau78, tau785, a.y4)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau998)
            - einsum("pq,pq,iq->ip", tau507, tau84, a.y4)
            + einsum("pq,pq,pq,iq->ip", tau139, tau78, tau84, a.y4)
            - 2 * einsum("pj,pji->ip", tau158, tau3)
            - einsum("pq,qpi->ip", tau227, tau999)
            - einsum("qp,pqi->ip", tau370, tau89) / 4
            + einsum("pq,qi->ip", tau467, tau968) / 2
            + einsum("pq,qpi->ip", tau401, tau913) / 2
            + einsum("qp,pqi->ip", tau759, tau909) / 2
            + einsum("pq,qpi->ip", tau307, tau902) / 2
            - einsum("pq,qpi->ip", tau716, tau957) / 4
            + einsum("qi,pq->ip", tau43, tau542) / 2
            + einsum("pq,pq,iq->ip", tau101, tau452, a.y4)
            - einsum("pq,pq,pqi->ip", tau101, tau79, tau899) / 2
            + einsum("pq,pq,pqi->ip", tau74, tau84, tau317) / 2
            - einsum("pq,qpi->ip", tau216, tau904) / 4
            + einsum("qp,pqi->ip", tau81, tau948) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau586, tau74, a.y3)
            + einsum("pq,pq,iq->ip", tau78, tau825, a.y3)
            - 2 * einsum("p,pi->ip", tau580, tau15)
            + einsum("qp,pqi->ip", tau610, tau845) / 2
            - einsum("qp,pqi->ip", tau517, tau849) / 4
            + einsum("qp,pqi->ip", tau302, tau884) / 2
            - einsum("qi,pq,pq,pq->ip", tau191, tau78, tau79, tau84)
            + einsum("pq,pq,pq,iq->ip", tau101, tau592, tau73, a.y3)
            - einsum("qp,pqi->ip", tau604, tau1000) / 4
            + einsum("pq,qpi->ip", tau575, tau111) / 2
            - einsum("pq,qpi->ip", tau537, tau911)
            - einsum("pq,qpi->ip", tau401, tau842) / 4
            + einsum("pq,pq,iq->ip", tau79, tau934, a.y4)
            - einsum("pq,qpi->ip", tau122, tau845) / 4
            - einsum("pq,qpi->ip", tau711, tau1001)
            - einsum("pq,pq,pqi->ip", tau101, tau79, tau338)
            - einsum("pq,pq,iq->ip", tau79, tau936, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau101, tau635, a.y3) / 2
            + einsum("pj,pji->ip", tau158, tau1002)
            + einsum("qp,pqi->ip", tau397, tau847) / 2
            - einsum("qp,pqi->ip", tau731, tau909) / 4
            + einsum("pq,pq,pq,iq->ip", tau73, tau79, tau793, a.y4)
            + einsum("pq,qpi->ip", tau280, tau948) / 2
            - einsum("pq,pq,pq,iq->ip", tau301, tau74, tau84, a.y3) / 2
            + einsum("pq,pq,iq->ip", tau73, tau829, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau74, tau760, tau78, a.y3) / 2
            + einsum("pq,qpi->ip", tau115, tau881) / 2
            - einsum("qp,pqi->ip", tau759, tau892)
            - einsum("qp,pqi->ip", tau188, tau845)
            + einsum("qp,pqi->ip", tau334, tau994) / 2
            + einsum("pq,qpi->ip", tau216, tau832) / 2
            - einsum("pq,qpi->ip", tau280, tau905)
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau1003) / 2
            + einsum("pq,pq,pq,iq->ip", tau676, tau73, tau74, a.y3)
            + 2 * einsum("pq,pq,pqi->ip", tau101, tau74, tau949)
            + einsum("pq,qpi->ip", tau145, tau845) / 2
            + einsum("pq,qpi->ip", tau86, tau999) / 2
            + 2 * einsum("jp,pji->ip", a.x3, tau969)
            + einsum("qp,pqi->ip", tau186, tau999) / 2
            - einsum("pi->ip", tau986)
            - einsum("qp,pqi->ip", tau449, tau1004)
            - einsum("pq,pq,pq,iq->ip", tau592, tau78, tau84, a.y3) / 2
            - einsum("pq,pq,pqi->ip", tau74, tau84, tau941) / 2
            - einsum("qp,pqi->ip", tau81, tau905)
            - einsum("pq,pq,pqi->ip", tau74, tau78, tau802)
            + einsum("pq,pq,pqi->ip", tau79, tau84, tau981)
            - 2 * einsum("pq,pq,pq,iq->ip", tau748, tau78, tau79, a.y4)
            + einsum("pq,qpi->ip", tau508, tau1005) / 2
            - einsum("pq,pq,iq->ip", tau73, tau819, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau225, tau78, tau84, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau74, tau78, tau971) / 2
            + einsum("pj,pji->ip", tau306, tau20)
            + einsum("pq,pq,pq,iq->ip", tau375, tau78, tau84, a.y3)
            + einsum("pq,pq,pqi->ip", tau78, tau79, tau843)
            - einsum("qp,pqi->ip", tau385, tau999) / 4
            + einsum("pq,qpi->ip", tau122, tau994) / 2
            + einsum("pq,pq,iq->ip", tau442, tau84, a.y3) / 2
            + einsum("qp,pqi->ip", tau115, tau848) / 2
            + einsum("pq,qi->ip", tau330, tau920) / 2
            + einsum("pq,qpi->ip", tau102, tau111) / 2
            - einsum("pq,pq,pq,qi->ip", tau101, tau73, tau79, tau945) / 2
            + einsum("pq,qpi->ip", tau604, tau983) / 2
            - einsum("qp,pqi->ip", tau272, tau893)
            + einsum("qp,pqi->ip", tau707, tau909) / 2
            + einsum("pq,qi->ip", tau517, tau912) / 2
            + 2 * einsum("pq,pq,pqi->ip", tau79, tau84, tau900)
            - einsum("jp,pji->ip", a.x3, tau889)
            + einsum("pq,qi->ip", tau467, tau860) / 2
            - einsum("qp,pqi->ip", tau738, tau957)
            - einsum("pq,pq,iq->ip", tau633, tau84, a.y4)
            - einsum("qp,pqi->ip", tau115, tau840)
            - einsum("pq,pq,iq->ip", tau74, tau868, a.y3) / 2
            + einsum("pq,pq,iq->ip", tau79, tau868, a.y4)
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau589, tau73, a.y3)
            + einsum("pq,qpi->ip", tau672, tau957) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau333, tau73, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau101, tau225, tau73, a.y3)
            + einsum("pq,qi->ip", tau330, tau831) / 2
            - einsum("pq,qi->ip", tau472, tau831) / 4
            - einsum("pq,qpi->ip", tau672, tau880)
            - einsum("qp,pqi->ip", tau247, tau964) / 4
            - 2 * einsum("pq,pq,pq,iq->ip", tau299, tau79, tau84, a.y4)
            - 2 * einsum("pq,pq,pq,qi->ip", tau101, tau73, tau74, tau861)
            + einsum("pq,qi->ip", tau609, tau844) / 2
            + einsum("pq,qi->ip", tau472, tau918) / 2
            + einsum("qp,qpi->ip", tau413, tau1006) / 2
            - einsum("pq,pq,pq,iq->ip", tau428, tau78, tau84, a.y4)
            + einsum("pq,pq,pq,iq->ip", tau589, tau78, tau84, a.y3)
            + einsum("pq,qpi->ip", tau362, tau863) / 2
            + einsum("qp,pqi->ip", tau738, tau959) / 2
            + einsum("pq,pq,pq,iq->ip", tau726, tau78, tau79, a.y4)
            - einsum("qi,pq->ip", tau844, tau94)
            + einsum("pq,qpi->ip", tau672, tau839) / 2
            - einsum("pq,pq,iq->ip", tau73, tau827, a.y4) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau1008) / 2
            + einsum("qp,pqi->ip", tau272, tau902) / 2
            - einsum("qi,pq->ip", tau43, tau449)
            + einsum("qp,pqi->ip", tau731, tau892) / 2
            - einsum("qp,pqi->ip", tau492, tau994) / 4
            - einsum("pq,pq,pqi->ip", tau78, tau79, tau854) / 2
            - einsum("pq,qpi->ip", tau474, tau1009) / 4
            + einsum("pq,pq,iq->ip", tau101, tau629, a.y3) / 2
            - einsum("pq,pq,pq,iq->ip", tau709, tau73, tau79, a.y4) / 2
            - einsum("pq,qpi->ip", tau100, tau973) / 4
            - einsum("qp,pqi->ip", tau467, tau846)
            - einsum("pq,qpi->ip", tau246, tau999)
            + einsum("pq,pq,pq,iq->ip", tau101, tau297, tau73, a.y4)
            - einsum("qp,pqi->ip", tau223, tau973) / 4
            + einsum("qp,pqi->ip", tau370, tau1004) / 2
            - einsum("qp,pqi->ip", tau166, tau864) / 4
            + einsum("pq,qpi->ip", tau711, tau984) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau79, tau688) / 2
            + einsum("pq,qpi->ip", tau537, tau973) / 2
            - einsum("pq,qpi->ip", tau276, tau897)
            + 2 * einsum("pq,pq,pq,iq->ip", tau101, tau476, tau74, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau255, tau79, tau84, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau79, tau84, tau93, a.y4)
            - einsum("pq,qpi->ip", tau537, tau907)
            + einsum("qi,pq->ip", tau1010, tau449) / 2
            - einsum("pq,qpi->ip", tau303, tau905)
            - einsum("pq,pq,pqi->ip", tau73, tau74, tau838) / 2
            + einsum("qp,pqi->ip", tau152, tau893) / 2
            + einsum("qp,qpi->ip", tau362, tau842) / 2
            + einsum("qp,qpi->ip", tau508, tau664) / 2
            + einsum("qp,pqi->ip", tau604, tau168) / 2
            + 2 * einsum("pq,qpi->ip", tau691, tau851)
            - einsum("pq,qpi->ip", tau398, tau948) / 4
            - einsum("qp,pqi->ip", tau397, tau951) / 4
            - einsum("p,pi->ip", tau580, tau45)
            + einsum("pq,qpi->ip", tau398, tau905) / 2
            - einsum("pq,pq,pq,iq->ip", tau275, tau74, tau84, a.y3)
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau79, tau954)
            - einsum("pq,qpi->ip", tau362, tau837) / 4
            + 2 * einsum("pq,pq,pq,iq->ip", tau78, tau781, tau79, a.y4)
            + einsum("pq,qpi->ip", tau231, tau893) / 2
            - einsum("qp,pqi->ip", tau115, tau903)
            + einsum("pq,qi->ip", tau517, tau975) / 2
            - einsum("pq,pq,pq,iq->ip", tau205, tau78, tau84, a.y4)
            + einsum("qp,pqi->ip", tau727, tau892) / 2
            - einsum("qp,pqi->ip", tau256, tau964) / 4
            - einsum("pq,qpi->ip", tau524, tau884)
            + einsum("pq,qpi->ip", tau216, tau908) / 2
            + einsum("pq,pq,iq->ip", tau190, tau84, a.y3)
            - einsum("pq,qpi->ip", tau575, tau983)
            - einsum("qp,pqi->ip", tau263, tau974)
            + einsum("pq,pq,pqi->ip", tau101, tau79, tau871)
            + einsum("pq,qpi->ip", tau307, tau883) / 2
            - einsum("pq,pq,pq,qi->ip", tau74, tau78, tau84, tau865) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau74, tau901)
            - einsum("pq,pq,pq,iq->ip", tau73, tau74, tau800, a.y3)
            - einsum("pq,pq,iq->ip", tau101, tau627, a.y3) / 2
            + einsum("pq,qpi->ip", tau100, tau911) / 2
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau74, tau940)
            - einsum("qp,pqi->ip", tau312, tau1000) / 4
            - einsum("pq,qpi->ip", tau266, tau964)
            + einsum("pj,pji->ip", tau712, tau34)
            - einsum("qp,pqi->ip", tau468, tau907) / 4
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau1011) / 2
            - einsum("pq,qpi->ip", tau312, tau111) / 4
            + einsum("pq,pq,iq->ip", tau611, tau84, a.y4)
            - einsum("pq,qpi->ip", tau469, tau847)
            + 2 * einsum("pq,qpi->ip", tau276, tau929)
            + einsum("qp,pqi->ip", tau397, tau929) / 2
            - einsum("pq,pq,iq->ip", tau744, tau78, a.y3)
            - einsum("pq,pq,pqi->ip", tau74, tau84, tau894) / 2
            - einsum("pq,pq,iq->ip", tau78, tau830, a.y3) / 2
            - einsum("pq,pq,pq,iq->ip", tau255, tau74, tau84, a.y3) / 2
            - einsum("pq,qpi->ip", tau115, tau850)
            + einsum("pq,qpi->ip", tau730, tau909) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau73, tau996) / 2
            - einsum("pq,pq,pq,iq->ip", tau760, tau78, tau79, a.y4)
            - einsum("pq,pq,pqi->ip", tau73, tau79, tau906)
            + 2 * einsum("qp,qpi->ip", tau691, tau869)
            + einsum("pq,qi,pq,pq->ip", tau101, tau321, tau73, tau79) / 2
            - einsum("pq,pq,pq,iq->ip", tau674, tau78, tau79, a.y4)
            + einsum("jp,pji->ip", a.x3, tau887)
            + einsum("pq,pq,iq->ip", tau73, tau813, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau73, tau826, a.y3)
            + einsum("qp,pqi->ip", tau478, tau89) / 2
            + einsum("pq,pq,pq,iq->ip", tau73, tau79, tau809, a.y4) / 2
            - einsum("pq,pq,pq,qi->ip", tau74, tau78, tau84, tau940) / 2
            + einsum("pq,pq,pq,iq->ip", tau74, tau748, tau78, a.y3)
            + einsum("qp,pqi->ip", tau359, tau948) / 2
            - einsum("qi,pq->ip", tau1010, tau542) / 4
            + einsum("pq,pq,pq,qi->ip", tau78, tau79, tau84, tau947)
            - einsum("pq,pq,pqi->ip", tau73, tau74, tau812)
            + einsum("qp,pqi->ip", tau385, tau884) / 2
            + einsum("pq,qpi->ip", tau666, tau892) / 2
            - einsum("pq,qpi->ip", tau100, tau841) / 4
            + 4 * einsum("p,pi->ip", tau16, tau158)
            - einsum("pq,pq,pqi->ip", tau73, tau74, tau977) / 2
            - einsum("qp,pqi->ip", tau364, tau847)
            - einsum("pq,pq,pqi->ip", tau73, tau79, tau958) / 2
            + einsum("pq,pq,iq->ip", tau743, tau78, a.y3) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau243, tau74, a.y3)
            - einsum("pq,pq,pqi->ip", tau78, tau79, tau879)
            - einsum("qi,pq->ip", tau1010, tau370) / 4
            - einsum("pq,pq,iq->ip", tau79, tau931, a.y4) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau205, tau73, a.y4) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau79, tau888) / 2
            + einsum("pq,pq,iq->ip", tau79, tau924, a.y4) / 2
            + einsum("pq,qi->ip", tau94, tau986) / 2
            - einsum("pq,qpi->ip", tau763, tau909)
            - einsum("qp,pqi->ip", tau737, tau959) / 4
            + einsum("pq,pq,pqi->ip", tau74, tau84, tau944)
            + einsum("pq,pq,pqi->ip", tau79, tau84, tau876)
            - einsum("qp,pqi->ip", tau359, tau905)
            + einsum("pq,pq,iq->ip", tau101, tau595, a.y4)
            + einsum("pq,pq,pqi->ip", tau73, tau74, tau910)
            + einsum("pq,pq,iq->ip", tau558, tau84, a.y3)
            - einsum("qp,qpi->ip", tau362, tau913) / 4
            - einsum("pq,qpi->ip", tau216, tau974) / 4
            - einsum("pq,pq,iq->ip", tau658, tau73, a.y4)
            - 2 * einsum("pq,pq,pq,iq->ip", tau297, tau78, tau84, a.y4)
            + einsum("pq,pq,pq,iq->ip", tau709, tau73, tau74, a.y3)
            - 2 * einsum("qi,pq,pq,pq->ip", tau486, tau78, tau79, tau84)
            + einsum("pq,pq,iq->ip", tau74, tau936, a.y3)
            + einsum("pq,pq,pqi->ip", tau101, tau79, tau966)
            + einsum("qp,pqi->ip", tau223, tau911) / 2
            - einsum("pq,pq,pqi->ip", tau74, tau84, tau488)
            - einsum("pq,qpi->ip", tau102, tau983)
            + einsum("p,pi->ip", tau580, tau30)
            - einsum("pq,pq,pq,iq->ip", tau101, tau273, tau74, a.y3)
            + einsum("qp,pqi->ip", tau467, tau882) / 2
            - einsum("qp,pqi->ip", tau575, tau168)
            - 2 * einsum("pq,pq,pqi->ip", tau101, tau74, tau946)
            - einsum("qp,pqi->ip", tau727, tau909) / 4
            - einsum("qp,pqi->ip", tau153, tau884)
            + 2 * einsum("qp,qpi->ip", tau684, tau1001)
            - einsum("pq,qpi->ip", tau276, tau951)
            - einsum("qp,pqi->ip", tau707, tau892)
            + einsum("pq,pq,pqi->ip", tau101, tau74, tau896)
            + einsum("pq,qpi->ip", tau546, tau1009) / 2
            + einsum("qp,pqi->ip", tau171, tau905) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau428, tau73, a.y4) / 2
            - einsum("pq,qi->ip", tau330, tau918)
            + einsum("pq,pq,pqi->ip", tau101, tau73, tau1003)
            - einsum("pq,pq,pqi->ip", tau78, tau79, tau742)
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau1008)
            + einsum("pq,qi->ip", tau370, tau43) / 2
            + einsum("qp,pqi->ip", tau188, tau994) / 2
            - einsum("pq,qpi->ip", tau90, tau832) / 4
            - einsum("pq,pq,pqi->ip", tau74, tau78, tau836)
            + einsum("pq,pq,pq,qi->ip", tau74, tau78, tau84, tau861)
            + einsum("qp,pqi->ip", tau166, tau903) / 2
            + 2 * einsum("pq,qpi->ip", tau246, tau884)
            - einsum("pq,qpi->ip", tau716, tau839) / 4
            + einsum("pq,qpi->ip", tau166, tau850) / 2
            - einsum("qp,qpi->ip", tau401, tau863) / 4
            - einsum("pq,pq,iq->ip", tau73, tau775, a.y4) / 2
            + einsum("qp,pqi->ip", tau223, tau907) / 2
            + einsum("pq,pq,pq,iq->ip", tau322, tau78, tau84, a.y3) / 2
            - einsum("pq,qpi->ip", tau515, tau994) / 4
            - einsum("pq,pq,pqi->ip", tau79, tau84, tau925)
            - einsum("pq,pq,pq,iq->ip", tau101, tau236, tau73, a.y4) / 2
            - einsum("qp,qpi->ip", tau668, tau851)
            + einsum("qp,pqi->ip", tau364, tau951) / 2
            + einsum("qp,pqi->ip", tau272, tau883) / 2
            - einsum("pq,pq,pqi->ip", tau73, tau74, tau980)
            - einsum("qp,pqi->ip", tau263, tau904)
            + einsum("qp,pqi->ip", tau197, tau904) / 2
            - einsum("pq,pq,pqi->ip", tau74, tau84, tau961)
            - einsum("qp,pqi->ip", tau738, tau839)
            - einsum("pq,qpi->ip", tau307, tau893)
            - einsum("pq,pq,iq->ip", tau101, tau177, a.y4) / 2
            + einsum("pq,pq,pq,iq->ip", tau101, tau273, tau79, a.y4) / 2
            - einsum("qp,pqi->ip", tau334, tau845)
            + einsum("pq,qpi->ip", tau182, tau905) / 2
            - einsum("pq,qpi->ip", tau90, tau908) / 4
            - 2 * einsum("pj,pij->ip", tau158, tau10)
            - einsum("qp,pqi->ip", tau517, tau882) / 4
            - 2 * einsum("p,pi->ip", tau31, tau158)
            - einsum("jp,pij->ip", a.x3, tau969)
            + einsum("pq,pq,pq,iq->ip", tau236, tau78, tau84, a.y4)
            + 2 * einsum("pq,pq,pq,iq->ip", tau275, tau79, tau84, a.y4)
            + einsum("pq,pq,pqi->ip", tau78, tau79, tau870) / 2
            - einsum("pq,pq,iq->ip", tau599, tau84, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau73, tau806, a.y4)
            - einsum("pq,qpi->ip", tau166, tau881) / 4
            + einsum("pq,qpi->ip", tau474, tau964) / 2
            - einsum("pq,qi->ip", tau467, tau975)
            + einsum("pq,pq,pq,iq->ip", tau101, tau104, tau79, a.y4) / 2
            - einsum("pq,pq,iq->ip", tau101, tau577, a.y4)
            - einsum("pq,pq,iq->ip", tau101, tau445, a.y4)
            + einsum("pq,qi,pq,pq->ip", tau101, tau191, tau73, tau79) / 2
            + einsum("pq,pq,iq->ip", tau79, tau875, a.y4) / 2
            - einsum("pq,pq,pq,iq->ip", tau74, tau78, tau820, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau74, tau950) / 2
            + einsum("qp,pqi->ip", tau492, tau845) / 2
            - einsum("qp,pqi->ip", tau292, tau948) / 4
            + 2 * einsum("pi->ip", tau844)
            - 2 * einsum("p,pi->ip", tau72, tau712)
            + einsum("pq,pq,pqi->ip", tau73, tau79, tau960)
            - einsum("pq,pq,pq,iq->ip", tau683, tau73, tau79, a.y4)
            - einsum("pq,qi->ip", tau517, tau968) / 4
            + einsum("qp,pqi->ip", tau737, tau957) / 2
            + einsum("pq,pq,iq->ip", tau101, tau584, a.y3)
            - einsum("pq,qpi->ip", tau508, tau1006) / 4
            + 2 * einsum("pq,pq,pqi->ip", tau101, tau74, tau576)
            - einsum("pq,pq,pqi->ip", tau101, tau74, tau932)
            + einsum("pq,pq,iq->ip", tau101, tau425, a.y4) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau586, tau79, a.y4) / 2
            + einsum("pq,pq,pq,iq->ip", tau74, tau84, tau93, a.y3) / 2
            + einsum("pq,pq,pqi->ip", tau101, tau79, tau862) / 2
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau998) / 2
            + einsum("qp,pqi->ip", tau517, tau846) / 2
            + einsum("qi,pq->ip", tau1010, tau478) / 2
            - einsum("pq,pq,pqi->ip", tau101, tau79, tau852) / 2
            + 2 * einsum("pq,pq,pqi->ip", tau79, tau84, tau108)
            + einsum("pq,qpi->ip", tau90, tau904) / 2
            + 2 * einsum("pq,pq,pq,iq->ip", tau683, tau73, tau74, a.y3)
            + einsum("pq,pq,pqi->ip", tau78, tau79, tau928)
            - einsum("pq,qi->ip", tau496, tau986) / 4
            - einsum("qp,pqi->ip", tau737, tau880) / 4
            - einsum("pq,pq,iq->ip", tau74, tau934, a.y3) / 2
            - einsum("qp,pqi->ip", tau478, tau1004)
            - einsum("pq,pq,iq->ip", tau461, tau84, a.y3) / 2
            - einsum("pq,qpi->ip", tau672, tau959)
            + einsum("qp,pqi->ip", tau364, tau897) / 2
            + einsum("pq,qpi->ip", tau763, tau892) / 2
            + einsum("pq,pq,iq->ip", tau755, tau78, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau301, tau79, tau84, a.y4)
            - einsum("pq,qpi->ip", tau145, tau994) / 4
            + einsum("qp,pqi->ip", tau292, tau905) / 2
            + einsum("qp,pqi->ip", tau197, tau974) / 2
            - einsum("pq,qpi->ip", tau684, tau151)
            - einsum("pq,pq,pq,iq->ip", tau676, tau73, tau79, a.y4) / 2
            - 2 * einsum("pq,pq,pq,qi->ip", tau78, tau79, tau84, tau954)
            - einsum("qp,pqi->ip", tau152, tau883) / 4
            + einsum("qp,pqi->ip", tau263, tau832) / 2
            - einsum("pq,pq,pq,iq->ip", tau73, tau74, tau809, a.y3)
            - einsum("pq,qpi->ip", tau413, tau664) / 4
            - einsum("pq,qpi->ip", tau115, tau914)
            - einsum("pq,pq,pq,iq->ip", tau262, tau78, tau84, a.y3) / 2
            - einsum("qp,pqi->ip", tau422, tau1009) / 4
            + einsum("pq,qpi->ip", tau115, tau926) / 2
            - einsum("pq,qpi->ip", tau666, tau909)
            + einsum("pq,pq,pqi->ip", tau73, tau79, tau866) / 2
            - einsum("pq,qpi->ip", tau291, tau1009) / 4
            + 2 * einsum("p,pi->ip", tau58, tau158)
            - einsum("pq,qi,pq,pq->ip", tau101, tau427, tau73, tau74)
            + einsum("pq,pq,iq->ip", tau772, tau78, a.y4)
            + einsum("pq,qpi->ip", tau761, tau909) / 2
            - einsum("qp,pqi->ip", tau171, tau948) / 4
            + 2 * einsum("jp,pij->ip", a.x3, tau889)
            - einsum("qp,pqi->ip", tau467, tau856)
            - einsum("qi,pq->ip", tau43, tau478)
            - einsum("pq,qi->ip", tau467, tau912)
            - einsum("qp,pqi->ip", tau542, tau89) / 4
            - einsum("pq,pq,pqi->ip", tau101, tau79, tau919)
            + einsum("pq,pq,iq->ip", tau101, tau181, a.y3)
            - einsum("pq,pq,iq->ip", tau79, tau873, a.y4)
            - einsum("qp,pqi->ip", tau186, tau884)
            - einsum("p,pi->ip", tau6, tau712)
            - einsum("pq,pq,iq->ip", tau79, tau953, a.y4)
            + einsum("qi,pq,pq,pq->ip", tau204, tau74, tau78, tau84) / 2
            + einsum("pq,pq,pqi->ip", tau78, tau84, tau1011)
            - einsum("pq,qpi->ip", tau761, tau892) / 4
            + einsum("pq,pq,pqi->ip", tau74, tau84, tau979)
            + einsum("pq,pq,iq->ip", tau320, tau84, a.y3) / 2
            + einsum("qp,pqi->ip", tau102, tau1000) / 2
            - einsum("qp,pqi->ip", tau302, tau999) / 4
            + 2 * einsum("pq,qpi->ip", tau684, tau135)
            + 2 * einsum("pq,pq,iq->ip", tau73, tau734, a.y3)
            + einsum("qp,pqi->ip", tau247, tau1009) / 2
            + einsum("qp,qpi->ip", tau711, tau151) / 2
            + einsum("pq,pq,pqi->ip", tau73, tau74, tau752) / 2
            - 2 * einsum("pq,pq,pqi->ip", tau79, tau84, tau890)
            - einsum("pq,pq,iq->ip", tau74, tau875, a.y3)
            + einsum("pq,pq,iq->ip", tau74, tau873, a.y3) / 2
            - 2 * einsum("ji,pj->ip", tau305, tau56)
            - einsum("qp,pqi->ip", tau364, tau929)
            - einsum("pq,qpi->ip", tau546, tau964)
            - einsum("pq,pq,iq->ip", tau101, tau506, a.y3)
            + einsum("pq,pq,pq,iq->ip", tau674, tau74, tau78, a.y3) / 2
            - einsum("pq,pq,pqi->ip", tau79, tau84, tau319)
            + einsum("pq,pq,pq,iq->ip", tau101, tau262, tau73, a.y3)
            + 2 * einsum("pq,pq,iq->ip", tau678, tau78, a.y4)
            + einsum("pq,qpi->ip", tau524, tau999) / 2
            + einsum("qp,pqi->ip", tau152, tau886) / 2
            - einsum("pq,pq,pqi->ip", tau74, tau78, tau956) / 2
            - einsum("qi,pq,pq,pq->ip", tau321, tau78, tau79, tau84)
            + einsum("qp,pqi->ip", tau467, tau849) / 2
            + einsum("pq,qpi->ip", tau515, tau845) / 2
            + einsum("pq,pq,pqi->ip", tau74, tau78, tau982)
            + einsum("pq,qpi->ip", tau231, tau886) / 2
            + 2 * einsum("pq,qpi->ip", tau227, tau884)
            - einsum("qp,pqi->ip", tau610, tau994) / 4
            + einsum("pq,qpi->ip", tau469, tau951) / 2
            - einsum("qp,qpi->ip", tau413, tau1005) / 4
            + einsum("pq,qpi->ip", tau266, tau1009) / 2
            + einsum("pq,qi->ip", tau472, tau972) / 2
            + einsum("pj,pij->ip", tau712, tau37)
            - einsum("pq,pq,pq,iq->ip", tau101, tau139, tau73, a.y4) / 2
            + einsum("pq,pq,pq,qi->ip", tau101, tau73, tau74, tau865)
            + einsum("qp,pqi->ip", tau115, tau864) / 2
            + einsum("pq,pq,pq,qi->ip", tau78, tau79, tau84, tau945)
            - einsum("pq,pq,pqi->ip", tau78, tau84, tau993) / 2
            - einsum("pq,pq,pq,iq->ip", tau101, tau637, tau79, a.y4) / 2
            - 2 * einsum("pq,pq,pq,iq->ip", tau101, tau375, tau73, a.y3)
            - 2 * einsum("pq,pq,pq,iq->ip", tau73, tau74, tau793, a.y3)
            + einsum("pq,qpi->ip", tau166, tau914) / 2
            + einsum("qp,pqi->ip", tau575, tau1000) / 2
            - einsum("qp,pqi->ip", tau468, tau911) / 4
            - 2 * einsum("pq,pq,iq->ip", tau78, tau796, a.y4)
            + einsum("pq,qi,pq,pq->ip", tau101, tau486, tau73, tau79)
            + einsum("qp,pqi->ip", tau542, tau1004) / 2
            + einsum("qp,pqi->ip", tau256, tau1009) / 2
            + einsum("pq,pq,pqi->ip", tau74, tau78, tau943)
            - 2 * einsum("pq,pq,iq->ip", tau73, tau804, a.y3)
            - einsum("pq,qpi->ip", tau307, tau886)
            - einsum("pq,pq,iq->ip", tau719, tau78, a.y3) / 2
            + einsum("pq,pq,iq->ip", tau698, tau73, a.y4) / 2
            + einsum("qp,pqi->ip", tau153, tau999) / 2
            + einsum("pq,pq,pqi->ip", tau74, tau78, tau784) / 2
            + einsum("p,pi->ip", tau29, tau712)
            + einsum("pq,qi->ip", tau446, tau986) / 2
            - einsum("qp,pqi->ip", tau197, tau832) / 4
            + einsum("pq,qpi->ip", tau469, tau897) / 2
            + einsum("pq,pq,pq,iq->ip", tau78, tau79, tau820, a.y4)
            - einsum("pq,pq,pq,iq->ip", tau74, tau78, tau781, a.y3)
            + einsum("pq,pq,iq->ip", tau706, tau73, a.y3)
            - einsum("pq,qpi->ip", tau390, tau845) / 4
            - einsum("qp,pqi->ip", tau407, tau1009) / 4
        )

        return self.types.RESIDUALS_TYPE(rt1, rz1,
                                         rx1, rx2, rx3, rx4,
                                         ry1, ry2, ry3, ry4)

    def solve_amps(self, h, a, g):
        """
        Solving for new T amlitudes using RHS and denominator
        It is assumed that the order of fields in the RHS
        is consistent with the order in amplitudes
        """
        pass
            
    def update_rhs(self, h, a, r):
        """
        Updates right hand side of the CC equations, commonly referred as G
        """
        pass
            
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
    rhf = scf.RHF(mol)
    rhf = scf.density_fit(scf.RHF(mol))
    rhf.scf()  # -76.0267656731
    from tcc.rccsd_cpd import RCCSD_CPD
    from tcc.cc_solvers import root_solver
    
    cc = RCCSD_CPD(rhf, rankt = 6)
    converged, energy, amps = root_solver(
        cc,
        options={'nit':50})

if __name__ == '__main__':
    test_cc()
