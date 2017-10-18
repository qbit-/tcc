from numpy import einsum
from tcc.tensors import Tensors


def _rccsdt_cpd_ls_t_calculate_energy(h, a):
    tau0 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau1 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau2 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau3 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau4 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau5 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau6 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau7 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau6)
    )

    energy = (
        - einsum("pw,pw->", tau0, tau1)
        + 2 * einsum("pw,pw->", tau2, tau3)
        + 2 * einsum("w,w->", tau4, tau5)
        + einsum("ai,ia->", a.t1, tau7)
    )
    return energy


def _rccsdt_cpd_ls_t_calc_residuals(h, a):

    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau2 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau3 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau4 = (
        einsum("ai,piw->paw", a.t1, tau3)
    )

    tau5 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("aj,wia->ijw", a.t1, h.l.pov)
    )

    tau6 = (
        einsum("jp,jiw->piw", a.t2.x3, tau5)
    )

    tau7 = (
        - einsum("ai,pw->piaw", a.t1, tau2)
        + 2 * einsum("ip,paw->piaw", a.t2.x3, tau4)
        + 2 * einsum("ap,piw->piaw", a.t2.x2, tau6)
    )

    tau8 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau9 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau10 = (
        einsum("ap,piw->piaw", a.t2.x2, tau9)
        + einsum("ip,paw->piaw", a.t2.x4, tau4)
    )

    tau11 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau12 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau11, h.l.pov)
    )

    tau13 = (
        einsum("ap,ia->pi", a.t2.x2, tau12)
    )

    tau14 = (
        2 * einsum("ip,jp->pij", a.t2.x3, a.t2.x4)
        - einsum("jp,ip->pij", a.t2.x3, a.t2.x4)
    )

    tau15 = (
        einsum("paw,piaw->pi", tau1, tau7)
        - einsum("paw,piaw->pi", tau8, tau10)
        - einsum("pj,pij->pi", tau13, tau14)
    )

    tau16 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau17 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau18 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau0, h.l.pov)
        - einsum("wib,baw->ia", h.l.pov, tau17)
    )

    tau19 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau20 = (
        - einsum("ap,piw->piaw", a.t2.x1, tau3)
        + 2 * einsum("ap,piw->piaw", a.t2.x2, tau19)
    )

    tau21 = (
        einsum("paw,piaw->pi", tau1, tau20)
    )

    tau22 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau0, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau16)
        + einsum("aj,ia->ij", a.t1, tau18)
        + einsum("jp,pi->ij", a.t2.x3, tau21)
    )

    tau23 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau24 = (
        2 * einsum("ap,bp->pab", a.t3.x2, a.t3.x3)
        - einsum("bp,ap->pab", a.t3.x2, a.t3.x3)
    )

    tau25 = (
        einsum("pbw,pab->paw", tau23, tau24)
    )

    tau26 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau27 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau28 = (
        einsum("ip,paw->piaw", a.t3.x4, tau26)
        - einsum("ip,paw->piaw", a.t3.x5, tau27)
    )

    tau29 = (
        einsum("paw,piaw->pi", tau25, tau28)
    )

    tau30 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("w,wab->ab", tau0, h.l.pvv)
        - einsum("wac,cbw->ab", h.l.pvv, tau17)
    )

    tau31 = (
        2 * einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        - einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau32 = (
        einsum("pbw,pab->paw", tau1, tau31)
    )

    tau33 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
        - einsum("ip,paw->iaw", a.t2.x3, tau32)
    )

    rt1 = (
        einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("w,wai->ai", tau0, h.l.pvo)
        - einsum("ap,pi->ai", a.t2.x1, tau15)
        - einsum("aj,ji->ai", a.t1, tau22)
        + einsum("ap,pi->ai", a.t3.x1, tau29)
        + einsum("bi,ab->ai", a.t1, tau30)
        - einsum("wab,ibw->ai", h.l.pvv, tau33)
    )
    tau0 = (
        einsum("bp,ab->pa", a.t2.x2, h.f.vv)
    )

    tau1 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau0)
    )

    tau2 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau3 = (
        einsum("aj,ijw->iaw", a.t1, tau2)
    )

    tau4 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau3)
    )

    tau5 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau6 = (
        einsum("pw,wai->pia", tau5, h.l.pvo)
    )

    tau7 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau8 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau2)
    )

    tau9 = (
        einsum("jp,jiw->piw", a.t2.x4, tau8)
    )

    tau10 = (
        einsum("paw,piw->pia", tau7, tau9)
    )

    tau11 = (
        einsum("pw,wab->pab", tau5, h.l.pvv)
    )

    tau12 = (
        einsum("pw,wia->pia", tau5, h.l.pov)
    )

    tau13 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau14 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau15 = (
        einsum("piw,paw->pia", tau13, tau14)
    )

    tau16 = (
        2 * einsum("pia->pia", tau12)
        - einsum("pia->pia", tau15)
    )

    tau17 = (
        einsum("ai,pib->pab", a.t1, tau16)
    )

    tau18 = (
        2 * einsum("pab->pab", tau11)
        - einsum("pab->pab", tau17)
    )

    tau19 = (
        einsum("bi,pab->pia", a.t1, tau18)
    )

    tau20 = (
        2 * einsum("pia->pia", tau6)
        - einsum("pia->pia", tau10)
        + einsum("pia->pia", tau19)
    )

    tau21 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau22 = (
        einsum("pw,wai->pia", tau21, h.l.pvo)
    )

    tau23 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau24 = (
        einsum("piw,qjw->pqij", tau13, tau23)
    )

    tau25 = (
        - einsum("pqij->pqij", tau24)
        + 2 * einsum("qpij->pqij", tau24)
    )

    tau26 = (
        einsum("iq,jp,pqij->pq", a.t2.x3, a.t2.x4, tau25)
    )

    tau27 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x3, tau26)
    )

    tau28 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau29 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau28)
    )

    tau30 = (
        einsum("pw,abw->pab", tau21, tau29)
    )

    tau31 = (
        einsum("bi,pab->pia", a.t1, tau30)
    )

    tau32 = (
        einsum("pia->pia", tau22)
        + einsum("pia->pia", tau27)
        + einsum("pia->pia", tau31)
    )

    tau33 = (
        einsum("ap,qia->pqi", a.t2.x2, tau16)
    )

    tau34 = (
        einsum("aq,ip,qpi->pqa", a.t2.x1, a.t2.x3, tau33)
    )

    tau35 = (
        einsum("iq,jq,pqa->pija", a.t2.x3, a.t2.x4, tau34)
    )

    tau36 = (
        - einsum("jp,pia->pija", a.t2.x3, tau20)
        + einsum("jp,pia->pija", a.t2.x4, tau32)
        + einsum("pija->pija", tau35)
    )

    tau37 = (
        einsum("ap,pijb->ijab", a.t2.x1, tau36)
    )

    tau38 = (
        - einsum("ijab->ijab", tau1)
        + einsum("ijab->ijab", tau4)
        + einsum("ijba->ijab", tau37)
    )

    tau39 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau40 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau39)
    )

    tau41 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau42 = (
        einsum("piw,paw->pia", tau13, tau41)
    )

    tau43 = (
        einsum("ai,pja->pij", a.t1, tau42)
    )

    tau44 = (
        einsum("aj,pij->pia", a.t1, tau43)
    )

    tau45 = (
        einsum("jp,jiw->piw", a.t2.x3, tau8)
    )

    tau46 = (
        einsum("piw,paw->pia", tau45, tau7)
    )

    tau47 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau48 = (
        einsum("bcw,caw->ab", tau29, tau47)
    )

    tau49 = (
        einsum("bp,ip,ba->pia", a.t2.x2, a.t2.x3, tau48)
    )

    tau50 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau51 = (
        einsum("w,wab->ab", tau50, h.l.pvv)
    )

    tau52 = (
        einsum("w,wia->ia", tau50, h.l.pov)
    )

    tau53 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau52)
    )

    tau54 = (
        einsum("ai,ib->ab", a.t1, tau53)
    )

    tau55 = (
        - 2 * einsum("ab->ab", tau51)
        + einsum("ab->ab", tau54)
    )

    tau56 = (
        einsum("bp,ab->pa", a.t2.x2, tau55)
    )

    tau57 = (
        - einsum("pia->pia", tau44)
        + einsum("pia->pia", tau46)
        + einsum("pia->pia", tau49)
        + einsum("ip,pa->pia", a.t2.x3, tau56)
    )

    tau58 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau57)
    )

    tau59 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau60 = (
        einsum("aj,jiw->iaw", a.t1, tau8)
    )

    tau61 = (
        einsum("iaw,jbw->ijab", tau59, tau60)
    )

    tau62 = (
        einsum("ijab->ijab", tau40)
        + einsum("jiba->ijab", tau58)
        + einsum("ijba->ijab", tau61)
    )

    tau63 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau64 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau63)
    )

    tau65 = (
        einsum("jp,jiw->piw", a.t2.x3, tau64)
    )

    tau66 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau67 = (
        einsum("pjw,piw->pij", tau23, tau66)
    )

    tau68 = (
        einsum("jq,pij->pqi", a.t2.x4, tau67)
    )

    tau69 = (
        - einsum("ap,piw->piaw", a.t2.x1, tau13)
        + 2 * einsum("ap,piw->piaw", a.t2.x2, tau66)
    )

    tau70 = (
        einsum("paw,piaw->pi", tau14, tau69)
    )

    tau71 = (
        einsum("aj,jiw->iaw", a.t1, tau64)
    )

    tau72 = (
        - einsum("ip,pj->ij", a.t2.x3, tau70)
        + einsum("wja,iaw->ij", h.l.pov, tau71)
    )

    tau73 = (
        einsum("iq,kq,qpj->pijk", a.t2.x3, a.t2.x4, tau68)
        + einsum("kp,ij->pijk", a.t2.x4, tau72)
    )

    tau74 = (
        einsum("wib,baw->ia", h.l.pov, tau28)
    )

    tau75 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau76 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau77 = (
        2 * einsum("ap,piw->piaw", a.t2.x1, tau23)
        - einsum("ap,piw->piaw", a.t2.x2, tau76)
    )

    tau78 = (
        einsum("paw,piaw->pi", tau75, tau77)
    )

    tau79 = (
        einsum("wia,jaw->ij", h.l.pov, tau39)
        + einsum("aj,ia->ij", a.t1, tau74)
        - einsum("jp,pi->ij", a.t2.x4, tau78)
    )

    tau80 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau81 = (
        einsum("w,wia->ia", tau80, h.l.pov)
    )

    tau82 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau81)
    )

    tau83 = (
        einsum("ai,ja->ij", a.t1, tau82)
    )

    tau84 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau80, h.l.poo)
        + einsum("ji->ij", tau83)
    )

    tau85 = (
        einsum("jp,ji->pi", a.t2.x4, tau84)
    )

    tau86 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau50, h.l.poo)
        + einsum("ji->ij", tau83)
    )

    tau87 = (
        einsum("jp,ji->pi", a.t2.x3, tau86)
    )

    tau88 = (
        einsum("piw,pjw->pij", tau65, tau9)
        + einsum("kp,pikj->pij", a.t2.x3, tau73)
        + einsum("ip,kp,kj->pij", a.t2.x3, a.t2.x4, tau79)
        - einsum("ip,pj->pij", a.t2.x3, tau85)
        - einsum("jp,pi->pij", a.t2.x4, tau87)
    )

    tau89 = (
        einsum("ip,jq,qpij->pq", a.t2.x4, a.t2.x4, tau25)
    )

    tau90 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau91 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau92 = (
        - einsum("piw,pjw->pij", tau13, tau90)
        + 2 * einsum("pw,wij->pij", tau91, h.l.poo)
    )

    tau93 = (
        - 2 * einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x3, tau89)
        + einsum("aj,pji->pia", a.t1, tau92)
    )

    tau94 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau95 = (
        einsum("pjw,piw->pij", tau23, tau94)
    )

    tau96 = (
        einsum("piw,paw->pia", tau23, tau75)
    )

    tau97 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x3, tau96)
    )

    tau98 = (
        einsum("aj,pij->pia", a.t1, tau95)
        + einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x4, tau97)
    )

    tau99 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau100 = (
        einsum("pw,wij->pij", tau99, h.l.poo)
    )

    tau101 = (
        einsum("qw,pw->pq", tau21, tau99)
    )

    tau102 = (
        einsum("aj,pji->pia", a.t1, tau100)
        + einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x4, tau101)
    )

    tau103 = (
        einsum("ap,pji->pija", a.t2.x2, tau88)
        - einsum("jp,pia->pija", a.t2.x3, tau93)
        + einsum("ip,pja->pija", a.t2.x4, tau98)
        + einsum("jp,pia->pija", a.t2.x4, tau102)
    )

    tau104 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau105 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau106 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau107 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau108 = (
        einsum("paw,pbw->pab", tau106, tau107)
    )

    tau109 = (
        einsum("ip,ia->pa", a.t3.x6, tau74)
    )

    tau110 = (
        einsum("ip,ia->pa", a.t3.x5, tau74)
    )

    tau111 = (
        einsum("jp,jiw->piw", a.t3.x6, tau8)
    )

    tau112 = (
        2 * einsum("piw,paw->pia", tau104, tau105)
        + 2 * einsum("bi,pba->pia", a.t1, tau108)
        + 2 * einsum("ip,pa->pia", a.t3.x5, tau109)
        - einsum("ip,pa->pia", a.t3.x6, tau110)
        - einsum("paw,piw->pia", tau106, tau111)
    )

    tau113 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau114 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau115 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau116 = (
        einsum("pbw,paw->pab", tau107, tau115)
    )

    tau117 = (
        einsum("ip,ia->pa", a.t3.x4, tau74)
    )

    tau118 = (
        einsum("jp,jiw->piw", a.t3.x4, tau64)
    )

    tau119 = (
        einsum("piw,paw->pia", tau113, tau114)
        + einsum("bi,pab->pia", a.t1, tau116)
        + einsum("ip,pa->pia", a.t3.x6, tau117)
        - 2 * einsum("paw,piw->pia", tau107, tau118)
    )

    tau120 = (
        einsum("jp,jiw->piw", a.t3.x5, tau8)
    )

    tau121 = (
        einsum("paw,piw->pia", tau115, tau120)
    )

    tau122 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau123 = (
        einsum("piw,paw->pia", tau118, tau122)
    )

    tau124 = (
        - einsum("jp,pia->pija", a.t3.x4, tau112)
        + einsum("ip,pja->pija", a.t3.x5, tau119)
        + einsum("jp,pia->pija", a.t3.x6, tau121)
        + einsum("ip,pja->pija", a.t3.x6, tau123)
    )

    tau125 = (
        einsum("ap,ia->pi", a.t3.x3, tau82)
    )

    tau126 = (
        2 * einsum("ip,jp->pij", a.t3.x4, a.t3.x6)
        - einsum("jp,ip->pij", a.t3.x4, a.t3.x6)
    )

    tau127 = (
        einsum("pj,pij->pi", tau125, tau126)
    )

    tau128 = (
        einsum("ip,jp,pj->pi", a.t3.x4, a.t3.x5, tau125)
    )

    tau129 = (
        einsum("ap,pjia->pij", a.t3.x3, tau124)
        + einsum("jp,pi->pij", a.t3.x5, tau127)
        - einsum("jp,pi->pij", a.t3.x6, tau128)
    )

    tau130 = (
        2 * einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        - einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau131 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau132 = (
        einsum("pij,pjw->piw", tau130, tau131)
    )

    tau133 = (
        einsum("bp,abw->paw", a.t3.x3, tau29)
    )

    tau134 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau135 = (
        einsum("bp,abw->paw", a.t3.x2, tau29)
    )

    tau136 = (
        - einsum("piw,paw->pia", tau132, tau133)
        + einsum("ip,pw,paw->pia", a.t3.x4, tau134, tau135)
    )

    tau137 = (
        einsum("ap,pij->pija", a.t3.x2, tau129)
        - einsum("jp,pia->pija", a.t3.x6, tau136)
    )

    tau138 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau59)
    )

    tau139 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau140 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau141 = (
        einsum("pij,pjw->piw", tau126, tau140)
    )

    tau142 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x6, h.l.pov)
    )

    tau143 = (
        einsum("paw,piw->pia", tau135, tau141)
        - einsum("ip,pw,paw->pia", a.t3.x4, tau142, tau133)
    )

    tau144 = (
        einsum("pw,wij->pij", tau21, h.l.poo)
    )

    tau145 = (
        einsum("aj,pji->pia", a.t1, tau144)
    )

    tau146 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau147 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau47)
    )

    tau148 = (
        einsum("bp,abw->paw", a.t2.x2, tau147)
    )

    tau149 = (
        einsum("paw,piw->pia", tau7, tau76)
        - einsum("aj,pij->pia", a.t1, tau67)
    )

    tau150 = (
        einsum("paw,pbw->pab", tau146, tau148)
        - einsum("ai,pib->pab", a.t1, tau149)
    )

    tau151 = (
        einsum("bp,pia->piab", a.t2.x1, tau145)
        + einsum("ip,pab->piab", a.t2.x3, tau150)
    )

    tau152 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau153 = (
        einsum("piw,pjw->pij", tau13, tau152)
    )

    tau154 = (
        einsum("aj,pji->pia", a.t1, tau153)
    )

    tau155 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau156 = (
        2 * einsum("pw,wij->pij", tau5, h.l.poo)
        - einsum("pjw,piw->pij", tau155, tau23)
    )

    tau157 = (
        einsum("aj,pji->pia", a.t1, tau156)
    )

    tau158 = (
        einsum("jp,pia->pija", a.t2.x4, tau154)
        - einsum("ip,pja->pija", a.t2.x3, tau157)
    )

    rt2 = (
        - einsum("ijab->abij", tau38)
        - einsum("jiba->abij", tau38)
        - einsum("ijba->abij", tau62)
        - einsum("jiab->abij", tau62)
        + einsum("ap,pjib->abij", a.t2.x1, tau103)
        + einsum("ap,pijb->abij", a.t3.x1, tau137)
        + einsum("iaw,jbw->abij", tau138, tau139)
        + einsum("jbw,iaw->abij", tau60, tau71)
        + einsum("bp,ip,pja->abij", a.t3.x1, a.t3.x5, tau143)
        + einsum("jp,piab->abij", a.t2.x4, tau151)
        + einsum("bp,pjia->abij", a.t2.x1, tau158)
    )
    tau0 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau1 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau1)
    )

    tau3 = (
        einsum("pw,abw->pab", tau0, tau2)
    )

    tau4 = (
        einsum("bq,pab->pqa", a.t2.x2, tau3)
    )

    tau5 = (
        einsum("bq,iq,qpa->piab", a.t2.x1, a.t2.x4, tau4)
    )

    tau6 = (
        - einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau5)
    )

    tau7 = (
        einsum("bp,abw->paw", a.t2.x2, tau2)
    )

    tau8 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau9 = (
        einsum("iq,piw->pqw", a.t2.x4, tau8)
    )

    tau10 = (
        einsum("paw,qpw->pqa", tau7, tau9)
    )

    tau11 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau12 = (
        einsum("pw,abw->pab", tau11, tau2)
    )

    tau13 = (
        einsum("bq,pab->pqa", a.t2.x2, tau12)
    )

    tau14 = (
        einsum("pqa->pqa", tau10)
        - 2 * einsum("pqa->pqa", tau13)
    )

    tau15 = (
        einsum("bq,iq,qpa->piab", a.t2.x1, a.t2.x3, tau14)
    )

    tau16 = (
        einsum("iq,piw->pqw", a.t2.x3, tau8)
    )

    tau17 = (
        einsum("qpw,paw->pqa", tau16, tau7)
    )

    tau18 = (
        einsum("bq,iq,jq,pqa->pijab", a.t2.x1, a.t2.x3, a.t2.x4, tau17)
    )

    tau19 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau20 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau19)
    )

    tau21 = (
        einsum("ai,piw->paw", a.t1, tau8)
    )

    tau22 = (
        einsum("iaw,pbw->piab", tau20, tau21)
    )

    tau23 = (
        einsum("pijab->pijab", tau18)
        + einsum("jp,piba->pijab", a.t2.x3, tau22)
    )

    tau24 = (
        einsum("bp,ip,pjac->pijabc", a.t2.x1, a.t2.x3, tau15)
        + einsum("cp,pijab->pijabc", a.t2.x1, tau23)
    )

    tau25 = (
        einsum("kp,pijabc->ijkabc", a.t2.x4, tau24)
    )

    tau26 = (
        einsum("bp,abw->paw", a.t2.x1, tau2)
    )

    tau27 = (
        einsum("pqw,paw->pqa", tau16, tau26)
    )

    tau28 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau29 = (
        einsum("iq,piw->pqw", a.t2.x3, tau28)
    )

    tau30 = (
        einsum("pqw,paw->pqa", tau29, tau7)
    )

    tau31 = (
        einsum("pqa->pqa", tau27)
        - einsum("pqa->pqa", tau30)
    )

    tau32 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau31)
    )

    tau33 = (
        einsum("pw,wij->pij", tau0, h.l.poo)
    )

    tau34 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau35 = (
        einsum("piw,pjw->pij", tau34, tau8)
    )

    tau36 = (
        einsum("pij->pij", tau33)
        - einsum("pji->pij", tau35)
    )

    tau37 = (
        einsum("jq,pji->pqi", a.t2.x3, tau36)
    )

    tau38 = (
        - einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau37)
    )

    tau39 = (
        - einsum("bp,pija->pijab", a.t2.x1, tau32)
        - einsum("ap,pijb->pijab", a.t2.x1, tau38)
    )

    tau40 = (
        einsum("cp,kp,pijab->ijkabc", a.t2.x2, a.t2.x4, tau39)
    )

    tau41 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau42 = (
        einsum("pw,wij->pij", tau41, h.l.poo)
    )

    tau43 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau44 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau45 = (
        einsum("pjw,piw->pij", tau43, tau44)
    )

    tau46 = (
        einsum("pij->pij", tau42)
        - einsum("pji->pij", tau45)
    )

    tau47 = (
        einsum("aj,pji->pia", a.t1, tau46)
    )

    tau48 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau47)
    )

    tau49 = (
        - einsum("ikjabc->ijkabc", tau40)
        - einsum("ijkabc->ijkabc", tau48)
    )

    tau50 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau51 = (
        einsum("pw,wia->pia", tau50, h.l.pov)
    )

    tau52 = (
        einsum("ip,pia->pa", a.t3.x4, tau51)
    )

    tau53 = (
        einsum("ap,qa->pq", a.t2.x2, tau52)
    )

    tau54 = (
        einsum("aq,iq,jq,qp->pija", a.t2.x1, a.t2.x3, a.t2.x4, tau53)
    )

    tau55 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau56 = (
        einsum("pw,wai->pia", tau55, h.l.pvo)
    )

    tau57 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau58 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau57)
    )

    tau59 = (
        einsum("jp,jiw->piw", a.t3.x6, tau58)
    )

    tau60 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau61 = (
        einsum("piw,paw->pia", tau59, tau60)
    )

    tau62 = (
        einsum("iq,piw->pqw", a.t3.x6, tau8)
    )

    tau63 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau64 = (
        einsum("ip,qiw->pqw", a.t2.x4, tau63)
    )

    tau65 = (
        einsum("ip,qiw->pqw", a.t2.x3, tau63)
    )

    tau66 = (
        2 * einsum("ip,pqw->pqiw", a.t2.x3, tau64)
        - einsum("ip,pqw->pqiw", a.t2.x4, tau65)
    )

    tau67 = (
        einsum("pqw,pqiw->pqi", tau62, tau66)
    )

    tau68 = (
        2 * einsum("ip,jp->pij", a.t2.x3, a.t2.x4)
        - einsum("jp,ip->pij", a.t2.x3, a.t2.x4)
    )

    tau69 = (
        einsum("pw,wia->pia", tau55, h.l.pov)
    )

    tau70 = (
        einsum("ap,qia->pqi", a.t2.x2, tau69)
    )

    tau71 = (
        einsum("pij,pqj->pqi", tau68, tau70)
    )

    tau72 = (
        einsum("pqi->pqi", tau67)
        - 2 * einsum("pqi->pqi", tau71)
    )

    tau73 = (
        einsum("aq,qpi->pia", a.t2.x1, tau72)
    )

    tau74 = (
        einsum("pw,wab->pab", tau55, h.l.pvv)
    )

    tau75 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau76 = (
        einsum("piw,paw->pia", tau63, tau75)
    )

    tau77 = (
        2 * einsum("pia->pia", tau69)
        - einsum("pia->pia", tau76)
    )

    tau78 = (
        einsum("bi,pia->pab", a.t1, tau77)
    )

    tau79 = (
        2 * einsum("pab->pab", tau74)
        - einsum("pba->pab", tau78)
    )

    tau80 = (
        einsum("bi,pab->pia", a.t1, tau79)
    )

    tau81 = (
        2 * einsum("pia->pia", tau56)
        - einsum("pia->pia", tau61)
        - einsum("pia->pia", tau73)
        + einsum("pia->pia", tau80)
    )

    tau82 = (
        einsum("pw,wai->pia", tau50, h.l.pvo)
    )

    tau83 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau84 = (
        einsum("qw,pw->pq", tau50, tau83)
    )

    tau85 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau86 = (
        einsum("piw,paw->pia", tau8, tau85)
    )

    tau87 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau88 = (
        einsum("pw,wia->pia", tau87, h.l.pov)
    )

    tau89 = (
        - einsum("pia->pia", tau86)
        + 2 * einsum("pia->pia", tau88)
    )

    tau90 = (
        einsum("aq,pia->pqi", a.t3.x3, tau89)
    )

    tau91 = (
        einsum("ip,jq,pqj->pqi", a.t2.x3, a.t3.x5, tau90)
    )

    tau92 = (
        einsum("ip,pq->pqi", a.t2.x4, tau84)
        - einsum("pqi->pqi", tau91)
    )

    tau93 = (
        einsum("aq,qpi->pia", a.t2.x1, tau92)
    )

    tau94 = (
        einsum("pw,abw->pab", tau50, tau2)
    )

    tau95 = (
        einsum("bi,pab->pia", a.t1, tau94)
    )

    tau96 = (
        einsum("pia->pia", tau82)
        - einsum("pia->pia", tau93)
        + einsum("pia->pia", tau95)
    )

    tau97 = (
        - einsum("jp,pia->pija", a.t3.x5, tau81)
        + einsum("jp,pia->pija", a.t3.x6, tau96)
    )

    tau98 = (
        einsum("aq,pia->pqi", a.t2.x2, tau77)
    )

    tau99 = (
        einsum("ap,iq,qpi->pqa", a.t2.x1, a.t3.x4, tau98)
    )

    tau100 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau99)
    )

    tau101 = (
        einsum("kp,pija->pijka", a.t3.x6, tau54)
        - einsum("jp,pika->pijka", a.t3.x4, tau97)
        - einsum("kp,pija->pijka", a.t3.x5, tau100)
    )

    tau102 = (
        einsum("bp,cp,pijka->ijkabc", a.t3.x1, a.t3.x2, tau101)
    )

    tau103 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau104 = (
        einsum("wai,pjw->pija", h.l.pvo, tau103)
    )

    tau105 = (
        einsum("jp,ijw->piw", a.t2.x3, tau57)
    )

    tau106 = (
        einsum("waj,piw->pija", h.l.pvo, tau105)
    )

    tau107 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau108 = (
        einsum("aj,ijw->iaw", a.t1, tau107)
    )

    tau109 = (
        einsum("pjw,iaw->pija", tau105, tau108)
    )

    tau110 = (
        einsum("jp,jiw->piw", a.t2.x3, tau58)
    )

    tau111 = (
        einsum("piw,jaw->pija", tau110, tau19)
    )

    tau112 = (
        einsum("pija->pija", tau104)
        - einsum("pija->pija", tau106)
        - einsum("pija->pija", tau109)
        + einsum("pjia->pija", tau111)
    )

    tau113 = (
        einsum("paw,pqw->pqa", tau26, tau9)
    )

    tau114 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau113)
    )

    tau115 = (
        einsum("kp,pija->pijka", a.t2.x4, tau112)
        - einsum("ip,pjka->pijka", a.t2.x3, tau114)
    )

    tau116 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau117 = (
        einsum("paw,piw->pia", tau116, tau8)
    )

    tau118 = (
        einsum("ai,pja->pij", a.t1, tau117)
    )

    tau119 = (
        einsum("jq,pij->pqi", a.t2.x4, tau118)
    )

    tau120 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau119)
    )

    tau121 = (
        einsum("wib,baw->ia", h.l.pov, tau1)
    )

    tau122 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau123 = (
        einsum("w,wia->ia", tau122, h.l.pov)
    )

    tau124 = (
        einsum("ia->ia", h.f.ov)
        - einsum("ia->ia", tau121)
        + 2 * einsum("ia->ia", tau123)
    )

    tau125 = (
        einsum("ap,ia->pi", a.t2.x2, tau124)
    )

    tau126 = (
        einsum("iq,pi->pq", a.t2.x4, tau125)
    )

    tau127 = (
        einsum("aq,iq,jq,qp->pija", a.t2.x1, a.t2.x3, a.t2.x4, tau126)
    )

    tau128 = (
        einsum("pw,wia->pia", tau11, h.l.pov)
    )

    tau129 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau130 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau131 = (
        einsum("piw,paw->pia", tau129, tau130)
    )

    tau132 = (
        2 * einsum("pia->pia", tau128)
        - einsum("pia->pia", tau131)
    )

    tau133 = (
        einsum("aj,pia->pij", a.t1, tau132)
    )

    tau134 = (
        einsum("jq,pji->pqi", a.t2.x3, tau133)
    )

    tau135 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x3, tau134)
    )

    tau136 = (
        einsum("jp,pika->pijka", a.t2.x3, tau120)
        + einsum("ip,pjka->pijka", a.t2.x3, tau127)
        + einsum("kp,pija->pijka", a.t2.x4, tau135)
    )

    tau137 = (
        einsum("bp,pijka->pijkab", a.t2.x1, tau115)
        + einsum("ap,pijkb->pijkab", a.t2.x1, tau136)
    )

    tau138 = (
        einsum("cp,pijkab->ijkabc", a.t2.x2, tau137)
    )

    tau139 = (
        - einsum("ijkabc->ijkabc", tau102)
        + einsum("ijkabc->ijkabc", tau138)
    )

    tau140 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau141 = (
        einsum("pw,wia->pia", tau140, h.l.pov)
    )

    tau142 = (
        einsum("ip,pia->pa", a.t3.x5, tau141)
    )

    tau143 = (
        einsum("ap,qa->pq", a.t2.x2, tau142)
    )

    tau144 = (
        einsum("aq,iq,jq,qp->pija", a.t2.x1, a.t2.x3, a.t2.x4, tau143)
    )

    tau145 = (
        einsum("pw,wai->pia", tau41, h.l.pvo)
    )

    tau146 = (
        einsum("iq,pia->pqa", a.t3.x4, tau86)
    )

    tau147 = (
        einsum("iq,pia->pqa", a.t3.x4, tau117)
    )

    tau148 = (
        einsum("ip,pqa->pqia", a.t2.x3, tau146)
        - einsum("ip,pqa->pqia", a.t2.x4, tau147)
    )

    tau149 = (
        einsum("bq,qpia->piab", a.t2.x1, tau148)
    )

    tau150 = (
        einsum("ip,pia->pa", a.t2.x3, tau128)
    )

    tau151 = (
        einsum("pw,wia->pia", tau83, h.l.pov)
    )

    tau152 = (
        einsum("ip,pia->pa", a.t2.x4, tau151)
    )

    tau153 = (
        2 * einsum("pa->pa", tau150)
        - einsum("pa->pa", tau152)
    )

    tau154 = (
        einsum("bp,pa->ab", a.t2.x1, tau153)
    )

    tau155 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau156 = (
        einsum("cbw,acw->ab", tau155, tau2)
    )

    tau157 = (
        einsum("ba->ab", tau154)
        + einsum("ab->ab", tau156)
    )

    tau158 = (
        einsum("piba->piab", tau149)
        + einsum("ip,ab->piab", a.t3.x4, tau157)
    )

    tau159 = (
        einsum("bp,piab->pia", a.t3.x3, tau158)
    )

    tau160 = (
        einsum("jp,jiw->piw", a.t3.x4, tau58)
    )

    tau161 = (
        einsum("piw,paw->pia", tau160, tau60)
    )

    tau162 = (
        einsum("pw,wab->pab", tau41, h.l.pvv)
    )

    tau163 = (
        einsum("pw,wia->pia", tau41, h.l.pov)
    )

    tau164 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau165 = (
        einsum("paw,piw->pia", tau164, tau63)
    )

    tau166 = (
        einsum("pia->pia", tau163)
        - einsum("pia->pia", tau165)
    )

    tau167 = (
        einsum("bi,pia->pab", a.t1, tau166)
    )

    tau168 = (
        einsum("pab->pab", tau162)
        - einsum("pba->pab", tau167)
    )

    tau169 = (
        einsum("bi,pab->pia", a.t1, tau168)
    )

    tau170 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau171 = (
        einsum("w,wab->ab", tau170, h.l.pvv)
    )

    tau172 = (
        einsum("w,wia->ia", tau170, h.l.pov)
    )

    tau173 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau172)
    )

    tau174 = (
        einsum("bi,ia->ab", a.t1, tau173)
    )

    tau175 = (
        - 2 * einsum("ab->ab", tau171)
        + einsum("ba->ab", tau174)
    )

    tau176 = (
        einsum("bp,ab->pa", a.t3.x3, tau175)
    )

    tau177 = (
        einsum("ap,qia->pqi", a.t2.x2, tau163)
    )

    tau178 = (
        einsum("pqj,pij->pqi", tau177, tau68)
    )

    tau179 = (
        einsum("aq,qpi->pia", a.t2.x1, tau178)
    )

    tau180 = (
        - einsum("pia->pia", tau145)
        + einsum("pia->pia", tau159)
        + einsum("pia->pia", tau161)
        - einsum("pia->pia", tau169)
        + einsum("ip,pa->pia", a.t3.x4, tau176)
        - einsum("pia->pia", tau179)
    )

    tau181 = (
        einsum("pija->pija", tau144)
        + einsum("jp,pia->pija", a.t3.x5, tau180)
    )

    tau182 = (
        einsum("ap,iq,qpi->pqa", a.t2.x1, a.t3.x5, tau98)
    )

    tau183 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau182)
    )

    tau184 = (
        einsum("kp,pija->pijka", a.t3.x6, tau181)
        - einsum("kp,pija->pijka", a.t3.x4, tau183)
    )

    tau185 = (
        einsum("bp,cp,pijka->ijkabc", a.t3.x1, a.t3.x2, tau184)
    )

    tau186 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau107)
    )

    tau187 = (
        einsum("aj,jiw->iaw", a.t1, tau186)
    )

    tau188 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau19)
        - einsum("iaw->iaw", tau187)
    )

    tau189 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau190 = (
        einsum("iaw,pjw->pija", tau188, tau189)
    )

    tau191 = (
        einsum("jp,ijw->piw", a.t2.x4, tau57)
    )

    tau192 = (
        einsum("bi,abw->iaw", a.t1, tau2)
    )

    tau193 = (
        einsum("piw,jaw->pija", tau191, tau192)
    )

    tau194 = (
        einsum("pija->pija", tau190)
        + einsum("pjia->pija", tau193)
    )

    tau195 = (
        einsum("iq,pi->pq", a.t2.x3, tau125)
    )

    tau196 = (
        einsum("pw,wia->pia", tau0, h.l.pov)
    )

    tau197 = (
        einsum("pia->pia", tau196)
        - einsum("pia->pia", tau117)
    )

    tau198 = (
        einsum("aj,pia->pij", a.t1, tau197)
    )

    tau199 = (
        einsum("jq,pji->pqi", a.t2.x3, tau198)
    )

    tau200 = (
        einsum("ip,pq->pqi", a.t2.x3, tau195)
        + einsum("pqi->pqi", tau199)
    )

    tau201 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau200)
    )

    tau202 = (
        - einsum("bp,kp,pija->pijkab", a.t2.x1, a.t2.x3, tau194)
        + einsum("ap,jp,pikb->pijkab", a.t2.x1, a.t2.x4, tau201)
    )

    tau203 = (
        einsum("cp,pijkab->ijkabc", a.t2.x2, tau202)
    )

    tau204 = (
        einsum("ijkabc->ijkabc", tau185)
        + einsum("ijkabc->ijkabc", tau203)
    )

    tau205 = (
        einsum("paw,pbw->pab", tau116, tau130)
    )

    tau206 = (
        einsum("bp,qab->pqa", a.t2.x2, tau205)
    )

    tau207 = (
        einsum("ai,pqa->pqi", a.t1, tau206)
    )

    tau208 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau207)
    )

    tau209 = (
        einsum("bp,jp,kp,piac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau208)
    )

    tau210 = (
        einsum("waj,piw->pija", h.l.pvo, tau191)
    )

    tau211 = (
        einsum("piw,pjw->pij", tau129, tau189)
    )

    tau212 = (
        einsum("pw,wij->pij", tau87, h.l.poo)
    )

    tau213 = (
        2 * einsum("pia->pia", tau128)
        - einsum("pia->pia", tau86)
    )

    tau214 = (
        einsum("aj,pia->pij", a.t1, tau213)
    )

    tau215 = (
        - einsum("pij->pij", tau211)
        + 2 * einsum("pij->pij", tau212)
        + einsum("pij->pij", tau214)
    )

    tau216 = (
        einsum("jq,pji->pqi", a.t2.x4, tau215)
    )

    tau217 = (
        einsum("pw,ijw->pij", tau83, tau58)
    )

    tau218 = (
        einsum("jq,pji->pqi", a.t2.x4, tau217)
    )

    tau219 = (
        - einsum("jp,pqi->pqij", a.t2.x3, tau216)
        + einsum("jp,pqi->pqij", a.t2.x4, tau218)
    )

    tau220 = (
        einsum("aq,qpij->pija", a.t2.x1, tau219)
    )

    tau221 = (
        - einsum("bp,kp,pija->pijkab", a.t2.x1, a.t2.x3, tau210)
        + einsum("ap,jp,pikb->pijkab", a.t2.x1, a.t2.x3, tau220)
    )

    tau222 = (
        einsum("cp,pijkab->ijkabc", a.t2.x2, tau221)
    )

    tau223 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x5, tau117)
    )

    tau224 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x4, tau223)
    )

    tau225 = (
        einsum("jp,jiw->piw", a.t3.x5, tau58)
    )

    tau226 = (
        einsum("bp,abw->paw", a.t3.x3, tau2)
    )

    tau227 = (
        einsum("piw,paw->pia", tau225, tau226)
    )

    tau228 = (
        einsum("pia->pia", tau224)
        - einsum("pia->pia", tau227)
    )

    tau229 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau228)
    )

    tau230 = (
        - einsum("ijkabc->ijkabc", tau222)
        + einsum("ijkabc->ijkabc", tau229)
    )

    tau231 = (
        einsum("bp,qba->pqa", a.t2.x2, tau205)
    )

    tau232 = (
        einsum("ai,pqa->pqi", a.t1, tau231)
    )

    tau233 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau232)
    )

    tau234 = (
        einsum("bp,jp,kp,piac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau233)
    )

    tau235 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x4, h.l.pov)
    )

    tau236 = (
        einsum("pw,wia->pia", tau235, h.l.pov)
    )

    tau237 = (
        einsum("ap,pia->pi", a.t3.x3, tau236)
    )

    tau238 = (
        einsum("ip,qi->pq", a.t2.x4, tau237)
    )

    tau239 = (
        einsum("aq,bq,iq,qp->piab", a.t2.x1, a.t2.x2, a.t2.x3, tau238)
    )

    tau240 = (
        einsum("bp,wab->paw", a.t3.x2, h.l.pvv)
    )

    tau241 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau242 = (
        einsum("paw,pbw->pab", tau240, tau241)
    )

    tau243 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau244 = (
        einsum("piw,pjw->pij", tau243, tau43)
    )

    tau245 = (
        einsum("ip,jp,qij->pq", a.t2.x3, a.t2.x4, tau244)
    )

    tau246 = (
        einsum("aq,bq,qp->pab", a.t2.x1, a.t2.x2, tau245)
    )

    tau247 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau248 = (
        einsum("piw,paw->pia", tau247, tau60)
    )

    tau249 = (
        einsum("aj,pij->pia", a.t1, tau244)
    )

    tau250 = (
        einsum("pia->pia", tau248)
        - einsum("pia->pia", tau249)
    )

    tau251 = (
        einsum("bi,pia->pab", a.t1, tau250)
    )

    tau252 = (
        einsum("pab->pab", tau242)
        + einsum("pab->pab", tau246)
        - einsum("pba->pab", tau251)
    )

    tau253 = (
        - einsum("piab->piab", tau239)
        + einsum("ip,pab->piab", a.t3.x4, tau252)
    )

    tau254 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau255 = (
        einsum("paw,piw->pia", tau254, tau63)
    )

    tau256 = (
        - einsum("pia->pia", tau51)
        + 2 * einsum("pia->pia", tau255)
    )

    tau257 = (
        einsum("iq,pia->pqa", a.t2.x4, tau256)
    )

    tau258 = (
        einsum("ip,aq,qpa->pqi", a.t2.x3, a.t3.x2, tau257)
    )

    tau259 = (
        einsum("aq,bq,qpi->piab", a.t2.x1, a.t2.x2, tau258)
    )

    tau260 = (
        einsum("jp,piab->pijab", a.t3.x5, tau253)
        + einsum("jp,piab->pijab", a.t3.x4, tau259)
    )

    tau261 = (
        einsum("cp,kp,pijab->ijkabc", a.t3.x1, a.t3.x6, tau260)
    )

    tau262 = (
        - einsum("ap,piw->piaw", a.t3.x2, tau63)
        + 2 * einsum("ap,piw->piaw", a.t3.x3, tau243)
    )

    tau263 = (
        einsum("paw,piaw->pi", tau75, tau262)
    )

    tau264 = (
        einsum("iq,ap,pi->pqa", a.t2.x3, a.t3.x1, tau263)
    )

    tau265 = (
        einsum("iq,jq,qpa->pija", a.t3.x4, a.t3.x5, tau264)
    )

    tau266 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau265)
    )

    tau267 = (
        2 * einsum("ap,piw->piaw", a.t3.x2, tau63)
        - einsum("ap,piw->piaw", a.t3.x3, tau243)
    )

    tau268 = (
        einsum("paw,piaw->pi", tau254, tau267)
    )

    tau269 = (
        einsum("iq,ap,pi->pqa", a.t2.x3, a.t3.x1, tau268)
    )

    tau270 = (
        einsum("iq,jq,qpa->pija", a.t3.x4, a.t3.x6, tau269)
    )

    tau271 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau270)
    )

    tau272 = (
        einsum("aj,jiw->iaw", a.t1, tau58)
    )

    tau273 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau274 = (
        einsum("iaw,pbw->piab", tau272, tau273)
    )

    tau275 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau274)
    )

    tau276 = (
        einsum("ai,piw->paw", a.t1, tau129)
    )

    tau277 = (
        einsum("aj,ijw->iaw", a.t1, tau57)
    )

    tau278 = (
        einsum("paw,ibw->piab", tau276, tau277)
    )

    tau279 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau280 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau279)
    )

    tau281 = (
        einsum("pbw,iaw->piab", tau273, tau280)
    )

    tau282 = (
        einsum("piab->piab", tau278)
        + einsum("piba->piab", tau281)
    )

    tau283 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau282)
    )

    tau284 = (
        einsum("iaw,pbw->piab", tau108, tau21)
    )

    tau285 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau286 = (
        einsum("iaw,pbw->piab", tau20, tau285)
    )

    tau287 = (
        einsum("piab->piab", tau284)
        + einsum("piab->piab", tau286)
    )

    tau288 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau287)
    )

    tau289 = (
        einsum("jq,pij->pqi", a.t2.x4, tau35)
    )

    tau290 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau289)
    )

    tau291 = (
        einsum("pjw,iaw->pija", tau103, tau187)
    )

    tau292 = (
        einsum("jp,jiw->piw", a.t2.x3, tau186)
    )

    tau293 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau294 = (
        einsum("piw,jaw->pija", tau292, tau293)
    )

    tau295 = (
        einsum("pija->pija", tau291)
        - einsum("pija->pija", tau294)
    )

    tau296 = (
        einsum("pw,wij->pij", tau11, h.l.poo)
    )

    tau297 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau298 = (
        einsum("piw,pjw->pij", tau297, tau8)
    )

    tau299 = (
        2 * einsum("pij->pij", tau296)
        - einsum("pji->pij", tau298)
    )

    tau300 = (
        einsum("jq,pji->pqi", a.t2.x3, tau299)
    )

    tau301 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x3, tau300)
    )

    tau302 = (
        - einsum("bp,pija->pijab", a.t2.x1, tau295)
        + einsum("ap,pijb->pijab", a.t2.x1, tau301)
    )

    tau303 = (
        einsum("ap,jp,pikb->pijkab", a.t2.x1, a.t2.x3, tau290)
        + einsum("kp,pijab->pijkab", a.t2.x4, tau302)
    )

    tau304 = (
        einsum("cp,pijkab->ijkabc", a.t2.x2, tau303)
    )

    tau305 = (
        einsum("pw,wij->pij", tau50, h.l.poo)
    )

    tau306 = (
        einsum("aj,pji->pia", a.t1, tau305)
    )

    tau307 = (
        einsum("pw,wij->pij", tau55, h.l.poo)
    )

    tau308 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau309 = (
        einsum("piw,pjw->pij", tau308, tau43)
    )

    tau310 = (
        2 * einsum("pij->pij", tau307)
        - einsum("pji->pij", tau309)
    )

    tau311 = (
        einsum("aj,pji->pia", a.t1, tau310)
    )

    tau312 = (
        einsum("jp,pia->pija", a.t3.x6, tau306)
        - einsum("jp,pia->pija", a.t3.x5, tau311)
    )

    tau313 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t3.x1, a.t3.x2, a.t3.x4, tau312)
    )

    tau314 = (
        einsum("ijkabc->ijkabc", tau304)
        - einsum("ikjabc->ijkabc", tau313)
    )

    tau315 = (
        einsum("pjw,piw->pij", tau103, tau129)
    )

    tau316 = (
        einsum("jq,pji->pqi", a.t2.x4, tau315)
    )

    tau317 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau316)
    )

    tau318 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau319 = (
        einsum("piw,jaw->pija", tau105, tau318)
    )

    tau320 = (
        einsum("iaw,pjw->pija", tau277, tau34)
    )

    tau321 = (
        einsum("pija->pija", tau319)
        - einsum("pija->pija", tau320)
    )

    tau322 = (
        - einsum("pij->pij", tau211)
        + 2 * einsum("pij->pij", tau212)
    )

    tau323 = (
        einsum("jq,pji->pqi", a.t2.x3, tau322)
    )

    tau324 = (
        einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x3, tau323)
    )

    tau325 = (
        einsum("bp,pija->pijab", a.t2.x1, tau321)
        + einsum("ap,pijb->pijab", a.t2.x1, tau324)
    )

    tau326 = (
        einsum("ap,jp,pikb->pijkab", a.t2.x1, a.t2.x3, tau317)
        + einsum("kp,pijab->pijkab", a.t2.x4, tau325)
    )

    tau327 = (
        einsum("cp,pijkab->ijkabc", a.t2.x2, tau326)
    )

    tau328 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau329 = (
        einsum("pw,wij->pij", tau328, h.l.poo)
    )

    tau330 = (
        einsum("aj,pji->pia", a.t1, tau329)
    )

    tau331 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau332 = (
        einsum("pjw,piw->pij", tau331, tau63)
    )

    tau333 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau334 = (
        einsum("pw,wij->pij", tau333, h.l.poo)
    )

    tau335 = (
        - einsum("pij->pij", tau332)
        + 2 * einsum("pij->pij", tau334)
    )

    tau336 = (
        einsum("aj,pji->pia", a.t1, tau335)
    )

    tau337 = (
        einsum("jp,pia->pija", a.t3.x6, tau330)
        - einsum("jp,pia->pija", a.t3.x5, tau336)
    )

    tau338 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t3.x1, a.t3.x2, a.t3.x4, tau337)
    )

    tau339 = (
        einsum("ijkabc->ijkabc", tau327)
        - einsum("ikjabc->ijkabc", tau338)
    )

    tau340 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau341 = (
        einsum("pjw,piw->pij", tau340, tau63)
    )

    tau342 = (
        einsum("pw,wij->pij", tau140, h.l.poo)
    )

    tau343 = (
        einsum("pij->pij", tau341)
        - einsum("pij->pij", tau342)
    )

    tau344 = (
        einsum("aj,pji->pia", a.t1, tau343)
    )

    tau345 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau344)
    )

    tau346 = (
        einsum("pw,wij->pij", tau83, h.l.poo)
    )

    tau347 = (
        einsum("pij->pij", tau315)
        - einsum("pij->pij", tau346)
    )

    tau348 = (
        einsum("jq,pji->pqi", a.t2.x3, tau347)
    )

    tau349 = (
        - einsum("aq,jq,qpi->pija", a.t2.x1, a.t2.x4, tau348)
    )

    tau350 = (
        - einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau349)
    )

    tau351 = (
        - einsum("ijkabc->ijkabc", tau345)
        - einsum("ikjbac->ijkabc", tau350)
    )

    tau352 = (
        einsum("paw,piw->pia", tau116, tau189)
    )

    tau353 = (
        einsum("ap,qia->pqi", a.t2.x2, tau352)
    )

    tau354 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau353)
    )

    tau355 = (
        einsum("bp,jp,kp,piac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau354)
    )

    tau356 = (
        einsum("paw,piw->pia", tau130, tau34)
    )

    tau357 = (
        einsum("ap,qia->pqi", a.t2.x2, tau356)
    )

    tau358 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau357)
    )

    tau359 = (
        einsum("bp,jp,kp,piac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau358)
    )

    tau360 = (
        einsum("piw,paw->pia", tau103, tau85)
    )

    tau361 = (
        einsum("ap,qia->pqi", a.t2.x2, tau360)
    )

    tau362 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau361)
    )

    tau363 = (
        einsum("bp,jp,kp,piac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau362)
    )

    tau364 = (
        einsum("paw,ibw->piab", tau276, tau293)
    )

    tau365 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau364)
    )

    tau366 = (
        einsum("pbw,iaw->piab", tau21, tau318)
    )

    tau367 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau366)
    )

    tau368 = (
        einsum("piw,jaw->pija", tau191, tau318)
    )

    tau369 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau368)
    )

    tau370 = (
        einsum("jp,ijw->piw", a.t2.x4, tau107)
    )

    tau371 = (
        einsum("jaw,piw->pija", tau293, tau370)
    )

    tau372 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau371)
    )

    tau373 = (
        einsum("bp,ab->pa", a.t3.x3, h.f.vv)
    )

    tau374 = (
        einsum("ap,bp,ip,jp,kp,pc->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau373)
    )

    tau375 = (
        einsum("paw,piw->pia", tau240, tau43)
    )

    tau376 = (
        einsum("ai,pib->pab", a.t1, tau375)
    )

    tau377 = (
        einsum("cp,ip,jp,kp,pab->ijkabc", a.t3.x1,
               a.t3.x4, a.t3.x5, a.t3.x6, tau376)
    )

    tau378 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau379 = (
        einsum("paw,piw->pia", tau164, tau378)
    )

    tau380 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau381 = (
        einsum("pbw,paw->pab", tau254, tau380)
    )

    tau382 = (
        einsum("bi,pab->pia", a.t1, tau381)
    )

    tau383 = (
        einsum("ai,ja->ij", a.t1, tau121)
    )

    tau384 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau385 = (
        einsum("piw,pjw->pij", tau384, tau8)
    )

    tau386 = (
        einsum("pij->pij", tau385)
        - einsum("piw,pjw->pij", tau129, tau28)
    )

    tau387 = (
        einsum("jq,pij->pqi", a.t3.x5, tau386)
    )

    tau388 = (
        einsum("wia,jaw->ij", h.l.pov, tau293)
    )

    tau389 = (
        2 * einsum("ap,piw->piaw", a.t2.x1, tau8)
        - einsum("ap,piw->piaw", a.t2.x2, tau28)
    )

    tau390 = (
        einsum("paw,piaw->pi", tau116, tau389)
    )

    tau391 = (
        einsum("jp,pi->ij", a.t2.x4, tau390)
    )

    tau392 = (
        einsum("ij->ij", tau388)
        - einsum("ij->ij", tau391)
    )

    tau393 = (
        2 * einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        - einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau394 = (
        einsum("pab,pbw->paw", tau393, tau85)
    )

    tau395 = (
        einsum("ip,paw->iaw", a.t2.x3, tau394)
    )

    tau396 = (
        einsum("iaw->iaw", tau318)
        - einsum("iaw->iaw", tau395)
    )

    tau397 = (
        einsum("wja,iaw->ij", h.l.pov, tau396)
    )

    tau398 = (
        einsum("kp,ji->pijk", a.t3.x5, tau383)
        - einsum("jp,ki->pijk", a.t3.x5, tau383)
        - einsum("kq,jq,qpi->pijk", a.t2.x3, a.t2.x4, tau387)
        + einsum("kp,ij->pijk", a.t3.x5, tau392)
        - einsum("jp,ki->pijk", a.t3.x5, tau397)
    )

    tau399 = (
        einsum("kp,ji->pijk", a.t3.x4, tau383)
        - einsum("jp,ki->pijk", a.t3.x4, tau383)
        + einsum("kp,ij->pijk", a.t3.x4, tau392)
        - einsum("jp,ki->pijk", a.t3.x4, tau397)
    )

    tau400 = (
        einsum("jp,jiw->piw", a.t3.x4, tau186)
    )

    tau401 = (
        einsum("jp,jiw->piw", a.t3.x5, tau186)
    )

    tau402 = (
        einsum("w,wij->ij", tau170, h.l.poo)
    )

    tau403 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau123)
    )

    tau404 = (
        einsum("aj,ia->ij", a.t1, tau403)
    )

    tau405 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau402)
        + einsum("ij->ij", tau404)
    )

    tau406 = (
        einsum("jp,ji->pi", a.t3.x4, tau405)
    )

    tau407 = (
        einsum("jp,ji->pi", a.t3.x5, tau405)
    )

    tau408 = (
        einsum("w,wij->ij", tau122, h.l.poo)
    )

    tau409 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau408)
        + einsum("ij->ij", tau404)
    )

    tau410 = (
        einsum("jp,ji->pi", a.t3.x5, tau409)
    )

    tau411 = (
        einsum("jp,ji->pi", a.t3.x4, tau409)
    )

    tau412 = (
        - einsum("aj,pia->pij", a.t1, tau379)
        - einsum("aj,pia->pij", a.t1, tau382)
        - einsum("kp,pkji->pij", a.t3.x4, tau398)
        + einsum("kp,pkji->pij", a.t3.x5, tau399)
        + einsum("pjw,piw->pij", tau225, tau400)
        - einsum("pjw,piw->pij", tau340, tau401)
        - einsum("jp,pi->pij", a.t3.x5, tau406)
        + einsum("jp,pi->pij", a.t3.x4, tau407)
        - einsum("ip,pj->pij", a.t3.x4, tau410)
        + einsum("ip,pj->pij", a.t3.x5, tau411)
    )

    tau413 = (
        einsum("ij->ij", h.f.oo)
        - einsum("ij->ij", tau388)
        + 2 * einsum("ij->ij", tau408)
        + einsum("aj,ia->ij", a.t1, tau124)
        + einsum("ij->ij", tau391)
    )

    tau414 = (
        einsum("jp,ji->pi", a.t3.x6, tau413)
    )

    tau415 = (
        - einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        + einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau416 = (
        einsum("iq,jq,pij->pq", a.t3.x5, a.t3.x6, tau385)
    )

    tau417 = (
        einsum("iq,jq,qp->pij", a.t2.x3, a.t2.x4, tau416)
    )

    tau418 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau419 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau420 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau421 = (
        einsum("paw,pbw->pab", tau420, tau75)
    )

    tau422 = (
        einsum("paw,piw->pia", tau418, tau419)
        + einsum("bi,pba->pia", a.t1, tau421)
    )

    tau423 = (
        einsum("pij->pij", tau417)
        + einsum("pjw,piw->pij", tau331, tau401)
        + einsum("aj,pia->pij", a.t1, tau422)
    )

    tau424 = (
        einsum("iq,jq,pij->pq", a.t3.x4, a.t3.x6, tau385)
    )

    tau425 = (
        einsum("iq,jq,qp->pij", a.t2.x3, a.t2.x4, tau424)
    )

    tau426 = (
        einsum("paw,pbw->pab", tau380, tau75)
    )

    tau427 = (
        einsum("piw,paw->pia", tau340, tau418)
        + einsum("bi,pba->pia", a.t1, tau426)
    )

    tau428 = (
        einsum("pij->pij", tau425)
        + einsum("pjw,piw->pij", tau331, tau400)
        + einsum("aj,pia->pij", a.t1, tau427)
    )

    tau429 = (
        einsum("pij->pij", tau425)
        + einsum("piw,pjw->pij", tau400, tau59)
    )

    tau430 = (
        einsum("pij->pij", tau417)
        + einsum("piw,pjw->pij", tau401, tau59)
    )

    tau431 = (
        - einsum("kp,pij->pijk", a.t3.x6, tau412)
        - einsum("pk,pij->pijk", tau414, tau415)
        - einsum("ip,pjk->pijk", a.t3.x4, tau423)
        + einsum("ip,pjk->pijk", a.t3.x5, tau428)
        - einsum("jp,pik->pijk", a.t3.x5, tau429)
        + einsum("jp,pik->pijk", a.t3.x4, tau430)
    )

    tau432 = (
        einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        - einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau433 = (
        einsum("iq,pia->pqa", a.t3.x6, tau86)
    )

    tau434 = (
        einsum("iq,pia->pqa", a.t3.x6, tau117)
    )

    tau435 = (
        2 * einsum("ip,pqa->pqia", a.t2.x3, tau433)
        - einsum("ip,pqa->pqia", a.t2.x4, tau434)
    )

    tau436 = (
        einsum("aq,qpib->piab", a.t2.x1, tau435)
        + einsum("ip,ab->piab", a.t3.x6, tau157)
    )

    tau437 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau171)
        - einsum("ba->ab", tau174)
    )

    tau438 = (
        einsum("bp,ab->pa", a.t3.x3, tau437)
    )

    tau439 = (
        einsum("pij->pij", tau334)
        + einsum("aj,pia->pij", a.t1, tau69)
    )

    tau440 = (
        einsum("paw,piw->pia", tau226, tau59)
        + einsum("bp,piab->pia", a.t3.x3, tau436)
        - einsum("ip,pa->pia", a.t3.x6, tau438)
        + 2 * einsum("aj,pji->pia", a.t1, tau439)
        - 2 * einsum("aq,qpi->pia", a.t2.x1, tau71)
        - 2 * einsum("pw,iaw->pia", tau55, tau20)
    )

    tau441 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x4, tau117)
    )

    tau442 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x4, tau441)
    )

    tau443 = (
        einsum("pia->pia", tau442)
        - einsum("piw,paw->pia", tau160, tau226)
    )

    tau444 = (
        einsum("pij->pij", tau45)
        + einsum("ai,pja->pij", a.t1, tau165)
    )

    tau445 = (
        einsum("pia->pia", tau442)
        - einsum("pia->pia", tau161)
        + einsum("aj,pij->pia", a.t1, tau444)
    )

    tau446 = (
        einsum("piw,pjw->pij", tau378, tau43)
        + einsum("ai,pja->pij", a.t1, tau255)
    )

    tau447 = (
        einsum("pia->pia", tau224)
        - einsum("piw,paw->pia", tau225, tau60)
        + einsum("aj,pij->pia", a.t1, tau446)
    )

    tau448 = (
        - einsum("jp,pia->pija", a.t3.x5, tau443)
        + einsum("jp,pia->pija", a.t3.x4, tau228)
        + einsum("ip,pja->pija", a.t3.x5, tau445)
        - einsum("ip,pja->pija", a.t3.x4, tau447)
    )

    tau449 = (
        einsum("ai,pja->pij", a.t1, tau51)
    )

    tau450 = (
        einsum("pij->pij", tau329)
        + einsum("pji->pij", tau449)
    )

    tau451 = (
        einsum("aj,pji->pia", a.t1, tau450)
    )

    tau452 = (
        einsum("pw,iaw->pia", tau50, tau20)
    )

    tau453 = (
        einsum("pia->pia", tau93)
        + einsum("pia->pia", tau451)
        - einsum("pia->pia", tau452)
    )

    tau454 = (
        einsum("qw,pw->pq", tau41, tau83)
    )

    tau455 = (
        einsum("ip,jq,pqj->pqi", a.t2.x3, a.t3.x4, tau90)
    )

    tau456 = (
        einsum("ip,pq->pqi", a.t2.x4, tau454)
        - einsum("pqi->pqi", tau455)
    )

    tau457 = (
        einsum("aq,qpi->pia", a.t2.x1, tau456)
    )

    tau458 = (
        einsum("ai,pja->pij", a.t1, tau163)
    )

    tau459 = (
        einsum("pij->pij", tau342)
        + einsum("pji->pij", tau458)
    )

    tau460 = (
        einsum("aj,pji->pia", a.t1, tau459)
    )

    tau461 = (
        einsum("pw,iaw->pia", tau41, tau20)
    )

    tau462 = (
        einsum("pia->pia", tau457)
        + einsum("pia->pia", tau460)
        - einsum("pia->pia", tau461)
    )

    tau463 = (
        einsum("pia->pia", tau51)
        - einsum("pia->pia", tau255)
    )

    tau464 = (
        einsum("aq,pia->pqi", a.t2.x2, tau463)
    )

    tau465 = (
        einsum("ap,iq,qpi->pqa", a.t2.x1, a.t3.x4, tau464)
    )

    tau466 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau465)
    )

    tau467 = (
        - einsum("jp,pia->pija", a.t3.x4, tau453)
        + einsum("jp,pia->pija", a.t3.x5, tau462)
        - einsum("pija->pija", tau466)
    )

    tau468 = (
        - einsum("ap,pijk->pijka", a.t3.x3, tau431)
        - einsum("pij,pka->pijka", tau432, tau440)
        + einsum("kp,pjia->pijka", a.t3.x6, tau448)
        + einsum("ip,pkja->pijka", a.t3.x6, tau467)
        - einsum("jp,pkia->pijka", a.t3.x6, tau467)
        + einsum("ip,pkja->pijka", a.t3.x5, tau100)
        - einsum("ip,pkja->pijka", a.t3.x4, tau183)
        + einsum("jp,pkia->pijka", a.t3.x4, tau183)
        - einsum("jp,pkia->pijka", a.t3.x5, tau100)
    )

    tau469 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau470 = (
        einsum("piw,paw->pia", tau297, tau469)
    )

    tau471 = (
        einsum("ap,qia->pqi", a.t2.x2, tau470)
    )

    tau472 = (
        einsum("pjw,piw->pij", tau247, tau63)
    )

    tau473 = (
        einsum("ip,jp,qij->pq", a.t2.x3, a.t2.x4, tau472)
    )

    tau474 = (
        einsum("jp,ijw->piw", a.t2.x3, tau107)
    )

    tau475 = (
        einsum("ap,ia->pi", a.t2.x2, tau121)
    )

    tau476 = (
        einsum("pji->pij", tau298)
        - 2 * einsum("pij->pij", tau296)
        + einsum("jp,pi->pij", a.t2.x4, tau475)
        - einsum("pij->pij", tau133)
    )

    tau477 = (
        einsum("ap,ia->pi", a.t2.x2, tau403)
    )

    tau478 = (
        einsum("iq,pi->pq", a.t2.x3, tau477)
    )

    tau479 = (
        - einsum("jq,pji->pqi", a.t2.x3, tau476)
        + einsum("ip,pq->pqi", a.t2.x4, tau478)
    )

    tau480 = (
        einsum("pij->pij", tau315)
        + einsum("pji->pij", tau118)
    )

    tau481 = (
        einsum("jq,pji->pqi", a.t2.x3, tau480)
    )

    tau482 = (
        einsum("pw,ijw->pij", tau0, tau186)
    )

    tau483 = (
        einsum("jq,pji->pqi", a.t2.x3, tau482)
    )

    tau484 = (
        - einsum("ip,pqj->pqij", a.t2.x3, tau479)
        + einsum("jp,pqi->pqij", a.t2.x4, tau481)
        + einsum("ip,pqj->pqij", a.t2.x4, tau483)
    )

    tau485 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x6, h.l.pov)
    )

    tau486 = (
        einsum("pw,wia->pia", tau485, h.l.pov)
    )

    tau487 = (
        einsum("ap,pia->pi", a.t3.x3, tau486)
    )

    tau488 = (
        einsum("ip,qi->pq", a.t2.x3, tau487)
    )

    tau489 = (
        2 * einsum("ip,jp->pij", a.t3.x4, a.t3.x6)
        - einsum("jp,ip->pij", a.t3.x4, a.t3.x6)
    )

    tau490 = (
        einsum("pjw,pij->piw", tau43, tau489)
    )

    tau491 = (
        einsum("ip,qiw->pqw", a.t2.x3, tau243)
    )

    tau492 = (
        einsum("iq,pq->pqi", a.t3.x4, tau488)
        - einsum("qiw,pqw->pqi", tau490, tau491)
    )

    tau493 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau30)
    )

    tau494 = (
        einsum("iaw,pjw->pija", tau293, tau474)
        + einsum("pija->pija", tau109)
        + einsum("aq,qpij->pija", a.t2.x1, tau484)
        - einsum("pjw,iaw->pija", tau110, tau20)
        + einsum("aq,jq,pqi->pija", a.t3.x1, a.t3.x5, tau492)
        + einsum("iaw,pjw->pija", tau272, tau34)
        - einsum("pjia->pija", tau493)
    )

    tau495 = (
        einsum("jp,jiw->piw", a.t2.x4, tau186)
    )

    tau496 = (
        einsum("pw,ijw->pij", tau11, tau186)
    )

    tau497 = (
        - einsum("piw,pqw->pqi", tau495, tau9)
        + einsum("ip,pq->pqi", a.t2.x4, tau126)
        + 2 * einsum("jq,pji->pqi", a.t2.x4, tau496)
    )

    tau498 = (
        einsum("jq,pji->pqi", a.t2.x4, tau480)
    )

    tau499 = (
        einsum("pij->pij", tau33)
        + einsum("aj,pia->pij", a.t1, tau151)
    )

    tau500 = (
        einsum("jq,pji->pqi", a.t2.x4, tau499)
    )

    tau501 = (
        - einsum("ip,pqj->pqij", a.t2.x3, tau497)
        + einsum("jp,pqi->pqij", a.t2.x4, tau498)
        + einsum("ip,pqj->pqij", a.t2.x4, tau500)
    )

    tau502 = (
        einsum("jp,jiw->piw", a.t2.x4, tau58)
    )

    tau503 = (
        einsum("ip,qi->pq", a.t2.x4, tau487)
    )

    tau504 = (
        einsum("ip,qiw->pqw", a.t2.x4, tau243)
    )

    tau505 = (
        einsum("iq,pq->pqi", a.t3.x4, tau503)
        - einsum("qiw,pqw->pqi", tau490, tau504)
    )

    tau506 = (
        einsum("iq,piw->pqw", a.t2.x4, tau28)
    )

    tau507 = (
        einsum("pqw,paw->pqa", tau506, tau7)
    )

    tau508 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau507)
    )

    tau509 = (
        einsum("pjia->pija", tau371)
        + einsum("iaw,pjw->pija", tau108, tau191)
        + einsum("aq,qpij->pija", a.t2.x1, tau501)
        - einsum("iaw,pjw->pija", tau20, tau502)
        + einsum("aq,jq,pqi->pija", a.t3.x1, a.t3.x5, tau505)
        + einsum("iaw,pjw->pija", tau272, tau297)
        - einsum("pjia->pija", tau508)
    )

    tau510 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau192)
    )

    tau511 = (
        einsum("iq,piw->pqw", a.t2.x3, tau129)
    )

    tau512 = (
        einsum("pij->pij", tau212)
        + einsum("aj,pia->pij", a.t1, tau128)
    )

    tau513 = (
        - einsum("piw,pqw->pqi", tau502, tau511)
        + einsum("ip,pq->pqi", a.t2.x4, tau195)
        + 2 * einsum("jq,pji->pqi", a.t2.x3, tau512)
    )

    tau514 = (
        einsum("pij->pij", tau346)
        + einsum("aj,pia->pij", a.t1, tau196)
    )

    tau515 = (
        einsum("jq,pji->pqi", a.t2.x3, tau514)
    )

    tau516 = (
        - einsum("ip,pqj->pqij", a.t2.x3, tau513)
        + einsum("jp,pqi->pqij", a.t2.x4, tau481)
        + einsum("ip,pqj->pqij", a.t2.x4, tau515)
    )

    tau517 = (
        einsum("ap,pia->pi", a.t3.x2, tau163)
    )

    tau518 = (
        einsum("ip,qi->pq", a.t2.x3, tau517)
    )

    tau519 = (
        2 * einsum("aq,pqw->pqaw", a.t3.x2, tau65)
        - einsum("aq,pqw->pqaw", a.t3.x3, tau491)
    )

    tau520 = (
        einsum("qaw,pqaw->pq", tau254, tau519)
    )

    tau521 = (
        einsum("jq,iq,pq->pqij", a.t3.x5, a.t3.x6, tau518)
        - einsum("iq,jq,pq->pqij", a.t3.x4, a.t3.x6, tau520)
    )

    tau522 = (
        - einsum("pjw,iaw->pija", tau110, tau510)
        + einsum("aq,qpij->pija", a.t2.x1, tau516)
        + einsum("pjia->pija", tau294)
        + einsum("aq,pqij->pija", a.t3.x1, tau521)
        - einsum("pjia->pija", tau493)
    )

    tau523 = (
        einsum("pij->pij", tau211)
        - 2 * einsum("pij->pij", tau212)
        + einsum("jp,pi->pij", a.t2.x4, tau475)
        - einsum("pij->pij", tau214)
    )

    tau524 = (
        einsum("iq,pi->pq", a.t2.x4, tau477)
    )

    tau525 = (
        - einsum("jq,pji->pqi", a.t2.x4, tau523)
        + einsum("ip,pq->pqi", a.t2.x4, tau524)
    )

    tau526 = (
        - einsum("ip,pqj->pqij", a.t2.x3, tau525)
        + einsum("jp,pqi->pqij", a.t2.x4, tau498)
        + einsum("ip,pqj->pqij", a.t2.x4, tau218)
    )

    tau527 = (
        einsum("ip,qi->pq", a.t2.x4, tau517)
    )

    tau528 = (
        2 * einsum("aq,pqw->pqaw", a.t3.x2, tau64)
        - einsum("aq,pqw->pqaw", a.t3.x3, tau504)
    )

    tau529 = (
        einsum("qaw,pqaw->pq", tau254, tau528)
    )

    tau530 = (
        einsum("jq,iq,pq->pqij", a.t3.x5, a.t3.x6, tau527)
        - einsum("iq,jq,pq->pqij", a.t3.x4, a.t3.x6, tau529)
    )

    tau531 = (
        - einsum("pjw,iaw->pija", tau502, tau510)
        + einsum("aq,qpij->pija", a.t2.x1, tau526)
        + einsum("iaw,pjw->pija", tau293, tau495)
        + einsum("aq,pqij->pija", a.t3.x1, tau530)
        - einsum("pjia->pija", tau508)
    )

    tau532 = (
        einsum("aq,jq,kq,qpi->pijka", a.t2.x1, a.t2.x3, a.t2.x4, tau471)
        + einsum("aq,jq,iq,kq,pq->pijka", a.t3.x1,
                 a.t3.x4, a.t3.x5, a.t3.x6, tau473)
        - einsum("kp,pjia->pijka", a.t2.x4, tau494)
        + einsum("kp,pjia->pijka", a.t2.x3, tau509)
        + einsum("ip,pjka->pijka", a.t2.x4, tau522)
        - einsum("ip,pjka->pijka", a.t2.x3, tau531)
    )

    tau533 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau155)
    )

    tau534 = (
        einsum("bp,abw->paw", a.t3.x3, tau533)
    )

    tau535 = (
        einsum("bp,abw->paw", a.t3.x2, tau533)
    )

    tau536 = (
        einsum("pij->pij", tau244)
        - einsum("pij->pij", tau472)
    )

    tau537 = (
        einsum("aj,pij->pia", a.t1, tau536)
    )

    tau538 = (
        einsum("pab->pab", tau376)
        - einsum("ai,pib->pab", a.t1, tau248)
        + einsum("pab->pab", tau246)
        + einsum("paw,pbw->pab", tau240, tau534)
        - einsum("pbw,paw->pab", tau535, tau60)
        + einsum("ai,pib->pab", a.t1, tau537)
    )

    tau539 = (
        einsum("iq,pia->pqa", a.t2.x3, tau166)
    )

    tau540 = (
        einsum("ip,aq,qpa->pqi", a.t2.x4, a.t3.x2, tau539)
    )

    tau541 = (
        einsum("aq,bq,qpi->piab", a.t2.x1, a.t2.x2, tau540)
    )

    tau542 = (
        einsum("cp,ip,pab->piabc", a.t3.x1, a.t3.x4, tau538)
        + einsum("bp,piac->piabc", a.t3.x1, tau541)
        - einsum("ap,pibc->piabc", a.t3.x1, tau541)
    )

    rt3 = (
        - einsum("kijacb->abcijk", tau6)
        + einsum("kijbca->abcijk", tau6)
        + einsum("jikcba->abcijk", tau6)
        - einsum("jikcab->abcijk", tau6)
        + einsum("kjiacb->abcijk", tau6)
        - einsum("kjibca->abcijk", tau6)
        - einsum("ijkcba->abcijk", tau6)
        + einsum("ijkcab->abcijk", tau6)
        + einsum("jkiabc->abcijk", tau6)
        - einsum("jkibac->abcijk", tau6)
        - einsum("ikjabc->abcijk", tau6)
        + einsum("ikjbac->abcijk", tau6)
        - einsum("ijkcab->abcijk", tau25)
        + einsum("ijkcba->abcijk", tau25)
        + einsum("ikjabc->abcijk", tau25)
        - einsum("ikjbac->abcijk", tau25)
        + einsum("jikcab->abcijk", tau25)
        - einsum("jikcba->abcijk", tau25)
        - einsum("jkiabc->abcijk", tau25)
        + einsum("jkibac->abcijk", tau25)
        + einsum("kijacb->abcijk", tau25)
        - einsum("kijbca->abcijk", tau25)
        - einsum("kjiacb->abcijk", tau25)
        + einsum("kjibca->abcijk", tau25)
        - einsum("ikjabc->abcijk", tau49)
        + einsum("ikjbac->abcijk", tau49)
        - einsum("ijkabc->abcijk", tau139)
        + einsum("ijkbac->abcijk", tau139)
        + einsum("jikabc->abcijk", tau139)
        - einsum("jikbac->abcijk", tau139)
        + einsum("ikjabc->abcijk", tau204)
        - einsum("ikjbac->abcijk", tau204)
        - einsum("jkiabc->abcijk", tau204)
        + einsum("jkibac->abcijk", tau204)
        + einsum("ijkabc->abcijk", tau209)
        - einsum("ijkbac->abcijk", tau209)
        + einsum("ikjacb->abcijk", tau209)
        - einsum("jikabc->abcijk", tau209)
        + einsum("jikbac->abcijk", tau209)
        - einsum("jkiacb->abcijk", tau209)
        - einsum("kijabc->abcijk", tau230)
        + einsum("kijbac->abcijk", tau230)
        + einsum("kjiabc->abcijk", tau230)
        - einsum("kjibac->abcijk", tau230)
        - einsum("ikjacb->abcijk", tau234)
        + einsum("jkiacb->abcijk", tau234)
        - einsum("kijabc->abcijk", tau234)
        + einsum("kijbac->abcijk", tau234)
        + einsum("kjiabc->abcijk", tau234)
        - einsum("kjibac->abcijk", tau234)
        - einsum("ijkacb->abcijk", tau261)
        + einsum("ijkbca->abcijk", tau261)
        + einsum("jikacb->abcijk", tau261)
        - einsum("jikbca->abcijk", tau261)
        - einsum("jikbac->abcijk", tau266)
        + einsum("jikabc->abcijk", tau266)
        + einsum("ijkbac->abcijk", tau271)
        - einsum("ijkabc->abcijk", tau271)
        - einsum("ijkacb->abcijk", tau275)
        + einsum("ijkbca->abcijk", tau275)
        - einsum("ikjabc->abcijk", tau275)
        + einsum("ikjbac->abcijk", tau275)
        + einsum("jikacb->abcijk", tau275)
        - einsum("jikbca->abcijk", tau275)
        + einsum("jkiabc->abcijk", tau275)
        - einsum("jkibac->abcijk", tau275)
        + einsum("kijcab->abcijk", tau275)
        - einsum("kijcba->abcijk", tau275)
        - einsum("kjicab->abcijk", tau275)
        + einsum("kjicba->abcijk", tau275)
        - einsum("ikjabc->abcijk", tau283)
        + einsum("jkiabc->abcijk", tau283)
        - einsum("kijacb->abcijk", tau283)
        + einsum("kijbca->abcijk", tau283)
        + einsum("kjiacb->abcijk", tau283)
        - einsum("kjibca->abcijk", tau283)
        + einsum("ijkacb->abcijk", tau288)
        - einsum("ijkbca->abcijk", tau288)
        + einsum("ikjabc->abcijk", tau288)
        - einsum("jikacb->abcijk", tau288)
        + einsum("jikbca->abcijk", tau288)
        - einsum("jkiabc->abcijk", tau288)
        - einsum("ijkabc->abcijk", tau314)
        + einsum("ijkbac->abcijk", tau314)
        + einsum("jikabc->abcijk", tau339)
        - einsum("jikbac->abcijk", tau339)
        - einsum("jkiabc->abcijk", tau351)
        + einsum("jkibac->abcijk", tau351)
        + einsum("jkiacb->abcijk", tau355)
        - einsum("kijabc->abcijk", tau355)
        + einsum("kijbac->abcijk", tau355)
        + einsum("kjiabc->abcijk", tau355)
        - einsum("kjibac->abcijk", tau355)
        + einsum("ijkabc->abcijk", tau359)
        - einsum("ijkbac->abcijk", tau359)
        + einsum("ikjacb->abcijk", tau359)
        - einsum("jikabc->abcijk", tau363)
        + einsum("jikbac->abcijk", tau363)
        - einsum("jkiacb->abcijk", tau363)
        - einsum("jikcab->abcijk", tau365)
        + einsum("jikcba->abcijk", tau365)
        + einsum("jkiabc->abcijk", tau365)
        - einsum("jkibac->abcijk", tau365)
        - einsum("kijacb->abcijk", tau365)
        + einsum("kijbca->abcijk", tau365)
        + einsum("kjiacb->abcijk", tau365)
        - einsum("kjibca->abcijk", tau365)
        + einsum("ijkacb->abcijk", tau367)
        - einsum("ijkbca->abcijk", tau367)
        + einsum("ikjabc->abcijk", tau367)
        - einsum("ikjbac->abcijk", tau367)
        + einsum("kijabc->abcijk", tau369)
        - einsum("kijbac->abcijk", tau369)
        - einsum("kjiabc->abcijk", tau372)
        + einsum("kjibac->abcijk", tau372)
        + einsum("ikjacb->abcijk", tau374)
        - einsum("ikjbca->abcijk", tau374)
        - einsum("jkiacb->abcijk", tau374)
        + einsum("jkibca->abcijk", tau374)
        + einsum("ijkcab->abcijk", tau377)
        - einsum("ijkcba->abcijk", tau377)
        - einsum("jikcab->abcijk", tau377)
        + einsum("jikcba->abcijk", tau377)
        + einsum("ap,bp,pijkc->abcijk", a.t3.x1, a.t3.x2, tau468)
        - einsum("ap,bp,pikjc->abcijk", a.t2.x1, a.t2.x2, tau532)
        + einsum("ip,jp,pkabc->abcijk", a.t3.x5, a.t3.x6, tau542)
    )

    return Tensors(t1=rt1, t2=rt2, t3=rt3)
