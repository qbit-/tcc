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


def _rccsdt_ncpd_ls_t_calculate_energy(h, a):

    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau3 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau2)
    )

    tau4 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau5 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau6 = (
        - einsum("ap,piw->piaw", a.t2.x1, tau4)
        + 2 * einsum("ap,piw->piaw", a.t2.x2, tau5)
    )

    tau7 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau8 = (
        einsum("ip,paw,piaw->p", a.t2.x3, tau7, tau6)
    )

    energy = (
        2 * einsum("w,w->", tau0, tau1)
        + einsum("ai,ia->", a.t1, tau3)
        + einsum("p,p->", a.t2.xlam[0, :], tau8)
    )

    return energy


def _rccsdt_ncpd_ls_t_calc_residuals(h, a):

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
        einsum("p,ip,wia->paw", a.t2.xlam[0, :], a.t2.x3, h.l.pov)
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
        - einsum("ip,jp->pij", a.t2.x3, a.t2.x4)
        + 2 * einsum("jp,ip->pij", a.t2.x3, a.t2.x4)
    )

    tau15 = (
        einsum("p,paw,piaw->pi", a.t2.xlam[0, :], tau1, tau7)
        - einsum("paw,piaw->pi", tau8, tau10)
        - einsum("p,pj,pji->pi", a.t2.xlam[0, :], tau13, tau14)
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
        - einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        + 2 * einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau20 = (
        einsum("pbw,pba->paw", tau1, tau19)
    )

    tau21 = (
        einsum("p,wia,paw->pi", a.t2.xlam[0, :], h.l.pov, tau20)
    )

    tau22 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau0, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau16)
        + einsum("aj,ia->ij", a.t1, tau18)
        + einsum("jp,pi->ij", a.t2.x3, tau21)
    )

    tau23 = (
        2 * einsum("ap,bp->pab", a.t3.x2, a.t3.x3)
        - einsum("bp,ap->pab", a.t3.x2, a.t3.x3)
    )

    tau24 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau25 = (
        einsum("p,pab,pbw->paw", a.t3.xlam[0, :], tau23, tau24)
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
        einsum("p,pbw,pab->paw", a.t2.xlam[0, :], tau1, tau31)
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
        einsum("p,bp,wab->paw", a.t2.xlam[0, :], a.t2.x1, h.l.pvv)
    )

    tau1 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau2 = (
        einsum("paw,pbw->pab", tau0, tau1)
    )

    tau3 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau4 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau5 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau6 = (
        einsum("aj,ijw->iaw", a.t1, tau5)
    )

    tau7 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau8 = (
        einsum("aj,ijw->iaw", a.t1, tau7)
    )

    tau9 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau10 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau11 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x1, h.l.pov)
    )

    tau12 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau13 = (
        einsum("piw,paw->pia", tau11, tau12)
    )

    tau14 = (
        einsum("ai,pib->pab", a.t1, tau13)
    )

    tau15 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau16 = (
        einsum("paw,piw->pia", tau0, tau15)
    )

    tau17 = (
        einsum("ai,pib->pab", a.t1, tau16)
    )

    tau18 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau19 = (
        einsum("pw,wab->pab", tau18, h.l.pvv)
    )

    tau20 = (
        einsum("bp,pab->pa", a.t3.x2, tau19)
    )

    tau21 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x4, h.l.pov)
    )

    tau22 = (
        einsum("pw,wab->pab", tau21, h.l.pvv)
    )

    tau23 = (
        einsum("bp,pab->pa", a.t3.x3, tau22)
    )

    tau24 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau25 = (
        einsum("pw,wab->pab", tau24, h.l.pvv)
    )

    tau26 = (
        einsum("bp,pab->pa", a.t3.x2, tau25)
    )

    tau27 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x6, h.l.pov)
    )

    tau28 = (
        einsum("pw,wab->pab", tau27, h.l.pvv)
    )

    tau29 = (
        einsum("bp,pab->pa", a.t3.x3, tau28)
    )

    tau30 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x5, h.l.pov)
    )

    tau31 = (
        einsum("pw,wab->pab", tau30, h.l.pvv)
    )

    tau32 = (
        einsum("bp,pab->pa", a.t3.x3, tau31)
    )

    tau33 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau34 = (
        einsum("pw,wab->pab", tau33, h.l.pvv)
    )

    tau35 = (
        einsum("bp,pab->pa", a.t3.x2, tau34)
    )

    tau36 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau37 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau38 = (
        einsum("piw,pjw->pij", tau36, tau37)
    )

    tau39 = (
        einsum("jp,ji->pi", a.t2.x4, h.f.oo)
    )

    tau40 = (
        einsum("jp,ji->pi", a.t2.x3, h.f.oo)
    )

    tau41 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau42 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau43 = (
        einsum("pw,qw->pq", tau41, tau42)
    )

    tau44 = (
        einsum("p,q,aq,iq,pq->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau43)
    )

    tau45 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau46 = (
        einsum("piw,paw->pia", tau15, tau45)
    )

    tau47 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x3, tau46)
    )

    tau48 = (
        einsum("p,q,aq,iq,pq->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau47)
    )

    tau49 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau50 = (
        einsum("pjw,piw->pij", tau15, tau49)
    )

    tau51 = (
        einsum("iq,jq,pij->pq", a.t2.x3, a.t2.x4, tau50)
    )

    tau52 = (
        einsum("p,q,iq,jq,qp->pij", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau51)
    )

    tau53 = (
        einsum("pw,wia->pia", tau41, h.l.pov)
    )

    tau54 = (
        einsum("ap,pia->pi", a.t2.x1, tau53)
    )

    tau55 = (
        einsum("iq,pi->pq", a.t2.x4, tau54)
    )

    tau56 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau55)
    )

    tau57 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau58 = (
        einsum("pw,wia->pia", tau57, h.l.pov)
    )

    tau59 = (
        einsum("ap,pia->pi", a.t2.x2, tau58)
    )

    tau60 = (
        einsum("iq,pi->pq", a.t2.x3, tau59)
    )

    tau61 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau60)
    )

    tau62 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau63 = (
        einsum("piw,paw->pia", tau15, tau62)
    )

    tau64 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x4, tau63)
    )

    tau65 = (
        einsum("p,q,aq,iq,pq->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau64)
    )

    tau66 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau67 = (
        einsum("pw,wia->pia", tau66, h.l.pov)
    )

    tau68 = (
        einsum("ap,pia->pi", a.t2.x2, tau67)
    )

    tau69 = (
        einsum("iq,pi->pq", a.t2.x4, tau68)
    )

    tau70 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau69)
    )

    tau71 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau72 = (
        einsum("pw,wia->pia", tau71, h.l.pov)
    )

    tau73 = (
        einsum("ap,pia->pi", a.t2.x1, tau72)
    )

    tau74 = (
        einsum("iq,pi->pq", a.t2.x3, tau73)
    )

    tau75 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau74)
    )

    tau76 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau77 = (
        einsum("qw,pw->pq", tau71, tau76)
    )

    tau78 = (
        einsum("p,q,aq,iq,pq->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau77)
    )

    tau79 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x2, h.l.pov)
    )

    tau80 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau81 = (
        einsum("pjw,piw->pij", tau79, tau80)
    )

    tau82 = (
        einsum("aj,pij->pia", a.t1, tau81)
    )

    tau83 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x2, h.l.pov)
    )

    tau84 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau85 = (
        einsum("piw,pjw->pij", tau83, tau84)
    )

    tau86 = (
        einsum("aj,pji->pia", a.t1, tau85)
    )

    tau87 = (
        einsum("p,ap,ip,wia->pw", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau88 = (
        einsum("pw,wij->pij", tau87, h.l.poo)
    )

    tau89 = (
        einsum("aj,pji->pia", a.t1, tau88)
    )

    tau90 = (
        einsum("pjw,piw->pij", tau37, tau83)
    )

    tau91 = (
        einsum("aj,pji->pia", a.t1, tau90)
    )

    tau92 = (
        einsum("p,ap,ip,wia->pw", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau93 = (
        einsum("pw,wij->pij", tau92, h.l.poo)
    )

    tau94 = (
        einsum("aj,pji->pia", a.t1, tau93)
    )

    tau95 = (
        einsum("piw,pjw->pij", tau36, tau79)
    )

    tau96 = (
        einsum("aj,pij->pia", a.t1, tau95)
    )

    tau97 = (
        einsum("p,ip,wia->paw", a.t2.xlam[0, :], a.t2.x3, h.l.pov)
    )

    tau98 = (
        einsum("piw,paw->pia", tau37, tau97)
    )

    tau99 = (
        einsum("ai,pja->pij", a.t1, tau98)
    )

    tau100 = (
        einsum("p,jp,wji->piw", a.t2.xlam[0, :], a.t2.x3, h.l.poo)
    )

    tau101 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau102 = (
        einsum("piw,paw->pia", tau100, tau101)
    )

    tau103 = (
        einsum("ai,pja->pij", a.t1, tau102)
    )

    tau104 = (
        einsum("wia,jaw->ij", h.l.pov, tau9)
    )

    tau105 = (
        einsum("jp,ji->pi", a.t2.x4, tau104)
    )

    tau106 = (
        einsum("wja,iaw->ij", h.l.pov, tau10)
    )

    tau107 = (
        einsum("jp,ij->pi", a.t2.x3, tau106)
    )

    tau108 = (
        einsum("p,ap,ip,wia->pw", a.t2.xlam[0, :], a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau109 = (
        einsum("pw,wij->pij", tau108, h.l.poo)
    )

    tau110 = (
        einsum("aj,pji->pia", a.t1, tau109)
    )

    tau111 = (
        einsum("p,ap,ip,wia->pw", a.t2.xlam[0, :], a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau112 = (
        einsum("pw,wij->pij", tau111, h.l.poo)
    )

    tau113 = (
        einsum("aj,pji->pia", a.t1, tau112)
    )

    tau114 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau115 = (
        einsum("w,wij->ij", tau114, h.l.poo)
    )

    tau116 = (
        einsum("jp,ji->pi", a.t2.x4, tau115)
    )

    tau117 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau118 = (
        einsum("w,wij->ij", tau117, h.l.poo)
    )

    tau119 = (
        einsum("jp,ji->pi", a.t2.x3, tau118)
    )

    tau120 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau121 = (
        einsum("pw,wij->pij", tau120, h.l.poo)
    )

    tau122 = (
        einsum("jp,pji->pi", a.t3.x6, tau121)
    )

    tau123 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau124 = (
        einsum("pw,wij->pij", tau123, h.l.poo)
    )

    tau125 = (
        einsum("jp,pji->pi", a.t3.x5, tau124)
    )

    tau126 = (
        einsum("pw,wij->pij", tau18, h.l.poo)
    )

    tau127 = (
        einsum("jp,pji->pi", a.t3.x4, tau126)
    )

    tau128 = (
        einsum("pw,wij->pij", tau24, h.l.poo)
    )

    tau129 = (
        einsum("jp,pji->pi", a.t3.x6, tau128)
    )

    tau130 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau131 = (
        einsum("pw,wij->pij", tau130, h.l.poo)
    )

    tau132 = (
        einsum("jp,pji->pi", a.t3.x5, tau131)
    )

    tau133 = (
        einsum("pw,wij->pij", tau33, h.l.poo)
    )

    tau134 = (
        einsum("jp,pji->pi", a.t3.x4, tau133)
    )

    tau135 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau136 = (
        einsum("jp,ij->pi", a.t2.x3, tau135)
    )

    tau137 = (
        einsum("jp,ij->pi", a.t2.x4, tau135)
    )

    tau138 = (
        einsum("ap,ia->pi", a.t3.x3, h.f.ov)
    )

    tau139 = (
        einsum("ip,pi->p", a.t3.x5, tau138)
    )

    tau140 = (
        einsum("ip,pi->p", a.t3.x4, tau138)
    )

    tau141 = (
        einsum("ip,pi->p", a.t3.x6, tau138)
    )

    tau142 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x1, h.l.pov)
    )

    tau143 = (
        einsum("piw,pjw->pij", tau142, tau15)
    )

    tau144 = (
        einsum("aj,pij->pia", a.t1, tau143)
    )

    tau145 = (
        einsum("ai,pib->pab", a.t1, tau144)
    )

    tau146 = (
        einsum("pbw,paw->pab", tau101, tau97)
    )

    tau147 = (
        einsum("bi,pab->pia", a.t1, tau146)
    )

    tau148 = (
        einsum("ai,pja->pij", a.t1, tau147)
    )

    tau149 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau150 = (
        einsum("wib,baw->ia", h.l.pov, tau149)
    )

    tau151 = (
        einsum("ai,ja->ij", a.t1, tau150)
    )

    tau152 = (
        einsum("jp,ij->pi", a.t2.x3, tau151)
    )

    tau153 = (
        einsum("jp,ij->pi", a.t2.x4, tau151)
    )

    tau154 = (
        einsum("w,wia->ia", tau114, h.l.pov)
    )

    tau155 = (
        einsum("ai,ja->ij", a.t1, tau154)
    )

    tau156 = (
        einsum("jp,ij->pi", a.t2.x3, tau155)
    )

    tau157 = (
        einsum("jp,ij->pi", a.t2.x4, tau155)
    )

    tau158 = (
        einsum("pw,wia->pia", tau24, h.l.pov)
    )

    tau159 = (
        einsum("ap,pia->pi", a.t3.x2, tau158)
    )

    tau160 = (
        einsum("ai,pi->pa", a.t1, tau159)
    )

    tau161 = (
        einsum("pw,wia->pia", tau27, h.l.pov)
    )

    tau162 = (
        einsum("ap,pia->pi", a.t3.x3, tau161)
    )

    tau163 = (
        einsum("ai,pi->pa", a.t1, tau162)
    )

    tau164 = (
        einsum("pw,wia->pia", tau18, h.l.pov)
    )

    tau165 = (
        einsum("ap,pia->pi", a.t3.x2, tau164)
    )

    tau166 = (
        einsum("ai,pi->pa", a.t1, tau165)
    )

    tau167 = (
        einsum("pw,wia->pia", tau21, h.l.pov)
    )

    tau168 = (
        einsum("ap,pia->pi", a.t3.x3, tau167)
    )

    tau169 = (
        einsum("ai,pi->pa", a.t1, tau168)
    )

    tau170 = (
        einsum("pw,wia->pia", tau123, h.l.pov)
    )

    tau171 = (
        einsum("ip,pia->pa", a.t3.x6, tau170)
    )

    tau172 = (
        einsum("ai,pa->pi", a.t1, tau171)
    )

    tau173 = (
        einsum("ip,pia->pa", a.t3.x4, tau164)
    )

    tau174 = (
        einsum("ai,pa->pi", a.t1, tau173)
    )

    tau175 = (
        einsum("pw,wia->pia", tau120, h.l.pov)
    )

    tau176 = (
        einsum("ip,pia->pa", a.t3.x6, tau175)
    )

    tau177 = (
        einsum("ai,pa->pi", a.t1, tau176)
    )

    tau178 = (
        einsum("ip,pia->pa", a.t3.x5, tau170)
    )

    tau179 = (
        einsum("ai,pa->pi", a.t1, tau178)
    )

    tau180 = (
        einsum("ap,ia->pi", a.t3.x3, tau150)
    )

    tau181 = (
        einsum("ip,pi->p", a.t3.x5, tau180)
    )

    tau182 = (
        einsum("ip,pi->p", a.t3.x4, tau180)
    )

    tau183 = (
        einsum("pw,wia->pia", tau33, h.l.pov)
    )

    tau184 = (
        einsum("ap,pia->pi", a.t3.x2, tau183)
    )

    tau185 = (
        einsum("ai,pi->pa", a.t1, tau184)
    )

    tau186 = (
        einsum("pw,wia->pia", tau30, h.l.pov)
    )

    tau187 = (
        einsum("ap,pia->pi", a.t3.x3, tau186)
    )

    tau188 = (
        einsum("ai,pi->pa", a.t1, tau187)
    )

    tau189 = (
        einsum("ip,pia->pa", a.t3.x4, tau183)
    )

    tau190 = (
        einsum("ai,pa->pi", a.t1, tau189)
    )

    tau191 = (
        einsum("ip,pia->pa", a.t3.x5, tau183)
    )

    tau192 = (
        einsum("ai,pa->pi", a.t1, tau191)
    )

    tau193 = (
        einsum("ip,pi->p", a.t3.x6, tau180)
    )

    tau194 = (
        einsum("ap,ia->pi", a.t3.x3, tau154)
    )

    tau195 = (
        einsum("ip,pi->p", a.t3.x5, tau194)
    )

    tau196 = (
        einsum("ip,pi->p", a.t3.x4, tau194)
    )

    tau197 = (
        einsum("ip,pi->p", a.t3.x6, tau194)
    )

    tau198 = (
        einsum("bp,ab->pa", a.t2.x2, h.f.vv)
    )

    tau199 = (
        einsum("p,ap,ip,jp,pb->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau198)
    )

    tau200 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau6)
    )

    tau201 = (
        einsum("p,bp,wab->paw", a.t2.xlam[0, :], a.t2.x2, h.l.pvv)
    )

    tau202 = (
        einsum("paw,piw->pia", tau201, tau37)
    )

    tau203 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau202)
    )

    tau204 = (
        einsum("pw,wai->pia", tau87, h.l.pvo)
    )

    tau205 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau204)
    )

    tau206 = (
        einsum("pw,wai->pia", tau108, h.l.pvo)
    )

    tau207 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau206)
    )

    tau208 = (
        einsum("pw,wab->pab", tau87, h.l.pvv)
    )

    tau209 = (
        einsum("bi,pab->pia", a.t1, tau208)
    )

    tau210 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau209)
    )

    tau211 = (
        einsum("pbw,paw->pab", tau101, tau201)
    )

    tau212 = (
        einsum("bi,pab->pia", a.t1, tau211)
    )

    tau213 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau212)
    )

    tau214 = (
        einsum("pw,wab->pab", tau108, h.l.pvv)
    )

    tau215 = (
        einsum("bi,pab->pia", a.t1, tau214)
    )

    tau216 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau215)
    )

    tau217 = (
        einsum("pw,wia->pia", tau108, h.l.pov)
    )

    tau218 = (
        einsum("ai,pja->pij", a.t1, tau217)
    )

    tau219 = (
        einsum("aj,pij->pia", a.t1, tau218)
    )

    tau220 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau219)
    )

    tau221 = (
        einsum("pw,wia->pia", tau87, h.l.pov)
    )

    tau222 = (
        einsum("ai,pja->pij", a.t1, tau221)
    )

    tau223 = (
        einsum("aj,pij->pia", a.t1, tau222)
    )

    tau224 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau223)
    )

    tau225 = (
        einsum("paw,piw->pia", tau101, tau83)
    )

    tau226 = (
        einsum("ai,pja->pij", a.t1, tau225)
    )

    tau227 = (
        einsum("aj,pij->pia", a.t1, tau226)
    )

    tau228 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau227)
    )

    tau229 = (
        einsum("ip,pia->pa", a.t2.x3, tau72)
    )

    tau230 = (
        einsum("aq,pa->pq", a.t2.x2, tau229)
    )

    tau231 = (
        einsum("q,aq,qp->pa", a.t2.xlam[0, :], a.t2.x1, tau230)
    )

    tau232 = (
        einsum("p,ap,ip,jp,pb->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau231)
    )

    tau233 = (
        einsum("aq,iq,pia->pq", a.t2.x2, a.t2.x4, tau46)
    )

    tau234 = (
        einsum("p,q,aq,iq,qp->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau233)
    )

    tau235 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau234)
    )

    tau236 = (
        einsum("ip,pia->pa", a.t2.x4, tau53)
    )

    tau237 = (
        einsum("aq,pa->pq", a.t2.x2, tau236)
    )

    tau238 = (
        einsum("q,aq,qp->pa", a.t2.xlam[0, :], a.t2.x1, tau237)
    )

    tau239 = (
        einsum("p,ap,ip,jp,pb->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau238)
    )

    tau240 = (
        einsum("pw,qw->pq", tau41, tau71)
    )

    tau241 = (
        einsum("p,q,aq,iq,qp->pia", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau240)
    )

    tau242 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau241)
    )

    tau243 = (
        - einsum("ijab->ijab", tau199)
        + einsum("ijab->ijab", tau200)
        + einsum("ijab->ijab", tau203)
        + einsum("ijab->ijab", tau205)
        - 2 * einsum("ijab->ijab", tau207)
        + einsum("ijab->ijab", tau210)
        + einsum("ijab->ijab", tau213)
        - 2 * einsum("ijab->ijab", tau216)
        + 2 * einsum("ijab->ijab", tau220)
        - einsum("ijab->ijab", tau224)
        - einsum("ijab->ijab", tau228)
        + 2 * einsum("ijab->ijab", tau232)
        - einsum("ijab->ijab", tau235)
        - einsum("ijab->ijab", tau239)
        + 2 * einsum("ijab->ijab", tau242)
    )

    tau244 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau9)
    )

    tau245 = (
        einsum("ap,ia->pi", a.t2.x2, h.f.ov)
    )

    tau246 = (
        einsum("ai,pi->pa", a.t1, tau245)
    )

    tau247 = (
        einsum("p,bp,ip,jp,pa->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau246)
    )

    tau248 = (
        einsum("ibw,jaw->ijab", tau3, tau9)
    )

    tau249 = (
        einsum("paw,piw->pia", tau201, tau84)
    )

    tau250 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau249)
    )

    tau251 = (
        einsum("ibw,jaw->ijab", tau3, tau6)
    )

    tau252 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau253 = (
        einsum("paw,pbw->pab", tau201, tau252)
    )

    tau254 = (
        einsum("bi,pab->pia", a.t1, tau253)
    )

    tau255 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau254)
    )

    tau256 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau257 = (
        einsum("wac,cbw->ab", h.l.pvv, tau256)
    )

    tau258 = (
        einsum("bp,ab->pa", a.t2.x2, tau257)
    )

    tau259 = (
        einsum("p,bp,ip,jp,pa->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau258)
    )

    tau260 = (
        einsum("w,wab->ab", tau117, h.l.pvv)
    )

    tau261 = (
        einsum("bp,ab->pa", a.t2.x2, tau260)
    )

    tau262 = (
        einsum("p,bp,ip,jp,pa->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau261)
    )

    tau263 = (
        einsum("paw,piw->pia", tau252, tau83)
    )

    tau264 = (
        einsum("ai,pja->pij", a.t1, tau263)
    )

    tau265 = (
        einsum("aj,pij->pia", a.t1, tau264)
    )

    tau266 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau265)
    )

    tau267 = (
        einsum("wib,baw->ia", h.l.pov, tau256)
    )

    tau268 = (
        einsum("ap,ia->pi", a.t2.x2, tau267)
    )

    tau269 = (
        einsum("ai,pi->pa", a.t1, tau268)
    )

    tau270 = (
        einsum("p,bp,ip,jp,pa->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau269)
    )

    tau271 = (
        einsum("w,wia->ia", tau117, h.l.pov)
    )

    tau272 = (
        einsum("ap,ia->pi", a.t2.x2, tau271)
    )

    tau273 = (
        einsum("ai,pi->pa", a.t1, tau272)
    )

    tau274 = (
        einsum("p,bp,ip,jp,pa->ijab", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau273)
    )

    tau275 = (
        einsum("ijab->ijab", tau244)
        + einsum("ijab->ijab", tau247)
        + einsum("ijab->ijab", tau248)
        + einsum("ijab->ijab", tau250)
        + einsum("ijab->ijab", tau251)
        + einsum("ijab->ijab", tau255)
        + einsum("ijab->ijab", tau259)
        - 2 * einsum("ijab->ijab", tau262)
        - einsum("ijab->ijab", tau266)
        - einsum("ijab->ijab", tau270)
        + 2 * einsum("ijab->ijab", tau274)
    )

    rt2 = (
        einsum("ip,jp,pab->abij", a.t2.x3, a.t2.x4, tau2)
        + einsum("wbj,iaw->abij", h.l.pvo, tau3)
        + einsum("wai,jbw->abij", h.l.pvo, tau4)
        + einsum("wai,wbj->abij", h.l.pvo, h.l.pvo)
        + einsum("jbw,iaw->abij", tau6, tau8)
        + einsum("iaw,jbw->abij", tau8, tau9)
        + einsum("iaw,jbw->abij", tau10, tau6)
        - einsum("ip,jp,pab->abij", a.t2.x3, a.t2.x4, tau14)
        - einsum("ip,jp,pba->abij", a.t2.x3, a.t2.x4, tau17)
        - einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x6, tau20)
        - einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x5, a.t3.x6, tau23)
        - einsum("p,bp,ip,jp,pa->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x5, a.t3.x6, tau26)
        - einsum("p,bp,jp,ip,pa->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x5, tau29)
        + 2 * einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x4, a.t3.x6, tau32)
        + 2 * einsum("p,bp,jp,ip,pa->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x4, a.t3.x5, tau35)
        + einsum("p,ap,bp,pij->abij", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau38)
        - einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x3, tau39)
        - einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x4, tau40)
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x4, tau44)
        + einsum("bp,ip,pja->abij", a.t2.x1, a.t2.x4, tau48)
        + einsum("ap,bp,pij->abij", a.t2.x1, a.t2.x2, tau52)
        + einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x3, tau56)
        + einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x4, tau61)
        - 2 * einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau65)
        - 2 * einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x3, tau70)
        - 2 * einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x4, tau75)
        + 4 * einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau78)
        + einsum("iaw,jbw->abij", tau3, tau4)
        + einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau82)
        + einsum("bp,ip,pja->abij", a.t2.x1, a.t2.x4, tau86)
        + einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x4, tau89)
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau91)
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x4, tau94)
        + einsum("ap,jp,pib->abij", a.t2.x1, a.t2.x4, tau96)
        + einsum("ap,bp,pij->abij", a.t2.x1, a.t2.x2, tau99)
        + einsum("ap,bp,pji->abij", a.t2.x1, a.t2.x2, tau103)
        + einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x3, tau105)
        + einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x4, tau107)
        - 2 * einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau110)
        - 2 * einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau113)
        - 2 * einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x3, tau116)
        - 2 * einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x4, tau119)
        + einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x4, tau122)
        + einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x6, tau125)
        + einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x6, tau127)
        + einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x5, tau129)
        - 2 * einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x4, tau132)
        - 2 * einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x5, tau134)
        + einsum("iaw,jbw->abij", tau10, tau9)
        - einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x4, tau136)
        - einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x3, tau137)
        - einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                 tau139, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6)
        - einsum("p,p,ap,bp,jp,ip->abij", a.t3.xlam[0, :],
                 tau140, a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6)
        + 2 * einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                     tau141, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5)
        + einsum("ip,jp,pab->abij", a.t2.x3, a.t2.x4, tau145)
        + einsum("ap,bp,pij->abij", a.t2.x1, a.t2.x2, tau148)
        + einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x4, tau152)
        + einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x2, a.t2.x3, tau153)
        - 2 * einsum("p,ap,bp,jp,pi->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x4, tau156)
        - 2 * einsum("p,ap,bp,ip,pj->abij", a.t2.xlam[0, :],
                     a.t2.x1, a.t2.x2, a.t2.x3, tau157)
        + einsum("p,bp,ip,jp,pa->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x5, a.t3.x6, tau160)
        + einsum("p,bp,jp,ip,pa->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x5, tau163)
        + einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x6, tau166)
        + einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x5, a.t3.x6, tau169)
        + einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x5, tau172)
        + einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x6, tau174)
        + einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x4, tau177)
        + einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x6, tau179)
        + einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                 tau181, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6)
        + einsum("p,p,ap,bp,jp,ip->abij", a.t3.xlam[0, :],
                 tau182, a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6)
        - 2 * einsum("p,bp,jp,ip,pa->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x4, a.t3.x5, tau185)
        - 2 * einsum("p,ap,ip,jp,pb->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x4, a.t3.x6, tau188)
        - 2 * einsum("p,ap,bp,jp,pi->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x5, tau190)
        - 2 * einsum("p,ap,bp,ip,pj->abij", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x4, tau192)
        - 2 * einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                     tau193, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5)
        - 2 * einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                     tau195, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6)
        - 2 * einsum("p,p,ap,bp,jp,ip->abij", a.t3.xlam[0, :],
                     tau196, a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6)
        + 4 * einsum("p,p,ap,bp,ip,jp->abij", a.t3.xlam[0, :],
                     tau197, a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5)
        - einsum("ijab->abij", tau243)
        - einsum("jiba->abij", tau243)
        - einsum("ijba->abij", tau275)
        - einsum("jiab->abij", tau275)
    )
    tau0 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau1 = (
        einsum("pw,wia->pia", tau0, h.l.pov)
    )

    tau2 = (
        einsum("ap,pia->pi", a.t2.x1, tau1)
    )

    tau3 = (
        einsum("iq,pi->pq", a.t3.x5, tau2)
    )

    tau4 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau3)
    )

    tau5 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau6 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau7 = (
        einsum("piw,pjw->pij", tau5, tau6)
    )

    tau8 = (
        einsum("iq,jq,pij->pq", a.t3.x4, a.t3.x5, tau7)
    )

    tau9 = (
        einsum("q,iq,jq,qp->pij", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau8)
    )

    tau10 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau11 = (
        einsum("pw,wia->pia", tau10, h.l.pov)
    )

    tau12 = (
        einsum("ap,pia->pi", a.t2.x2, tau11)
    )

    tau13 = (
        einsum("iq,pi->pq", a.t3.x4, tau12)
    )

    tau14 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau13)
    )

    tau15 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x6, h.l.pov)
    )

    tau16 = (
        einsum("pw,wia->pia", tau15, h.l.pov)
    )

    tau17 = (
        einsum("ap,pia->pi", a.t3.x3, tau16)
    )

    tau18 = (
        einsum("ip,qi->pq", a.t2.x4, tau17)
    )

    tau19 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x5, tau18)
    )

    tau20 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau21 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau22 = (
        einsum("pjw,piw->pij", tau20, tau21)
    )

    tau23 = (
        einsum("ip,jp,qij->pq", a.t2.x3, a.t2.x4, tau22)
    )

    tau24 = (
        einsum("q,aq,bq,qp->pab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau23)
    )

    tau25 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau26 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau27 = (
        einsum("pjw,piw->pij", tau25, tau26)
    )

    tau28 = (
        einsum("iq,jq,pij->pq", a.t3.x4, a.t3.x5, tau27)
    )

    tau29 = (
        einsum("q,iq,jq,qp->pij", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau28)
    )

    tau30 = (
        einsum("iq,pi->pq", a.t3.x4, tau2)
    )

    tau31 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau30)
    )

    tau32 = (
        einsum("iq,pi->pq", a.t3.x5, tau12)
    )

    tau33 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau32)
    )

    tau34 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau35 = (
        einsum("pw,wia->pia", tau34, h.l.pov)
    )

    tau36 = (
        einsum("ap,pia->pi", a.t2.x1, tau35)
    )

    tau37 = (
        einsum("iq,pi->pq", a.t3.x4, tau36)
    )

    tau38 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau37)
    )

    tau39 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau40 = (
        einsum("pw,wia->pia", tau39, h.l.pov)
    )

    tau41 = (
        einsum("ap,pia->pi", a.t2.x2, tau40)
    )

    tau42 = (
        einsum("iq,pi->pq", a.t3.x5, tau41)
    )

    tau43 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau42)
    )

    tau44 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau45 = (
        einsum("pw,wia->pia", tau44, h.l.pov)
    )

    tau46 = (
        einsum("ap,pia->pi", a.t3.x2, tau45)
    )

    tau47 = (
        einsum("ip,qi->pq", a.t2.x4, tau46)
    )

    tau48 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x5, tau47)
    )

    tau49 = (
        einsum("iq,pi->pq", a.t3.x5, tau36)
    )

    tau50 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x3, tau49)
    )

    tau51 = (
        einsum("iq,pi->pq", a.t3.x4, tau41)
    )

    tau52 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau51)
    )

    tau53 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau54 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau55 = (
        einsum("piw,pjw->pij", tau53, tau54)
    )

    tau56 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau57 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau58 = (
        einsum("pjw,piw->pij", tau56, tau57)
    )

    tau59 = (
        einsum("p,ap,wia->piw", a.t3.xlam[0, :], a.t3.x2, h.l.pov)
    )

    tau60 = (
        einsum("piw,pjw->pij", tau21, tau59)
    )

    tau61 = (
        einsum("aj,pij->pia", a.t1, tau60)
    )

    tau62 = (
        einsum("ai,pib->pab", a.t1, tau61)
    )

    tau63 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau64 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau65 = (
        einsum("paw,piw->pia", tau63, tau64)
    )

    tau66 = (
        einsum("ap,qia->pqi", a.t2.x2, tau65)
    )

    tau67 = (
        einsum("q,aq,bq,pqi->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau66)
    )

    tau68 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau69 = (
        einsum("pw,wij->pij", tau68, h.l.poo)
    )

    tau70 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau69)
    )

    tau71 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau70)
    )

    tau72 = (
        einsum("pjw,piw->pij", tau6, tau64)
    )

    tau73 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau72)
    )

    tau74 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau73)
    )

    tau75 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau76 = (
        einsum("pw,wij->pij", tau75, h.l.poo)
    )

    tau77 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau76)
    )

    tau78 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau77)
    )

    tau79 = (
        einsum("p,jp,wji->piw", a.t2.xlam[0, :], a.t2.x4, h.l.poo)
    )

    tau80 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau81 = (
        einsum("aj,ijw->iaw", a.t1, tau80)
    )

    tau82 = (
        einsum("pjw,iaw->pija", tau79, tau81)
    )

    tau83 = (
        einsum("p,ip,wia->paw", a.t3.xlam[0, :], a.t3.x4, h.l.pov)
    )

    tau84 = (
        einsum("piw,paw->pia", tau54, tau83)
    )

    tau85 = (
        einsum("ai,pja->pij", a.t1, tau84)
    )

    tau86 = (
        einsum("p,jp,wji->piw", a.t3.xlam[0, :], a.t3.x4, h.l.poo)
    )

    tau87 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau88 = (
        einsum("piw,paw->pia", tau86, tau87)
    )

    tau89 = (
        einsum("ai,pja->pij", a.t1, tau88)
    )

    tau90 = (
        einsum("p,jp,wji->piw", a.t3.xlam[0, :], a.t3.x5, h.l.poo)
    )

    tau91 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau92 = (
        einsum("piw,paw->pia", tau90, tau91)
    )

    tau93 = (
        einsum("ai,pja->pij", a.t1, tau92)
    )

    tau94 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau95 = (
        einsum("piw,paw->pia", tau86, tau94)
    )

    tau96 = (
        einsum("ai,pja->pij", a.t1, tau95)
    )

    tau97 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau98 = (
        einsum("wia,jaw->ij", h.l.pov, tau97)
    )

    tau99 = (
        einsum("jp,ji->pi", a.t3.x5, tau98)
    )

    tau100 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau101 = (
        einsum("wja,iaw->ij", h.l.pov, tau100)
    )

    tau102 = (
        einsum("jp,ij->pi", a.t3.x4, tau101)
    )

    tau103 = (
        einsum("p,ap,wia->piw", a.t3.xlam[0, :], a.t3.x3, h.l.pov)
    )

    tau104 = (
        einsum("pjw,piw->pij", tau103, tau57)
    )

    tau105 = (
        einsum("aj,pij->pia", a.t1, tau104)
    )

    tau106 = (
        einsum("p,jp,wji->piw", a.t3.xlam[0, :], a.t3.x4, h.l.poo)
    )

    tau107 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau108 = (
        einsum("piw,paw->pia", tau106, tau107)
    )

    tau109 = (
        einsum("ai,pja->pij", a.t1, tau108)
    )

    tau110 = (
        einsum("p,ip,wia->paw", a.t3.xlam[0, :], a.t3.x4, h.l.pov)
    )

    tau111 = (
        einsum("paw,piw->pia", tau110, tau57)
    )

    tau112 = (
        einsum("ai,pja->pij", a.t1, tau111)
    )

    tau113 = (
        einsum("piw,paw->pia", tau106, tau91)
    )

    tau114 = (
        einsum("ai,pja->pij", a.t1, tau113)
    )

    tau115 = (
        einsum("p,jp,wji->piw", a.t3.xlam[0, :], a.t3.x5, h.l.poo)
    )

    tau116 = (
        einsum("piw,paw->pia", tau115, tau94)
    )

    tau117 = (
        einsum("ai,pja->pij", a.t1, tau116)
    )

    tau118 = (
        einsum("jp,ji->pi", a.t3.x4, tau98)
    )

    tau119 = (
        einsum("jp,ij->pi", a.t3.x5, tau101)
    )

    tau120 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau121 = (
        einsum("w,wij->ij", tau120, h.l.poo)
    )

    tau122 = (
        einsum("jp,ji->pi", a.t3.x5, tau121)
    )

    tau123 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau124 = (
        einsum("w,wij->ij", tau123, h.l.poo)
    )

    tau125 = (
        einsum("jp,ji->pi", a.t3.x4, tau124)
    )

    tau126 = (
        einsum("jp,ji->pi", a.t3.x4, tau121)
    )

    tau127 = (
        einsum("jp,ji->pi", a.t3.x5, tau124)
    )

    tau128 = (
        einsum("bp,wab->paw", a.t3.x2, h.l.pvv)
    )

    tau129 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau130 = (
        einsum("pbw,paw->pab", tau128, tau129)
    )

    tau131 = (
        einsum("pw,wab->pab", tau68, h.l.pvv)
    )

    tau132 = (
        einsum("q,bq,pab->pqa", a.t2.xlam[0, :], a.t2.x2, tau131)
    )

    tau133 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau132)
    )

    tau134 = (
        einsum("bp,ip,jp,pkac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau133)
    )

    tau135 = (
        einsum("pw,wia->pia", tau68, h.l.pov)
    )

    tau136 = (
        einsum("q,aq,pia->pqi", a.t2.xlam[0, :], a.t2.x2, tau135)
    )

    tau137 = (
        einsum("ai,pqi->pqa", a.t1, tau136)
    )

    tau138 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau137)
    )

    tau139 = (
        einsum("bp,ip,jp,pkac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau138)
    )

    tau140 = (
        einsum("ijkabc->ijkabc", tau134)
        - einsum("ijkabc->ijkabc", tau139)
    )

    tau141 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x2, h.l.pov)
    )

    tau142 = (
        einsum("ai,piw->paw", a.t1, tau141)
    )

    tau143 = (
        einsum("wbi,paw->piab", h.l.pvo, tau142)
    )

    tau144 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau143)
    )

    tau145 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau146 = (
        einsum("paw,ibw->piab", tau142, tau145)
    )

    tau147 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau146)
    )

    tau148 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau149 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau150 = (
        einsum("paw,pbw->pab", tau148, tau149)
    )

    tau151 = (
        einsum("q,bq,pab->pqa", a.t2.xlam[0, :], a.t2.x2, tau150)
    )

    tau152 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau151)
    )

    tau153 = (
        einsum("bp,ip,kp,pjac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau152)
    )

    tau154 = (
        einsum("paw,pbw->pab", tau148, tau63)
    )

    tau155 = (
        einsum("q,bq,pab->pqa", a.t2.xlam[0, :], a.t2.x2, tau154)
    )

    tau156 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau155)
    )

    tau157 = (
        einsum("bp,ip,jp,pkac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau156)
    )

    tau158 = (
        einsum("pw,wab->pab", tau75, h.l.pvv)
    )

    tau159 = (
        einsum("q,bq,pab->pqa", a.t2.xlam[0, :], a.t2.x2, tau158)
    )

    tau160 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau159)
    )

    tau161 = (
        einsum("bp,ip,kp,pjac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau160)
    )

    tau162 = (
        einsum("piw,paw->pia", tau26, tau63)
    )

    tau163 = (
        einsum("q,aq,pia->pqi", a.t2.xlam[0, :], a.t2.x2, tau162)
    )

    tau164 = (
        einsum("ai,pqi->pqa", a.t1, tau163)
    )

    tau165 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau164)
    )

    tau166 = (
        einsum("bp,ip,jp,pkac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau165)
    )

    tau167 = (
        einsum("paw,piw->pia", tau149, tau26)
    )

    tau168 = (
        einsum("q,aq,pia->pqi", a.t2.xlam[0, :], a.t2.x2, tau167)
    )

    tau169 = (
        einsum("ai,pqi->pqa", a.t1, tau168)
    )

    tau170 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau169)
    )

    tau171 = (
        einsum("bp,ip,kp,pjac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau170)
    )

    tau172 = (
        einsum("pw,wia->pia", tau75, h.l.pov)
    )

    tau173 = (
        einsum("q,aq,pia->pqi", a.t2.xlam[0, :], a.t2.x2, tau172)
    )

    tau174 = (
        einsum("ai,pqi->pqa", a.t1, tau173)
    )

    tau175 = (
        einsum("q,bq,iq,qpa->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau174)
    )

    tau176 = (
        einsum("bp,ip,kp,pjac->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau175)
    )

    tau177 = (
        einsum("ijkabc->ijkabc", tau144)
        + einsum("ijkabc->ijkabc", tau147)
        + einsum("ijkabc->ijkabc", tau153)
        + einsum("ijkabc->ijkabc", tau157)
        - 2 * einsum("ijkabc->ijkabc", tau161)
        - einsum("ijkabc->ijkabc", tau166)
        - einsum("ijkabc->ijkabc", tau171)
        + 2 * einsum("ijkabc->ijkabc", tau176)
    )

    tau178 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau179 = (
        einsum("pw,wij->pij", tau178, h.l.poo)
    )

    tau180 = (
        einsum("aj,pji->pia", a.t1, tau179)
    )

    tau181 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau180)
    )

    tau182 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau183 = (
        einsum("piw,pjw->pij", tau182, tau6)
    )

    tau184 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau183)
    )

    tau185 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau184)
    )

    tau186 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau185)
    )

    tau187 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau188 = (
        einsum("paw,piw->pia", tau187, tau6)
    )

    tau189 = (
        einsum("q,iq,pia->pqa", a.t2.xlam[0, :], a.t2.x3, tau188)
    )

    tau190 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau189)
    )

    tau191 = (
        einsum("bp,cp,jp,pika->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau190)
    )

    tau192 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau7)
    )

    tau193 = (
        einsum("ai,pqi->pqa", a.t1, tau192)
    )

    tau194 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau193)
    )

    tau195 = (
        einsum("bp,cp,jp,pika->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau194)
    )

    tau196 = (
        einsum("ijkabc->ijkabc", tau181)
        - einsum("ijkabc->ijkabc", tau186)
        - einsum("ijkabc->ijkabc", tau191)
        + einsum("ijkabc->ijkabc", tau195)
    )

    tau197 = (
        einsum("q,iq,pia->pqa", a.t2.xlam[0, :], a.t2.x4, tau188)
    )

    tau198 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau197)
    )

    tau199 = (
        einsum("bp,cp,ip,pjka->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau198)
    )

    tau200 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau7)
    )

    tau201 = (
        einsum("ai,pqi->pqa", a.t1, tau200)
    )

    tau202 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau201)
    )

    tau203 = (
        einsum("bp,cp,ip,pjka->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau202)
    )

    tau204 = (
        einsum("ijkabc->ijkabc", tau199)
        - einsum("ijkabc->ijkabc", tau203)
    )

    tau205 = (
        einsum("paw,piw->pia", tau148, tau25)
    )

    tau206 = (
        einsum("q,iq,pia->pqa", a.t2.xlam[0, :], a.t2.x3, tau205)
    )

    tau207 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau206)
    )

    tau208 = (
        einsum("bp,cp,jp,pika->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau207)
    )

    tau209 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau27)
    )

    tau210 = (
        einsum("ai,pqi->pqa", a.t1, tau209)
    )

    tau211 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau210)
    )

    tau212 = (
        einsum("bp,cp,jp,pika->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau211)
    )

    tau213 = (
        einsum("ijkabc->ijkabc", tau208)
        - einsum("ijkabc->ijkabc", tau212)
    )

    tau214 = (
        einsum("q,iq,pia->pqa", a.t2.xlam[0, :], a.t2.x4, tau205)
    )

    tau215 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau214)
    )

    tau216 = (
        einsum("bp,cp,ip,pjka->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau215)
    )

    tau217 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau27)
    )

    tau218 = (
        einsum("ai,pqi->pqa", a.t1, tau217)
    )

    tau219 = (
        einsum("q,iq,jq,qpa->pija", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau218)
    )

    tau220 = (
        einsum("bp,cp,ip,pjka->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau219)
    )

    tau221 = (
        einsum("ijkabc->ijkabc", tau216)
        - einsum("ijkabc->ijkabc", tau220)
    )

    tau222 = (
        einsum("ap,ia->pi", a.t2.x2, h.f.ov)
    )

    tau223 = (
        einsum("iq,pi->pq", a.t2.x3, tau222)
    )

    tau224 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau223)
    )

    tau225 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau224)
    )

    tau226 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau227 = (
        einsum("paw,piw->pia", tau226, tau6)
    )

    tau228 = (
        einsum("ai,pja->pij", a.t1, tau227)
    )

    tau229 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau228)
    )

    tau230 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau229)
    )

    tau231 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau230)
    )

    tau232 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau233 = (
        einsum("wib,baw->ia", h.l.pov, tau232)
    )

    tau234 = (
        einsum("ap,ia->pi", a.t2.x2, tau233)
    )

    tau235 = (
        einsum("iq,pi->pq", a.t2.x3, tau234)
    )

    tau236 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau235)
    )

    tau237 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau236)
    )

    tau238 = (
        einsum("w,wia->ia", tau120, h.l.pov)
    )

    tau239 = (
        einsum("ap,ia->pi", a.t2.x2, tau238)
    )

    tau240 = (
        einsum("iq,pi->pq", a.t2.x3, tau239)
    )

    tau241 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau240)
    )

    tau242 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau241)
    )

    tau243 = (
        einsum("ijkabc->ijkabc", tau225)
        - einsum("ijkabc->ijkabc", tau231)
        - einsum("ijkabc->ijkabc", tau237)
        + 2 * einsum("ijkabc->ijkabc", tau242)
    )

    tau244 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau228)
    )

    tau245 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau244)
    )

    tau246 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau245)
    )

    tau247 = (
        einsum("ai,pja->pij", a.t1, tau135)
    )

    tau248 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau247)
    )

    tau249 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau248)
    )

    tau250 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau249)
    )

    tau251 = (
        einsum("iq,pi->pq", a.t2.x4, tau222)
    )

    tau252 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau251)
    )

    tau253 = (
        einsum("ap,cp,ip,pjkb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau252)
    )

    tau254 = (
        einsum("ai,pja->pij", a.t1, tau167)
    )

    tau255 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau254)
    )

    tau256 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau255)
    )

    tau257 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau256)
    )

    tau258 = (
        einsum("p,ip,wia->paw", a.t2.xlam[0, :], a.t2.x3, h.l.pov)
    )

    tau259 = (
        einsum("pbw,paw->pab", tau149, tau258)
    )

    tau260 = (
        einsum("bp,qab->pqa", a.t2.x2, tau259)
    )

    tau261 = (
        einsum("ai,pqa->pqi", a.t1, tau260)
    )

    tau262 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau261)
    )

    tau263 = (
        einsum("p,bp,jp,kp,piac->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau262)
    )

    tau264 = (
        einsum("ai,pja->pij", a.t1, tau172)
    )

    tau265 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau264)
    )

    tau266 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau265)
    )

    tau267 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau266)
    )

    tau268 = (
        einsum("iq,pi->pq", a.t2.x4, tau234)
    )

    tau269 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau268)
    )

    tau270 = (
        einsum("ap,cp,ip,pjkb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau269)
    )

    tau271 = (
        einsum("iq,pi->pq", a.t2.x4, tau239)
    )

    tau272 = (
        einsum("p,q,aq,iq,jq,qp->pija", a.t2.xlam[0, :],
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau271)
    )

    tau273 = (
        einsum("ap,cp,ip,pjkb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau272)
    )

    tau274 = (
        einsum("ijkabc->ijkabc", tau253)
        - einsum("ijkabc->ijkabc", tau257)
        - einsum("ijkabc->ijkabc", tau263)
        + 2 * einsum("ijkabc->ijkabc", tau267)
        - einsum("ijkabc->ijkabc", tau270)
        + 2 * einsum("ijkabc->ijkabc", tau273)
    )

    tau275 = (
        einsum("ai,pja->pij", a.t1, tau1)
    )

    tau276 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau275)
    )

    tau277 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau276)
    )

    tau278 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau277)
    )

    tau279 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau280 = (
        einsum("paw,piw->pia", tau279, tau6)
    )

    tau281 = (
        einsum("ai,pja->pij", a.t1, tau280)
    )

    tau282 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau281)
    )

    tau283 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau282)
    )

    tau284 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau283)
    )

    tau285 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau264)
    )

    tau286 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau285)
    )

    tau287 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau286)
    )

    tau288 = (
        einsum("ijkabc->ijkabc", tau278)
        + einsum("ijkabc->ijkabc", tau284)
        - 2 * einsum("ijkabc->ijkabc", tau287)
    )

    tau289 = (
        einsum("bp,qba->pqa", a.t2.x2, tau259)
    )

    tau290 = (
        einsum("ai,pqa->pqi", a.t1, tau289)
    )

    tau291 = (
        einsum("aq,bq,pqi->piab", a.t2.x1, a.t2.x2, tau290)
    )

    tau292 = (
        einsum("p,bp,jp,kp,piac->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau291)
    )

    tau293 = (
        einsum("p,jp,wji->piw", a.t2.xlam[0, :], a.t2.x3, h.l.poo)
    )

    tau294 = (
        einsum("wai,pjw->pija", h.l.pvo, tau293)
    )

    tau295 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau294)
    )

    tau296 = (
        einsum("iaw,pjw->pija", tau145, tau293)
    )

    tau297 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau296)
    )

    tau298 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau299 = (
        einsum("paw,piw->pia", tau129, tau298)
    )

    tau300 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau299)
    )

    tau301 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau302 = (
        einsum("pw,wai->pia", tau301, h.l.pvo)
    )

    tau303 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau302)
    )

    tau304 = (
        einsum("pw,wai->pia", tau44, h.l.pvo)
    )

    tau305 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau304)
    )

    tau306 = (
        einsum("p,jp,ijw->piw", a.t2.xlam[0, :], a.t2.x3, tau80)
    )

    tau307 = (
        einsum("iaw,pjw->pija", tau145, tau306)
    )

    tau308 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau307)
    )

    tau309 = (
        einsum("p,bp,wab->paw", a.t3.xlam[0, :], a.t3.x3, h.l.pvv)
    )

    tau310 = (
        einsum("paw,pbw->pab", tau309, tau94)
    )

    tau311 = (
        einsum("bi,pab->pia", a.t1, tau310)
    )

    tau312 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau311)
    )

    tau313 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau314 = (
        einsum("pw,wab->pab", tau313, h.l.pvv)
    )

    tau315 = (
        einsum("bi,pab->pia", a.t1, tau314)
    )

    tau316 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau315)
    )

    tau317 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau318 = (
        einsum("pw,wab->pab", tau317, h.l.pvv)
    )

    tau319 = (
        einsum("bi,pab->pia", a.t1, tau318)
    )

    tau320 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau319)
    )

    tau321 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau322 = (
        einsum("aj,ijw->iaw", a.t1, tau321)
    )

    tau323 = (
        einsum("pjw,iaw->pija", tau306, tau322)
    )

    tau324 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau323)
    )

    tau325 = (
        einsum("p,ap,wia->piw", a.t3.xlam[0, :], a.t3.x3, h.l.pov)
    )

    tau326 = (
        einsum("piw,paw->pia", tau325, tau94)
    )

    tau327 = (
        einsum("ai,pja->pij", a.t1, tau326)
    )

    tau328 = (
        einsum("aj,pij->pia", a.t1, tau327)
    )

    tau329 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau328)
    )

    tau330 = (
        einsum("pw,wia->pia", tau313, h.l.pov)
    )

    tau331 = (
        einsum("ai,pja->pij", a.t1, tau330)
    )

    tau332 = (
        einsum("aj,pij->pia", a.t1, tau331)
    )

    tau333 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau332)
    )

    tau334 = (
        einsum("pw,wia->pia", tau317, h.l.pov)
    )

    tau335 = (
        einsum("ai,pja->pij", a.t1, tau334)
    )

    tau336 = (
        einsum("aj,pij->pia", a.t1, tau335)
    )

    tau337 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau336)
    )

    tau338 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau339 = (
        einsum("pw,wia->pia", tau338, h.l.pov)
    )

    tau340 = (
        einsum("ip,pia->pa", a.t3.x6, tau339)
    )

    tau341 = (
        einsum("ap,qa->pq", a.t2.x2, tau340)
    )

    tau342 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x5, tau341)
    )

    tau343 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau342)
    )

    tau344 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x5, tau280)
    )

    tau345 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau344)
    )

    tau346 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau345)
    )

    tau347 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x6, tau280)
    )

    tau348 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau347)
    )

    tau349 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau348)
    )

    tau350 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x6, tau227)
    )

    tau351 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau350)
    )

    tau352 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau351)
    )

    tau353 = (
        einsum("pw,wia->pia", tau301, h.l.pov)
    )

    tau354 = (
        einsum("ip,pia->pa", a.t3.x4, tau353)
    )

    tau355 = (
        einsum("ap,qa->pq", a.t2.x2, tau354)
    )

    tau356 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x6, tau355)
    )

    tau357 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau356)
    )

    tau358 = (
        einsum("ip,pia->pa", a.t3.x4, tau45)
    )

    tau359 = (
        einsum("ap,qa->pq", a.t2.x2, tau358)
    )

    tau360 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x5, tau359)
    )

    tau361 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau360)
    )

    tau362 = (
        einsum("qw,pw->pq", tau301, tau34)
    )

    tau363 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau362)
    )

    tau364 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau363)
    )

    tau365 = (
        einsum("pw,qw->pq", tau0, tau301)
    )

    tau366 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau365)
    )

    tau367 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau366)
    )

    tau368 = (
        einsum("pw,qw->pq", tau0, tau44)
    )

    tau369 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau368)
    )

    tau370 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau369)
    )

    tau371 = (
        einsum("pw,qw->pq", tau34, tau44)
    )

    tau372 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau371)
    )

    tau373 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x5, tau372)
    )

    tau374 = (
        einsum("ijkabc->ijkabc", tau295)
        + einsum("ijkabc->ijkabc", tau297)
        + einsum("ijkabc->ijkabc", tau300)
        + einsum("ijkabc->ijkabc", tau303)
        - 2 * einsum("ijkabc->ijkabc", tau305)
        + einsum("ijkabc->ijkabc", tau308)
        + einsum("ijkabc->ijkabc", tau312)
        + einsum("ijkabc->ijkabc", tau316)
        - 2 * einsum("ijkabc->ijkabc", tau320)
        - einsum("ijkabc->ijkabc", tau324)
        - einsum("ijkabc->ijkabc", tau329)
        - einsum("ijkabc->ijkabc", tau333)
        + 2 * einsum("ijkabc->ijkabc", tau337)
        - einsum("ijkabc->ijkabc", tau343)
        - einsum("ijkabc->ijkabc", tau346)
        + 2 * einsum("ijkabc->ijkabc", tau349)
        - einsum("ijkabc->ijkabc", tau352)
        - einsum("ijkabc->ijkabc", tau357)
        + 2 * einsum("ijkabc->ijkabc", tau361)
        + 2 * einsum("ijkabc->ijkabc", tau364)
        - einsum("ijkabc->ijkabc", tau367)
        + 2 * einsum("ijkabc->ijkabc", tau370)
        - 4 * einsum("ijkabc->ijkabc", tau373)
    )

    tau375 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau376 = (
        einsum("wai,pjw->pija", h.l.pvo, tau375)
    )

    tau377 = (
        einsum("p,bp,cp,kp,pija->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x2, a.t2.x3, tau376)
    )

    tau378 = (
        einsum("p,jp,wji->piw", a.t2.xlam[0, :], a.t2.x4, h.l.poo)
    )

    tau379 = (
        einsum("iaw,pjw->pija", tau145, tau378)
    )

    tau380 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau379)
    )

    tau381 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau382 = (
        einsum("pw,wai->pia", tau381, h.l.pvo)
    )

    tau383 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau382)
    )

    tau384 = (
        einsum("p,jp,ijw->piw", a.t2.xlam[0, :], a.t2.x4, tau80)
    )

    tau385 = (
        einsum("iaw,pjw->pija", tau145, tau384)
    )

    tau386 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau385)
    )

    tau387 = (
        einsum("pw,wab->pab", tau178, h.l.pvv)
    )

    tau388 = (
        einsum("bi,pab->pia", a.t1, tau387)
    )

    tau389 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau388)
    )

    tau390 = (
        einsum("iaw,pjw->pija", tau322, tau384)
    )

    tau391 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau390)
    )

    tau392 = (
        einsum("pw,wia->pia", tau178, h.l.pov)
    )

    tau393 = (
        einsum("ai,pja->pij", a.t1, tau392)
    )

    tau394 = (
        einsum("aj,pij->pia", a.t1, tau393)
    )

    tau395 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau394)
    )

    tau396 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau397 = (
        einsum("pw,wia->pia", tau396, h.l.pov)
    )

    tau398 = (
        einsum("ip,pia->pa", a.t3.x6, tau397)
    )

    tau399 = (
        einsum("ap,qa->pq", a.t2.x2, tau398)
    )

    tau400 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x4, tau399)
    )

    tau401 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau400)
    )

    tau402 = (
        einsum("ip,pia->pa", a.t3.x5, tau339)
    )

    tau403 = (
        einsum("ap,qa->pq", a.t2.x2, tau402)
    )

    tau404 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x6, tau403)
    )

    tau405 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau404)
    )

    tau406 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x4, tau280)
    )

    tau407 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau406)
    )

    tau408 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau407)
    )

    tau409 = (
        einsum("ip,pia->pa", a.t3.x5, tau45)
    )

    tau410 = (
        einsum("ap,qa->pq", a.t2.x2, tau409)
    )

    tau411 = (
        einsum("p,q,aq,bq,iq,pq->piab", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, a.t3.x4, tau410)
    )

    tau412 = (
        einsum("ap,ip,jp,pkbc->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau411)
    )

    tau413 = (
        einsum("pw,qw->pq", tau34, tau381)
    )

    tau414 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau413)
    )

    tau415 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau414)
    )

    tau416 = (
        einsum("pw,qw->pq", tau0, tau381)
    )

    tau417 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau416)
    )

    tau418 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau417)
    )

    tau419 = (
        einsum("ijkabc->ijkabc", tau377)
        + einsum("ijkabc->ijkabc", tau380)
        + einsum("ijkabc->ijkabc", tau383)
        + einsum("ijkabc->ijkabc", tau386)
        + einsum("ijkabc->ijkabc", tau389)
        - einsum("ijkabc->ijkabc", tau391)
        - einsum("ijkabc->ijkabc", tau395)
        - einsum("ijkabc->ijkabc", tau401)
        - einsum("ijkabc->ijkabc", tau405)
        - einsum("ijkabc->ijkabc", tau408)
        + 2 * einsum("ijkabc->ijkabc", tau412)
        + 2 * einsum("ijkabc->ijkabc", tau415)
        - einsum("ijkabc->ijkabc", tau418)
    )

    tau420 = (
        einsum("ap,ia->pi", a.t3.x3, h.f.ov)
    )

    tau421 = (
        einsum("ai,pi->pa", a.t1, tau420)
    )

    tau422 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau421)
    )

    tau423 = (
        einsum("paw,piw->pia", tau129, tau56)
    )

    tau424 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau423)
    )

    tau425 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau426 = (
        einsum("paw,pbw->pab", tau309, tau425)
    )

    tau427 = (
        einsum("bi,pab->pia", a.t1, tau426)
    )

    tau428 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau427)
    )

    tau429 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau430 = (
        einsum("wac,cbw->ab", h.l.pvv, tau429)
    )

    tau431 = (
        einsum("bp,ab->pa", a.t3.x3, tau430)
    )

    tau432 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau431)
    )

    tau433 = (
        einsum("w,wab->ab", tau123, h.l.pvv)
    )

    tau434 = (
        einsum("bp,ab->pa", a.t3.x3, tau433)
    )

    tau435 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau434)
    )

    tau436 = (
        einsum("w,wia->ia", tau123, h.l.pov)
    )

    tau437 = (
        einsum("ap,ia->pi", a.t3.x3, tau436)
    )

    tau438 = (
        einsum("ai,pi->pa", a.t1, tau437)
    )

    tau439 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau438)
    )

    tau440 = (
        einsum("wib,baw->ia", h.l.pov, tau429)
    )

    tau441 = (
        einsum("ap,ia->pi", a.t3.x3, tau440)
    )

    tau442 = (
        einsum("ai,pi->pa", a.t1, tau441)
    )

    tau443 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau442)
    )

    tau444 = (
        einsum("piw,paw->pia", tau325, tau425)
    )

    tau445 = (
        einsum("ai,pja->pij", a.t1, tau444)
    )

    tau446 = (
        einsum("aj,pij->pia", a.t1, tau445)
    )

    tau447 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau446)
    )

    tau448 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x4, tau227)
    )

    tau449 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau448)
    )

    tau450 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x5, a.t3.x6, tau449)
    )

    tau451 = (
        einsum("ip,pia->pa", a.t2.x3, tau172)
    )

    tau452 = (
        einsum("aq,pa->pq", a.t3.x3, tau451)
    )

    tau453 = (
        einsum("q,aq,qp->pa", a.t2.xlam[0, :], a.t2.x1, tau452)
    )

    tau454 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau453)
    )

    tau455 = (
        einsum("ip,pia->pa", a.t2.x4, tau1)
    )

    tau456 = (
        einsum("aq,pa->pq", a.t3.x3, tau455)
    )

    tau457 = (
        einsum("q,aq,qp->pa", a.t2.xlam[0, :], a.t2.x1, tau456)
    )

    tau458 = (
        einsum("p,bp,cp,ip,jp,kp,pa->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau457)
    )

    tau459 = (
        einsum("ijkabc->ijkabc", tau422)
        + einsum("ijkabc->ijkabc", tau424)
        + einsum("ijkabc->ijkabc", tau428)
        + einsum("ijkabc->ijkabc", tau432)
        - 2 * einsum("ijkabc->ijkabc", tau435)
        + 2 * einsum("ijkabc->ijkabc", tau439)
        - einsum("ijkabc->ijkabc", tau443)
        - einsum("ijkabc->ijkabc", tau447)
        - einsum("ijkabc->ijkabc", tau450)
        + 2 * einsum("ijkabc->ijkabc", tau454)
        - einsum("ijkabc->ijkabc", tau458)
    )

    tau460 = (
        einsum("paw,piw->pia", tau129, tau54)
    )

    tau461 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau460)
    )

    tau462 = (
        einsum("paw,pbw->pab", tau309, tau87)
    )

    tau463 = (
        einsum("bi,pab->pia", a.t1, tau462)
    )

    tau464 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau463)
    )

    tau465 = (
        einsum("piw,paw->pia", tau325, tau87)
    )

    tau466 = (
        einsum("ai,pja->pij", a.t1, tau465)
    )

    tau467 = (
        einsum("aj,pij->pia", a.t1, tau466)
    )

    tau468 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau467)
    )

    tau469 = (
        einsum("aq,iq,pia->pq", a.t3.x3, a.t3.x5, tau227)
    )

    tau470 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau469)
    )

    tau471 = (
        einsum("p,bp,cp,jp,kp,pia->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau470)
    )

    tau472 = (
        einsum("ijkabc->ijkabc", tau461)
        + einsum("ijkabc->ijkabc", tau464)
        - einsum("ijkabc->ijkabc", tau468)
        - einsum("ijkabc->ijkabc", tau471)
    )

    tau473 = (
        einsum("jp,ji->pi", a.t3.x6, h.f.oo)
    )

    tau474 = (
        einsum("p,ap,bp,cp,ip,jp,pk->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau473)
    )

    tau475 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau476 = (
        einsum("jp,ij->pi", a.t3.x5, tau475)
    )

    tau477 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau476)
    )

    tau478 = (
        einsum("jp,ij->pi", a.t3.x4, tau475)
    )

    tau479 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau478)
    )

    tau480 = (
        einsum("paw,pbw->pab", tau83, tau87)
    )

    tau481 = (
        einsum("bi,pab->pia", a.t1, tau480)
    )

    tau482 = (
        einsum("ai,pja->pij", a.t1, tau481)
    )

    tau483 = (
        einsum("ap,bp,cp,kp,pij->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x6, tau482)
    )

    tau484 = (
        einsum("ai,ja->ij", a.t1, tau233)
    )

    tau485 = (
        einsum("jp,ij->pi", a.t3.x4, tau484)
    )

    tau486 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau485)
    )

    tau487 = (
        einsum("ai,ja->ij", a.t1, tau238)
    )

    tau488 = (
        einsum("jp,ij->pi", a.t3.x5, tau487)
    )

    tau489 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau488)
    )

    tau490 = (
        einsum("jp,ij->pi", a.t3.x5, tau484)
    )

    tau491 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau490)
    )

    tau492 = (
        einsum("jp,ij->pi", a.t3.x4, tau487)
    )

    tau493 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau492)
    )

    tau494 = (
        einsum("pw,wia->pia", tau381, h.l.pov)
    )

    tau495 = (
        einsum("ap,pia->pi", a.t3.x2, tau494)
    )

    tau496 = (
        einsum("ip,qi->pq", a.t2.x4, tau495)
    )

    tau497 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x5, a.t3.x6, tau496)
    )

    tau498 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau497)
    )

    tau499 = (
        - einsum("ijkabc->ijkabc", tau474)
        + einsum("ijkabc->ijkabc", tau477)
        - einsum("ijkabc->ijkabc", tau479)
        + einsum("ijkabc->ijkabc", tau483)
        + einsum("ijkabc->ijkabc", tau486)
        + 2 * einsum("ijkabc->ijkabc", tau489)
        - einsum("ijkabc->ijkabc", tau491)
        - 2 * einsum("ijkabc->ijkabc", tau493)
        + einsum("ijkabc->ijkabc", tau498)
    )

    tau500 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x5, h.l.pov)
    )

    tau501 = (
        einsum("pw,wia->pia", tau500, h.l.pov)
    )

    tau502 = (
        einsum("ap,pia->pi", a.t3.x3, tau501)
    )

    tau503 = (
        einsum("ip,qi->pq", a.t2.x4, tau502)
    )

    tau504 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x6, tau503)
    )

    tau505 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau504)
    )

    tau506 = (
        einsum("ap,pia->pi", a.t3.x2, tau353)
    )

    tau507 = (
        einsum("ip,qi->pq", a.t2.x4, tau506)
    )

    tau508 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x6, tau507)
    )

    tau509 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau508)
    )

    tau510 = (
        2 * einsum("ijkabc->ijkabc", tau505)
        - einsum("ijkabc->ijkabc", tau509)
    )

    tau511 = (
        einsum("bp,wab->paw", a.t3.x2, h.l.pvv)
    )

    tau512 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau513 = (
        einsum("paw,pbw->pab", tau511, tau512)
    )

    tau514 = (
        einsum("p,cp,ip,jp,kp,pab->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x4, a.t3.x5, a.t3.x6, tau513)
    )

    tau515 = (
        einsum("p,ap,wia->piw", a.t3.xlam[0, :], a.t3.x2, h.l.pov)
    )

    tau516 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau517 = (
        einsum("piw,pjw->pij", tau515, tau516)
    )

    tau518 = (
        einsum("aj,pij->pia", a.t1, tau517)
    )

    tau519 = (
        einsum("ai,pib->pab", a.t1, tau518)
    )

    tau520 = (
        einsum("cp,ip,jp,kp,pab->ijkabc", a.t3.x1,
               a.t3.x4, a.t3.x5, a.t3.x6, tau519)
    )

    tau521 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau522 = (
        einsum("pjw,piw->pij", tau516, tau521)
    )

    tau523 = (
        einsum("ip,jp,qij->pq", a.t2.x3, a.t2.x4, tau522)
    )

    tau524 = (
        einsum("q,aq,bq,qp->pab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau523)
    )

    tau525 = (
        einsum("p,cp,ip,jp,kp,pab->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x4, a.t3.x5, a.t3.x6, tau524)
    )

    tau526 = (
        einsum("ijkabc->ijkabc", tau514)
        + einsum("ijkabc->ijkabc", tau520)
        + einsum("ijkabc->ijkabc", tau525)
    )

    tau527 = (
        einsum("ip,qi->pq", a.t2.x3, tau17)
    )

    tau528 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x5, tau527)
    )

    tau529 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau528)
    )

    tau530 = (
        einsum("ip,qi->pq", a.t2.x3, tau46)
    )

    tau531 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x5, tau530)
    )

    tau532 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau531)
    )

    tau533 = (
        - einsum("ijkabc->ijkabc", tau529)
        + 2 * einsum("ijkabc->ijkabc", tau532)
    )

    tau534 = (
        einsum("ip,qi->pq", a.t2.x3, tau495)
    )

    tau535 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x5, a.t3.x6, tau534)
    )

    tau536 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau535)
    )

    tau537 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x4, h.l.pov)
    )

    tau538 = (
        einsum("pw,wia->pia", tau537, h.l.pov)
    )

    tau539 = (
        einsum("ap,pia->pi", a.t3.x3, tau538)
    )

    tau540 = (
        einsum("ip,qi->pq", a.t2.x4, tau539)
    )

    tau541 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x5, a.t3.x6, tau540)
    )

    tau542 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau541)
    )

    tau543 = (
        einsum("jp,ij->pi", a.t3.x6, tau475)
    )

    tau544 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau543)
    )

    tau545 = (
        einsum("jp,ji->pi", a.t3.x6, tau98)
    )

    tau546 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau545)
    )

    tau547 = (
        einsum("jp,ji->pi", a.t3.x6, tau121)
    )

    tau548 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau547)
    )

    tau549 = (
        einsum("jp,ij->pi", a.t3.x6, tau487)
    )

    tau550 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau549)
    )

    tau551 = (
        einsum("jp,ij->pi", a.t3.x6, tau484)
    )

    tau552 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau551)
    )

    tau553 = (
        einsum("iq,pi->pq", a.t3.x6, tau2)
    )

    tau554 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau553)
    )

    tau555 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau554)
    )

    tau556 = (
        einsum("iq,pi->pq", a.t3.x6, tau41)
    )

    tau557 = (
        einsum("q,iq,qp->pi", a.t2.xlam[0, :], a.t2.x4, tau556)
    )

    tau558 = (
        einsum("p,ap,bp,cp,jp,kp,pi->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x5, tau557)
    )

    tau559 = (
        einsum("ijkabc->ijkabc", tau544)
        - einsum("ijkabc->ijkabc", tau546)
        + 2 * einsum("ijkabc->ijkabc", tau548)
        + 2 * einsum("ijkabc->ijkabc", tau550)
        - einsum("ijkabc->ijkabc", tau552)
        - einsum("ijkabc->ijkabc", tau555)
        + 2 * einsum("ijkabc->ijkabc", tau558)
    )

    tau560 = (
        einsum("ip,qi->pq", a.t2.x3, tau539)
    )

    tau561 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x5, a.t3.x6, tau560)
    )

    tau562 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau561)
    )

    tau563 = (
        einsum("ip,qi->pq", a.t2.x3, tau502)
    )

    tau564 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x6, tau563)
    )

    tau565 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau564)
    )

    tau566 = (
        einsum("ip,qi->pq", a.t2.x3, tau506)
    )

    tau567 = (
        einsum("p,q,aq,iq,jq,pq->pija", a.t2.xlam[0, :],
               a.t3.xlam[0, :], a.t3.x1, a.t3.x4, a.t3.x6, tau566)
    )

    tau568 = (
        einsum("ap,bp,ip,pjkc->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau567)
    )

    tau569 = (
        2 * einsum("ijkabc->ijkabc", tau565)
        - einsum("ijkabc->ijkabc", tau568)
    )

    tau570 = (
        einsum("jp,ji->pi", a.t3.x5, h.f.oo)
    )

    tau571 = (
        einsum("p,ap,bp,cp,ip,jp,pk->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau570)
    )

    tau572 = (
        einsum("jp,ji->pi", a.t3.x4, h.f.oo)
    )

    tau573 = (
        einsum("p,ap,bp,cp,ip,jp,pk->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau572)
    )

    tau574 = (
        einsum("pjw,piw->pij", tau298, tau57)
    )

    tau575 = (
        einsum("p,ap,bp,cp,kp,pij->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, tau574)
    )

    tau576 = (
        einsum("pjw,piw->pij", tau298, tau53)
    )

    tau577 = (
        einsum("p,ap,bp,cp,kp,pij->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, tau576)
    )

    tau578 = (
        einsum("p,ip,wia->paw", a.t3.xlam[0, :], a.t3.x5, h.l.pov)
    )

    tau579 = (
        einsum("piw,paw->pia", tau298, tau578)
    )

    tau580 = (
        einsum("ai,pja->pij", a.t1, tau579)
    )

    tau581 = (
        einsum("ap,bp,cp,kp,pij->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, tau580)
    )

    tau582 = (
        einsum("piw,paw->pia", tau298, tau83)
    )

    tau583 = (
        einsum("ai,pja->pij", a.t1, tau582)
    )

    tau584 = (
        einsum("ap,bp,cp,kp,pij->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, tau583)
    )

    tau585 = (
        einsum("paw,pbw->pab", tau578, tau94)
    )

    tau586 = (
        einsum("bi,pab->pia", a.t1, tau585)
    )

    tau587 = (
        einsum("ai,pja->pij", a.t1, tau586)
    )

    tau588 = (
        einsum("ap,bp,cp,kp,pij->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x4, tau587)
    )

    tau589 = (
        einsum("paw,pbw->pab", tau83, tau94)
    )

    tau590 = (
        einsum("bi,pab->pia", a.t1, tau589)
    )

    tau591 = (
        einsum("ai,pja->pij", a.t1, tau590)
    )

    tau592 = (
        einsum("ap,bp,cp,kp,pij->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x3, a.t3.x5, tau591)
    )

    tau593 = (
        einsum("iq,jq,pij->pq", a.t3.x5, a.t3.x6, tau7)
    )

    tau594 = (
        einsum("q,iq,jq,qp->pij", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau593)
    )

    tau595 = (
        einsum("p,ap,bp,cp,kp,pij->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, tau594)
    )

    tau596 = (
        einsum("iq,jq,pij->pq", a.t3.x4, a.t3.x6, tau7)
    )

    tau597 = (
        einsum("q,iq,jq,qp->pij", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau596)
    )

    tau598 = (
        einsum("p,ap,bp,cp,kp,pij->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, tau597)
    )

    tau599 = (
        einsum("ijkabc->ijkabc", tau571)
        - einsum("ijkabc->ijkabc", tau573)
        + einsum("ijkabc->ijkabc", tau575)
        - einsum("ijkabc->ijkabc", tau577)
        + einsum("ijkabc->ijkabc", tau581)
        - einsum("ijkabc->ijkabc", tau584)
        + einsum("ijkabc->ijkabc", tau588)
        - einsum("ijkabc->ijkabc", tau592)
        + einsum("ijkabc->ijkabc", tau595)
        - einsum("ijkabc->ijkabc", tau598)
    )

    tau600 = (
        einsum("p,bp,wab->paw", a.t2.xlam[0, :], a.t2.x2, h.l.pvv)
    )

    tau601 = (
        einsum("pbw,iaw->piab", tau600, tau97)
    )

    tau602 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau601)
    )

    tau603 = (
        einsum("pbw,iaw->piab", tau600, tau81)
    )

    tau604 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau603)
    )

    tau605 = (
        einsum("ijkabc->ijkabc", tau602)
        + einsum("ijkabc->ijkabc", tau604)
    )

    tau606 = (
        einsum("waj,piw->pija", h.l.pvo, tau384)
    )

    tau607 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau606)
    )

    tau608 = (
        einsum("waj,piw->pija", h.l.pvo, tau306)
    )

    tau609 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau608)
    )

    tau610 = (
        einsum("iaw,pjw->pija", tau100, tau293)
    )

    tau611 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau610)
    )

    tau612 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau613 = (
        einsum("pjw,piw->pij", tau103, tau612)
    )

    tau614 = (
        einsum("aj,pij->pia", a.t1, tau613)
    )

    tau615 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau614)
    )

    tau616 = (
        einsum("pw,wij->pij", tau313, h.l.poo)
    )

    tau617 = (
        einsum("aj,pji->pia", a.t1, tau616)
    )

    tau618 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau617)
    )

    tau619 = (
        einsum("pw,wij->pij", tau317, h.l.poo)
    )

    tau620 = (
        einsum("aj,pji->pia", a.t1, tau619)
    )

    tau621 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau620)
    )

    tau622 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x4, tau183)
    )

    tau623 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau622)
    )

    tau624 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau623)
    )

    tau625 = (
        einsum("ijkabc->ijkabc", tau611)
        + einsum("ijkabc->ijkabc", tau615)
        + einsum("ijkabc->ijkabc", tau618)
        - 2 * einsum("ijkabc->ijkabc", tau621)
        - einsum("ijkabc->ijkabc", tau624)
    )

    tau626 = (
        einsum("iaw,pjw->pija", tau100, tau378)
    )

    tau627 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau626)
    )

    tau628 = (
        einsum("p,jp,wji->piw", a.t2.xlam[0, :], a.t2.x3, h.l.poo)
    )

    tau629 = (
        einsum("piw,jaw->pija", tau628, tau97)
    )

    tau630 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau629)
    )

    tau631 = (
        einsum("p,jp,ijw->piw", a.t2.xlam[0, :], a.t2.x3, tau321)
    )

    tau632 = (
        einsum("piw,jaw->pija", tau631, tau97)
    )

    tau633 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau632)
    )

    tau634 = (
        einsum("ijkabc->ijkabc", tau630)
        + einsum("ijkabc->ijkabc", tau633)
    )

    tau635 = (
        einsum("piw,jaw->pija", tau79, tau97)
    )

    tau636 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau635)
    )

    tau637 = (
        einsum("wbi,paw->piab", h.l.pvo, tau600)
    )

    tau638 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau637)
    )

    tau639 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau640 = (
        einsum("paw,ibw->piab", tau600, tau639)
    )

    tau641 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau640)
    )

    tau642 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x2, h.l.pov)
    )

    tau643 = (
        einsum("ai,piw->paw", a.t1, tau642)
    )

    tau644 = (
        einsum("paw,ibw->piab", tau643, tau81)
    )

    tau645 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau644)
    )

    tau646 = (
        einsum("ijkabc->ijkabc", tau638)
        + einsum("ijkabc->ijkabc", tau641)
        + einsum("ijkabc->ijkabc", tau645)
    )

    tau647 = (
        einsum("p,bp,wab->paw", a.t2.xlam[0, :], a.t2.x2, h.l.pvv)
    )

    tau648 = (
        einsum("wai,pbw->piab", h.l.pvo, tau647)
    )

    tau649 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau648)
    )

    tau650 = (
        einsum("iaw,pbw->piab", tau145, tau647)
    )

    tau651 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau650)
    )

    tau652 = (
        einsum("pbw,iaw->piab", tau142, tau322)
    )

    tau653 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau652)
    )

    tau654 = (
        einsum("ijkabc->ijkabc", tau649)
        + einsum("ijkabc->ijkabc", tau651)
        + einsum("ijkabc->ijkabc", tau653)
    )

    tau655 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau656 = (
        einsum("piw,pjw->pij", tau26, tau655)
    )

    tau657 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau656)
    )

    tau658 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau657)
    )

    tau659 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau658)
    )

    tau660 = (
        einsum("piw,pjw->pij", tau26, tau375)
    )

    tau661 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau660)
    )

    tau662 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau661)
    )

    tau663 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau662)
    )

    tau664 = (
        einsum("pw,wij->pij", tau0, h.l.poo)
    )

    tau665 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau664)
    )

    tau666 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau665)
    )

    tau667 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau666)
    )

    tau668 = (
        einsum("pw,wij->pij", tau34, h.l.poo)
    )

    tau669 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x4, tau668)
    )

    tau670 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau669)
    )

    tau671 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau670)
    )

    tau672 = (
        einsum("ijkabc->ijkabc", tau663)
        + einsum("ijkabc->ijkabc", tau667)
        - 2 * einsum("ijkabc->ijkabc", tau671)
    )

    tau673 = (
        einsum("q,jq,pij->pqi", a.t2.xlam[0, :], a.t2.x3, tau72)
    )

    tau674 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau673)
    )

    tau675 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau674)
    )

    tau676 = (
        einsum("paw,piw->pia", tau149, tau182)
    )

    tau677 = (
        einsum("ap,qia->pqi", a.t2.x2, tau676)
    )

    tau678 = (
        einsum("q,aq,bq,pqi->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau677)
    )

    tau679 = (
        einsum("p,bp,jp,kp,piac->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau678)
    )

    tau680 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau76)
    )

    tau681 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau680)
    )

    tau682 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau681)
    )

    tau683 = (
        einsum("ijkabc->ijkabc", tau675)
        + einsum("ijkabc->ijkabc", tau679)
        - 2 * einsum("ijkabc->ijkabc", tau682)
    )

    tau684 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau656)
    )

    tau685 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau684)
    )

    tau686 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau685)
    )

    tau687 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau69)
    )

    tau688 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau687)
    )

    tau689 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau688)
    )

    tau690 = (
        einsum("paw,piw->pia", tau226, tau375)
    )

    tau691 = (
        einsum("ap,qia->pqi", a.t2.x2, tau690)
    )

    tau692 = (
        einsum("q,aq,bq,pqi->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau691)
    )

    tau693 = (
        einsum("p,bp,jp,kp,piac->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau692)
    )

    tau694 = (
        einsum("paw,piw->pia", tau279, tau655)
    )

    tau695 = (
        einsum("ap,qia->pqi", a.t2.x2, tau694)
    )

    tau696 = (
        einsum("q,aq,bq,pqi->piab", a.t2.xlam[0, :], a.t2.x1, a.t2.x2, tau695)
    )

    tau697 = (
        einsum("p,bp,jp,kp,piac->ijkabc", a.t2.xlam[0, :],
               a.t2.x1, a.t2.x3, a.t2.x4, tau696)
    )

    tau698 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau660)
    )

    tau699 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau698)
    )

    tau700 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau699)
    )

    tau701 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau668)
    )

    tau702 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x3, tau701)
    )

    tau703 = (
        einsum("ap,cp,kp,pijb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau702)
    )

    tau704 = (
        einsum("ijkabc->ijkabc", tau697)
        + einsum("ijkabc->ijkabc", tau700)
        - 2 * einsum("ijkabc->ijkabc", tau703)
    )

    tau705 = (
        einsum("q,jq,pji->pqi", a.t2.xlam[0, :], a.t2.x3, tau664)
    )

    tau706 = (
        einsum("q,aq,jq,qpi->pija", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau705)
    )

    tau707 = (
        einsum("ap,cp,jp,pikb->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau706)
    )

    tau708 = (
        einsum("paw,ibw->piab", tau643, tau97)
    )

    tau709 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau708)
    )

    tau710 = (
        einsum("iaw,pbw->piab", tau100, tau142)
    )

    tau711 = (
        einsum("cp,jp,kp,piab->ijkabc", a.t2.x1, a.t2.x3, a.t2.x4, tau710)
    )

    tau712 = (
        einsum("pjw,iaw->pija", tau293, tau322)
    )

    tau713 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau712)
    )

    tau714 = (
        einsum("iaw,pjw->pija", tau322, tau378)
    )

    tau715 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau714)
    )

    tau716 = (
        einsum("jaw,piw->pija", tau100, tau306)
    )

    tau717 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau716)
    )

    tau718 = (
        einsum("jaw,piw->pija", tau100, tau384)
    )

    tau719 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau718)
    )

    tau720 = (
        einsum("pjw,iaw->pija", tau628, tau81)
    )

    tau721 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x4, tau720)
    )

    tau722 = (
        einsum("p,jp,ijw->piw", a.t2.xlam[0, :], a.t2.x4, tau321)
    )

    tau723 = (
        einsum("piw,jaw->pija", tau722, tau97)
    )

    tau724 = (
        einsum("bp,cp,kp,pija->ijkabc", a.t2.x1, a.t2.x2, a.t2.x3, tau723)
    )

    tau725 = (
        einsum("piw,pjw->pij", tau325, tau56)
    )

    tau726 = (
        einsum("aj,pji->pia", a.t1, tau725)
    )

    tau727 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau726)
    )

    tau728 = (
        einsum("piw,pjw->pij", tau325, tau54)
    )

    tau729 = (
        einsum("aj,pji->pia", a.t1, tau728)
    )

    tau730 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau729)
    )

    tau731 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau732 = (
        einsum("pw,wij->pij", tau731, h.l.poo)
    )

    tau733 = (
        einsum("aj,pji->pia", a.t1, tau732)
    )

    tau734 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x6, tau733)
    )

    tau735 = (
        einsum("pjw,piw->pij", tau298, tau325)
    )

    tau736 = (
        einsum("aj,pji->pia", a.t1, tau735)
    )

    tau737 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau736)
    )

    tau738 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau739 = (
        einsum("pw,wij->pij", tau738, h.l.poo)
    )

    tau740 = (
        einsum("aj,pji->pia", a.t1, tau739)
    )

    tau741 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, tau740)
    )

    tau742 = (
        einsum("ijkabc->ijkabc", tau734)
        + einsum("ijkabc->ijkabc", tau737)
        - 2 * einsum("ijkabc->ijkabc", tau741)
    )

    tau743 = (
        einsum("p,ap,ip,wia->pw", a.t3.xlam[0, :], a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau744 = (
        einsum("pw,wij->pij", tau743, h.l.poo)
    )

    tau745 = (
        einsum("aj,pji->pia", a.t1, tau744)
    )

    tau746 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau745)
    )

    tau747 = (
        einsum("pjw,piw->pij", tau103, tau53)
    )

    tau748 = (
        einsum("aj,pij->pia", a.t1, tau747)
    )

    tau749 = (
        einsum("bp,cp,jp,kp,pia->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau748)
    )

    tau750 = (
        einsum("bp,ab->pa", a.t3.x3, h.f.vv)
    )

    tau751 = (
        einsum("p,ap,bp,ip,jp,kp,pc->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau750)
    )

    tau752 = (
        einsum("paw,piw->pia", tau129, tau59)
    )

    tau753 = (
        einsum("ai,pib->pab", a.t1, tau752)
    )

    tau754 = (
        einsum("cp,ip,jp,kp,pab->ijkabc", a.t3.x1,
               a.t3.x4, a.t3.x5, a.t3.x6, tau753)
    )

    tau755 = (
        einsum("p,bp,wab->paw", a.t3.xlam[0, :], a.t3.x2, h.l.pvv)
    )

    tau756 = (
        einsum("piw,paw->pia", tau516, tau755)
    )

    tau757 = (
        einsum("ai,pib->pab", a.t1, tau756)
    )

    tau758 = (
        einsum("cp,ip,jp,kp,pab->ijkabc", a.t3.x1,
               a.t3.x4, a.t3.x5, a.t3.x6, tau757)
    )

    rt3 = (
        einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau4)
        + einsum("p,ap,bp,cp,kp,pij->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x6, tau9)
        + einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau14)
        - einsum("ap,bp,jp,pkic->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau19)
        - einsum("p,cp,kp,ip,jp,pab->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x5, a.t3.x6, tau24)
        - einsum("p,ap,bp,cp,kp,pij->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x6, tau29)
        - einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau31)
        - einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau33)
        - 2 * einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau38)
        - 2 * einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau43)
        + 2 * einsum("ap,bp,jp,pkic->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau48)
        + 2 * einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau50)
        + 2 * einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau52)
        + einsum("p,ap,bp,cp,kp,pij->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x6, tau55)
        - einsum("p,ap,bp,cp,kp,pij->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x6, tau58)
        - einsum("cp,kp,ip,jp,pab->abcijk", a.t3.x1,
                 a.t3.x4, a.t3.x5, a.t3.x6, tau62)
        - einsum("p,cp,kp,jp,piab->abcijk", a.t2.xlam[0, :],
                 a.t2.x1, a.t2.x3, a.t2.x4, tau67)
        - einsum("ap,bp,jp,pikc->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau71)
        - einsum("ap,bp,jp,pikc->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau74)
        + 2 * einsum("ap,bp,jp,pikc->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau78)
        - einsum("ap,bp,jp,pkic->abcijk", a.t2.x1, a.t2.x2, a.t2.x3, tau82)
        + einsum("ap,bp,cp,kp,pij->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x6, tau85)
        + einsum("ap,bp,cp,kp,pji->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x6, tau89)
        + einsum("ap,bp,cp,ip,pkj->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x4, tau93)
        + einsum("ap,bp,cp,jp,pki->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, tau96)
        + einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau99)
        + einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau102)
        - einsum("ap,bp,jp,kp,pic->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x4, a.t3.x6, tau105)
        - einsum("ap,bp,cp,kp,pij->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x6, tau109)
        - einsum("ap,bp,cp,kp,pji->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x6, tau112)
        - einsum("ap,bp,cp,ip,pkj->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, tau114)
        - einsum("ap,bp,cp,jp,pki->abcijk", a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x4, tau117)
        - einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau118)
        - einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :], a.t3.x1,
                 a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau119)
        - 2 * einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau122)
        - 2 * einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau125)
        + 2 * einsum("p,ap,bp,cp,ip,kp,pj->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x5, a.t3.x6, tau126)
        + 2 * einsum("p,ap,bp,cp,jp,kp,pi->abcijk", a.t3.xlam[0, :],
                     a.t3.x1, a.t3.x2, a.t3.x3, a.t3.x4, a.t3.x6, tau127)
        - einsum("p,cp,kp,ip,jp,pab->abcijk", a.t3.xlam[0, :],
                 a.t3.x1, a.t3.x4, a.t3.x5, a.t3.x6, tau130)
        + einsum("ijkabc->abcijk", tau140)
        - einsum("ijkbac->abcijk", tau140)
        - einsum("ikjcab->abcijk", tau140)
        + einsum("ikjcba->abcijk", tau140)
        - einsum("jikabc->abcijk", tau140)
        + einsum("jikbac->abcijk", tau140)
        + einsum("jkicab->abcijk", tau140)
        - einsum("jkicba->abcijk", tau140)
        - einsum("kijacb->abcijk", tau140)
        + einsum("kijbca->abcijk", tau140)
        + einsum("kjiacb->abcijk", tau140)
        - einsum("kjibca->abcijk", tau140)
        - einsum("ijkcab->abcijk", tau177)
        + einsum("ijkcba->abcijk", tau177)
        + einsum("ikjabc->abcijk", tau177)
        - einsum("ikjbac->abcijk", tau177)
        + einsum("jikcab->abcijk", tau177)
        - einsum("jikcba->abcijk", tau177)
        - einsum("jkiabc->abcijk", tau177)
        + einsum("jkibac->abcijk", tau177)
        + einsum("kijacb->abcijk", tau177)
        - einsum("kijbca->abcijk", tau177)
        - einsum("kjiacb->abcijk", tau177)
        + einsum("kjibca->abcijk", tau177)
        + einsum("ikjabc->abcijk", tau196)
        - einsum("ikjbac->abcijk", tau196)
        + einsum("ijkabc->abcijk", tau204)
        - einsum("ijkbac->abcijk", tau204)
        - einsum("jikabc->abcijk", tau204)
        + einsum("jikbac->abcijk", tau204)
        - einsum("ijkcab->abcijk", tau213)
        + einsum("ikjabc->abcijk", tau213)
        - einsum("ikjbac->abcijk", tau213)
        + einsum("jikcab->abcijk", tau213)
        - einsum("ijkcab->abcijk", tau221)
        + einsum("jikcab->abcijk", tau221)
        + einsum("ikjabc->abcijk", tau243)
        - einsum("ikjbac->abcijk", tau243)
        - einsum("jkiabc->abcijk", tau243)
        + einsum("jkibac->abcijk", tau243)
        + einsum("kijacb->abcijk", tau243)
        - einsum("kjiacb->abcijk", tau243)
        - einsum("ijkabc->abcijk", tau246)
        + einsum("ijkbac->abcijk", tau246)
        + einsum("jikabc->abcijk", tau246)
        - einsum("jikbac->abcijk", tau246)
        + einsum("kijacb->abcijk", tau246)
        - einsum("kjiacb->abcijk", tau246)
        + einsum("ijkacb->abcijk", tau250)
        + einsum("ikjabc->abcijk", tau250)
        - einsum("ikjbac->abcijk", tau250)
        - einsum("jikacb->abcijk", tau250)
        - einsum("jkiabc->abcijk", tau250)
        + einsum("jkibac->abcijk", tau250)
        - einsum("ijkabc->abcijk", tau274)
        + einsum("ijkbac->abcijk", tau274)
        - einsum("ikjacb->abcijk", tau274)
        + einsum("jikabc->abcijk", tau274)
        - einsum("jikbac->abcijk", tau274)
        + einsum("jkiacb->abcijk", tau274)
        - einsum("ijkacb->abcijk", tau288)
        + einsum("jikacb->abcijk", tau288)
        + einsum("kijabc->abcijk", tau288)
        - einsum("kijbac->abcijk", tau288)
        - einsum("kjiabc->abcijk", tau288)
        + einsum("kjibac->abcijk", tau288)
        - einsum("ikjacb->abcijk", tau292)
        + einsum("jkiacb->abcijk", tau292)
        - einsum("kijabc->abcijk", tau292)
        + einsum("kijbac->abcijk", tau292)
        + einsum("kjiabc->abcijk", tau292)
        - einsum("kjibac->abcijk", tau292)
        - einsum("ijkabc->abcijk", tau374)
        + einsum("ijkbac->abcijk", tau374)
        + einsum("jikabc->abcijk", tau374)
        - einsum("jikbac->abcijk", tau374)
        - einsum("kijcab->abcijk", tau374)
        + einsum("kjicab->abcijk", tau374)
        - einsum("ikjabc->abcijk", tau419)
        + einsum("ikjbac->abcijk", tau419)
        + einsum("jkiabc->abcijk", tau419)
        - einsum("jkibac->abcijk", tau419)
        + einsum("kijcab->abcijk", tau419)
        - einsum("kjicab->abcijk", tau419)
        - einsum("ijkcab->abcijk", tau459)
        + einsum("ikjabc->abcijk", tau459)
        - einsum("ikjbac->abcijk", tau459)
        + einsum("jikcab->abcijk", tau459)
        - einsum("jkiabc->abcijk", tau459)
        + einsum("jkibac->abcijk", tau459)
        + einsum("ijkcab->abcijk", tau472)
        - einsum("jikcab->abcijk", tau472)
        + einsum("kijabc->abcijk", tau472)
        - einsum("kijbac->abcijk", tau472)
        - einsum("kjiabc->abcijk", tau472)
        + einsum("kjibac->abcijk", tau472)
        + einsum("ijkabc->abcijk", tau499)
        - einsum("jikabc->abcijk", tau499)
        - einsum("ijkacb->abcijk", tau510)
        + einsum("ijkbca->abcijk", tau510)
        - einsum("ikjabc->abcijk", tau510)
        + einsum("jikacb->abcijk", tau510)
        - einsum("jikbca->abcijk", tau510)
        - einsum("ijkacb->abcijk", tau526)
        + einsum("ijkbca->abcijk", tau526)
        + einsum("jikacb->abcijk", tau526)
        - einsum("jikbca->abcijk", tau526)
        + einsum("kijabc->abcijk", tau526)
        - einsum("jkiabc->abcijk", tau533)
        - einsum("kjiacb->abcijk", tau533)
        + einsum("kjibca->abcijk", tau533)
        - einsum("ijkabc->abcijk", tau536)
        + einsum("jikabc->abcijk", tau536)
        + einsum("kijacb->abcijk", tau536)
        - einsum("kijbca->abcijk", tau536)
        + einsum("ijkacb->abcijk", tau542)
        - einsum("ijkbca->abcijk", tau542)
        - einsum("jikacb->abcijk", tau542)
        + einsum("jikbca->abcijk", tau542)
        - einsum("kijabc->abcijk", tau559)
        + einsum("kjiabc->abcijk", tau559)
        - einsum("kijacb->abcijk", tau562)
        + einsum("kijbca->abcijk", tau562)
        + einsum("ikjabc->abcijk", tau569)
        + einsum("kijacb->abcijk", tau569)
        - einsum("kijbca->abcijk", tau569)
        - einsum("ikjabc->abcijk", tau599)
        + einsum("jkiabc->abcijk", tau599)
        - einsum("ijkacb->abcijk", tau605)
        + einsum("ijkbca->abcijk", tau605)
        - einsum("ikjabc->abcijk", tau605)
        + einsum("ikjbac->abcijk", tau605)
        + einsum("jikacb->abcijk", tau605)
        - einsum("jikbca->abcijk", tau605)
        + einsum("jkiabc->abcijk", tau605)
        - einsum("jkibac->abcijk", tau605)
        + einsum("kijcab->abcijk", tau605)
        - einsum("kijcba->abcijk", tau605)
        - einsum("kjicab->abcijk", tau605)
        + einsum("kjicba->abcijk", tau605)
        + einsum("ikjcab->abcijk", tau607)
        - einsum("jkicab->abcijk", tau607)
        - einsum("kijabc->abcijk", tau607)
        + einsum("kijbac->abcijk", tau607)
        + einsum("kjiabc->abcijk", tau607)
        - einsum("kjibac->abcijk", tau607)
        + einsum("ijkabc->abcijk", tau609)
        - einsum("ijkbac->abcijk", tau609)
        - einsum("ikjcab->abcijk", tau609)
        - einsum("jikabc->abcijk", tau609)
        + einsum("jikbac->abcijk", tau609)
        + einsum("jkicab->abcijk", tau609)
        + einsum("ijkabc->abcijk", tau625)
        - einsum("ijkbac->abcijk", tau625)
        + einsum("ikjabc->abcijk", tau627)
        - einsum("ikjbac->abcijk", tau627)
        - einsum("jkiabc->abcijk", tau627)
        + einsum("jkibac->abcijk", tau627)
        - einsum("ijkabc->abcijk", tau634)
        + einsum("ijkbac->abcijk", tau634)
        + einsum("ikjcab->abcijk", tau634)
        - einsum("jkicab->abcijk", tau634)
        - einsum("ikjcab->abcijk", tau636)
        + einsum("jkicab->abcijk", tau636)
        - einsum("ikjabc->abcijk", tau646)
        + einsum("jkiabc->abcijk", tau646)
        - einsum("kijacb->abcijk", tau646)
        + einsum("kijbca->abcijk", tau646)
        + einsum("kjiacb->abcijk", tau646)
        - einsum("kjibca->abcijk", tau646)
        + einsum("ijkacb->abcijk", tau654)
        - einsum("ijkbca->abcijk", tau654)
        + einsum("ikjabc->abcijk", tau654)
        - einsum("jikacb->abcijk", tau654)
        + einsum("jikbca->abcijk", tau654)
        - einsum("jkiabc->abcijk", tau654)
        + einsum("jikabc->abcijk", tau659)
        - einsum("jikbac->abcijk", tau659)
        + einsum("kijacb->abcijk", tau659)
        - einsum("kjiacb->abcijk", tau659)
        + einsum("jikacb->abcijk", tau672)
        + einsum("kijabc->abcijk", tau672)
        - einsum("kijbac->abcijk", tau672)
        - einsum("kjiabc->abcijk", tau672)
        + einsum("kjibac->abcijk", tau672)
        + einsum("ijkabc->abcijk", tau683)
        - einsum("ijkbac->abcijk", tau683)
        + einsum("ikjacb->abcijk", tau683)
        + einsum("jkiabc->abcijk", tau686)
        - einsum("jkibac->abcijk", tau686)
        - einsum("kijacb->abcijk", tau686)
        + einsum("kjiacb->abcijk", tau686)
        + einsum("ijkacb->abcijk", tau689)
        + einsum("ikjabc->abcijk", tau689)
        - einsum("ikjbac->abcijk", tau689)
        + einsum("jkiacb->abcijk", tau693)
        - einsum("kijabc->abcijk", tau693)
        + einsum("kijbac->abcijk", tau693)
        + einsum("kjiabc->abcijk", tau693)
        - einsum("kjibac->abcijk", tau693)
        - einsum("jikabc->abcijk", tau704)
        + einsum("jikbac->abcijk", tau704)
        - einsum("jkiacb->abcijk", tau704)
        - einsum("jikacb->abcijk", tau707)
        - einsum("jkiabc->abcijk", tau707)
        + einsum("jkibac->abcijk", tau707)
        - einsum("jikcab->abcijk", tau709)
        + einsum("jikcba->abcijk", tau709)
        + einsum("jkiabc->abcijk", tau709)
        - einsum("jkibac->abcijk", tau709)
        - einsum("kijacb->abcijk", tau709)
        + einsum("kijbca->abcijk", tau709)
        + einsum("kjiacb->abcijk", tau709)
        - einsum("kjibca->abcijk", tau709)
        + einsum("ijkacb->abcijk", tau711)
        - einsum("ijkbca->abcijk", tau711)
        + einsum("ikjabc->abcijk", tau711)
        - einsum("ikjbac->abcijk", tau711)
        + einsum("ijkabc->abcijk", tau713)
        - einsum("ijkbac->abcijk", tau713)
        - einsum("kjicab->abcijk", tau713)
        + einsum("ikjabc->abcijk", tau715)
        - einsum("ikjbac->abcijk", tau715)
        - einsum("jkiabc->abcijk", tau715)
        + einsum("jkibac->abcijk", tau715)
        + einsum("kjicab->abcijk", tau715)
        + einsum("jikabc->abcijk", tau717)
        - einsum("jikbac->abcijk", tau717)
        + einsum("kijabc->abcijk", tau719)
        - einsum("kijbac->abcijk", tau719)
        - einsum("jikabc->abcijk", tau721)
        + einsum("jikbac->abcijk", tau721)
        + einsum("kijcab->abcijk", tau721)
        - einsum("ikjcab->abcijk", tau724)
        + einsum("jkicab->abcijk", tau724)
        - einsum("kjiabc->abcijk", tau724)
        + einsum("kjibac->abcijk", tau724)
        - einsum("jikcab->abcijk", tau727)
        + einsum("jkiabc->abcijk", tau727)
        - einsum("jkibac->abcijk", tau727)
        + einsum("jikcab->abcijk", tau730)
        - einsum("kijabc->abcijk", tau730)
        + einsum("kijbac->abcijk", tau730)
        + einsum("kjiabc->abcijk", tau730)
        - einsum("kjibac->abcijk", tau730)
        - einsum("jikabc->abcijk", tau742)
        + einsum("jikbac->abcijk", tau742)
        + einsum("kijcab->abcijk", tau742)
        - einsum("kjicab->abcijk", tau742)
        - einsum("jkiabc->abcijk", tau746)
        + einsum("jkibac->abcijk", tau746)
        - einsum("kijcab->abcijk", tau746)
        + einsum("kjicab->abcijk", tau746)
        + einsum("ijkcab->abcijk", tau749)
        - einsum("ikjabc->abcijk", tau749)
        + einsum("ikjbac->abcijk", tau749)
        + einsum("ijkabc->abcijk", tau751)
        + einsum("ikjacb->abcijk", tau751)
        - einsum("ikjbca->abcijk", tau751)
        - einsum("jikabc->abcijk", tau751)
        - einsum("jkiacb->abcijk", tau751)
        + einsum("jkibca->abcijk", tau751)
        + einsum("ijkacb->abcijk", tau754)
        - einsum("ijkbca->abcijk", tau754)
        - einsum("jikacb->abcijk", tau754)
        + einsum("jikbca->abcijk", tau754)
        - einsum("kijabc->abcijk", tau754)
        + einsum("kijbac->abcijk", tau754)
        + einsum("ijkcab->abcijk", tau758)
        - einsum("ijkcba->abcijk", tau758)
        - einsum("jikcab->abcijk", tau758)
        + einsum("jikcba->abcijk", tau758)
        + einsum("kijabc->abcijk", tau758)
        - einsum("kijbac->abcijk", tau758)
    )
    return Tensors(t1=rt1, t2=rt2, t3=rt3)


def _rccsdt_ncpd_t2f_ls_t_calculate_energy(h, a):
    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau2 = (
        2 * einsum("w,ai->iaw", tau0, a.t1)
        + einsum("wjb,jiba->iaw", h.l.pov, tau1)
    )

    tau3 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau4 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau3)
    )

    energy = (
        einsum("wia,iaw->", h.l.pov, tau2)
        + einsum("ai,ia->", a.t1, tau4)
    )

    return energy


def _rccsdt_ncpd_t2f_ls_t_calc_residuals(h, a):
    tau0 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau3 = (
        einsum("ia->ia", h.f.ov)
        - einsum("wib,baw->ia", h.l.pov, tau1)
        + 2 * einsum("w,wia->ia", tau2, h.l.pov)
    )

    tau4 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau5 = (
        einsum("wjb,jiba->iaw", h.l.pov, tau4)
    )

    tau6 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau7 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau6)
    )

    tau8 = (
        - einsum("iaw->iaw", tau5)
        + einsum("aj,jiw->iaw", a.t1, tau7)
    )

    tau9 = (
        einsum("aj,ijw->iaw", a.t1, tau6)
    )

    tau10 = (
        einsum("iaw->iaw", tau9)
        - einsum("iaw->iaw", tau5)
    )

    tau11 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau0, h.l.pov)
    )

    tau12 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau0, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau10)
        + einsum("aj,ia->ij", a.t1, tau11)
    )

    tau13 = (
        einsum("wjb,abji->iaw", h.l.pov, a.t2)
    )

    tau14 = (
        einsum("wjb,baji->iaw", h.l.pov, a.t2)
    )

    tau15 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("w,wab->ab", tau0, h.l.pvv)
        + einsum("wib,iaw->ab", h.l.pov, tau13)
        - 2 * einsum("wib,iaw->ab", h.l.pov, tau14)
    )

    tau16 = (
        2 * einsum("ap,bp->pab", a.t3.x2, a.t3.x3)
        - einsum("bp,ap->pab", a.t3.x2, a.t3.x3)
    )

    tau17 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau18 = (
        einsum("p,pab,pbw->paw", a.t3.xlam[0, :], tau16, tau17)
    )

    tau19 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau20 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau21 = (
        einsum("ip,paw->piaw", a.t3.x4, tau19)
        - einsum("ip,paw->piaw", a.t3.x5, tau20)
    )

    tau22 = (
        einsum("paw,piaw->pi", tau18, tau21)
    )

    rt1 = (
        einsum("ia->ai", h.f.ov.conj())
        + 2 * einsum("w,wai->ai", tau0, h.l.pvo)
        + einsum("jb,jiba->ai", tau3, tau4)
        - einsum("wab,ibw->ai", h.l.pvv, tau8)
        + einsum("wji,jaw->ai", h.l.poo, tau10)
        - einsum("aj,ji->ai", a.t1, tau12)
        + einsum("bi,ab->ai", a.t1, tau15)
        + einsum("ap,pi->ai", a.t3.x1, tau22)
    )
    tau0 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("wij,abw->ijab", h.l.poo, tau0)
    )

    tau2 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau3 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau4 = (
        einsum("aj,ijw->iaw", a.t1, tau3)
    )

    tau5 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau6 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau5)
    )

    tau7 = (
        einsum("wab,wij->ijab", h.l.pvv, h.l.poo)
    )

    tau8 = (
        einsum("acki,kjbc->ijab", a.t2, tau7)
    )

    tau9 = (
        einsum("wbc,adw->abcd", h.l.pvv, tau0)
    )

    tau10 = (
        einsum("dcji,abcd->ijab", a.t2, tau9)
    )

    tau11 = (
        einsum("wab,ijw->ijab", h.l.pvv, tau3)
    )

    tau12 = (
        einsum("cakj,ikbc->ijab", a.t2, tau11)
    )

    tau13 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau14 = (
        einsum("abw,ijw->ijab", tau13, tau3)
    )

    tau15 = (
        einsum("bckj,ikac->ijab", a.t2, tau14)
    )

    tau16 = (
        einsum("wib,baw->ia", h.l.pov, tau13)
    )

    tau17 = (
        einsum("ai,ja->ij", a.t1, tau16)
    )

    tau18 = (
        einsum("ik,bakj->ijab", tau17, a.t2)
    )

    tau19 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau20 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau21 = (
        einsum("wjb,jiba->iaw", h.l.pov, tau20)
    )

    tau22 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau3)
    )

    tau23 = (
        einsum("aj,jiw->iaw", a.t1, tau22)
    )

    tau24 = (
        - einsum("iaw->iaw", tau21)
        + einsum("iaw->iaw", tau23)
    )

    tau25 = (
        einsum("iaw,jbw->ijab", tau19, tau24)
    )

    tau26 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau27 = (
        einsum("w,wia->ia", tau26, h.l.pov)
    )

    tau28 = (
        einsum("wib,baw->ia", h.l.pov, tau0)
    )

    tau29 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau27)
        - einsum("ia->ia", tau28)
    )

    tau30 = (
        einsum("bi,ia->ab", a.t1, tau29)
    )

    tau31 = (
        einsum("cb,caij->ijab", tau30, a.t2)
    )

    tau32 = (
        einsum("ijab->ijab", tau6)
        + einsum("ijab->ijab", tau8)
        + einsum("ijab->ijab", tau10)
        + einsum("ijab->ijab", tau12)
        - einsum("ijab->ijab", tau15)
        - einsum("ijab->ijab", tau18)
        + einsum("ijba->ijab", tau25)
        + einsum("jiba->ijab", tau31)
    )

    tau33 = (
        einsum("caki,kjbc->ijab", a.t2, tau7)
    )

    tau34 = (
        einsum("ackj,ikbc->ijab", a.t2, tau11)
    )

    tau35 = (
        einsum("cbkj,ikac->ijab", a.t2, tau14)
    )

    tau36 = (
        einsum("w,wab->ab", tau26, h.l.pvv)
    )

    tau37 = (
        einsum("wjb,baji->iaw", h.l.pov, a.t2)
    )

    tau38 = (
        einsum("wib,iaw->ab", h.l.pov, tau37)
    )

    tau39 = (
        einsum("wjb,abji->iaw", h.l.pov, a.t2)
    )

    tau40 = (
        einsum("iaw->iaw", tau19)
        - einsum("iaw->iaw", tau39)
    )

    tau41 = (
        einsum("wia,ibw->ab", h.l.pov, tau40)
    )

    tau42 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau36)
        - 2 * einsum("ab->ab", tau38)
        - einsum("ba->ab", tau41)
    )

    tau43 = (
        einsum("bc,caij->ijab", tau42, a.t2)
    )

    tau44 = (
        einsum("iaw->iaw", tau4)
        - einsum("iaw->iaw", tau21)
    )

    tau45 = (
        einsum("wai,jbw->ijab", h.l.pvo, tau44)
    )

    tau46 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau47 = (
        einsum("aj,ijw->iaw", a.t1, tau46)
    )

    tau48 = (
        einsum("jbw,iaw->ijab", tau21, tau47)
    )

    tau49 = (
        einsum("wia,jaw->ij", h.l.pov, tau21)
    )

    tau50 = (
        einsum("kj,abki->ijab", tau49, a.t2)
    )

    tau51 = (
        einsum("ijab->ijab", tau33)
        + einsum("ijab->ijab", tau34)
        - einsum("ijab->ijab", tau35)
        - einsum("jiab->ijab", tau43)
        + einsum("jiba->ijab", tau45)
        + einsum("ijab->ijab", tau48)
        + einsum("ijba->ijab", tau50)
    )

    tau52 = (
        einsum("wia,wjb->ijab", h.l.pov, h.l.pov)
    )

    tau53 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau46)
    )

    tau54 = (
        einsum("balj,ikab->ijkl", a.t2, tau52)
        + einsum("klw,ijw->ijkl", tau22, tau53)
    )

    tau55 = (
        einsum("jiab->ijab", tau1)
        - einsum("kica,kjbc->ijab", tau20, tau52)
    )

    tau56 = (
        einsum("wij,abw->ijab", h.l.poo, tau13)
    )

    tau57 = (
        einsum("jiab->ijab", tau56)
        + einsum("acki,kjbc->ijab", a.t2, tau52)
    )

    tau58 = (
        einsum("jiab->ijab", tau56)
        + einsum("acki,jkcb->ijab", a.t2, tau52)
    )

    tau59 = (
        einsum("wab,wcd->abcd", h.l.pvv, h.l.pvv)
        + einsum("cdw,abw->abcd", tau0, tau13)
    )

    tau60 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau61 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau62 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau63 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau64 = (
        einsum("paw,pbw->pab", tau62, tau63)
    )

    tau65 = (
        einsum("ip,ia->pa", a.t3.x6, tau16)
    )

    tau66 = (
        einsum("ip,ia->pa", a.t3.x5, tau16)
    )

    tau67 = (
        einsum("jp,jiw->piw", a.t3.x6, tau22)
    )

    tau68 = (
        2 * einsum("piw,paw->pia", tau60, tau61)
        + 2 * einsum("bi,pba->pia", a.t1, tau64)
        + 2 * einsum("ip,pa->pia", a.t3.x5, tau65)
        - einsum("ip,pa->pia", a.t3.x6, tau66)
        - einsum("paw,piw->pia", tau62, tau67)
    )

    tau69 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau70 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau71 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau72 = (
        einsum("pbw,paw->pab", tau63, tau71)
    )

    tau73 = (
        einsum("ip,ia->pa", a.t3.x4, tau16)
    )

    tau74 = (
        einsum("jp,jiw->piw", a.t3.x4, tau53)
    )

    tau75 = (
        einsum("paw,piw->pia", tau69, tau70)
        + einsum("bi,pab->pia", a.t1, tau72)
        + einsum("ip,pa->pia", a.t3.x6, tau73)
        - 2 * einsum("paw,piw->pia", tau63, tau74)
    )

    tau76 = (
        einsum("jp,jiw->piw", a.t3.x5, tau22)
    )

    tau77 = (
        einsum("paw,piw->pia", tau71, tau76)
    )

    tau78 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau79 = (
        einsum("piw,paw->pia", tau74, tau78)
    )

    tau80 = (
        - einsum("jp,pia->pija", a.t3.x4, tau68)
        + einsum("ip,pja->pija", a.t3.x5, tau75)
        + einsum("jp,pia->pija", a.t3.x6, tau77)
        + einsum("ip,pja->pija", a.t3.x6, tau79)
    )

    tau81 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau82 = (
        einsum("w,wia->ia", tau81, h.l.pov)
    )

    tau83 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau82)
    )

    tau84 = (
        einsum("ap,ia->pi", a.t3.x3, tau83)
    )

    tau85 = (
        - einsum("ip,jp->pij", a.t3.x4, a.t3.x6)
        + 2 * einsum("jp,ip->pij", a.t3.x4, a.t3.x6)
    )

    tau86 = (
        einsum("pj,pji->pi", tau84, tau85)
    )

    tau87 = (
        einsum("ip,jp,pj->pi", a.t3.x4, a.t3.x5, tau84)
    )

    tau88 = (
        einsum("ap,pjia->pij", a.t3.x3, tau80)
        + einsum("jp,pi->pij", a.t3.x5, tau86)
        - einsum("jp,pi->pij", a.t3.x6, tau87)
    )

    tau89 = (
        - einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        + 2 * einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau90 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau91 = (
        einsum("pji,pjw->piw", tau89, tau90)
    )

    tau92 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau13)
    )

    tau93 = (
        einsum("bp,abw->paw", a.t3.x3, tau92)
    )

    tau94 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau95 = (
        einsum("bp,abw->paw", a.t3.x2, tau92)
    )

    tau96 = (
        - einsum("piw,paw->pia", tau91, tau93)
        + einsum("ip,pw,paw->pia", a.t3.x4, tau94, tau95)
    )

    tau97 = (
        einsum("ap,pij->pija", a.t3.x2, tau88)
        - einsum("jp,pia->pija", a.t3.x6, tau96)
    )

    tau98 = (
        einsum("iaw->iaw", tau2)
        - einsum("wjb,jiba->iaw", h.l.pov, tau20)
    )

    tau99 = (
        einsum("iaw->iaw", tau5)
        - einsum("iaw->iaw", tau21)
    )

    tau100 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau19)
    )

    tau101 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau102 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau103 = (
        einsum("pjw,pji->piw", tau102, tau85)
    )

    tau104 = (
        einsum("ap,ip,wia->pw", a.t3.x2, a.t3.x6, h.l.pov)
    )

    tau105 = (
        einsum("piw,paw->pia", tau103, tau95)
        - einsum("ip,pw,paw->pia", a.t3.x4, tau104, tau93)
    )

    tau106 = (
        einsum("aj,ia->ij", a.t1, tau83)
    )

    tau107 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau26, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau2)
        + einsum("ij->ij", tau106)
    )

    tau108 = (
        einsum("ij->ij", h.f.oo)
        - einsum("wia,jaw->ij", h.l.pov, tau5)
        + 2 * einsum("w,wij->ij", tau81, h.l.poo)
        + einsum("ij->ij", tau106)
    )

    rt2 = (
        einsum("ackj,kibc->abij", a.t2, tau1)
        + einsum("iaw,jbw->abij", tau2, tau4)
        - einsum("ijba->abij", tau32)
        - einsum("jiab->abij", tau32)
        - einsum("ijab->abij", tau51)
        - einsum("jiba->abij", tau51)
        + einsum("bakl,likj->abij", a.t2, tau54)
        + einsum("cbkj,ikac->abij", a.t2, tau55)
        + einsum("bcki,jkac->abij", a.t2, tau57)
        + einsum("caki,jkbc->abij", a.t2, tau58)
        + einsum("cdji,adbc->abij", a.t2, tau59)
        + einsum("p,ap,pijb->abij", a.t3.xlam[0, :], a.t3.x1, tau97)
        + einsum("iaw,jbw->abij", tau98, tau99)
        + einsum("iaw,jbw->abij", tau100, tau101)
        + einsum("p,bp,ip,pja->abij",
                 a.t3.xlam[0, :], a.t3.x1, a.t3.x5, tau105)
        + einsum("jbw,iaw->abij", tau23, tau47)
        - einsum("ki,abkj->abij", tau107, a.t2)
        - einsum("kj,baki->abij", tau108, a.t2)
    )
    tau0 = (
        einsum("wia,wjk->ijka", h.l.pov, h.l.poo)
    )

    tau1 = (
        einsum("bali,jlkb->ijka", a.t2, tau0)
    )

    tau2 = (
        - einsum("abji->ijab", a.t2)
        + 2 * einsum("baji->ijab", a.t2)
    )

    tau3 = (
        einsum("wjb,jiba->iaw", h.l.pov, tau2)
    )

    tau4 = (
        einsum("wij,kaw->ijka", h.l.poo, tau3)
    )

    tau5 = (
        einsum("ijka->ijka", tau1)
        - einsum("jkia->ijka", tau4)
    )

    tau6 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau5)
    )

    tau7 = (
        einsum("bakj,jkic->iabc", a.t2, tau0)
    )

    tau8 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau9 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau10 = (
        einsum("acw,ibw->iabc", tau8, tau9)
    )

    tau11 = (
        einsum("iabc->iabc", tau7)
        + einsum("iabc->iabc", tau10)
    )

    tau12 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau11)
    )

    tau13 = (
        einsum("wij,wka->ijka", h.l.poo, h.l.pov)
    )

    tau14 = (
        einsum("bali,ljkb->ijka", a.t2, tau13)
    )

    tau15 = (
        einsum("wjb,jiba->iaw", h.l.pov, tau2)
    )

    tau16 = (
        einsum("wij,kaw->ijka", h.l.poo, tau15)
    )

    tau17 = (
        einsum("ikja->ijka", tau14)
        - einsum("jkia->ijka", tau16)
    )

    tau18 = (
        einsum("abli,jlkc->ijkabc", a.t2, tau17)
    )

    tau19 = (
        einsum("abli,jlkb->ijka", a.t2, tau0)
    )

    tau20 = (
        einsum("bali,jlkc->ijkabc", a.t2, tau19)
    )

    tau21 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau22 = (
        einsum("aj,ijw->iaw", a.t1, tau21)
    )

    tau23 = (
        einsum("wjk,iaw->ijka", h.l.poo, tau22)
    )

    tau24 = (
        einsum("cblj,ilka->ijkabc", a.t2, tau23)
    )

    tau25 = (
        einsum("ijkabc->ijkabc", tau20)
        + einsum("ijkabc->ijkabc", tau24)
    )

    tau26 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau27 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau28 = (
        einsum("jaw,ikw->ijka", tau26, tau27)
    )

    tau29 = (
        einsum("cblj,ikla->ijkabc", a.t2, tau28)
    )

    tau30 = (
        einsum("bakj,jikc->iabc", a.t2, tau13)
    )

    tau31 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau32 = (
        einsum("iaw,bcw->iabc", tau26, tau31)
    )

    tau33 = (
        einsum("iabc->iabc", tau30)
        + einsum("iabc->iabc", tau32)
    )

    tau34 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau33)
    )

    tau35 = (
        einsum("ijkabc->ijkabc", tau29)
        + einsum("jikcab->ijkabc", tau34)
    )

    tau36 = (
        einsum("abli,ljkb->ijka", a.t2, tau13)
    )

    tau37 = (
        einsum("bali,jklc->ijkabc", a.t2, tau36)
    )

    tau38 = (
        einsum("aj,ijw->iaw", a.t1, tau27)
    )

    tau39 = (
        einsum("wjk,iaw->ijka", h.l.poo, tau38)
    )

    tau40 = (
        einsum("cblj,ilka->ijkabc", a.t2, tau39)
    )

    tau41 = (
        einsum("ijkabc->ijkabc", tau37)
        + einsum("ijkabc->ijkabc", tau40)
    )

    tau42 = (
        einsum("ijw,kaw->ijka", tau21, tau9)
    )

    tau43 = (
        einsum("cblj,ilka->ijkabc", a.t2, tau42)
    )

    tau44 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau45 = (
        einsum("pw,wia->pia", tau44, h.l.pov)
    )

    tau46 = (
        einsum("ip,pia->pa", a.t3.x4, tau45)
    )

    tau47 = (
        einsum("pb,baji->pija", tau46, a.t2)
    )

    tau48 = (
        2 * einsum("ip,jp->pij", a.t3.x5, a.t3.x6)
        - einsum("jp,ip->pij", a.t3.x5, a.t3.x6)
    )

    tau49 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau50 = (
        einsum("jaw,piw->pija", tau3, tau49)
    )

    tau51 = (
        einsum("pjk,pkia->pija", tau48, tau50)
    )

    tau52 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau53 = (
        einsum("pw,wai->pia", tau52, h.l.pvo)
    )

    tau54 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau55 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau27)
    )

    tau56 = (
        einsum("jp,jiw->piw", a.t3.x6, tau55)
    )

    tau57 = (
        einsum("paw,piw->pia", tau54, tau56)
    )

    tau58 = (
        einsum("ap,wia->piw", a.t3.x3, h.l.pov)
    )

    tau59 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau60 = (
        einsum("piw,paw->pia", tau58, tau59)
    )

    tau61 = (
        einsum("pjb,jiba->pia", tau60, tau2)
    )

    tau62 = (
        einsum("pw,wab->pab", tau52, h.l.pvv)
    )

    tau63 = (
        einsum("pw,wia->pia", tau52, h.l.pov)
    )

    tau64 = (
        2 * einsum("pia->pia", tau63)
        - einsum("pia->pia", tau60)
    )

    tau65 = (
        einsum("ai,pib->pab", a.t1, tau64)
    )

    tau66 = (
        2 * einsum("pab->pab", tau62)
        - einsum("pab->pab", tau65)
    )

    tau67 = (
        einsum("bi,pab->pia", a.t1, tau66)
    )

    tau68 = (
        - 2 * einsum("pia->pia", tau53)
        + einsum("pia->pia", tau57)
        + einsum("pia->pia", tau61)
        - einsum("pia->pia", tau67)
    )

    tau69 = (
        einsum("pw,wai->pia", tau44, h.l.pvo)
    )

    tau70 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau71 = (
        einsum("piw,paw->pia", tau58, tau70)
    )

    tau72 = (
        einsum("pjb,baji->pia", tau71, a.t2)
    )

    tau73 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau8)
    )

    tau74 = (
        einsum("pw,abw->pab", tau44, tau73)
    )

    tau75 = (
        einsum("bi,pab->pia", a.t1, tau74)
    )

    tau76 = (
        einsum("pia->pia", tau69)
        - einsum("pia->pia", tau72)
        + einsum("pia->pia", tau75)
    )

    tau77 = (
        - einsum("pija->pija", tau51)
        + einsum("jp,pia->pija", a.t3.x5, tau68)
        + einsum("jp,pia->pija", a.t3.x6, tau76)
    )

    tau78 = (
        einsum("ip,pia->pa", a.t3.x4, tau63)
    )

    tau79 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau80 = (
        einsum("pw,wia->pia", tau79, h.l.pov)
    )

    tau81 = (
        einsum("ip,pia->pa", a.t3.x6, tau80)
    )

    tau82 = (
        2 * einsum("pa->pa", tau78)
        - einsum("pa->pa", tau81)
    )

    tau83 = (
        einsum("pb,baij->pija", tau82, a.t2)
    )

    tau84 = (
        einsum("kp,pija->pijka", a.t3.x6, tau47)
        - einsum("jp,pika->pijka", a.t3.x4, tau77)
        - einsum("kp,pjia->pijka", a.t3.x5, tau83)
    )

    tau85 = (
        einsum("p,ap,bp,pijkc->ijkabc",
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, tau84)
    )

    tau86 = (
        einsum("pjb,abji->pia", tau71, a.t2)
    )

    tau87 = (
        einsum("jp,jiw->piw", a.t3.x5, tau55)
    )

    tau88 = (
        einsum("bp,abw->paw", a.t3.x3, tau73)
    )

    tau89 = (
        einsum("piw,paw->pia", tau87, tau88)
    )

    tau90 = (
        einsum("pia->pia", tau86)
        - einsum("pia->pia", tau89)
    )

    tau91 = (
        einsum("p,cp,ap,ip,jp,pkb->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, a.t3.x6, tau90)
    )

    tau92 = (
        einsum("ip,pia->pa", a.t3.x5, tau80)
    )

    tau93 = (
        einsum("pb,baji->pija", tau92, a.t2)
    )

    tau94 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau95 = (
        einsum("ibw,paw->piab", tau3, tau94)
    )

    tau96 = (
        einsum("wjb,baji->iaw", h.l.pov, a.t2)
    )

    tau97 = (
        einsum("wib,iaw->ab", h.l.pov, tau96)
    )

    tau98 = (
        einsum("wjb,abji->iaw", h.l.pov, a.t2)
    )

    tau99 = (
        einsum("wib,iaw->ab", h.l.pov, tau98)
    )

    tau100 = (
        2 * einsum("ab->ab", tau97)
        - einsum("ab->ab", tau99)
    )

    tau101 = (
        einsum("piba->piab", tau95)
        - einsum("ip,ab->piab", a.t3.x4, tau100)
    )

    tau102 = (
        einsum("bp,ip,pjab->pija", a.t3.x3, a.t3.x5, tau101)
    )

    tau103 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x4, h.l.pov)
    )

    tau104 = (
        einsum("pw,wai->pia", tau103, h.l.pvo)
    )

    tau105 = (
        einsum("jp,jiw->piw", a.t3.x4, tau55)
    )

    tau106 = (
        einsum("piw,paw->pia", tau105, tau54)
    )

    tau107 = (
        einsum("piw,paw->pia", tau58, tau94)
    )

    tau108 = (
        einsum("abji->ijab", a.t2)
        - einsum("baji->ijab", a.t2)
    )

    tau109 = (
        einsum("pjb,jiba->pia", tau107, tau108)
    )

    tau110 = (
        einsum("caw,bcw->ab", tau31, tau73)
    )

    tau111 = (
        einsum("bp,ip,ba->pia", a.t3.x3, a.t3.x4, tau110)
    )

    tau112 = (
        einsum("pw,wab->pab", tau103, h.l.pvv)
    )

    tau113 = (
        einsum("pw,wia->pia", tau103, h.l.pov)
    )

    tau114 = (
        einsum("pia->pia", tau113)
        - einsum("pia->pia", tau107)
    )

    tau115 = (
        einsum("ai,pib->pab", a.t1, tau114)
    )

    tau116 = (
        einsum("pab->pab", tau112)
        - einsum("pab->pab", tau115)
    )

    tau117 = (
        einsum("bi,pab->pia", a.t1, tau116)
    )

    tau118 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau119 = (
        einsum("w,wab->ab", tau118, h.l.pvv)
    )

    tau120 = (
        einsum("w,wia->ia", tau118, h.l.pov)
    )

    tau121 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau120)
    )

    tau122 = (
        einsum("ai,ib->ab", a.t1, tau121)
    )

    tau123 = (
        - 2 * einsum("ab->ab", tau119)
        + einsum("ab->ab", tau122)
    )

    tau124 = (
        einsum("bp,ab->pa", a.t3.x3, tau123)
    )

    tau125 = (
        - einsum("pia->pia", tau104)
        + einsum("pia->pia", tau106)
        - einsum("pia->pia", tau109)
        + einsum("pia->pia", tau111)
        - einsum("pia->pia", tau117)
        + einsum("ip,pa->pia", a.t3.x4, tau124)
    )

    tau126 = (
        einsum("pija->pija", tau93)
        - einsum("pjia->pija", tau102)
        + einsum("jp,pia->pija", a.t3.x5, tau125)
    )

    tau127 = (
        einsum("ip,pia->pa", a.t3.x5, tau63)
    )

    tau128 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x5, h.l.pov)
    )

    tau129 = (
        einsum("pw,wia->pia", tau128, h.l.pov)
    )

    tau130 = (
        einsum("ip,pia->pa", a.t3.x6, tau129)
    )

    tau131 = (
        2 * einsum("pa->pa", tau127)
        - einsum("pa->pa", tau130)
    )

    tau132 = (
        einsum("pb,baij->pija", tau131, a.t2)
    )

    tau133 = (
        einsum("kp,pija->pijka", a.t3.x6, tau126)
        - einsum("kp,pjia->pijka", a.t3.x4, tau132)
    )

    tau134 = (
        einsum("p,ap,bp,pijkc->ijkabc",
               a.t3.xlam[0, :], a.t3.x1, a.t3.x2, tau133)
    )

    tau135 = (
        einsum("wai,wjk->ijka", h.l.pvo, h.l.poo)
    )

    tau136 = (
        einsum("bali,jlkc->ijkabc", a.t2, tau135)
    )

    tau137 = (
        einsum("wab,wic->iabc", h.l.pvv, h.l.pov)
    )

    tau138 = (
        einsum("daji,jbdc->iabc", a.t2, tau137)
    )

    tau139 = (
        einsum("wib,baw->ia", h.l.pov, tau8)
    )

    tau140 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau141 = (
        einsum("w,wia->ia", tau140, h.l.pov)
    )

    tau142 = (
        einsum("ia->ia", h.f.ov)
        - einsum("ia->ia", tau139)
        + 2 * einsum("ia->ia", tau141)
    )

    tau143 = (
        einsum("jc,abji->iabc", tau142, a.t2)
    )

    tau144 = (
        einsum("wab,icw->iabc", h.l.pvv, tau15)
    )

    tau145 = (
        einsum("iabc->iabc", tau138)
        + einsum("ibac->iabc", tau143)
        - einsum("ibca->iabc", tau144)
    )

    tau146 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau145)
    )

    tau147 = (
        einsum("wic,abw->iabc", h.l.pov, tau8)
    )

    tau148 = (
        einsum("cbji,kabc->ijka", a.t2, tau147)
    )

    tau149 = (
        einsum("iaw,jkw->ijka", tau22, tau27)
    )

    tau150 = (
        einsum("ijka->ijka", tau148)
        + einsum("ijka->ijka", tau149)
    )

    tau151 = (
        einsum("abli,jklc->ijkabc", a.t2, tau150)
    )

    tau152 = (
        einsum("ijkabc->ijkabc", tau136)
        + einsum("jikabc->ijkabc", tau146)
        - einsum("kijcba->ijkabc", tau151)
    )

    tau153 = (
        einsum("ap,pia->pi", a.t3.x2, tau45)
    )

    tau154 = (
        einsum("p,ap,wia->piw", a.t3.xlam[0, :], a.t3.x2, h.l.pov)
    )

    tau155 = (
        - einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        + 2 * einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau156 = (
        einsum("pjw,pji->piw", tau154, tau155)
    )

    tau157 = (
        einsum("pjw,piw->pij", tau156, tau58)
    )

    tau158 = (
        einsum("p,jp,pi->pij", a.t3.xlam[0, :], a.t3.x4, tau153)
        - einsum("pij->pij", tau157)
    )

    tau159 = (
        einsum("ap,ip,pjk->ijka", a.t3.x1, a.t3.x6, tau158)
    )

    tau160 = (
        einsum("abli,kljc->ijkabc", a.t2, tau159)
    )

    tau161 = (
        einsum("bp,wab->paw", a.t3.x2, h.l.pvv)
    )

    tau162 = (
        einsum("bp,wab->paw", a.t3.x3, h.l.pvv)
    )

    tau163 = (
        einsum("paw,pbw->pab", tau161, tau162)
    )

    tau164 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau165 = (
        einsum("piw,pjw->pij", tau164, tau49)
    )

    tau166 = (
        einsum("pij,baji->pab", tau165, a.t2)
    )

    tau167 = (
        einsum("ap,wia->piw", a.t3.x2, h.l.pov)
    )

    tau168 = (
        einsum("piw,paw->pia", tau167, tau54)
    )

    tau169 = (
        einsum("aj,pij->pia", a.t1, tau165)
    )

    tau170 = (
        einsum("pia->pia", tau168)
        - einsum("pia->pia", tau169)
    )

    tau171 = (
        einsum("ai,pib->pab", a.t1, tau170)
    )

    tau172 = (
        einsum("pab->pab", tau163)
        + einsum("pab->pab", tau166)
        - einsum("pab->pab", tau171)
    )

    tau173 = (
        einsum("p,cp,ip,jp,kp,pab->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x4, a.t3.x5, a.t3.x6, tau172)
    )

    tau174 = (
        einsum("p,ap,bp,pib->pia", a.t3.xlam[0, :], a.t3.x1, a.t3.x2, tau64)
    )

    tau175 = (
        einsum("ip,jp,pka->ijka", a.t3.x4, a.t3.x5, tau174)
    )

    tau176 = (
        einsum("abli,jklc->ijkabc", a.t2, tau175)
    )

    tau177 = (
        einsum("ap,pia->pi", a.t3.x2, tau113)
    )

    tau178 = (
        einsum("p,ap,jp,kp,pi->ijka", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x5, a.t3.x6, tau177)
    )

    tau179 = (
        einsum("bali,ljkc->ijkabc", a.t2, tau178)
    )

    tau180 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau181 = (
        einsum("ibw,acw->iabc", tau180, tau31)
    )

    tau182 = (
        einsum("dbkj,iacd->ijkabc", a.t2, tau181)
    )

    tau183 = (
        einsum("cbji,kabc->ijka", a.t2, tau137)
    )

    tau184 = (
        einsum("iaw,jkw->ijka", tau180, tau27)
    )

    tau185 = (
        einsum("ijka->ijka", tau183)
        + einsum("ijka->ijka", tau184)
    )

    tau186 = (
        einsum("abli,jklc->ijkabc", a.t2, tau185)
    )

    tau187 = (
        einsum("ijkabc->ijkabc", tau182)
        + einsum("kijbac->ijkabc", tau186)
    )

    tau188 = (
        einsum("adji,jbdc->iabc", a.t2, tau137)
    )

    tau189 = (
        einsum("daji,kbcd->ijkabc", a.t2, tau188)
    )

    tau190 = (
        einsum("wbc,iaw->iabc", h.l.pvv, tau38)
    )

    tau191 = (
        einsum("dbkj,iacd->ijkabc", a.t2, tau190)
    )

    tau192 = (
        einsum("ijkabc->ijkabc", tau189)
        + einsum("ijkabc->ijkabc", tau191)
    )

    tau193 = (
        einsum("wbc,iaw->iabc", h.l.pvv, tau180)
    )

    tau194 = (
        einsum("dakj,ibcd->ijkabc", a.t2, tau193)
    )

    tau195 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau196 = (
        einsum("wab,icw->iabc", h.l.pvv, tau195)
    )

    tau197 = (
        einsum("dakj,ibdc->ijkabc", a.t2, tau196)
    )

    tau198 = (
        einsum("wbc,iaw->iabc", h.l.pvv, tau9)
    )

    tau199 = (
        einsum("bdji,jadc->iabc", a.t2, tau147)
    )

    tau200 = (
        - einsum("iabc->iabc", tau198)
        + einsum("iabc->iabc", tau199)
    )

    tau201 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau200)
    )

    tau202 = (
        einsum("waj,ikw->ijka", h.l.pvo, tau27)
    )

    tau203 = (
        einsum("balj,iklc->ijkabc", a.t2, tau202)
    )

    tau204 = (
        einsum("wbi,acw->iabc", h.l.pvo, tau31)
    )

    tau205 = (
        einsum("dbji,jadc->iabc", a.t2, tau147)
    )

    tau206 = (
        einsum("icw,abw->iabc", tau15, tau8)
    )

    tau207 = (
        einsum("iabc->iabc", tau204)
        - einsum("iabc->iabc", tau205)
        + einsum("iacb->iabc", tau206)
    )

    tau208 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau207)
    )

    tau209 = (
        einsum("ijkabc->ijkabc", tau203)
        + einsum("jikbac->ijkabc", tau208)
    )

    tau210 = (
        einsum("wjk,iaw->ijka", h.l.poo, tau180)
    )

    tau211 = (
        einsum("balj,ilkc->ijkabc", a.t2, tau210)
    )

    tau212 = (
        einsum("wab,wci->iabc", h.l.pvv, h.l.pvo)
    )

    tau213 = (
        einsum("daji,kbdc->ijkabc", a.t2, tau212)
    )

    tau214 = (
        einsum("wai,wbc->iabc", h.l.pvo, h.l.pvv)
    )

    tau215 = (
        einsum("daji,kbcd->ijkabc", a.t2, tau214)
    )

    tau216 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau217 = (
        einsum("pjw,piw->pij", tau216, tau58)
    )

    tau218 = (
        einsum("pw,wij->pij", tau79, h.l.poo)
    )

    tau219 = (
        einsum("pij->pij", tau217)
        - einsum("pij->pij", tau218)
    )

    tau220 = (
        einsum("p,aj,pji->pia", a.t3.xlam[0, :], a.t1, tau219)
    )

    tau221 = (
        einsum("cp,ap,ip,jp,pkb->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau220)
    )

    tau222 = (
        einsum("pw,wij->pij", tau103, h.l.poo)
    )

    tau223 = (
        einsum("jp,wji->piw", a.t3.x4, h.l.poo)
    )

    tau224 = (
        einsum("piw,pjw->pij", tau223, tau49)
    )

    tau225 = (
        einsum("pij->pij", tau222)
        - einsum("pji->pij", tau224)
    )

    tau226 = (
        einsum("p,aj,pji->pia", a.t3.xlam[0, :], a.t1, tau225)
    )

    tau227 = (
        einsum("cp,ap,ip,jp,pkb->ijkabc", a.t3.x1,
               a.t3.x2, a.t3.x5, a.t3.x6, tau226)
    )

    tau228 = (
        einsum("pw,wij->pij", tau44, h.l.poo)
    )

    tau229 = (
        einsum("aj,pji->pia", a.t1, tau228)
    )

    tau230 = (
        einsum("pw,wij->pij", tau52, h.l.poo)
    )

    tau231 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau232 = (
        einsum("piw,pjw->pij", tau231, tau49)
    )

    tau233 = (
        2 * einsum("pij->pij", tau230)
        - einsum("pji->pij", tau232)
    )

    tau234 = (
        einsum("aj,pji->pia", a.t1, tau233)
    )

    tau235 = (
        einsum("jp,pia->pija", a.t3.x6, tau229)
        - einsum("jp,pia->pija", a.t3.x5, tau234)
    )

    tau236 = (
        einsum("p,ap,bp,ip,pjkc->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, tau235)
    )

    tau237 = (
        einsum("pw,wij->pij", tau128, h.l.poo)
    )

    tau238 = (
        einsum("aj,pji->pia", a.t1, tau237)
    )

    tau239 = (
        einsum("jp,wji->piw", a.t3.x6, h.l.poo)
    )

    tau240 = (
        einsum("pjw,piw->pij", tau239, tau58)
    )

    tau241 = (
        einsum("ap,ip,wia->pw", a.t3.x3, a.t3.x6, h.l.pov)
    )

    tau242 = (
        einsum("pw,wij->pij", tau241, h.l.poo)
    )

    tau243 = (
        - einsum("pij->pij", tau240)
        + 2 * einsum("pij->pij", tau242)
    )

    tau244 = (
        einsum("aj,pji->pia", a.t1, tau243)
    )

    tau245 = (
        einsum("jp,pia->pija", a.t3.x6, tau238)
        - einsum("jp,pia->pija", a.t3.x5, tau244)
    )

    tau246 = (
        einsum("p,ap,bp,ip,pjkc->ijkabc", a.t3.xlam[0, :],
               a.t3.x1, a.t3.x2, a.t3.x4, tau245)
    )

    tau247 = (
        einsum("bp,ab->pa", a.t3.x3, h.f.vv)
    )

    tau248 = (
        einsum("p,ap,bp,ip,jp,kp,pc->ijkabc", a.t3.xlam[0, :], a.t3.x1,
               a.t3.x2, a.t3.x4, a.t3.x5, a.t3.x6, tau247)
    )

    tau249 = (
        einsum("wjk,iaw->ijka", h.l.poo, tau26)
    )

    tau250 = (
        einsum("cbli,jlka->ijkabc", a.t2, tau249)
    )

    tau251 = (
        einsum("wij,kaw->ijka", h.l.poo, tau9)
    )

    tau252 = (
        einsum("cbli,ljka->ijkabc", a.t2, tau251)
    )

    tau253 = (
        einsum("wja,ikw->ijka", h.l.pov, tau27)
    )

    tau254 = (
        einsum("balj,iklb->ijka", a.t2, tau253)
    )

    tau255 = (
        einsum("kaw,ijw->ijka", tau15, tau21)
    )

    tau256 = (
        einsum("ijka->ijka", tau254)
        - einsum("ikja->ijka", tau255)
    )

    tau257 = (
        einsum("abli,jklc->ijkabc", a.t2, tau256)
    )

    tau258 = (
        einsum("wka,ijw->ijka", h.l.pov, tau21)
    )

    tau259 = (
        einsum("ablj,ilkb->ijka", a.t2, tau258)
    )

    tau260 = (
        einsum("balj,iklc->ijkabc", a.t2, tau259)
    )

    tau261 = (
        einsum("bakj,ijkc->iabc", a.t2, tau258)
    )

    tau262 = (
        einsum("iaw,bcw->iabc", tau22, tau31)
    )

    tau263 = (
        einsum("iabc->iabc", tau261)
        + einsum("iabc->iabc", tau262)
    )

    tau264 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau263)
    )

    tau265 = (
        einsum("balj,ilkb->ijka", a.t2, tau258)
    )

    tau266 = (
        einsum("ijw,kaw->ijka", tau27, tau3)
    )

    tau267 = (
        einsum("ijka->ijka", tau265)
        - einsum("ikja->ijka", tau266)
    )

    tau268 = (
        einsum("abli,jklc->ijkabc", a.t2, tau267)
    )

    tau269 = (
        einsum("bakj,ijkc->iabc", a.t2, tau253)
    )

    tau270 = (
        einsum("ibw,acw->iabc", tau38, tau8)
    )

    tau271 = (
        einsum("iabc->iabc", tau269)
        + einsum("iabc->iabc", tau270)
    )

    tau272 = (
        einsum("daij,kbcd->ijkabc", a.t2, tau271)
    )

    tau273 = (
        einsum("ablj,iklb->ijka", a.t2, tau253)
    )

    tau274 = (
        einsum("balj,iklc->ijkabc", a.t2, tau273)
    )

    tau275 = (
        einsum("p,bp,wab->paw", a.t3.xlam[0, :], a.t3.x2, h.l.pvv)
    )

    tau276 = (
        einsum("paw,piw->pia", tau275, tau49)
    )

    tau277 = (
        einsum("ai,pib->pab", a.t1, tau276)
    )

    tau278 = (
        einsum("cp,ip,jp,kp,pab->ijkabc", a.t3.x1,
               a.t3.x4, a.t3.x5, a.t3.x6, tau277)
    )

    tau279 = (
        einsum("ip,wia->paw", a.t3.x5, h.l.pov)
    )

    tau280 = (
        einsum("paw,pbw->pab", tau279, tau59)
    )

    tau281 = (
        einsum("pab,baji->pij", tau280, a.t2)
    )

    tau282 = (
        einsum("ip,wia->paw", a.t3.x4, h.l.pov)
    )

    tau283 = (
        einsum("paw,pbw->pab", tau282, tau59)
    )

    tau284 = (
        einsum("pab,baji->pij", tau283, a.t2)
    )

    tau285 = (
        einsum("kp,pij->pijk", a.t3.x4, tau281)
        - einsum("kp,pij->pijk", a.t3.x5, tau284)
    )

    tau286 = (
        einsum("paw,pbw->pab", tau282, tau70)
    )

    tau287 = (
        einsum("pab,baji->pij", tau286, a.t2)
    )

    tau288 = (
        einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        - einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau289 = (
        einsum("wia,jaw->ij", h.l.pov, tau3)
    )

    tau290 = (
        - einsum("ip,jp->pij", a.t3.x4, a.t3.x5)
        + einsum("jp,ip->pij", a.t3.x4, a.t3.x5)
    )

    tau291 = (
        - einsum("lp,kp,ji->pijkl", a.t3.x4, a.t3.x5, tau289)
        + einsum("kp,lp,ji->pijkl", a.t3.x4, a.t3.x5, tau289)
    )

    tau292 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau293 = (
        einsum("piw,paw->pia", tau292, tau94)
    )

    tau294 = (
        einsum("bi,pab->pia", a.t1, tau286)
    )

    tau295 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau21)
    )

    tau296 = (
        einsum("jp,jiw->piw", a.t3.x4, tau295)
    )

    tau297 = (
        einsum("wia,jaw->ij", h.l.pov, tau9)
    )

    tau298 = (
        einsum("wja,iaw->ij", h.l.pov, tau26)
    )

    tau299 = (
        einsum("ai,ja->ij", a.t1, tau139)
    )

    tau300 = (
        einsum("kp,ij->pijk", a.t3.x5, tau297)
        - einsum("jp,ki->pijk", a.t3.x5, tau298)
        + einsum("kp,ji->pijk", a.t3.x5, tau299)
        - einsum("jp,ki->pijk", a.t3.x5, tau299)
    )

    tau301 = (
        einsum("kp,ij->pijk", a.t3.x4, tau297)
        - einsum("jp,ki->pijk", a.t3.x4, tau298)
        + einsum("kp,ji->pijk", a.t3.x4, tau299)
        - einsum("jp,ki->pijk", a.t3.x4, tau299)
    )

    tau302 = (
        einsum("jp,jiw->piw", a.t3.x5, tau295)
    )

    tau303 = (
        einsum("w,wij->ij", tau118, h.l.poo)
    )

    tau304 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau141)
    )

    tau305 = (
        einsum("ai,ja->ij", a.t1, tau304)
    )

    tau306 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau303)
        + einsum("ji->ij", tau305)
    )

    tau307 = (
        einsum("jp,ji->pi", a.t3.x4, tau306)
    )

    tau308 = (
        einsum("jp,ji->pi", a.t3.x5, tau306)
    )

    tau309 = (
        einsum("w,wij->ij", tau140, h.l.poo)
    )

    tau310 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("ij->ij", tau309)
        + einsum("ji->ij", tau305)
    )

    tau311 = (
        einsum("jp,ji->pi", a.t3.x5, tau310)
    )

    tau312 = (
        einsum("jp,ji->pi", a.t3.x4, tau310)
    )

    tau313 = (
        - einsum("aj,pia->pij", a.t1, tau293)
        - einsum("aj,pia->pij", a.t1, tau294)
        + einsum("piw,pjw->pij", tau296, tau87)
        - einsum("kp,pkji->pij", a.t3.x4, tau300)
        + einsum("kp,pkji->pij", a.t3.x5, tau301)
        - einsum("pjw,piw->pij", tau216, tau302)
        - einsum("jp,pi->pij", a.t3.x5, tau307)
        + einsum("jp,pi->pij", a.t3.x4, tau308)
        - einsum("ip,pj->pij", a.t3.x4, tau311)
        + einsum("ip,pj->pij", a.t3.x5, tau312)
    )

    tau314 = (
        einsum("ij->ij", h.f.oo)
        - einsum("ij->ij", tau297)
        + 2 * einsum("ij->ij", tau309)
        + einsum("aj,ia->ij", a.t1, tau142)
    )

    tau315 = (
        einsum("jp,ji->pi", a.t3.x6, tau314)
    )

    tau316 = (
        einsum("jp,wji->piw", a.t3.x5, h.l.poo)
    )

    tau317 = (
        einsum("ip,wia->paw", a.t3.x6, h.l.pov)
    )

    tau318 = (
        einsum("piw,paw->pia", tau316, tau317)
        + einsum("bi,pba->pia", a.t1, tau280)
    )

    tau319 = (
        einsum("pjw,piw->pij", tau239, tau302)
        + einsum("aj,pia->pij", a.t1, tau318)
    )

    tau320 = (
        einsum("piw,paw->pia", tau216, tau317)
        + einsum("bi,pba->pia", a.t1, tau283)
    )

    tau321 = (
        einsum("pjw,piw->pij", tau239, tau296)
        + einsum("aj,pia->pij", a.t1, tau320)
    )

    tau322 = (
        einsum("piw,pjw->pij", tau296, tau56)
    )

    tau323 = (
        einsum("piw,pjw->pij", tau302, tau56)
    )

    tau324 = (
        einsum("pikj->pijk", tau285)
        - einsum("pjki->pijk", tau285)
        - einsum("kp,pij->pijk", a.t3.x6, tau287)
        + einsum("kp,pji->pijk", a.t3.x6, tau287)
        + einsum("kp,li,plj->pijk", a.t3.x6, tau289, tau288)
        - einsum("kp,lj,pil->pijk", a.t3.x6, tau289, tau290)
        - einsum("lp,pklji->pijk", a.t3.x6, tau291)
        - einsum("kp,pij->pijk", a.t3.x6, tau313)
        + einsum("pk,pji->pijk", tau315, tau290)
        - einsum("ip,pjk->pijk", a.t3.x4, tau319)
        + einsum("ip,pjk->pijk", a.t3.x5, tau321)
        - einsum("jp,pik->pijk", a.t3.x5, tau322)
        + einsum("jp,pik->pijk", a.t3.x4, tau323)
    )

    tau325 = (
        einsum("iaw->iaw", tau98)
        - einsum("bi,abw->iaw", a.t1, tau73)
    )

    tau326 = (
        2 * einsum("ab->ab", tau97)
        - einsum("wib,iaw->ab", h.l.pov, tau325)
    )

    tau327 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau119)
        - einsum("ab->ab", tau122)
    )

    tau328 = (
        einsum("bp,ab->pa", a.t3.x3, tau327)
    )

    tau329 = (
        einsum("pij->pij", tau242)
        + einsum("aj,pia->pij", a.t1, tau63)
    )

    tau330 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau180)
    )

    tau331 = (
        einsum("piw,paw->pia", tau56, tau88)
        - einsum("pjb,jiba->pia", tau64, tau2)
        + einsum("bp,ip,ab->pia", a.t3.x3, a.t3.x6, tau326)
        - einsum("ip,pa->pia", a.t3.x6, tau328)
        + 2 * einsum("aj,pji->pia", a.t1, tau329)
        - 2 * einsum("pw,iaw->pia", tau52, tau330)
    )

    tau332 = (
        einsum("pjb,abji->pia", tau107, a.t2)
    )

    tau333 = (
        einsum("jp,pia->pija", a.t3.x4, tau86)
        - einsum("jp,pia->pija", a.t3.x5, tau332)
    )

    tau334 = (
        einsum("piw,paw->pia", tau105, tau88)
    )

    tau335 = (
        einsum("pij->pij", tau224)
        + einsum("ai,pja->pij", a.t1, tau107)
    )

    tau336 = (
        einsum("pia->pia", tau106)
        - einsum("aj,pij->pia", a.t1, tau335)
    )

    tau337 = (
        einsum("piw,pjw->pij", tau292, tau49)
        + einsum("ai,pja->pij", a.t1, tau71)
    )

    tau338 = (
        einsum("paw,piw->pia", tau54, tau87)
        - einsum("aj,pij->pia", a.t1, tau337)
    )

    tau339 = (
        einsum("pija->pija", tau333)
        - einsum("pjia->pija", tau333)
        + einsum("jp,pia->pija", a.t3.x5, tau334)
        - einsum("jp,pia->pija", a.t3.x4, tau89)
        - einsum("ip,pja->pija", a.t3.x5, tau336)
        + einsum("ip,pja->pija", a.t3.x4, tau338)
    )

    tau340 = (
        einsum("pik,pkja->pija", tau290, tau50)
    )

    tau341 = (
        einsum("ai,pja->pij", a.t1, tau45)
    )

    tau342 = (
        einsum("pij->pij", tau237)
        + einsum("pji->pij", tau341)
    )

    tau343 = (
        einsum("aj,pji->pia", a.t1, tau342)
    )

    tau344 = (
        einsum("pw,iaw->pia", tau44, tau330)
    )

    tau345 = (
        einsum("pia->pia", tau72)
        + einsum("pia->pia", tau343)
        - einsum("pia->pia", tau344)
    )

    tau346 = (
        einsum("pjb,baji->pia", tau107, a.t2)
    )

    tau347 = (
        einsum("ai,pja->pij", a.t1, tau113)
    )

    tau348 = (
        einsum("pij->pij", tau218)
        + einsum("pji->pij", tau347)
    )

    tau349 = (
        einsum("aj,pji->pia", a.t1, tau348)
    )

    tau350 = (
        einsum("pw,iaw->pia", tau103, tau330)
    )

    tau351 = (
        einsum("pia->pia", tau346)
        + einsum("pia->pia", tau349)
        - einsum("pia->pia", tau350)
    )

    tau352 = (
        einsum("pa->pa", tau46)
        - einsum("pa->pa", tau92)
    )

    tau353 = (
        einsum("pb,baij->pija", tau352, a.t2)
    )

    tau354 = (
        - einsum("pjia->pija", tau340)
        - einsum("jp,pia->pija", a.t3.x4, tau345)
        + einsum("jp,pia->pija", a.t3.x5, tau351)
        - einsum("pjia->pija", tau353)
    )

    tau355 = (
        einsum("ap,pijk->pijka", a.t3.x3, tau324)
        - einsum("pji,pka->pijka", tau288, tau331)
        - einsum("kp,pjia->pijka", a.t3.x6, tau339)
        + einsum("jp,pkia->pijka", a.t3.x6, tau354)
        - einsum("ip,pkja->pijka", a.t3.x6, tau354)
        + einsum("jp,pika->pijka", a.t3.x5, tau83)
        - einsum("jp,pika->pijka", a.t3.x4, tau132)
        + einsum("ip,pjka->pijka", a.t3.x4, tau132)
        - einsum("ip,pjka->pijka", a.t3.x5, tau83)
    )

    tau356 = (
        2 * einsum("ap,piw->piaw", a.t3.x2, tau58)
        - einsum("ap,piw->piaw", a.t3.x3, tau164)
    )

    tau357 = (
        einsum("paw,piaw->pi", tau70, tau356)
    )

    tau358 = (
        einsum("pj,abji->piab", tau357, a.t2)
    )

    tau359 = (
        einsum("paw,piw->pia", tau161, tau49)
    )

    tau360 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau31)
    )

    tau361 = (
        einsum("bp,abw->paw", a.t3.x3, tau360)
    )

    tau362 = (
        einsum("bp,abw->paw", a.t3.x2, tau360)
    )

    tau363 = (
        einsum("pjw,piw->pij", tau167, tau58)
    )

    tau364 = (
        einsum("pij->pij", tau165)
        - einsum("pij->pij", tau363)
    )

    tau365 = (
        einsum("aj,pij->pia", a.t1, tau364)
    )

    tau366 = (
        einsum("ai,pib->pab", a.t1, tau359)
        - einsum("ai,pib->pab", a.t1, tau168)
        + einsum("paw,pbw->pab", tau161, tau361)
        - einsum("pbw,paw->pab", tau362, tau54)
        + einsum("pji,baij->pab", tau364, a.t2)
        + einsum("ai,pib->pab", a.t1, tau365)
    )

    tau367 = (
        - einsum("piba->piab", tau358)
        + einsum("piab->piab", tau358)
        + einsum("ip,pab->piab", a.t3.x5, tau366)
    )

    tau368 = (
        - einsum("ap,piw->piaw", a.t3.x2, tau58)
        + 2 * einsum("ap,piw->piaw", a.t3.x3, tau164)
    )

    tau369 = (
        einsum("paw,piaw->pi", tau59, tau368)
    )

    tau370 = (
        einsum("pj,abji->piab", tau369, a.t2)
    )

    tau371 = (
        einsum("piba->piab", tau370)
        - einsum("piab->piab", tau370)
    )

    tau372 = (
        einsum("jp,piab->pijab", a.t3.x6, tau367)
        - einsum("ip,pjba->pijab", a.t3.x5, tau371)
    )

    rt3 = (
        einsum("ijkcab->abcijk", tau6)
        - einsum("ijkcba->abcijk", tau6)
        + einsum("ikjbac->abcijk", tau6)
        - einsum("ikjabc->abcijk", tau6)
        - einsum("jikcab->abcijk", tau6)
        + einsum("jikcba->abcijk", tau6)
        - einsum("kijacb->abcijk", tau6)
        + einsum("kijbca->abcijk", tau6)
        - einsum("jikbac->abcijk", tau12)
        + einsum("jikabc->abcijk", tau12)
        - einsum("kijbca->abcijk", tau12)
        + einsum("kijacb->abcijk", tau12)
        + einsum("ijkbac->abcijk", tau12)
        - einsum("ijkabc->abcijk", tau12)
        + einsum("ikjcab->abcijk", tau12)
        - einsum("ikjcba->abcijk", tau12)
        - einsum("jkibac->abcijk", tau18)
        + einsum("jkiabc->abcijk", tau18)
        + einsum("kjiacb->abcijk", tau18)
        - einsum("kjibca->abcijk", tau18)
        + einsum("ijkabc->abcijk", tau25)
        - einsum("ijkbac->abcijk", tau25)
        + einsum("ikjacb->abcijk", tau25)
        - einsum("ikjbca->abcijk", tau25)
        - einsum("jikabc->abcijk", tau25)
        + einsum("jikbac->abcijk", tau25)
        + einsum("kijcab->abcijk", tau25)
        - einsum("kijcba->abcijk", tau25)
        + einsum("jkiacb->abcijk", tau35)
        - einsum("jkibca->abcijk", tau35)
        + einsum("kjiabc->abcijk", tau35)
        - einsum("kjibac->abcijk", tau35)
        - einsum("jkiacb->abcijk", tau41)
        + einsum("jkibca->abcijk", tau41)
        - einsum("kjicab->abcijk", tau41)
        + einsum("kjicba->abcijk", tau41)
        - einsum("ijkcab->abcijk", tau43)
        + einsum("ijkcba->abcijk", tau43)
        - einsum("ikjacb->abcijk", tau43)
        + einsum("ikjbca->abcijk", tau43)
        + einsum("jikcab->abcijk", tau43)
        - einsum("jikcba->abcijk", tau43)
        - einsum("kijabc->abcijk", tau43)
        + einsum("kijbac->abcijk", tau43)
        + einsum("ijkbca->abcijk", tau85)
        - einsum("ijkacb->abcijk", tau85)
        - einsum("jikbca->abcijk", tau85)
        + einsum("jikacb->abcijk", tau85)
        - einsum("ijkcab->abcijk", tau91)
        + einsum("ijkcba->abcijk", tau91)
        + einsum("jikcab->abcijk", tau91)
        - einsum("jikcba->abcijk", tau91)
        + einsum("ikjbca->abcijk", tau134)
        - einsum("ikjacb->abcijk", tau134)
        - einsum("jkibca->abcijk", tau134)
        + einsum("jkiacb->abcijk", tau134)
        - einsum("ijkacb->abcijk", tau152)
        + einsum("ijkbca->abcijk", tau152)
        - einsum("ikjabc->abcijk", tau152)
        + einsum("ikjbac->abcijk", tau152)
        + einsum("jikacb->abcijk", tau152)
        - einsum("jikbca->abcijk", tau152)
        + einsum("jkiabc->abcijk", tau152)
        - einsum("jkibac->abcijk", tau152)
        + einsum("kijcab->abcijk", tau152)
        - einsum("kijcba->abcijk", tau152)
        - einsum("kjicab->abcijk", tau152)
        + einsum("kjicba->abcijk", tau152)
        + einsum("ijkcab->abcijk", tau160)
        - einsum("ijkcba->abcijk", tau160)
        - einsum("jikcab->abcijk", tau160)
        + einsum("jikcba->abcijk", tau160)
        - einsum("kijacb->abcijk", tau160)
        + einsum("kijbca->abcijk", tau160)
        - einsum("ijkacb->abcijk", tau173)
        + einsum("ijkbca->abcijk", tau173)
        + einsum("jikacb->abcijk", tau173)
        - einsum("jikbca->abcijk", tau173)
        - einsum("kjiacb->abcijk", tau176)
        + einsum("kjibca->abcijk", tau176)
        + einsum("ijkabc->abcijk", tau179)
        - einsum("ijkbac->abcijk", tau179)
        - einsum("jikabc->abcijk", tau179)
        + einsum("jikbac->abcijk", tau179)
        + einsum("kijcab->abcijk", tau179)
        - einsum("kijcba->abcijk", tau179)
        + einsum("ijkcab->abcijk", tau187)
        - einsum("ijkcba->abcijk", tau187)
        + einsum("ikjacb->abcijk", tau187)
        - einsum("ikjbca->abcijk", tau187)
        - einsum("jikcab->abcijk", tau187)
        + einsum("jikcba->abcijk", tau187)
        - einsum("jkiacb->abcijk", tau187)
        + einsum("jkibca->abcijk", tau187)
        + einsum("kijabc->abcijk", tau187)
        - einsum("kijbac->abcijk", tau187)
        - einsum("kjiabc->abcijk", tau187)
        + einsum("kjibac->abcijk", tau187)
        - einsum("ijkabc->abcijk", tau192)
        + einsum("ijkbac->abcijk", tau192)
        - einsum("ikjacb->abcijk", tau192)
        + einsum("ikjbca->abcijk", tau192)
        + einsum("jikabc->abcijk", tau192)
        - einsum("jikbac->abcijk", tau192)
        + einsum("jkiacb->abcijk", tau192)
        - einsum("jkibca->abcijk", tau192)
        - einsum("kijcab->abcijk", tau192)
        + einsum("kijcba->abcijk", tau192)
        + einsum("kjicab->abcijk", tau192)
        - einsum("kjicba->abcijk", tau192)
        - einsum("ijkabc->abcijk", tau194)
        + einsum("ijkbac->abcijk", tau194)
        + einsum("ikjcab->abcijk", tau194)
        + einsum("jikabc->abcijk", tau194)
        - einsum("jikbac->abcijk", tau194)
        - einsum("jkicab->abcijk", tau194)
        - einsum("ikjcab->abcijk", tau197)
        + einsum("jkicab->abcijk", tau197)
        + einsum("kijabc->abcijk", tau197)
        - einsum("kijbac->abcijk", tau197)
        - einsum("kjiabc->abcijk", tau197)
        + einsum("kjibac->abcijk", tau197)
        + einsum("jikacb->abcijk", tau201)
        - einsum("jikbca->abcijk", tau201)
        - einsum("kijbac->abcijk", tau201)
        + einsum("kijabc->abcijk", tau201)
        - einsum("ijkacb->abcijk", tau201)
        + einsum("ijkbca->abcijk", tau201)
        + einsum("kjibac->abcijk", tau201)
        - einsum("kjiabc->abcijk", tau201)
        - einsum("ikjcab->abcijk", tau201)
        + einsum("ikjcba->abcijk", tau201)
        + einsum("jkicab->abcijk", tau201)
        - einsum("jkicba->abcijk", tau201)
        + einsum("ijkabc->abcijk", tau209)
        - einsum("ijkbac->abcijk", tau209)
        - einsum("ikjcab->abcijk", tau209)
        + einsum("ikjcba->abcijk", tau209)
        - einsum("jikabc->abcijk", tau209)
        + einsum("jikbac->abcijk", tau209)
        + einsum("jkicab->abcijk", tau209)
        - einsum("jkicba->abcijk", tau209)
        - einsum("kijacb->abcijk", tau209)
        + einsum("kijbca->abcijk", tau209)
        + einsum("kjiacb->abcijk", tau209)
        - einsum("kjibca->abcijk", tau209)
        + einsum("ijkacb->abcijk", tau211)
        - einsum("ijkbca->abcijk", tau211)
        + einsum("ikjcab->abcijk", tau211)
        - einsum("ikjcba->abcijk", tau211)
        - einsum("jikacb->abcijk", tau211)
        + einsum("jikbca->abcijk", tau211)
        - einsum("jkicab->abcijk", tau211)
        + einsum("jkicba->abcijk", tau211)
        - einsum("kijabc->abcijk", tau211)
        + einsum("kijbac->abcijk", tau211)
        + einsum("kjiabc->abcijk", tau211)
        - einsum("kjibac->abcijk", tau211)
        + einsum("ijkabc->abcijk", tau213)
        - einsum("ijkbac->abcijk", tau213)
        - einsum("jikabc->abcijk", tau213)
        + einsum("jikbac->abcijk", tau213)
        + einsum("kijcab->abcijk", tau213)
        - einsum("kjicab->abcijk", tau213)
        + einsum("ikjabc->abcijk", tau215)
        - einsum("ikjbac->abcijk", tau215)
        - einsum("jkiabc->abcijk", tau215)
        + einsum("jkibac->abcijk", tau215)
        - einsum("kijcab->abcijk", tau215)
        + einsum("kjicab->abcijk", tau215)
        + einsum("kijcab->abcijk", tau221)
        - einsum("kijcba->abcijk", tau221)
        + einsum("kjicab->abcijk", tau227)
        - einsum("kjicba->abcijk", tau227)
        + einsum("jikbca->abcijk", tau236)
        - einsum("jikacb->abcijk", tau236)
        - einsum("ijkbca->abcijk", tau246)
        + einsum("ijkacb->abcijk", tau246)
        + einsum("ikjacb->abcijk", tau248)
        - einsum("ikjbca->abcijk", tau248)
        - einsum("jkiacb->abcijk", tau248)
        + einsum("jkibca->abcijk", tau248)
        - einsum("ijkabc->abcijk", tau250)
        + einsum("ijkbac->abcijk", tau250)
        + einsum("jikabc->abcijk", tau250)
        - einsum("jikbac->abcijk", tau250)
        + einsum("kijacb->abcijk", tau250)
        - einsum("kijbca->abcijk", tau250)
        + einsum("ijkcab->abcijk", tau252)
        - einsum("ijkcba->abcijk", tau252)
        - einsum("jikcab->abcijk", tau252)
        + einsum("jikcba->abcijk", tau252)
        - einsum("kijacb->abcijk", tau252)
        + einsum("kijbca->abcijk", tau252)
        - einsum("jikbac->abcijk", tau257)
        + einsum("jikabc->abcijk", tau257)
        + einsum("kijacb->abcijk", tau257)
        + einsum("ijkbac->abcijk", tau257)
        - einsum("ijkabc->abcijk", tau257)
        - einsum("kjiacb->abcijk", tau257)
        + einsum("ikjcab->abcijk", tau257)
        - einsum("jkicab->abcijk", tau257)
        + einsum("ijkbca->abcijk", tau260)
        + einsum("ikjcba->abcijk", tau260)
        - einsum("jikbca->abcijk", tau260)
        - einsum("jkicba->abcijk", tau260)
        + einsum("kjibac->abcijk", tau264)
        - einsum("kjiabc->abcijk", tau264)
        + einsum("jkicab->abcijk", tau264)
        - einsum("kijbac->abcijk", tau264)
        + einsum("kijabc->abcijk", tau264)
        - einsum("ikjcab->abcijk", tau264)
        - einsum("kijbca->abcijk", tau268)
        + einsum("kjibca->abcijk", tau268)
        - einsum("ikjcba->abcijk", tau268)
        + einsum("jkicba->abcijk", tau268)
        - einsum("jkicab->abcijk", tau272)
        + einsum("ikjcab->abcijk", tau272)
        - einsum("jikbac->abcijk", tau272)
        + einsum("jikabc->abcijk", tau272)
        + einsum("ijkbac->abcijk", tau272)
        - einsum("ijkabc->abcijk", tau272)
        - einsum("ijkacb->abcijk", tau274)
        - einsum("ikjcab->abcijk", tau274)
        + einsum("jikacb->abcijk", tau274)
        + einsum("jkicab->abcijk", tau274)
        + einsum("kijabc->abcijk", tau274)
        - einsum("kijbac->abcijk", tau274)
        - einsum("kjiabc->abcijk", tau274)
        + einsum("kjibac->abcijk", tau274)
        + einsum("ijkcab->abcijk", tau278)
        - einsum("ijkcba->abcijk", tau278)
        - einsum("jikcab->abcijk", tau278)
        + einsum("jikcba->abcijk", tau278)
        - einsum("p,ap,bp,pijkc->abcijk",
                 a.t3.xlam[0, :], a.t3.x1, a.t3.x2, tau355)
        + einsum("p,cp,kp,pijab->abcijk",
                 a.t3.xlam[0, :], a.t3.x1, a.t3.x4, tau372)
    )

    return Tensors(t1=rt1, t2=rt2, t3=rt3)
