from numpy import einsum
from tcc.tensors import Tensors


def _rccsd_cpd_ls_t_calculate_energy(h, a):
    tau0 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau1 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau2 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau3 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau4 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau5 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau6 = (
        einsum("ai,oja->oij", a.t1, h.l.pov)
    )

    tau7 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("oji,oja->ia", tau6, h.l.pov)
    )

    energy = (
        - einsum("po,po->", tau0, tau1)
        + 2 * einsum("po,po->", tau2, tau3)
        + 2 * einsum("o,o->", tau4, tau5)
        + einsum("ai,ia->", a.t1, tau7)
    )

    return energy


def _rccsd_cpd_ls_t_calc_residuals(h, a):
    tau0 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("o,oij->ij", tau0, h.l.poo)
    )

    tau2 = (
        einsum("ai,oja->oij", a.t1, h.l.pov)
    )

    tau3 = (
        einsum("okj,oki->ij", tau2, h.l.poo)
    )

    tau4 = (
        einsum("o,oia->ia", tau0, h.l.pov)
    )

    tau5 = (
        einsum("ai,oja->oij", a.t1, h.l.pov)
    )

    tau6 = (
        einsum("oji,oja->ia", tau5, h.l.pov)
    )

    tau7 = (
        2 * einsum("ia->ia", tau4)
        - einsum("ia->ia", tau6)
        + einsum("ia->ia", h.f.ov)
    )

    tau8 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau9 = (
        einsum("po,oia->pia", tau8, h.l.pov)
    )

    tau10 = (
        einsum("ap,oia->poi", a.t2.x1, h.l.pov)
    )

    tau11 = (
        einsum("ip,oia->poa", a.t2.x4, h.l.pov)
    )

    tau12 = (
        einsum("poi,poa->pia", tau10, tau11)
    )

    tau13 = (
        - einsum("pia->pia", tau9)
        + 2 * einsum("pia->pia", tau12)
    )

    tau14 = (
        einsum("ap,pia->pi", a.t2.x2, tau13)
    )

    tau15 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau16 = (
        einsum("po,oia->pia", tau15, h.l.pov)
    )

    tau17 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau18 = (
        einsum("po,oia->pia", tau17, h.l.pov)
    )

    tau19 = (
        einsum("ap,pia->pi", a.t2.x1, tau18)
    )

    tau20 = (
        2 * einsum("ap,pia->pi", a.t2.x2, tau16)
        - einsum("pi->pi", tau19)
    )

    tau21 = (
        4 * einsum("ij->ij", tau1)
        + 2 * einsum("ij->ij", h.f.oo)
        - 2 * einsum("ji->ij", tau3)
        + 2 * einsum("aj,ia->ij", a.t1, tau7)
        + einsum("pi,jp->ij", tau14, a.t2.x3)
        + einsum("pi,jp->ij", tau20, a.t2.x4)
    )

    tau22 = (
        einsum("po,oij->pij", tau8, tau5)
    )

    tau23 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau24 = (
        einsum("o,oia->ia", tau23, h.l.pov)
    )

    tau25 = (
        einsum("oji,oja->ia", tau2, h.l.pov)
    )

    tau26 = (
        2 * einsum("ia->ia", tau24)
        - einsum("ia->ia", tau25)
        + einsum("ia->ia", h.f.ov)
    )

    tau27 = (
        einsum("ia,ap->pi", tau26, a.t2.x1)
    )

    tau28 = (
        einsum("pi,ip->p", tau27, a.t2.x3)
    )

    tau29 = (
        einsum("pi,ip->p", tau27, a.t2.x4)
    )

    tau30 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau31 = (
        einsum("jp,oji->poi", a.t2.x3, h.l.poo)
    )

    tau32 = (
        einsum("ap,oia->poi", a.t2.x1, h.l.pov)
    )

    tau33 = (
        einsum("poi,poj->pij", tau31, tau32)
    )

    tau34 = (
        einsum("po,oij->pij", tau15, h.l.poo)
    )

    tau35 = (
        2 * einsum("po,oji->pij", tau30, tau2)
        - einsum("pji->pij", tau33)
        + 2 * einsum("pij->pij", tau34)
    )

    tau36 = (
        - einsum("jp,pij->pi", a.t2.x3, tau22)
        - 2 * einsum("p,ip->pi", tau28, a.t2.x4)
        + einsum("p,ip->pi", tau29, a.t2.x3)
        + einsum("jp,pji->pi", a.t2.x4, tau35)
    )

    tau37 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau38 = (
        einsum("po,oij->pij", tau17, h.l.poo)
    )

    tau39 = (
        einsum("ia,ap->pi", tau24, a.t2.x2)
    )

    tau40 = (
        einsum("po,oji->pij", tau37, tau2)
        + einsum("pij->pij", tau38)
        + 4 * einsum("pi,jp->pij", tau39, a.t2.x3)
    )

    tau41 = (
        einsum("ia,ap->pi", h.f.ov, a.t2.x2)
    )

    tau42 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau43 = (
        einsum("oij->oij", h.l.poo)
        + einsum("oji->oij", tau5)
    )

    tau44 = (
        einsum("pi,jp->pij", tau41, a.t2.x4)
        + 2 * einsum("po,oij->pij", tau42, tau43)
    )

    tau45 = (
        2 * einsum("ia->ia", tau24)
        - einsum("ia->ia", tau25)
    )

    tau46 = (
        einsum("ia,ap->pi", tau45, a.t2.x2)
    )

    tau47 = (
        einsum("pi,ip->p", tau46, a.t2.x3)
    )

    tau48 = (
        - einsum("ia->ia", tau25)
        + einsum("ia->ia", h.f.ov)
    )

    tau49 = (
        einsum("ia,ap->pi", tau48, a.t2.x2)
    )

    tau50 = (
        einsum("pi,ip->p", tau49, a.t2.x4)
    )

    tau51 = (
        - einsum("jp,pji->pi", a.t2.x4, tau40)
        + einsum("jp,pji->pi", a.t2.x3, tau44)
        + einsum("p,ip->pi", tau47, a.t2.x4)
        - 2 * einsum("p,ip->pi", tau50, a.t2.x3)
    )

    tau52 = (
        einsum("oij->oij", h.l.poo)
        + einsum("oji->oij", tau2)
    )

    tau53 = (
        einsum("po,ap,ip->oia", tau17, a.t2.x1, a.t2.x4)
        + einsum("po,ap,ip->oia", tau8, a.t2.x2, a.t2.x3)
        - 2 * einsum("po,ap,ip->oia", tau42, a.t2.x1, a.t2.x3)
        - 2 * einsum("po,ap,ip->oia", tau15, a.t2.x2, a.t2.x4)
        + 2 * einsum("aj,oji->oia", a.t1, tau52)
    )

    tau54 = (
        einsum("o,oab->ab", tau0, h.l.pvv)
    )

    tau55 = (
        2 * einsum("ab->ab", tau54)
        + einsum("ab->ab", h.f.vv)
    )

    tau56 = (
        einsum("bi,oab->oia", a.t1, h.l.pvv)
    )

    tau57 = (
        einsum("oib,oaj->ijab", tau56, h.l.pvo)
    )

    tau58 = (
        einsum("ap,oia->poi", a.t2.x2, h.l.pov)
    )

    tau59 = (
        einsum("poj,poi->pij", tau32, tau58)
    )

    tau60 = (
        einsum("kp,lp,pij->ijkl", a.t2.x3, a.t2.x4, tau59)
    )

    tau61 = (
        einsum("al,iljk->ijka", a.t1, tau60)
    )

    tau62 = (
        einsum("ak,kijb->ijab", a.t1, tau61)
    )

    tau63 = (
        einsum("ap,oia->poi", a.t2.x2, h.l.pov)
    )

    tau64 = (
        einsum("iq,poi->pqo", a.t2.x3, tau63)
    )

    tau65 = (
        einsum("poi,qpo->pqi", tau58, tau64)
    )

    tau66 = (
        einsum("iq,pqi->pq", a.t2.x3, tau65)
    )

    tau67 = (
        einsum("qp,aq,iq->pia", tau66, a.t2.x1, a.t2.x4)
    )

    tau68 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau67)
    )

    tau69 = (
        einsum("bp,oab->poa", a.t2.x2, h.l.pvv)
    )

    tau70 = (
        einsum("bp,oab->poa", a.t2.x1, h.l.pvv)
    )

    tau71 = (
        einsum("poa,pob->pab", tau69, tau70)
    )

    tau72 = (
        einsum("ip,jp,pab->ijab", a.t2.x3, a.t2.x4, tau71)
    )

    tau73 = (
        einsum("iq,poi->pqo", a.t2.x4, tau32)
    )

    tau74 = (
        einsum("poi,qpo->pqi", tau10, tau73)
    )

    tau75 = (
        einsum("iq,pqi->pq", a.t2.x4, tau74)
    )

    tau76 = (
        einsum("qp,aq,iq->pia", tau75, a.t2.x2, a.t2.x3)
    )

    tau77 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau76)
    )

    tau78 = (
        4 * einsum("ijab->ijab", tau57)
        + 2 * einsum("ijab->ijab", tau62)
        + einsum("ijab->ijab", tau68)
        + 2 * einsum("ijab->ijab", tau72)
        + einsum("ijab->ijab", tau77)
    )

    tau79 = (
        einsum("poi,poj->pij", tau10, tau63)
    )

    tau80 = (
        einsum("jq,iq,pij->pq", a.t2.x3, a.t2.x4, tau79)
    )

    tau81 = (
        einsum("qp,iq,jq->pij", tau80, a.t2.x3, a.t2.x4)
    )

    tau82 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau81)
    )

    tau83 = (
        einsum("ap,ip,oia->po", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau84 = (
        einsum("qo,po->pq", tau17, tau83)
    )

    tau85 = (
        einsum("pq,aq,iq->pia", tau84, a.t2.x1, a.t2.x4)
    )

    tau86 = (
        einsum("bp,ip,pja->ijab", a.t2.x2, a.t2.x3, tau85)
    )

    tau87 = (
        einsum("jp,oij->poi", a.t2.x4, tau2)
    )

    tau88 = (
        einsum("poj,poi->pij", tau10, tau87)
    )

    tau89 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau88)
    )

    tau90 = (
        einsum("jp,oij->poi", a.t2.x3, tau2)
    )

    tau91 = (
        einsum("poj,poi->pij", tau58, tau90)
    )

    tau92 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau91)
    )

    tau93 = (
        einsum("bi,oab->oia", a.t1, h.l.pvv)
    )

    tau94 = (
        einsum("ojk,oia->ijka", tau2, tau93)
    )

    tau95 = (
        einsum("bp,oab->poa", a.t2.x1, h.l.pvv)
    )

    tau96 = (
        einsum("poi,poa->pia", tau63, tau95)
    )

    tau97 = (
        einsum("ia,ap->pi", tau7, a.t2.x2)
    )

    tau98 = (
        einsum("pia->pia", tau96)
        + einsum("pi,ap->pia", tau97, a.t2.x1)
    )

    tau99 = (
        einsum("ip,jp,pka->ijka", a.t2.x3, a.t2.x4, tau98)
    )

    tau100 = (
        einsum("oia->oia", tau93)
        + einsum("oai->oia", h.l.pvo)
    )

    tau101 = (
        einsum("oia,ojk->ijka", tau100, h.l.poo)
    )

    tau102 = (
        - einsum("jika->ijka", tau89)
        - einsum("jika->ijka", tau92)
        + 2 * einsum("jkia->ijka", tau94)
        + einsum("jkia->ijka", tau99)
        + 2 * einsum("jika->ijka", tau101)
    )

    tau103 = (
        einsum("ak,kijb->ijab", a.t1, tau102)
    )

    tau104 = (
        einsum("ai,ja->ij", a.t1, tau26)
    )

    tau105 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau104)
    )

    tau106 = (
        einsum("ji,jp->pi", tau105, a.t2.x4)
    )

    tau107 = (
        einsum("jp,oji->poi", a.t2.x4, tau52)
    )

    tau108 = (
        einsum("poi,poa->pia", tau107, tau95)
    )

    tau109 = (
        einsum("pi,ap->pia", tau106, a.t2.x1)
        + einsum("pia->pia", tau108)
    )

    tau110 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau109)
    )

    tau111 = (
        einsum("oia,oib->ab", tau93, h.l.pov)
    )

    tau112 = (
        2 * einsum("ab->ab", tau54)
        + einsum("ab->ab", h.f.vv)
        - einsum("ab->ab", tau111)
    )

    tau113 = (
        einsum("ab,bp->pa", tau112, a.t2.x2)
    )

    tau114 = (
        einsum("jp,oji->poi", a.t2.x3, tau52)
    )

    tau115 = (
        einsum("poi,poa->pia", tau114, tau69)
    )

    tau116 = (
        - einsum("pa,ip->pia", tau113, a.t2.x3)
        + einsum("pia->pia", tau115)
    )

    tau117 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau116)
    )

    tau118 = (
        einsum("ip,pia->pa", a.t2.x3, tau13)
    )

    tau119 = (
        einsum("qa,ap->pq", tau118, a.t2.x1)
    )

    tau120 = (
        einsum("pq,aq->pa", tau119, a.t2.x2)
    )

    tau121 = (
        einsum("pb,ap,jp,ip->ijab", tau120, a.t2.x2, a.t2.x3, a.t2.x4)
    )

    tau122 = (
        einsum("po,oia->pia", tau30, h.l.pov)
    )

    tau123 = (
        einsum("ap,pia->pi", a.t2.x2, tau122)
    )

    tau124 = (
        2 * einsum("pi->pi", tau123)
        - einsum("pi->pi", tau19)
    )

    tau125 = (
        einsum("pj,ip->ij", tau124, a.t2.x4)
    )

    tau126 = (
        einsum("ij,jp->pi", tau125, a.t2.x3)
    )

    tau127 = (
        einsum("pj,bp,ap,ip->ijab", tau126, a.t2.x1, a.t2.x2, a.t2.x4)
    )

    tau128 = (
        - einsum("ijab->ijab", tau82)
        - einsum("ijab->ijab", tau86)
        + 2 * einsum("ijab->ijab", tau103)
        + 2 * einsum("jiba->ijab", tau110)
        + 2 * einsum("jiba->ijab", tau117)
        + einsum("jiab->ijab", tau121)
        + einsum("ijba->ijab", tau127)
    )

    tau129 = (
        einsum("poj,poi->pij", tau31, tau87)
    )

    tau130 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau129)
    )

    tau131 = (
        einsum("oil,ojk->ijkl", tau2, h.l.poo)
    )

    tau132 = (
        einsum("al,ijkl->ijka", a.t1, tau131)
    )

    tau133 = (
        einsum("jp,oji->poi", a.t2.x4, h.l.poo)
    )

    tau134 = (
        einsum("poi,poj->pij", tau10, tau133)
    )

    tau135 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau134)
    )

    tau136 = (
        einsum("jp,oji->poi", a.t2.x3, h.l.poo)
    )

    tau137 = (
        einsum("poj,poi->pij", tau136, tau58)
    )

    tau138 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau137)
    )

    tau139 = (
        2 * einsum("ijka->ijka", tau132)
        + einsum("jika->ijka", tau135)
        + einsum("jika->ijka", tau138)
    )

    tau140 = (
        einsum("ak,ikjb->ijab", a.t1, tau139)
    )

    tau141 = (
        einsum("o,oij->ij", tau23, h.l.poo)
    )

    tau142 = (
        einsum("oki,okj->ij", tau5, h.l.poo)
    )

    tau143 = (
        2 * einsum("ij->ij", tau141)
        - einsum("ij->ij", tau142)
    )

    tau144 = (
        einsum("ji,jp->pi", tau143, a.t2.x4)
    )

    tau145 = (
        einsum("pj,bp,ap,ip->ijab", tau144, a.t2.x1, a.t2.x2, a.t2.x3)
    )

    tau146 = (
        einsum("ijab->ijab", tau130)
        + einsum("ijab->ijab", tau140)
        - einsum("jiba->ijab", tau145)
    )

    tau147 = (
        einsum("poi,poj->pij", tau31, tau63)
    )

    tau148 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau147)
    )

    tau149 = (
        einsum("jp,oji->poi", a.t2.x4, h.l.poo)
    )

    tau150 = (
        einsum("poi,poj->pij", tau149, tau32)
    )

    tau151 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau150)
    )

    tau152 = (
        einsum("ijka->ijka", tau148)
        + einsum("ijka->ijka", tau151)
    )

    tau153 = (
        einsum("ak,ikjb->ijab", a.t1, tau152)
    )

    tau154 = (
        2 * einsum("ij->ij", tau1)
        - einsum("ji->ij", tau3)
    )

    tau155 = (
        einsum("ji,jp->pi", tau154, a.t2.x4)
    )

    tau156 = (
        einsum("pj,bp,ap,ip->ijab", tau155, a.t2.x1, a.t2.x2, a.t2.x3)
    )

    tau157 = (
        einsum("jp,oji->poi", a.t2.x4, tau43)
    )

    tau158 = (
        einsum("poj,poi->pij", tau136, tau157)
    )

    tau159 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau158)
    )

    tau160 = (
        einsum("ijab->ijab", tau153)
        - einsum("jiba->ijab", tau156)
        + einsum("ijab->ijab", tau159)
    )

    tau161 = (
        einsum("bp,oab->poa", a.t2.x2, h.l.pvv)
    )

    tau162 = (
        einsum("pob,poa->pab", tau161, tau95)
    )

    tau163 = (
        einsum("ip,jp,pab->ijab", a.t2.x3, a.t2.x4, tau162)
    )

    tau164 = (
        einsum("oai,obj->ijab", h.l.pvo, h.l.pvo)
    )

    tau165 = (
        einsum("qo,po->pq", tau8, tau83)
    )

    tau166 = (
        einsum("pq,aq,iq->pia", tau165, a.t2.x2, a.t2.x3)
    )

    tau167 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau166)
    )

    tau168 = (
        einsum("qo,po->pq", tau17, tau37)
    )

    tau169 = (
        einsum("pq,aq,iq->pia", tau168, a.t2.x1, a.t2.x4)
    )

    tau170 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau169)
    )

    tau171 = (
        einsum("oai->oia", h.l.pvo)
        + einsum("oia->oia", tau56)
    )

    tau172 = (
        einsum("oia,ojb->ijab", tau171, tau93)
    )

    tau173 = (
        einsum("kp,lp,pij->ijkl", a.t2.x3, a.t2.x4, tau79)
    )

    tau174 = (
        einsum("ojl,oik->ijkl", tau2, tau5)
    )

    tau175 = (
        einsum("klij->ijkl", tau173)
        + 2 * einsum("ijkl->ijkl", tau174)
    )

    tau176 = (
        einsum("al,ijlk->ijka", a.t1, tau175)
    )

    tau177 = (
        einsum("ak,ijkb->ijab", a.t1, tau176)
    )

    tau178 = (
        einsum("iq,poi->pqo", a.t2.x4, tau63)
    )

    tau179 = (
        einsum("qpo,poi->pqi", tau178, tau58)
    )

    tau180 = (
        einsum("iq,pqi->pq", a.t2.x4, tau179)
    )

    tau181 = (
        einsum("ap,ip,oia->po", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau182 = (
        einsum("po,qo->pq", tau181, tau42)
    )

    tau183 = (
        - einsum("qp->pq", tau180)
        + 2 * einsum("pq->pq", tau182)
    )

    tau184 = (
        einsum("pq,aq,iq->pia", tau183, a.t2.x1, a.t2.x3)
    )

    tau185 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau184)
    )

    tau186 = (
        einsum("ip,oia->poa", a.t2.x3, h.l.pov)
    )

    tau187 = (
        einsum("poi,poa->pia", tau10, tau186)
    )

    tau188 = (
        - einsum("pia->pia", tau187)
        + 2 * einsum("pia->pia", tau16)
    )

    tau189 = (
        einsum("ap,qia->pqi", a.t2.x1, tau188)
    )

    tau190 = (
        einsum("ip,pqi->pq", a.t2.x3, tau189)
    )

    tau191 = (
        einsum("qp,aq,iq->pia", tau190, a.t2.x2, a.t2.x4)
    )

    tau192 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x4, tau191)
    )

    tau193 = (
        2 * einsum("ijab->ijab", tau163)
        + 4 * einsum("ijab->ijab", tau164)
        + einsum("ijab->ijab", tau167)
        + einsum("ijab->ijab", tau170)
        + 4 * einsum("jiba->ijab", tau172)
        + 2 * einsum("ijba->ijab", tau177)
        + 2 * einsum("ijab->ijab", tau185)
        + 2 * einsum("jiba->ijab", tau192)
    )

    tau194 = (
        einsum("poj,poi->pij", tau149, tau90)
    )

    tau195 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau194)
    )

    tau196 = (
        einsum("po,oij->pij", tau37, h.l.poo)
    )

    tau197 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau196)
    )

    tau198 = (
        einsum("po,oij->pij", tau83, h.l.poo)
    )

    tau199 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau198)
    )

    tau200 = (
        einsum("po,oij->pij", tau181, h.l.poo)
    )

    tau201 = (
        einsum("poj,poi->pij", tau133, tau58)
    )

    tau202 = (
        2 * einsum("pij->pij", tau200)
        - einsum("pij->pij", tau201)
    )

    tau203 = (
        einsum("ap,ip,pjk->ijka", a.t2.x1, a.t2.x3, tau202)
    )

    tau204 = (
        einsum("poi,poj->pij", tau10, tau136)
    )

    tau205 = (
        einsum("po,oij->pij", tau30, h.l.poo)
    )

    tau206 = (
        - einsum("pij->pij", tau204)
        + 2 * einsum("pij->pij", tau205)
    )

    tau207 = (
        einsum("ap,ip,pjk->ijka", a.t2.x2, a.t2.x4, tau206)
    )

    tau208 = (
        einsum("ijka->ijka", tau197)
        + einsum("ijka->ijka", tau199)
        - einsum("kija->ijka", tau203)
        - einsum("kija->ijka", tau207)
    )

    tau209 = (
        einsum("ak,kijb->ijab", a.t1, tau208)
    )

    tau210 = (
        einsum("ji,jp->pi", tau143, a.t2.x3)
    )

    tau211 = (
        einsum("pj,ap,bp,ip->ijab", tau210, a.t2.x1, a.t2.x2, a.t2.x4)
    )

    tau212 = (
        einsum("ijab->ijab", tau195)
        + einsum("ijab->ijab", tau209)
        - einsum("jiab->ijab", tau211)
    )

    tau213 = (
        einsum("poi,qpo->pqi", tau10, tau178)
    )

    tau214 = (
        einsum("iq,pqi->pq", a.t2.x3, tau213)
    )

    tau215 = (
        einsum("pq,aq,iq->pia", tau214, a.t2.x1, a.t2.x4)
    )

    tau216 = (
        einsum("bp,ip,pja->ijab", a.t2.x2, a.t2.x3, tau215)
    )

    tau217 = (
        einsum("po,oia->pia", tau83, h.l.pov)
    )

    tau218 = (
        2 * einsum("pia->pia", tau12)
        - einsum("pia->pia", tau217)
    )

    tau219 = (
        einsum("ap,pia->pi", a.t2.x2, tau218)
    )

    tau220 = (
        einsum("pj,ip->ij", tau219, a.t2.x3)
    )

    tau221 = (
        2 * einsum("ij->ij", h.f.oo)
        + 2 * einsum("ji->ij", tau104)
        + einsum("ji->ij", tau220)
    )

    tau222 = (
        einsum("ji,jp->pi", tau221, a.t2.x3)
    )

    tau223 = (
        einsum("ab,bp->pa", tau112, a.t2.x1)
    )

    tau224 = (
        einsum("ip,oia->poa", a.t2.x3, h.l.pov)
    )

    tau225 = (
        einsum("poa,poi->pia", tau224, tau63)
    )

    tau226 = (
        - einsum("pia->pia", tau18)
        + 2 * einsum("pia->pia", tau225)
    )

    tau227 = (
        einsum("ip,pia->pa", a.t2.x4, tau226)
    )

    tau228 = (
        einsum("qa,ap->pq", tau227, a.t2.x1)
    )

    tau229 = (
        einsum("pq,aq->pa", tau228, a.t2.x1)
    )

    tau230 = (
        2 * einsum("pa->pa", tau223)
        - einsum("pa->pa", tau229)
    )

    tau231 = (
        einsum("qo,po->pq", tau17, tau30)
    )

    tau232 = (
        einsum("poi,qpo->pqi", tau10, tau64)
    )

    tau233 = (
        einsum("iq,pqi->pq", a.t2.x3, tau232)
    )

    tau234 = (
        2 * einsum("pq->pq", tau231)
        - einsum("pq->pq", tau233)
    )

    tau235 = (
        einsum("aq,pia->pqi", a.t2.x2, tau122)
    )

    tau236 = (
        2 * einsum("pqi->pqi", tau235)
        - einsum("pqi->pqi", tau232)
    )

    tau237 = (
        einsum("ip,qpi->pq", a.t2.x4, tau236)
    )

    tau238 = (
        einsum("pq,iq->pqi", tau234, a.t2.x4)
        - 2 * einsum("qp,iq->pqi", tau237, a.t2.x3)
    )

    tau239 = (
        einsum("aq,pqi->pia", a.t2.x1, tau238)
    )

    tau240 = (
        einsum("po,oia->pia", tau15, tau100)
    )

    tau241 = (
        einsum("poa,poi->pia", tau224, tau32)
    )

    tau242 = (
        - einsum("pia->pia", tau241)
        + 2 * einsum("pia->pia", tau122)
    )

    tau243 = (
        einsum("ap,qia->pqi", a.t2.x1, tau242)
    )

    tau244 = (
        einsum("ip,pqi->pq", a.t2.x4, tau243)
    )

    tau245 = (
        einsum("qp,aq,iq->pia", tau244, a.t2.x2, a.t2.x3)
    )

    tau246 = (
        einsum("poi,poa->pia", tau114, tau95)
    )

    tau247 = (
        einsum("pi,ap->pia", tau222, a.t2.x1)
        - einsum("pa,ip->pia", tau230, a.t2.x3)
        + einsum("pia->pia", tau239)
        - 4 * einsum("pia->pia", tau240)
        + einsum("pia->pia", tau245)
        + 2 * einsum("pia->pia", tau246)
    )

    tau248 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x4, tau247)
    )

    tau249 = (
        einsum("ji->ij", tau220)
        + einsum("ji->ij", tau125)
    )

    tau250 = (
        einsum("ji,jp->pi", tau249, a.t2.x4)
    )

    tau251 = (
        einsum("ip,pia->pa", a.t2.x3, tau218)
    )

    tau252 = (
        einsum("qa,ap->pq", tau251, a.t2.x2)
    )

    tau253 = (
        einsum("pq,aq->pa", tau252, a.t2.x2)
    )

    tau254 = (
        einsum("po,oia->pia", tau37, h.l.pov)
    )

    tau255 = (
        2 * einsum("pia->pia", tau225)
        - einsum("pia->pia", tau254)
    )

    tau256 = (
        einsum("ip,pia->pa", a.t2.x4, tau255)
    )

    tau257 = (
        einsum("qa,ap->pq", tau256, a.t2.x2)
    )

    tau258 = (
        einsum("pq,aq->pa", tau257, a.t2.x1)
    )

    tau259 = (
        einsum("pa->pa", tau253)
        + einsum("pa->pa", tau258)
    )

    tau260 = (
        einsum("poa,poi->pia", tau11, tau58)
    )

    tau261 = (
        einsum("po,oia->pia", tau42, h.l.pov)
    )

    tau262 = (
        - einsum("pia->pia", tau260)
        + 2 * einsum("pia->pia", tau261)
    )

    tau263 = (
        einsum("ap,qia->pqi", a.t2.x2, tau262)
    )

    tau264 = (
        einsum("ip,pqi->pq", a.t2.x3, tau263)
    )

    tau265 = (
        einsum("qp,aq,iq->pia", tau264, a.t2.x1, a.t2.x4)
    )

    tau266 = (
        einsum("iq,pqi->pq", a.t2.x4, tau213)
    )

    tau267 = (
        einsum("qo,po->pq", tau42, tau83)
    )

    tau268 = (
        - einsum("pq->pq", tau266)
        + 2 * einsum("pq->pq", tau267)
    )

    tau269 = (
        einsum("qp,aq,iq->pia", tau268, a.t2.x2, a.t2.x3)
    )

    tau270 = (
        einsum("pi,ap->pia", tau250, a.t2.x2)
        + einsum("pa,ip->pia", tau259, a.t2.x4)
        + einsum("pia->pia", tau265)
        + einsum("pia->pia", tau269)
    )

    tau271 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau270)
    )

    tau272 = (
        einsum("po,oij->pij", tau17, tau5)
    )

    tau273 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau272)
    )

    tau274 = (
        einsum("oik,oaj->ijka", tau2, h.l.pvo)
    )

    tau275 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau22)
    )

    tau276 = (
        einsum("poi,poa->pia", tau32, tau69)
    )

    tau277 = (
        einsum("jp,kp,pia->ijka", a.t2.x3, a.t2.x4, tau276)
    )

    tau278 = (
        einsum("poj,poi->pij", tau10, tau90)
    )

    tau279 = (
        einsum("po,oij->pij", tau15, tau5)
    )

    tau280 = (
        einsum("ia,ap->pi", tau7, a.t2.x1)
    )

    tau281 = (
        - einsum("pji->pij", tau278)
        + 2 * einsum("pji->pij", tau279)
        + einsum("pi,jp->pij", tau280, a.t2.x3)
    )

    tau282 = (
        einsum("ap,ip,pjk->ijka", a.t2.x2, a.t2.x4, tau281)
    )

    tau283 = (
        einsum("po,oij->pij", tau42, tau5)
    )

    tau284 = (
        einsum("poj,poi->pij", tau58, tau87)
    )

    tau285 = (
        2 * einsum("pij->pij", tau283)
        - einsum("pij->pij", tau284)
    )

    tau286 = (
        einsum("ap,ip,pjk->ijka", a.t2.x1, a.t2.x3, tau285)
    )

    tau287 = (
        - einsum("jika->ijka", tau273)
        + 2 * einsum("jkia->ijka", tau274)
        - einsum("jika->ijka", tau275)
        + einsum("ijka->ijka", tau277)
        + einsum("kija->ijka", tau282)
        + einsum("kjia->ijka", tau286)
    )

    tau288 = (
        einsum("ak,kijb->ijab", a.t1, tau287)
    )

    tau289 = (
        einsum("po,oia->pia", tau42, tau100)
    )

    tau290 = (
        einsum("poi,poa->pia", tau107, tau69)
    )

    tau291 = (
        2 * einsum("pia->pia", tau289)
        - einsum("pia->pia", tau290)
    )

    tau292 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau291)
    )

    tau293 = (
        einsum("po,oia->pia", tau17, tau100)
    )

    tau294 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau293)
    )

    tau295 = (
        einsum("po,oia->pia", tau8, tau100)
    )

    tau296 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau295)
    )

    tau297 = (
        einsum("iq,jq,pij->pq", a.t2.x3, a.t2.x4, tau79)
    )

    tau298 = (
        einsum("qp,iq,jq->pij", tau297, a.t2.x3, a.t2.x4)
    )

    tau299 = (
        einsum("jp,oij->poi", a.t2.x3, tau5)
    )

    tau300 = (
        einsum("poi,poj->pij", tau299, tau87)
    )

    tau301 = (
        einsum("pij->pij", tau298)
        + 2 * einsum("pij->pij", tau300)
    )

    tau302 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau301)
    )

    tau303 = (
        - einsum("ijab->ijab", tau216)
        + einsum("jiba->ijab", tau248)
        + einsum("ijab->ijab", tau271)
        + 2 * einsum("ijab->ijab", tau288)
        - 2 * einsum("jiba->ijab", tau292)
        + 2 * einsum("jiba->ijab", tau294)
        + 2 * einsum("jiba->ijab", tau296)
        - einsum("ijab->ijab", tau302)
    )

    tau304 = (
        einsum("ij,jp->pi", tau3, a.t2.x3)
    )

    tau305 = (
        einsum("pi,ap,bp,jp->ijab", tau304, a.t2.x1, a.t2.x2, a.t2.x4)
    )

    tau306 = (
        einsum("po,oij->pij", tau8, h.l.poo)
    )

    tau307 = (
        einsum("ap,kp,pij->ijka", a.t2.x2, a.t2.x3, tau306)
    )

    tau308 = (
        einsum("ap,kp,pij->ijka", a.t2.x1, a.t2.x4, tau38)
    )

    tau309 = (
        - einsum("pji->pij", tau33)
        + 2 * einsum("pij->pij", tau34)
    )

    tau310 = (
        einsum("ap,ip,pjk->ijka", a.t2.x2, a.t2.x4, tau309)
    )

    tau311 = (
        einsum("po,oij->pij", tau42, h.l.poo)
    )

    tau312 = (
        einsum("poi,poj->pij", tau149, tau63)
    )

    tau313 = (
        2 * einsum("pij->pij", tau311)
        - einsum("pji->pij", tau312)
    )

    tau314 = (
        einsum("ap,ip,pjk->ijka", a.t2.x1, a.t2.x3, tau313)
    )

    tau315 = (
        einsum("okl,oij->ijkl", tau43, h.l.poo)
    )

    tau316 = (
        einsum("al,lkij->ijka", a.t1, tau315)
    )

    tau317 = (
        einsum("jika->ijka", tau307)
        + einsum("jika->ijka", tau308)
        - einsum("kjia->ijka", tau310)
        - einsum("kjia->ijka", tau314)
        + 2 * einsum("jika->ijka", tau316)
    )

    tau318 = (
        einsum("ak,ikjb->ijab", a.t1, tau317)
    )

    tau319 = (
        einsum("ji,jp->pi", tau1, a.t2.x3)
    )

    tau320 = (
        einsum("jp,oji->poi", a.t2.x3, tau43)
    )

    tau321 = (
        einsum("poj,poi->pij", tau133, tau320)
    )

    tau322 = (
        - 2 * einsum("pi,jp->pij", tau319, a.t2.x4)
        + einsum("pij->pij", tau321)
    )

    tau323 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau322)
    )

    tau324 = (
        einsum("ijab->ijab", tau305)
        + einsum("ijab->ijab", tau318)
        + einsum("ijab->ijab", tau323)
    )

    rt1 = (
        2 * einsum("ia->ai", h.f.ov.conj())
        + 4 * einsum("o,oai->ai", tau0, h.l.pvo)
        - einsum("aj,ji->ai", a.t1, tau21)
        - einsum("pi,ap->ai", tau36, a.t2.x2)
        - einsum("pi,ap->ai", tau51, a.t2.x1)
        - einsum("oib,oab->ai", tau53, h.l.pvv)
        + 2 * einsum("bi,ab->ai", a.t1, tau55)
    )

    rt2 = (
        - einsum("ijab->abij", tau78) / 2
        + einsum("jiab->abij", tau78)
        + einsum("ijab->abij", tau128) / 2
        - einsum("ijba->abij", tau128)
        - einsum("jiab->abij", tau128)
        + einsum("jiba->abij", tau128) / 2
        + 2 * einsum("jiab->abij", tau146)
        - einsum("jiba->abij", tau146)
        - einsum("ijab->abij", tau160)
        + 2 * einsum("ijba->abij", tau160)
        + einsum("ijab->abij", tau193)
        - einsum("jiab->abij", tau193) / 2
        - einsum("jiab->abij", tau212)
        + 2 * einsum("jiba->abij", tau212)
        - einsum("ijab->abij", tau303)
        + einsum("ijba->abij", tau303) / 2
        + einsum("jiab->abij", tau303) / 2
        - einsum("jiba->abij", tau303)
        + 2 * einsum("ijab->abij", tau324)
        - einsum("ijba->abij", tau324)
    )

    return Tensors(t1=rt1, t2=rt2)


def _rccsd_cpd_ls_t_unf_calculate_energy(h, a):
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


def _rccsd_cpd_ls_t_unf_calc_residuals(h, a):

    tau0 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau1 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau3 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau4 = (
        einsum("ai,piw->paw", a.t1, tau3)
    )

    tau5 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau6 = (
        einsum("ai,piw->paw", a.t1, tau5)
    )

    tau7 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau8 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau7)
    )

    tau9 = (
        einsum("bp,abw->paw", a.t2.x2, tau8)
    )

    tau10 = (
        einsum("bp,abw->paw", a.t2.x1, tau8)
    )

    tau11 = (
        2 * einsum("ap,pbw->pabw", a.t2.x1, tau4)
        - einsum("ap,pbw->pabw", a.t2.x2, tau6)
        + einsum("bp,paw->pabw", a.t2.x1, tau9)
        - 2 * einsum("bp,paw->pabw", a.t2.x2, tau10)
    )

    tau12 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau13 = (
        einsum("w,wia->ia", tau12, h.l.pov)
    )

    tau14 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau13)
    )

    tau15 = (
        einsum("ip,ia->pa", a.t2.x4, tau14)
    )

    tau16 = (
        - einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        + 2 * einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau17 = (
        - einsum("pbw,pabw->pa", tau2, tau11)
        + einsum("pb,pba->pa", tau15, tau16)
    )

    tau18 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau19 = (
        - einsum("ap,pbw->pabw", a.t2.x1, tau4)
        + 2 * einsum("ap,pbw->pabw", a.t2.x2, tau6)
        + einsum("bp,paw->pabw", a.t2.x2, tau10)
        - 2 * einsum("bp,paw->pabw", a.t2.x1, tau9)
    )

    tau20 = (
        einsum("ip,ia->pa", a.t2.x3, tau14)
    )

    tau21 = (
        2 * einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        - einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau22 = (
        - einsum("pbw,pabw->pa", tau18, tau19)
        + einsum("pb,pba->pa", tau20, tau21)
    )

    tau23 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau24 = (
        einsum("pw,wia->pia", tau23, h.l.pov)
    )

    tau25 = (
        einsum("ip,pia->pa", a.t2.x3, tau24)
    )

    tau26 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau27 = (
        einsum("pw,wia->pia", tau26, h.l.pov)
    )

    tau28 = (
        einsum("ip,pia->pa", a.t2.x3, tau27)
    )

    tau29 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau30 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau1, h.l.pov)
        - einsum("wib,baw->ia", h.l.pov, tau29)
    )

    tau31 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau32 = (
        einsum("pba,pbw->paw", tau21, tau31)
    )

    tau33 = (
        2 * einsum("bi,wab->iaw", a.t1, h.l.pvv)
        + einsum("ip,paw->iaw", a.t2.x4, tau32)
    )

    tau34 = (
        - 2 * einsum("ab->ab", h.f.vv)
        - 4 * einsum("w,wab->ab", tau1, h.l.pvv)
        + 2 * einsum("ap,pb->ab", a.t2.x1, tau25)
        - einsum("ap,pb->ab", a.t2.x2, tau28)
        + 2 * einsum("ai,ib->ab", a.t1, tau30)
        + einsum("wib,iaw->ab", h.l.pov, tau33)
    )

    tau35 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau36 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau1, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau35)
    )

    tau37 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau38 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau39 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau40 = (
        - einsum("jp,piw->pijw", a.t2.x3, tau38)
        + 2 * einsum("jp,piw->pijw", a.t2.x4, tau39)
    )

    tau41 = (
        einsum("pjw,pijw->pi", tau37, tau40)
    )

    tau42 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau43 = (
        2 * einsum("jp,piw->pijw", a.t2.x3, tau38)
        - einsum("jp,piw->pijw", a.t2.x4, tau39)
    )

    tau44 = (
        einsum("pjw,pijw->pi", tau42, tau43)
    )

    rt1 = (
        einsum("ia->ai", h.f.ov.conj())
        - einsum("wab,ibw->ai", h.l.pvv, tau0)
        + 2 * einsum("w,wai->ai", tau1, h.l.pvo)
        + einsum("ip,pa->ai", a.t2.x3, tau17) / 2
        + einsum("ip,pa->ai", a.t2.x4, tau22) / 2
        - einsum("bi,ab->ai", a.t1, tau34) / 2
        - einsum("aj,ji->ai", a.t1, tau36)
        - einsum("ap,pi->ai", a.t2.x1, tau41) / 2
        - einsum("ap,pi->ai", a.t2.x2, tau44) / 2
    )
    tau0 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau1 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau2 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau3 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau4 = (
        einsum("paw,pbw->pab", tau2, tau3)
    )

    tau5 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau6 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau7 = (
        einsum("pbw,paw->pab", tau5, tau6)
    )

    tau8 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau9 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau10 = (
        einsum("aj,ijw->iaw", a.t1, tau9)
    )

    tau11 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau12 = (
        einsum("aj,ijw->iaw", a.t1, tau11)
    )

    tau13 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau14 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau15 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau16 = (
        einsum("piw,pjw->pij", tau14, tau15)
    )

    tau17 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau18 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau19 = (
        einsum("pjw,piw->pij", tau17, tau18)
    )

    tau20 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau21 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau22 = (
        einsum("pw,qw->pq", tau20, tau21)
    )

    tau23 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x3, tau22)
    )

    tau24 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau25 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau26 = (
        einsum("pw,qw->pq", tau24, tau25)
    )

    tau27 = (
        einsum("aq,iq,pq->pia", a.t2.x2, a.t2.x4, tau26)
    )

    tau28 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau29 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau30 = (
        einsum("piw,paw->pia", tau28, tau29)
    )

    tau31 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x4, tau30)
    )

    tau32 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x3, tau31)
    )

    tau33 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau34 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau35 = (
        einsum("piw,paw->pia", tau33, tau34)
    )

    tau36 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x3, tau35)
    )

    tau37 = (
        einsum("aq,iq,pq->pia", a.t2.x2, a.t2.x4, tau36)
    )

    tau38 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau39 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau40 = (
        einsum("pw,qw->pq", tau38, tau39)
    )

    tau41 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x4, tau40)
    )

    tau42 = (
        einsum("piw,paw->pia", tau28, tau34)
    )

    tau43 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x3, tau42)
    )

    tau44 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x4, tau43)
    )

    tau45 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau46 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau47 = (
        einsum("pw,qw->pq", tau45, tau46)
    )

    tau48 = (
        einsum("aq,iq,pq->pia", a.t2.x2, a.t2.x3, tau47)
    )

    tau49 = (
        einsum("paw,piw->pia", tau29, tau33)
    )

    tau50 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x4, tau49)
    )

    tau51 = (
        einsum("aq,iq,pq->pia", a.t2.x2, a.t2.x3, tau50)
    )

    tau52 = (
        einsum("piw,pjw->pij", tau18, tau28)
    )

    tau53 = (
        einsum("aj,pij->pia", a.t1, tau52)
    )

    tau54 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau55 = (
        einsum("pjw,piw->pij", tau17, tau54)
    )

    tau56 = (
        einsum("aj,pji->pia", a.t1, tau55)
    )

    tau57 = (
        einsum("pw,wij->pij", tau39, h.l.poo)
    )

    tau58 = (
        einsum("aj,pji->pia", a.t1, tau57)
    )

    tau59 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau60 = (
        einsum("pjw,piw->pij", tau15, tau59)
    )

    tau61 = (
        einsum("aj,pji->pia", a.t1, tau60)
    )

    tau62 = (
        einsum("pw,wij->pij", tau46, h.l.poo)
    )

    tau63 = (
        einsum("aj,pji->pia", a.t1, tau62)
    )

    tau64 = (
        einsum("piw,pjw->pij", tau14, tau33)
    )

    tau65 = (
        einsum("aj,pij->pia", a.t1, tau64)
    )

    tau66 = (
        einsum("pjw,piw->pij", tau15, tau54)
    )

    tau67 = (
        einsum("aj,pji->pia", a.t1, tau66)
    )

    tau68 = (
        einsum("pw,wij->pij", tau38, h.l.poo)
    )

    tau69 = (
        einsum("aj,pji->pia", a.t1, tau68)
    )

    tau70 = (
        einsum("piw,pjw->pij", tau14, tau28)
    )

    tau71 = (
        einsum("aj,pij->pia", a.t1, tau70)
    )

    tau72 = (
        einsum("pw,wij->pij", tau45, h.l.poo)
    )

    tau73 = (
        einsum("aj,pji->pia", a.t1, tau72)
    )

    tau74 = (
        einsum("piw,pjw->pij", tau18, tau33)
    )

    tau75 = (
        einsum("aj,pij->pia", a.t1, tau74)
    )

    tau76 = (
        einsum("pjw,piw->pij", tau17, tau59)
    )

    tau77 = (
        einsum("aj,pji->pia", a.t1, tau76)
    )

    tau78 = (
        einsum("piw,paw->pia", tau15, tau34)
    )

    tau79 = (
        einsum("ai,pja->pij", a.t1, tau78)
    )

    tau80 = (
        einsum("piw,paw->pia", tau17, tau29)
    )

    tau81 = (
        einsum("ai,pja->pij", a.t1, tau80)
    )

    tau82 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau83 = (
        einsum("piw,paw->pia", tau14, tau82)
    )

    tau84 = (
        einsum("ai,pja->pij", a.t1, tau83)
    )

    tau85 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau86 = (
        einsum("piw,paw->pia", tau18, tau85)
    )

    tau87 = (
        einsum("ai,pja->pij", a.t1, tau86)
    )

    tau88 = (
        einsum("wia,jaw->ij", h.l.pov, tau8)
    )

    tau89 = (
        einsum("jp,ji->pi", a.t2.x4, tau88)
    )

    tau90 = (
        einsum("wja,iaw->ij", h.l.pov, tau13)
    )

    tau91 = (
        einsum("jp,ij->pi", a.t2.x3, tau90)
    )

    tau92 = (
        einsum("jp,ij->pi", a.t2.x4, tau90)
    )

    tau93 = (
        einsum("jp,ji->pi", a.t2.x3, tau88)
    )

    tau94 = (
        einsum("pw,wij->pij", tau21, h.l.poo)
    )

    tau95 = (
        einsum("aj,pji->pia", a.t1, tau94)
    )

    tau96 = (
        einsum("pw,wij->pij", tau25, h.l.poo)
    )

    tau97 = (
        einsum("aj,pji->pia", a.t1, tau96)
    )

    tau98 = (
        einsum("pw,wij->pij", tau20, h.l.poo)
    )

    tau99 = (
        einsum("aj,pji->pia", a.t1, tau98)
    )

    tau100 = (
        einsum("pw,wij->pij", tau24, h.l.poo)
    )

    tau101 = (
        einsum("aj,pji->pia", a.t1, tau100)
    )

    tau102 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau103 = (
        einsum("w,wij->ij", tau102, h.l.poo)
    )

    tau104 = (
        einsum("jp,ji->pi", a.t2.x4, tau103)
    )

    tau105 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau106 = (
        einsum("w,wij->ij", tau105, h.l.poo)
    )

    tau107 = (
        einsum("jp,ji->pi", a.t2.x3, tau106)
    )

    tau108 = (
        einsum("jp,ji->pi", a.t2.x4, tau106)
    )

    tau109 = (
        einsum("jp,ji->pi", a.t2.x3, tau103)
    )

    tau110 = (
        einsum("pjw,piw->pij", tau28, tau59)
    )

    tau111 = (
        einsum("aj,pij->pia", a.t1, tau110)
    )

    tau112 = (
        einsum("ai,pib->pab", a.t1, tau111)
    )

    tau113 = (
        einsum("pjw,piw->pij", tau33, tau54)
    )

    tau114 = (
        einsum("aj,pij->pia", a.t1, tau113)
    )

    tau115 = (
        einsum("ai,pib->pab", a.t1, tau114)
    )

    tau116 = (
        einsum("jp,ji->pi", a.t2.x4, h.f.oo)
    )

    tau117 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau116)
    )

    tau118 = (
        einsum("bp,ab->pa", a.t2.x2, h.f.vv)
    )

    tau119 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau118)
    )

    tau120 = (
        einsum("ap,ia->pi", a.t2.x1, h.f.ov)
    )

    tau121 = (
        einsum("ai,pi->pa", a.t1, tau120)
    )

    tau122 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau121)
    )

    tau123 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau124 = (
        einsum("jp,ij->pi", a.t2.x3, tau123)
    )

    tau125 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau124)
    )

    tau126 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau12)
    )

    tau127 = (
        einsum("piw,paw->pia", tau15, tau6)
    )

    tau128 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau127)
    )

    tau129 = (
        einsum("piw,paw->pia", tau17, tau2)
    )

    tau130 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau129)
    )

    tau131 = (
        einsum("pw,wai->pia", tau39, h.l.pvo)
    )

    tau132 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau131)
    )

    tau133 = (
        einsum("pw,wai->pia", tau46, h.l.pvo)
    )

    tau134 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau133)
    )

    tau135 = (
        einsum("pw,wai->pia", tau21, h.l.pvo)
    )

    tau136 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau135)
    )

    tau137 = (
        einsum("pw,wai->pia", tau25, h.l.pvo)
    )

    tau138 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau137)
    )

    tau139 = (
        einsum("piw,paw->pia", tau33, tau6)
    )

    tau140 = (
        einsum("ai,pib->pab", a.t1, tau139)
    )

    tau141 = (
        einsum("ip,jp,pab->ijab", a.t2.x3, a.t2.x4, tau140)
    )

    tau142 = (
        einsum("pw,wab->pab", tau39, h.l.pvv)
    )

    tau143 = (
        einsum("bi,pab->pia", a.t1, tau142)
    )

    tau144 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau143)
    )

    tau145 = (
        einsum("pw,wab->pab", tau46, h.l.pvv)
    )

    tau146 = (
        einsum("bi,pab->pia", a.t1, tau145)
    )

    tau147 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau146)
    )

    tau148 = (
        einsum("paw,pbw->pab", tau6, tau82)
    )

    tau149 = (
        einsum("bi,pab->pia", a.t1, tau148)
    )

    tau150 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau149)
    )

    tau151 = (
        einsum("paw,pbw->pab", tau2, tau85)
    )

    tau152 = (
        einsum("bi,pab->pia", a.t1, tau151)
    )

    tau153 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau152)
    )

    tau154 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau155 = (
        einsum("wac,cbw->ab", h.l.pvv, tau154)
    )

    tau156 = (
        einsum("bp,ab->pa", a.t2.x1, tau155)
    )

    tau157 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau156)
    )

    tau158 = (
        einsum("pw,wab->pab", tau21, h.l.pvv)
    )

    tau159 = (
        einsum("bi,pab->pia", a.t1, tau158)
    )

    tau160 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau159)
    )

    tau161 = (
        einsum("pw,wab->pab", tau25, h.l.pvv)
    )

    tau162 = (
        einsum("bi,pab->pia", a.t1, tau161)
    )

    tau163 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau162)
    )

    tau164 = (
        einsum("w,wab->ab", tau105, h.l.pvv)
    )

    tau165 = (
        einsum("bp,ab->pa", a.t2.x1, tau164)
    )

    tau166 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau165)
    )

    tau167 = (
        einsum("pw,wia->pia", tau21, h.l.pov)
    )

    tau168 = (
        einsum("ai,pja->pij", a.t1, tau167)
    )

    tau169 = (
        einsum("aj,pij->pia", a.t1, tau168)
    )

    tau170 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau169)
    )

    tau171 = (
        einsum("pw,wia->pia", tau25, h.l.pov)
    )

    tau172 = (
        einsum("ai,pja->pij", a.t1, tau171)
    )

    tau173 = (
        einsum("aj,pij->pia", a.t1, tau172)
    )

    tau174 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau173)
    )

    tau175 = (
        einsum("w,wia->ia", tau105, h.l.pov)
    )

    tau176 = (
        einsum("ap,ia->pi", a.t2.x1, tau175)
    )

    tau177 = (
        einsum("ai,pi->pa", a.t1, tau176)
    )

    tau178 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau177)
    )

    tau179 = (
        einsum("w,wia->ia", tau102, h.l.pov)
    )

    tau180 = (
        einsum("ai,ja->ij", a.t1, tau179)
    )

    tau181 = (
        einsum("jp,ij->pi", a.t2.x3, tau180)
    )

    tau182 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau181)
    )

    tau183 = (
        einsum("pw,wia->pia", tau39, h.l.pov)
    )

    tau184 = (
        einsum("ai,pja->pij", a.t1, tau183)
    )

    tau185 = (
        einsum("aj,pij->pia", a.t1, tau184)
    )

    tau186 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau185)
    )

    tau187 = (
        einsum("pw,wia->pia", tau46, h.l.pov)
    )

    tau188 = (
        einsum("ai,pja->pij", a.t1, tau187)
    )

    tau189 = (
        einsum("aj,pij->pia", a.t1, tau188)
    )

    tau190 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau189)
    )

    tau191 = (
        einsum("piw,paw->pia", tau54, tau82)
    )

    tau192 = (
        einsum("ai,pja->pij", a.t1, tau191)
    )

    tau193 = (
        einsum("aj,pij->pia", a.t1, tau192)
    )

    tau194 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x3, tau193)
    )

    tau195 = (
        einsum("piw,paw->pia", tau59, tau85)
    )

    tau196 = (
        einsum("ai,pja->pij", a.t1, tau195)
    )

    tau197 = (
        einsum("aj,pij->pia", a.t1, tau196)
    )

    tau198 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x4, tau197)
    )

    tau199 = (
        einsum("wib,baw->ia", h.l.pov, tau154)
    )

    tau200 = (
        einsum("ap,ia->pi", a.t2.x1, tau199)
    )

    tau201 = (
        einsum("ai,pi->pa", a.t1, tau200)
    )

    tau202 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau201)
    )

    tau203 = (
        einsum("paw,pbw->pab", tau34, tau82)
    )

    tau204 = (
        einsum("bi,pab->pia", a.t1, tau203)
    )

    tau205 = (
        einsum("ai,pja->pij", a.t1, tau204)
    )

    tau206 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau205)
    )

    tau207 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau208 = (
        einsum("wib,baw->ia", h.l.pov, tau207)
    )

    tau209 = (
        einsum("ai,ja->ij", a.t1, tau208)
    )

    tau210 = (
        einsum("jp,ij->pi", a.t2.x3, tau209)
    )

    tau211 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau210)
    )

    tau212 = (
        einsum("ip,pia->pa", a.t2.x3, tau167)
    )

    tau213 = (
        einsum("aq,pa->pq", a.t2.x2, tau212)
    )

    tau214 = (
        einsum("aq,qp->pa", a.t2.x1, tau213)
    )

    tau215 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau214)
    )

    tau216 = (
        einsum("aq,iq,pia->pq", a.t2.x2, a.t2.x4, tau42)
    )

    tau217 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x4, tau216)
    )

    tau218 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau217)
    )

    tau219 = (
        einsum("pw,wia->pia", tau38, h.l.pov)
    )

    tau220 = (
        einsum("ip,pia->pa", a.t2.x4, tau219)
    )

    tau221 = (
        einsum("aq,pa->pq", a.t2.x2, tau220)
    )

    tau222 = (
        einsum("aq,qp->pa", a.t2.x1, tau221)
    )

    tau223 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau222)
    )

    tau224 = (
        einsum("qw,pw->pq", tau21, tau38)
    )

    tau225 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x4, tau224)
    )

    tau226 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau225)
    )

    tau227 = (
        einsum("ap,pia->pi", a.t2.x1, tau167)
    )

    tau228 = (
        einsum("iq,pi->pq", a.t2.x4, tau227)
    )

    tau229 = (
        einsum("iq,qp->pi", a.t2.x3, tau228)
    )

    tau230 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau229)
    )

    tau231 = (
        einsum("pw,wia->pia", tau45, h.l.pov)
    )

    tau232 = (
        einsum("ap,pia->pi", a.t2.x2, tau231)
    )

    tau233 = (
        einsum("iq,pi->pq", a.t2.x4, tau232)
    )

    tau234 = (
        einsum("iq,qp->pi", a.t2.x3, tau233)
    )

    tau235 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau234)
    )

    tau236 = (
        einsum("pw,wia->pia", tau24, h.l.pov)
    )

    tau237 = (
        einsum("ap,pia->pi", a.t2.x2, tau236)
    )

    tau238 = (
        einsum("iq,pi->pq", a.t2.x4, tau237)
    )

    tau239 = (
        einsum("iq,qp->pi", a.t2.x4, tau238)
    )

    tau240 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau239)
    )

    tau241 = (
        einsum("iq,pi->pq", a.t2.x3, tau227)
    )

    tau242 = (
        einsum("iq,qp->pi", a.t2.x3, tau241)
    )

    tau243 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau242)
    )

    tau244 = (
        einsum("iq,jq,pij->pq", a.t2.x3, a.t2.x4, tau110)
    )

    tau245 = (
        einsum("iq,jq,qp->pij", a.t2.x3, a.t2.x4, tau244)
    )

    tau246 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau245)
    )

    tau247 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x4, tau30)
    )

    tau248 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x3, tau247)
    )

    tau249 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau248)
    )

    tau250 = (
        einsum("qw,pw->pq", tau21, tau45)
    )

    tau251 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x3, tau250)
    )

    tau252 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau251)
    )

    tau253 = (
        einsum("ip,pia->pa", a.t2.x4, tau236)
    )

    tau254 = (
        einsum("aq,pa->pq", a.t2.x2, tau253)
    )

    tau255 = (
        einsum("aq,qp->pa", a.t2.x2, tau254)
    )

    tau256 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau255)
    )

    tau257 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x4, tau42)
    )

    tau258 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x3, tau257)
    )

    tau259 = (
        einsum("ap,jp,pib->ijab", a.t2.x1, a.t2.x4, tau258)
    )

    tau260 = (
        einsum("ap,qa->pq", a.t2.x1, tau212)
    )

    tau261 = (
        einsum("aq,pq->pa", a.t2.x1, tau260)
    )

    tau262 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau261)
    )

    tau263 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x3, tau42)
    )

    tau264 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau263)
    )

    tau265 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau264)
    )

    tau266 = (
        einsum("pw,qw->pq", tau24, tau39)
    )

    tau267 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau266)
    )

    tau268 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x4, tau267)
    )

    tau269 = (
        einsum("ap,pia->pi", a.t2.x1, tau219)
    )

    tau270 = (
        einsum("iq,pi->pq", a.t2.x4, tau269)
    )

    tau271 = (
        einsum("iq,qp->pi", a.t2.x4, tau270)
    )

    tau272 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau271)
    )

    tau273 = (
        einsum("ap,pia->pi", a.t2.x2, tau187)
    )

    tau274 = (
        einsum("iq,pi->pq", a.t2.x3, tau273)
    )

    tau275 = (
        einsum("iq,qp->pi", a.t2.x3, tau274)
    )

    tau276 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau275)
    )

    tau277 = (
        einsum("ip,pia->pa", a.t2.x3, tau187)
    )

    tau278 = (
        einsum("aq,pa->pq", a.t2.x2, tau277)
    )

    tau279 = (
        einsum("aq,qp->pa", a.t2.x2, tau278)
    )

    tau280 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau279)
    )

    tau281 = (
        einsum("aq,iq,pia->pq", a.t2.x2, a.t2.x4, tau35)
    )

    tau282 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau281)
    )

    tau283 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau282)
    )

    tau284 = (
        einsum("ap,qa->pq", a.t2.x1, tau220)
    )

    tau285 = (
        einsum("aq,pq->pa", a.t2.x1, tau284)
    )

    tau286 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau285)
    )

    tau287 = (
        einsum("aq,iq,pia->pq", a.t2.x1, a.t2.x4, tau35)
    )

    tau288 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau287)
    )

    tau289 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau288)
    )

    tau290 = (
        einsum("pw,qw->pq", tau24, tau46)
    )

    tau291 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau290)
    )

    tau292 = (
        einsum("ap,ip,pjb->ijab", a.t2.x2, a.t2.x3, tau291)
    )

    tau293 = (
        einsum("qw,pw->pq", tau21, tau24)
    )

    tau294 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau293)
    )

    tau295 = (
        einsum("ap,ip,pjb->ijab", a.t2.x1, a.t2.x3, tau294)
    )

    tau296 = (
        2 * einsum("ijab->ijab", tau117)
        - 2 * einsum("ijab->ijab", tau119)
        + 2 * einsum("ijab->ijab", tau122)
        + 2 * einsum("ijab->ijab", tau125)
        + 4 * einsum("ijab->ijab", tau126)
        + 2 * einsum("ijab->ijab", tau128)
        + 2 * einsum("ijab->ijab", tau130)
        + 2 * einsum("ijab->ijab", tau132)
        + 2 * einsum("ijab->ijab", tau134)
        - 4 * einsum("ijab->ijab", tau136)
        - 4 * einsum("ijab->ijab", tau138)
        + 2 * einsum("ijab->ijab", tau141)
        + 2 * einsum("ijab->ijab", tau144)
        + 2 * einsum("ijab->ijab", tau147)
        + 2 * einsum("ijab->ijab", tau150)
        + 2 * einsum("ijab->ijab", tau153)
        + 2 * einsum("ijab->ijab", tau157)
        - 4 * einsum("ijab->ijab", tau160)
        - 4 * einsum("ijab->ijab", tau163)
        - 4 * einsum("ijab->ijab", tau166)
        + 4 * einsum("ijab->ijab", tau170)
        + 4 * einsum("ijab->ijab", tau174)
        + 4 * einsum("ijab->ijab", tau178)
        + 4 * einsum("ijab->ijab", tau182)
        - 2 * einsum("ijab->ijab", tau186)
        - 2 * einsum("ijab->ijab", tau190)
        - 2 * einsum("ijab->ijab", tau194)
        - 2 * einsum("ijab->ijab", tau198)
        - 2 * einsum("ijab->ijab", tau202)
        - 2 * einsum("ijab->ijab", tau206)
        - 2 * einsum("ijab->ijab", tau211)
        + 2 * einsum("ijab->ijab", tau215)
        - einsum("ijab->ijab", tau218)
        - einsum("ijab->ijab", tau223)
        + 2 * einsum("ijab->ijab", tau226)
        + 2 * einsum("ijab->ijab", tau230)
        - einsum("ijab->ijab", tau235)
        + 2 * einsum("ijab->ijab", tau240)
        + 2 * einsum("ijab->ijab", tau243)
        - einsum("ijab->ijab", tau246)
        - einsum("ijab->ijab", tau249)
        + 2 * einsum("ijab->ijab", tau252)
        + 2 * einsum("ijab->ijab", tau256)
        - einsum("ijab->ijab", tau259)
        + 2 * einsum("ijab->ijab", tau262)
        - einsum("ijab->ijab", tau265)
        + 2 * einsum("ijab->ijab", tau268)
        - einsum("ijab->ijab", tau272)
        - einsum("ijab->ijab", tau276)
        - einsum("ijab->ijab", tau280)
        + 2 * einsum("ijab->ijab", tau283)
        - einsum("ijab->ijab", tau286)
        - einsum("ijab->ijab", tau289)
        + 2 * einsum("ijab->ijab", tau292)
        - 4 * einsum("ijab->ijab", tau295)
    )

    tau297 = (
        einsum("jp,ji->pi", a.t2.x3, h.f.oo)
    )

    tau298 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau297)
    )

    tau299 = (
        einsum("bp,ab->pa", a.t2.x1, h.f.vv)
    )

    tau300 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau299)
    )

    tau301 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau8)
    )

    tau302 = (
        einsum("ap,ia->pi", a.t2.x2, h.f.ov)
    )

    tau303 = (
        einsum("ai,pi->pa", a.t1, tau302)
    )

    tau304 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau303)
    )

    tau305 = (
        einsum("jp,ij->pi", a.t2.x4, tau123)
    )

    tau306 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau305)
    )

    tau307 = (
        einsum("ibw,jaw->ijab", tau0, tau8)
    )

    tau308 = (
        einsum("piw,paw->pia", tau17, tau6)
    )

    tau309 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau308)
    )

    tau310 = (
        einsum("piw,paw->pia", tau15, tau2)
    )

    tau311 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau310)
    )

    tau312 = (
        einsum("ibw,jaw->ijab", tau0, tau12)
    )

    tau313 = (
        einsum("paw,piw->pia", tau2, tau28)
    )

    tau314 = (
        einsum("ai,pib->pab", a.t1, tau313)
    )

    tau315 = (
        einsum("ip,jp,pab->ijab", a.t2.x3, a.t2.x4, tau314)
    )

    tau316 = (
        einsum("paw,pbw->pab", tau6, tau85)
    )

    tau317 = (
        einsum("bi,pab->pia", a.t1, tau316)
    )

    tau318 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau317)
    )

    tau319 = (
        einsum("paw,pbw->pab", tau2, tau82)
    )

    tau320 = (
        einsum("bi,pab->pia", a.t1, tau319)
    )

    tau321 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau320)
    )

    tau322 = (
        einsum("bp,ab->pa", a.t2.x2, tau155)
    )

    tau323 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau322)
    )

    tau324 = (
        einsum("bp,ab->pa", a.t2.x2, tau164)
    )

    tau325 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau324)
    )

    tau326 = (
        einsum("ap,ia->pi", a.t2.x2, tau175)
    )

    tau327 = (
        einsum("ai,pi->pa", a.t1, tau326)
    )

    tau328 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau327)
    )

    tau329 = (
        einsum("jp,ij->pi", a.t2.x4, tau180)
    )

    tau330 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau329)
    )

    tau331 = (
        einsum("piw,paw->pia", tau54, tau85)
    )

    tau332 = (
        einsum("ai,pja->pij", a.t1, tau331)
    )

    tau333 = (
        einsum("aj,pij->pia", a.t1, tau332)
    )

    tau334 = (
        einsum("bp,jp,pia->ijab", a.t2.x1, a.t2.x4, tau333)
    )

    tau335 = (
        einsum("piw,paw->pia", tau59, tau82)
    )

    tau336 = (
        einsum("ai,pja->pij", a.t1, tau335)
    )

    tau337 = (
        einsum("aj,pij->pia", a.t1, tau336)
    )

    tau338 = (
        einsum("bp,jp,pia->ijab", a.t2.x2, a.t2.x3, tau337)
    )

    tau339 = (
        einsum("ap,ia->pi", a.t2.x2, tau199)
    )

    tau340 = (
        einsum("ai,pi->pa", a.t1, tau339)
    )

    tau341 = (
        einsum("bp,ip,jp,pa->ijab", a.t2.x1, a.t2.x3, a.t2.x4, tau340)
    )

    tau342 = (
        einsum("jp,ij->pi", a.t2.x4, tau209)
    )

    tau343 = (
        einsum("ap,bp,jp,pi->ijab", a.t2.x1, a.t2.x2, a.t2.x3, tau342)
    )

    tau344 = (
        einsum("ap,pia->pi", a.t2.x1, tau183)
    )

    tau345 = (
        einsum("iq,pi->pq", a.t2.x3, tau344)
    )

    tau346 = (
        einsum("iq,qp->pi", a.t2.x4, tau345)
    )

    tau347 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau346)
    )

    tau348 = (
        einsum("iq,pi->pq", a.t2.x3, tau237)
    )

    tau349 = (
        einsum("iq,qp->pi", a.t2.x4, tau348)
    )

    tau350 = (
        einsum("ap,bp,ip,pj->ijab", a.t2.x1, a.t2.x2, a.t2.x4, tau349)
    )

    tau351 = (
        einsum("iq,jq,pij->pq", a.t2.x3, a.t2.x4, tau113)
    )

    tau352 = (
        einsum("iq,jq,qp->pij", a.t2.x3, a.t2.x4, tau351)
    )

    tau353 = (
        einsum("ap,bp,pij->ijab", a.t2.x1, a.t2.x2, tau352)
    )

    tau354 = (
        einsum("qw,pw->pq", tau38, tau46)
    )

    tau355 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x3, tau354)
    )

    tau356 = (
        einsum("ap,jp,pib->ijab", a.t2.x1, a.t2.x4, tau355)
    )

    tau357 = (
        einsum("aq,pa->pq", a.t2.x1, tau277)
    )

    tau358 = (
        einsum("aq,qp->pa", a.t2.x2, tau357)
    )

    tau359 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau358)
    )

    tau360 = (
        einsum("aq,pa->pq", a.t2.x1, tau253)
    )

    tau361 = (
        einsum("aq,qp->pa", a.t2.x2, tau360)
    )

    tau362 = (
        einsum("ap,ip,jp,pb->ijab", a.t2.x2, a.t2.x3, a.t2.x4, tau361)
    )

    tau363 = (
        2 * einsum("ijab->ijab", tau298)
        - 2 * einsum("ijab->ijab", tau300)
        + 4 * einsum("ijab->ijab", tau301)
        + 2 * einsum("ijab->ijab", tau304)
        + 2 * einsum("ijab->ijab", tau306)
        + 4 * einsum("ijab->ijab", tau307)
        + 2 * einsum("ijab->ijab", tau309)
        + 2 * einsum("ijab->ijab", tau311)
        + 4 * einsum("ijab->ijab", tau312)
        + 2 * einsum("ijab->ijab", tau315)
        + 2 * einsum("ijab->ijab", tau318)
        + 2 * einsum("ijab->ijab", tau321)
        + 2 * einsum("ijab->ijab", tau323)
        - 4 * einsum("ijab->ijab", tau325)
        + 4 * einsum("ijab->ijab", tau328)
        + 4 * einsum("ijab->ijab", tau330)
        - 2 * einsum("ijab->ijab", tau334)
        - 2 * einsum("ijab->ijab", tau338)
        - 2 * einsum("ijab->ijab", tau341)
        - 2 * einsum("ijab->ijab", tau343)
        - einsum("ijab->ijab", tau347)
        + 2 * einsum("ijab->ijab", tau350)
        - einsum("ijab->ijab", tau353)
        - einsum("ijab->ijab", tau356)
        - einsum("ijab->ijab", tau359)
        + 2 * einsum("ijab->ijab", tau362)
    )

    rt2 = (
        einsum("wbj,iaw->abij", h.l.pvo, tau0)
        + einsum("wai,jbw->abij", h.l.pvo, tau1)
        + einsum("ip,jp,pab->abij", a.t2.x3, a.t2.x4, tau4) / 2
        + einsum("jp,ip,pab->abij", a.t2.x3, a.t2.x4, tau7) / 2
        + einsum("wai,wbj->abij", h.l.pvo, h.l.pvo)
        + einsum("iaw,jbw->abij", tau10, tau8)
        + einsum("jbw,iaw->abij", tau12, tau13)
        + einsum("ap,bp,pij->abij", a.t2.x1, a.t2.x2, tau16) / 2
        + einsum("bp,ap,pij->abij", a.t2.x1, a.t2.x2, tau19) / 2
        + einsum("iaw,jbw->abij", tau10, tau12)
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau23)
        + einsum("ap,ip,pjb->abij", a.t2.x2, a.t2.x4, tau27)
        - einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau32) / 2
        - einsum("bp,jp,pia->abij", a.t2.x2, a.t2.x4, tau37) / 2
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x4, tau41) / 4
        + einsum("bp,ip,pja->abij", a.t2.x1, a.t2.x4, tau44) / 4
        + einsum("ap,ip,pjb->abij", a.t2.x2, a.t2.x3, tau48) / 4
        + einsum("bp,ip,pja->abij", a.t2.x2, a.t2.x3, tau51) / 4
        + einsum("iaw,jbw->abij", tau0, tau1)
        + einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau53) / 2
        + einsum("bp,ip,pja->abij", a.t2.x1, a.t2.x4, tau56) / 2
        + einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x4, tau58) / 2
        + einsum("bp,ip,pja->abij", a.t2.x2, a.t2.x3, tau61) / 2
        + einsum("bp,jp,pia->abij", a.t2.x2, a.t2.x3, tau63) / 2
        + einsum("bp,jp,pia->abij", a.t2.x2, a.t2.x4, tau65) / 2
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau67) / 2
        + einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x4, tau69) / 2
        + einsum("ap,jp,pib->abij", a.t2.x1, a.t2.x4, tau71) / 2
        + einsum("ap,ip,pjb->abij", a.t2.x2, a.t2.x3, tau73) / 2
        + einsum("ap,jp,pib->abij", a.t2.x2, a.t2.x3, tau75) / 2
        + einsum("ap,ip,pjb->abij", a.t2.x2, a.t2.x4, tau77) / 2
        + einsum("ap,bp,pij->abij", a.t2.x1, a.t2.x2, tau79) / 2
        + einsum("bp,ap,pij->abij", a.t2.x1, a.t2.x2, tau81) / 2
        + einsum("ap,bp,pji->abij", a.t2.x1, a.t2.x2, tau84) / 2
        + einsum("bp,ap,pji->abij", a.t2.x1, a.t2.x2, tau87) / 2
        + einsum("ap,bp,ip,pj->abij", a.t2.x1, a.t2.x2, a.t2.x3, tau89) / 2
        + einsum("ap,bp,jp,pi->abij", a.t2.x1, a.t2.x2, a.t2.x4, tau91) / 2
        + einsum("bp,ap,jp,pi->abij", a.t2.x1, a.t2.x2, a.t2.x3, tau92) / 2
        + einsum("bp,ap,ip,pj->abij", a.t2.x1, a.t2.x2, a.t2.x4, tau93) / 2
        - einsum("bp,jp,pia->abij", a.t2.x1, a.t2.x3, tau95)
        - einsum("bp,jp,pia->abij", a.t2.x2, a.t2.x4, tau97)
        - einsum("ap,ip,pjb->abij", a.t2.x1, a.t2.x3, tau99)
        - einsum("ap,ip,pjb->abij", a.t2.x2, a.t2.x4, tau101)
        - einsum("ap,bp,ip,pj->abij", a.t2.x1, a.t2.x2, a.t2.x3, tau104)
        - einsum("ap,bp,jp,pi->abij", a.t2.x1, a.t2.x2, a.t2.x4, tau107)
        - einsum("bp,ap,jp,pi->abij", a.t2.x1, a.t2.x2, a.t2.x3, tau108)
        - einsum("bp,ap,ip,pj->abij", a.t2.x1, a.t2.x2, a.t2.x4, tau109)
        + einsum("ip,jp,pab->abij", a.t2.x3, a.t2.x4, tau112) / 2
        + einsum("jp,ip,pab->abij", a.t2.x3, a.t2.x4, tau115) / 2
        + einsum("iaw,jbw->abij", tau13, tau8)
        - einsum("ijab->abij", tau296) / 4
        - einsum("jiba->abij", tau296) / 4
        - einsum("ijba->abij", tau363) / 4
        - einsum("jiab->abij", tau363) / 4
    )

    return Tensors(t1=rt1, t2=rt2)


def _rccsd_ncpd_ls_t_unf_calculate_energy(h, a):
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


def _rccsd_ncpd_ls_t_unf_calc_residuals(h, a):
    tau0 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau1 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau3 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau4 = (
        einsum("ai,piw->paw", a.t1, tau3)
    )

    tau5 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau6 = (
        einsum("ai,piw->paw", a.t1, tau5)
    )

    tau7 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau8 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau7)
    )

    tau9 = (
        einsum("bp,abw->paw", a.t2.x1, tau8)
    )

    tau10 = (
        einsum("bp,abw->paw", a.t2.x2, tau8)
    )

    tau11 = (
        - einsum("ap,pbw->pabw", a.t2.x1, tau4)
        + 2 * einsum("ap,pbw->pabw", a.t2.x2, tau6)
        + einsum("bp,paw->pabw", a.t2.x2, tau9)
        - 2 * einsum("bp,paw->pabw", a.t2.x1, tau10)
    )

    tau12 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau13 = (
        einsum("w,wia->ia", tau12, h.l.pov)
    )

    tau14 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau13)
    )

    tau15 = (
        einsum("ip,ia->pa", a.t2.x3, tau14)
    )

    tau16 = (
        - einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        + 2 * einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau17 = (
        - einsum("p,pbw,pabw->pa", a.t2.xlam[0, :], tau2, tau11)
        + einsum("p,pb,pab->pa", a.t2.xlam[0, :], tau15, tau16)
    )

    tau18 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau19 = (
        2 * einsum("ap,pbw->pabw", a.t2.x1, tau4)
        - einsum("ap,pbw->pabw", a.t2.x2, tau6)
        + einsum("bp,paw->pabw", a.t2.x1, tau10)
        - 2 * einsum("bp,paw->pabw", a.t2.x2, tau9)
    )

    tau20 = (
        einsum("ip,ia->pa", a.t2.x4, tau14)
    )

    tau21 = (
        2 * einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        - einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau22 = (
        - einsum("p,pbw,pabw->pa", a.t2.xlam[0, :], tau18, tau19)
        + einsum("p,pb,pab->pa", a.t2.xlam[0, :], tau20, tau21)
    )

    tau23 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau24 = (
        einsum("pw,wia->pia", tau23, h.l.pov)
    )

    tau25 = (
        einsum("ip,pia->pa", a.t2.x3, tau24)
    )

    tau26 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau27 = (
        einsum("pw,wia->pia", tau26, h.l.pov)
    )

    tau28 = (
        einsum("ip,pia->pa", a.t2.x3, tau27)
    )

    tau29 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau30 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("w,wia->ia", tau1, h.l.pov)
        - einsum("wib,baw->ia", h.l.pov, tau29)
    )

    tau31 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau32 = (
        einsum("p,pab,pbw->paw", a.t2.xlam[0, :], tau16, tau31)
    )

    tau33 = (
        2 * einsum("bi,wab->iaw", a.t1, h.l.pvv)
        + einsum("ip,paw->iaw", a.t2.x4, tau32)
    )

    tau34 = (
        - 2 * einsum("ab->ab", h.f.vv)
        - 4 * einsum("w,wab->ab", tau1, h.l.pvv)
        + 2 * einsum("p,ap,pb->ab", a.t2.xlam[0, :], a.t2.x1, tau25)
        - einsum("p,ap,pb->ab", a.t2.xlam[0, :], a.t2.x2, tau28)
        + 2 * einsum("ai,ib->ab", a.t1, tau30)
        + einsum("wib,iaw->ab", h.l.pov, tau33)
    )

    tau35 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau36 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("w,wij->ij", tau1, h.l.poo)
        - einsum("wia,jaw->ij", h.l.pov, tau35)
    )

    tau37 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau38 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau39 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau40 = (
        2 * einsum("jp,piw->pijw", a.t2.x3, tau38)
        - einsum("jp,piw->pijw", a.t2.x4, tau39)
    )

    tau41 = (
        einsum("pjw,pijw->pi", tau37, tau40)
    )

    tau42 = (
        2 * einsum("ip,jp->pij", a.t2.x3, a.t2.x4)
        - einsum("jp,ip->pij", a.t2.x3, a.t2.x4)
    )

    tau43 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau44 = (
        einsum("pij,pjw->piw", tau42, tau43)
    )

    tau45 = (
        einsum("p,wji,pjw->pi", a.t2.xlam[0, :], h.l.poo, tau44)
    )

    rt1 = (
        einsum("ia->ai", h.f.ov.conj())
        - einsum("wab,ibw->ai", h.l.pvv, tau0)
        + 2 * einsum("w,wai->ai", tau1, h.l.pvo)
        + einsum("ip,pa->ai", a.t2.x4, tau17) / 2
        + einsum("ip,pa->ai", a.t2.x3, tau22) / 2
        - einsum("bi,ab->ai", a.t1, tau34) / 2
        - einsum("aj,ji->ai", a.t1, tau36)
        - einsum("p,ap,pi->ai", a.t2.xlam[0, :], a.t2.x2, tau41) / 2
        - einsum("ap,pi->ai", a.t2.x1, tau45) / 2
    )
    tau0 = (
        einsum("bp,ab->pa", a.t2.x2, h.f.vv)
    )

    tau1 = (
        einsum("bp,ab->pa", a.t2.x1, h.f.vv)
    )

    tau2 = (
        einsum("jp,ji->pi", a.t2.x4, h.f.oo)
    )

    tau3 = (
        einsum("jp,ji->pi", a.t2.x3, h.f.oo)
    )

    tau4 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau5 = (
        einsum("w,wab->ab", tau4, h.l.pvv)
    )

    tau6 = (
        einsum("bp,ab->pa", a.t2.x2, tau5)
    )

    tau7 = (
        einsum("bp,ab->pa", a.t2.x1, tau5)
    )

    tau8 = (
        einsum("w,wia->ia", tau4, h.l.pov)
    )

    tau9 = (
        einsum("ap,ia->pi", a.t2.x2, tau8)
    )

    tau10 = (
        einsum("ai,pi->pa", a.t1, tau9)
    )

    tau11 = (
        einsum("ap,ia->pi", a.t2.x1, tau8)
    )

    tau12 = (
        einsum("ai,pi->pa", a.t1, tau11)
    )

    tau13 = (
        einsum("ai,wia->w", a.t1, h.l.pov)
    )

    tau14 = (
        einsum("w,wia->ia", tau13, h.l.pov)
    )

    tau15 = (
        einsum("ai,ja->ij", a.t1, tau14)
    )

    tau16 = (
        einsum("jp,ij->pi", a.t2.x3, tau15)
    )

    tau17 = (
        einsum("jp,ij->pi", a.t2.x4, tau15)
    )

    tau18 = (
        einsum("ap,ia->pi", a.t2.x2, h.f.ov)
    )

    tau19 = (
        einsum("ai,pi->pa", a.t1, tau18)
    )

    tau20 = (
        einsum("ap,ia->pi", a.t2.x1, h.f.ov)
    )

    tau21 = (
        einsum("ai,pi->pa", a.t1, tau20)
    )

    tau22 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau23 = (
        einsum("jp,ij->pi", a.t2.x3, tau22)
    )

    tau24 = (
        einsum("jp,ij->pi", a.t2.x4, tau22)
    )

    tau25 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau26 = (
        einsum("wbi,jaw->ijab", h.l.pvo, tau25)
    )

    tau27 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau28 = (
        einsum("pw,wia->pia", tau27, h.l.pov)
    )

    tau29 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau30 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau31 = (
        einsum("piw,paw->pia", tau29, tau30)
    )

    tau32 = (
        - einsum("pia->pia", tau28)
        + 2 * einsum("pia->pia", tau31)
    )

    tau33 = (
        einsum("ip,qia->pqa", a.t2.x3, tau32)
    )

    tau34 = (
        einsum("ap,bq,pqb->pqa", a.t2.x1, a.t2.x1, tau33)
    )

    tau35 = (
        einsum("p,q,bq,iq,qpa->piab",
               a.t2.xlam[0, :], a.t2.xlam[0, :], a.t2.x2, a.t2.x4, tau34)
    )

    tau36 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau37 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau38 = (
        einsum("pw,qw->pq", tau36, tau37)
    )

    tau39 = (
        einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, tau38)
    )

    tau40 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau41 = (
        einsum("piw,paw->pia", tau29, tau40)
    )

    tau42 = (
        einsum("ai,pib->pab", a.t1, tau41)
    )

    tau43 = (
        einsum("pw,wia->pia", tau36, h.l.pov)
    )

    tau44 = (
        einsum("ip,pia->pa", a.t2.x3, tau43)
    )

    tau45 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau46 = (
        einsum("pw,wia->pia", tau45, h.l.pov)
    )

    tau47 = (
        einsum("ip,pia->pa", a.t2.x4, tau46)
    )

    tau48 = (
        - einsum("pa->pa", tau44)
        + 2 * einsum("pa->pa", tau47)
    )

    tau49 = (
        einsum("bp,aq,qb->pqa", a.t2.x1, a.t2.x2, tau48)
    )

    tau50 = (
        einsum("q,ap,pqb->pab", a.t2.xlam[0, :], a.t2.x2, tau49)
    )

    tau51 = (
        einsum("p,ip,wia->paw", a.t2.xlam[0, :], a.t2.x3, h.l.pov)
    )

    tau52 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau53 = (
        einsum("paw,pbw->pab", tau51, tau52)
    )

    tau54 = (
        einsum("bp,qab->pqa", a.t2.x1, tau53)
    )

    tau55 = (
        einsum("bq,cq,pqa->pabc", a.t2.x1, a.t2.x2, tau54)
    )

    tau56 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau57 = (
        einsum("ai,wib->abw", a.t1, h.l.pov)
    )

    tau58 = (
        einsum("wab->abw", h.l.pvv)
        - einsum("abw->abw", tau57)
    )

    tau59 = (
        einsum("caw,bcw->ab", tau56, tau58)
    )

    tau60 = (
        einsum("pbac->pabc", tau55)
        - 2 * einsum("cp,ba->pabc", a.t2.x1, tau59)
    )

    tau61 = (
        einsum("cp,pacb->pab", a.t2.x2, tau60)
    )

    tau62 = (
        2 * einsum("pab->pab", tau42)
        + einsum("pab->pab", tau50)
        - einsum("pab->pab", tau61)
    )

    tau63 = (
        einsum("ap,wia->piw", a.t2.x2, h.l.pov)
    )

    tau64 = (
        einsum("ip,wia->paw", a.t2.x3, h.l.pov)
    )

    tau65 = (
        einsum("piw,paw->pia", tau63, tau64)
    )

    tau66 = (
        einsum("ai,pja->pij", a.t1, tau65)
    )

    tau67 = (
        einsum("aj,pij->pia", a.t1, tau66)
    )

    tau68 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau69 = (
        einsum("ai,wja->ijw", a.t1, h.l.pov)
    )

    tau70 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("jiw->ijw", tau69)
    )

    tau71 = (
        einsum("jp,jiw->piw", a.t2.x3, tau70)
    )

    tau72 = (
        einsum("paw,piw->pia", tau68, tau71)
    )

    tau73 = (
        - einsum("pia->pia", tau67)
        + einsum("pia->pia", tau72)
    )

    tau74 = (
        - einsum("ap,pib->piab", a.t2.x1, tau39)
        + einsum("ip,pab->piab", a.t2.x3, tau62)
        + 2 * einsum("bp,pia->piab", a.t2.x1, tau73)
    )

    tau75 = (
        einsum("piab->piab", tau35)
        + einsum("p,piab->piab", a.t2.xlam[0, :], tau74)
    )

    tau76 = (
        einsum("ip,pjab->ijab", a.t2.x4, tau75)
    )

    tau77 = (
        einsum("jp,jiw->piw", a.t2.x4, tau70)
    )

    tau78 = (
        einsum("paw,piw->pia", tau40, tau77)
    )

    tau79 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau80 = (
        einsum("paw,piw->pia", tau52, tau79)
    )

    tau81 = (
        einsum("ai,pib->pab", a.t1, tau80)
    )

    tau82 = (
        einsum("wib,baw->ia", h.l.pov, tau57)
    )

    tau83 = (
        einsum("ip,ia->pa", a.t2.x4, tau82)
    )

    tau84 = (
        einsum("pab->pab", tau81)
        + einsum("ap,pb->pab", a.t2.x1, tau83)
    )

    tau85 = (
        einsum("bi,pab->pia", a.t1, tau84)
    )

    tau86 = (
        einsum("pia->pia", tau78)
        - einsum("pia->pia", tau85)
    )

    tau87 = (
        einsum("p,ap,ip,pjb->ijab", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, tau86)
    )

    tau88 = (
        einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau89 = (
        einsum("aj,jiw->iaw", a.t1, tau70)
    )

    tau90 = (
        einsum("iaw,jbw->ijab", tau88, tau89)
    )

    tau91 = (
        4 * einsum("ijab->ijab", tau26)
        + einsum("jiab->ijab", tau76)
        + 2 * einsum("jiba->ijab", tau87)
        + 4 * einsum("ijba->ijab", tau90)
    )

    tau92 = (
        einsum("aj,ijw->iaw", a.t1, tau69)
    )

    tau93 = (
        einsum("wbj,iaw->ijab", h.l.pvo, tau92)
    )

    tau94 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x1, h.l.pov)
    )

    tau95 = (
        einsum("paw,piw->pia", tau68, tau94)
    )

    tau96 = (
        einsum("ai,pib->pab", a.t1, tau95)
    )

    tau97 = (
        einsum("ip,jp,pab->ijab", a.t2.x3, a.t2.x4, tau96)
    )

    tau98 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x4, tau31)
    )

    tau99 = (
        einsum("q,aq,iq,pq->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau98)
    )

    tau100 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau101 = (
        einsum("pw,wai->pia", tau100, h.l.pvo)
    )

    tau102 = (
        einsum("qw,pw->pq", tau27, tau45)
    )

    tau103 = (
        einsum("q,aq,iq,pq->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau102)
    )

    tau104 = (
        einsum("p,ap,wia->piw", a.t2.xlam[0, :], a.t2.x1, h.l.pov)
    )

    tau105 = (
        einsum("piw,paw->pia", tau104, tau52)
    )

    tau106 = (
        einsum("iq,pia->pqa", a.t2.x3, tau105)
    )

    tau107 = (
        einsum("bq,iq,qpa->piab", a.t2.x2, a.t2.x3, tau106)
    )

    tau108 = (
        einsum("iq,pia->pqa", a.t2.x3, tau31)
    )

    tau109 = (
        einsum("paw,piw->pia", tau52, tau63)
    )

    tau110 = (
        einsum("iq,pia->pqa", a.t2.x3, tau109)
    )

    tau111 = (
        - einsum("ip,pqa->pqia", a.t2.x4, tau108)
        + 2 * einsum("ip,pqa->pqia", a.t2.x3, tau110)
    )

    tau112 = (
        einsum("q,aq,qpib->piab", a.t2.xlam[0, :], a.t2.x1, tau111)
    )

    tau113 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau114 = (
        einsum("pw,wia->pia", tau113, h.l.pov)
    )

    tau115 = (
        einsum("ip,pia->pa", a.t2.x3, tau114)
    )

    tau116 = (
        einsum("pw,wia->pia", tau37, h.l.pov)
    )

    tau117 = (
        einsum("ip,pia->pa", a.t2.x4, tau116)
    )

    tau118 = (
        2 * einsum("pa->pa", tau115)
        - einsum("pa->pa", tau117)
    )

    tau119 = (
        einsum("p,ap,pb->ab", a.t2.xlam[0, :], a.t2.x1, tau118)
    )

    tau120 = (
        einsum("ab->ab", tau119)
        + 2 * einsum("ba->ab", tau59)
    )

    tau121 = (
        einsum("piba->piab", tau107)
        - einsum("piab->piab", tau112)
        - einsum("ip,ab->piab", a.t2.x3, tau120)
    )

    tau122 = (
        einsum("bp,piab->pia", a.t2.x1, tau121)
    )

    tau123 = (
        einsum("paw,piw->pia", tau40, tau71)
    )

    tau124 = (
        einsum("ai,ja->ij", a.t1, tau82)
    )

    tau125 = (
        2 * einsum("ap,bp->pab", a.t2.x1, a.t2.x2)
        - einsum("bp,ap->pab", a.t2.x1, a.t2.x2)
    )

    tau126 = (
        einsum("pab,pbw->paw", tau125, tau52)
    )

    tau127 = (
        einsum("p,wia,paw->pi", a.t2.xlam[0, :], h.l.pov, tau126)
    )

    tau128 = (
        einsum("ip,pj->ij", a.t2.x3, tau127)
    )

    tau129 = (
        2 * einsum("ij->ij", tau124)
        - einsum("ij->ij", tau128)
    )

    tau130 = (
        einsum("ap,jp,ij->pia", a.t2.x1, a.t2.x3, tau129)
    )

    tau131 = (
        einsum("pw,wab->pab", tau100, h.l.pvv)
    )

    tau132 = (
        einsum("pw,wia->pia", tau100, h.l.pov)
    )

    tau133 = (
        einsum("paw,piw->pia", tau64, tau79)
    )

    tau134 = (
        2 * einsum("pia->pia", tau132)
        - einsum("pia->pia", tau133)
    )

    tau135 = (
        einsum("ai,pib->pab", a.t1, tau134)
    )

    tau136 = (
        2 * einsum("pab->pab", tau131)
        - einsum("pab->pab", tau135)
    )

    tau137 = (
        einsum("bi,pab->pia", a.t1, tau136)
    )

    tau138 = (
        einsum("ap,qia->pqi", a.t2.x1, tau114)
    )

    tau139 = (
        einsum("aq,pia->pqi", a.t2.x1, tau43)
    )

    tau140 = (
        2 * einsum("ap,qpi->pqia", a.t2.x1, tau138)
        - einsum("ap,pqi->pqia", a.t2.x2, tau139)
    )

    tau141 = (
        einsum("p,iq,pqia->pqa", a.t2.xlam[0, :], a.t2.x3, tau140)
    )

    tau142 = (
        einsum("iq,qpa->pia", a.t2.x3, tau141)
    )

    tau143 = (
        4 * einsum("pia->pia", tau101)
        - 2 * einsum("pia->pia", tau103)
        + einsum("pia->pia", tau122)
        - 2 * einsum("pia->pia", tau123)
        + einsum("pia->pia", tau130)
        + 2 * einsum("pia->pia", tau137)
        + 2 * einsum("pia->pia", tau142)
    )

    tau144 = (
        einsum("paw,pbw->pab", tau30, tau52)
    )

    tau145 = (
        einsum("bi,pab->pia", a.t1, tau144)
    )

    tau146 = (
        einsum("ai,pja->pij", a.t1, tau145)
    )

    tau147 = (
        einsum("pjw,piw->pij", tau29, tau79)
    )

    tau148 = (
        einsum("iq,jq,pij->pq", a.t2.x3, a.t2.x4, tau147)
    )

    tau149 = (
        einsum("q,iq,jq,qp->pij", a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau148)
    )

    tau150 = (
        2 * einsum("pia->pia", tau31)
        - einsum("pia->pia", tau116)
    )

    tau151 = (
        einsum("ip,qia->pqa", a.t2.x4, tau150)
    )

    tau152 = (
        einsum("ip,wia->paw", a.t2.x4, h.l.pov)
    )

    tau153 = (
        einsum("paw,piw->pia", tau152, tau29)
    )

    tau154 = (
        2 * einsum("pia->pia", tau114)
        - einsum("pia->pia", tau153)
    )

    tau155 = (
        einsum("ip,qia->pqa", a.t2.x4, tau154)
    )

    tau156 = (
        einsum("ip,qpa->pqia", a.t2.x4, tau151)
        + einsum("ip,qpa->pqia", a.t2.x3, tau155)
    )

    tau157 = (
        einsum("ap,pqia->pqi", a.t2.x1, tau156)
    )

    tau158 = (
        einsum("q,jp,qpi->pij", a.t2.xlam[0, :], a.t2.x3, tau157)
    )

    tau159 = (
        2 * einsum("pij->pij", tau146)
        + einsum("pij->pij", tau149)
        - einsum("pji->pij", tau158)
    )

    tau160 = (
        einsum("pw,wai->pia", tau36, h.l.pvo)
    )

    tau161 = (
        einsum("iq,piw->pqw", a.t2.x4, tau29)
    )

    tau162 = (
        - einsum("ip,pqw->pqiw", a.t2.x4, tau161)
        + 2 * einsum("iq,pw->pqiw", a.t2.x4, tau113)
    )

    tau163 = (
        einsum("q,piw,qpiw->pq", a.t2.xlam[0, :], tau79, tau162)
    )

    tau164 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x3, tau163)
    )

    tau165 = (
        einsum("pw,abw->pab", tau36, tau58)
    )

    tau166 = (
        einsum("bi,pab->pia", a.t1, tau165)
    )

    tau167 = (
        2 * einsum("pia->pia", tau160)
        + einsum("pia->pia", tau164)
        + 2 * einsum("pia->pia", tau166)
    )

    tau168 = (
        einsum("qjw,piw->pqij", tau29, tau79)
    )

    tau169 = (
        einsum("ap,wia->piw", a.t2.x1, h.l.pov)
    )

    tau170 = (
        einsum("pjw,qiw->pqij", tau169, tau63)
    )

    tau171 = (
        2 * einsum("pqij->pqij", tau168)
        - einsum("pqij->pqij", tau170)
    )

    tau172 = (
        einsum("q,jp,pqij->pqi", a.t2.xlam[0, :], a.t2.x4, tau171)
    )

    tau173 = (
        einsum("ap,iq,qpi->pqa", a.t2.x1, a.t2.x3, tau172)
    )

    tau174 = (
        einsum("iq,jq,qpa->pija", a.t2.x3, a.t2.x4, tau173)
    )

    tau175 = (
        einsum("ip,pja->pija", a.t2.x3, tau99)
        + einsum("jp,pia->pija", a.t2.x4, tau143)
        + einsum("ap,pij->pija", a.t2.x1, tau159)
        - einsum("jp,pia->pija", a.t2.x3, tau167)
        - einsum("pija->pija", tau174)
    )

    tau176 = (
        einsum("p,ap,pijb->ijab", a.t2.xlam[0, :], a.t2.x2, tau175)
    )

    tau177 = (
        2 * einsum("pia->pia", tau114)
        - einsum("pia->pia", tau109)
    )

    tau178 = (
        einsum("ap,qia->pqi", a.t2.x2, tau177)
    )

    tau179 = (
        einsum("ap,iq,pqi->pqa", a.t2.x1, a.t2.x3, tau178)
    )

    tau180 = (
        einsum("p,q,iq,jq,qpa->pija",
               a.t2.xlam[0, :], a.t2.xlam[0, :], a.t2.x3, a.t2.x4, tau179)
    )

    tau181 = (
        einsum("pw,wai->pia", tau113, h.l.pvo)
    )

    tau182 = (
        einsum("paw,piw->pia", tau68, tau77)
    )

    tau183 = (
        einsum("pw,wab->pab", tau113, h.l.pvv)
    )

    tau184 = (
        einsum("ai,pib->pab", a.t1, tau177)
    )

    tau185 = (
        2 * einsum("pab->pab", tau183)
        - einsum("pab->pab", tau184)
    )

    tau186 = (
        einsum("bi,pab->pia", a.t1, tau185)
    )

    tau187 = (
        2 * einsum("pia->pia", tau181)
        - einsum("pia->pia", tau182)
        + einsum("pia->pia", tau186)
    )

    tau188 = (
        einsum("pw,wai->pia", tau27, h.l.pvo)
    )

    tau189 = (
        - einsum("ap,qpw->pqaw", a.t2.x2, tau161)
        + 2 * einsum("aq,pw->pqaw", a.t2.x2, tau113)
    )

    tau190 = (
        einsum("q,paw,qpaw->pq", a.t2.xlam[0, :], tau30, tau189)
    )

    tau191 = (
        einsum("aq,iq,pq->pia", a.t2.x1, a.t2.x3, tau190)
    )

    tau192 = (
        einsum("pw,abw->pab", tau27, tau58)
    )

    tau193 = (
        einsum("bi,pab->pia", a.t1, tau192)
    )

    tau194 = (
        2 * einsum("pia->pia", tau188)
        + einsum("pia->pia", tau191)
        + 2 * einsum("pia->pia", tau193)
    )

    tau195 = (
        - 2 * einsum("jp,pia->pija", a.t2.x3, tau187)
        + einsum("jp,pia->pija", a.t2.x4, tau194)
    )

    tau196 = (
        einsum("pija->pija", tau180)
        + einsum("p,pija->pija", a.t2.xlam[0, :], tau195)
    )

    tau197 = (
        einsum("ap,pijb->ijab", a.t2.x1, tau196)
    )

    tau198 = (
        4 * einsum("ijab->ijab", tau93)
        + 2 * einsum("ijab->ijab", tau97)
        - einsum("ijba->ijab", tau176)
        + einsum("ijba->ijab", tau197)
    )

    tau199 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau200 = (
        einsum("paw,piw->pia", tau152, tau199)
    )

    tau201 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau202 = (
        einsum("w,wij->ij", tau4, h.l.poo)
    )

    tau203 = (
        einsum("aj,wji->iaw", a.t1, h.l.poo)
    )

    tau204 = (
        einsum("wja,iaw->ij", h.l.pov, tau203)
    )

    tau205 = (
        2 * einsum("ij->ij", tau202)
        - einsum("ji->ij", tau204)
    )

    tau206 = (
        einsum("jp,ji->pi", a.t2.x4, tau205)
    )

    tau207 = (
        einsum("wia,jaw->ij", h.l.pov, tau25)
    )

    tau208 = (
        einsum("w,wij->ij", tau13, h.l.poo)
    )

    tau209 = (
        - einsum("ij->ij", tau207)
        + 2 * einsum("ij->ij", tau208)
    )

    tau210 = (
        einsum("jp,ji->pi", a.t2.x3, tau209)
    )

    tau211 = (
        einsum("ai,pja->pij", a.t1, tau200)
        + einsum("piw,pjw->pij", tau201, tau71)
        - einsum("jp,pi->pij", a.t2.x3, tau206)
        - einsum("ip,pj->pij", a.t2.x4, tau210)
    )

    tau212 = (
        einsum("iq,piw->pqw", a.t2.x3, tau169)
    )

    tau213 = (
        - einsum("ap,qpw->pqaw", a.t2.x1, tau212)
        + 2 * einsum("aq,pw->pqaw", a.t2.x1, tau100)
    )

    tau214 = (
        einsum("p,qaw,pqaw->pq", a.t2.xlam[0, :], tau30, tau213)
    )

    tau215 = (
        - einsum("pjw,piw->pij", tau199, tau79)
        + 2 * einsum("pw,wij->pij", tau45, h.l.poo)
    )

    tau216 = (
        einsum("aq,iq,qp->pia", a.t2.x2, a.t2.x4, tau214)
        - einsum("aj,pji->pia", a.t1, tau215)
    )

    tau217 = (
        einsum("ap,ip,wia->pw", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau218 = (
        einsum("pw,wij->pij", tau217, h.l.poo)
    )

    tau219 = (
        einsum("pw,qw->pq", tau217, tau36)
    )

    tau220 = (
        2 * einsum("aj,pji->pia", a.t1, tau218)
        + einsum("q,aq,iq,pq->pia", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, tau219)
    )

    tau221 = (
        einsum("pjw,piw->pij", tau169, tau201)
    )

    tau222 = (
        einsum("paw,piw->pia", tau152, tau169)
    )

    tau223 = (
        einsum("ap,ip,qia->pq", a.t2.x1, a.t2.x4, tau222)
    )

    tau224 = (
        2 * einsum("aj,pij->pia", a.t1, tau221)
        + einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x2, a.t2.x3, tau223)
    )

    tau225 = (
        2 * einsum("ap,pij->pija", a.t2.x1, tau211)
        + 2 * einsum("ip,pja->pija", a.t2.x4, tau216)
        + einsum("ip,pja->pija", a.t2.x3, tau220)
        + einsum("jp,pia->pija", a.t2.x3, tau224)
    )

    tau226 = (
        einsum("jp,wji->piw", a.t2.x4, h.l.poo)
    )

    tau227 = (
        einsum("piw,paw->pia", tau226, tau30)
    )

    tau228 = (
        einsum("jp,wji->piw", a.t2.x3, h.l.poo)
    )

    tau229 = (
        einsum("jp,ji->pi", a.t2.x3, tau205)
    )

    tau230 = (
        einsum("jp,ji->pi", a.t2.x4, tau209)
    )

    tau231 = (
        einsum("ai,pja->pij", a.t1, tau227)
        + einsum("piw,pjw->pij", tau228, tau77)
        - einsum("jp,pi->pij", a.t2.x4, tau229)
        - einsum("ip,pj->pij", a.t2.x3, tau230)
    )

    tau232 = (
        einsum("p,qaw,pqaw->pq", a.t2.xlam[0, :], tau152, tau189)
    )

    tau233 = (
        einsum("ap,ip,wia->pw", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau234 = (
        - einsum("pjw,piw->pij", tau226, tau63)
        + 2 * einsum("pw,wij->pij", tau233, h.l.poo)
    )

    tau235 = (
        einsum("aq,iq,qp->pia", a.t2.x1, a.t2.x3, tau232)
        - einsum("aj,pji->pia", a.t1, tau234)
    )

    tau236 = (
        einsum("pw,wij->pij", tau37, h.l.poo)
    )

    tau237 = (
        einsum("qw,pw->pq", tau27, tau37)
    )

    tau238 = (
        2 * einsum("aj,pji->pia", a.t1, tau236)
        + einsum("q,aq,iq,pq->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau237)
    )

    tau239 = (
        einsum("piw,pjw->pij", tau228, tau29)
    )

    tau240 = (
        einsum("ap,ip,qia->pq", a.t2.x2, a.t2.x3, tau31)
    )

    tau241 = (
        2 * einsum("aj,pij->pia", a.t1, tau239)
        + einsum("q,aq,iq,qp->pia", a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau240)
    )

    tau242 = (
        2 * einsum("ap,pji->pija", a.t2.x2, tau231)
        + 2 * einsum("jp,pia->pija", a.t2.x3, tau235)
        + einsum("jp,pia->pija", a.t2.x4, tau238)
        + einsum("ip,pja->pija", a.t2.x4, tau241)
    )

    tau243 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("iaw->iaw", tau88)
    )

    tau244 = (
        einsum("wai->iaw", h.l.pvo)
        + einsum("bi,wab->iaw", a.t1, h.l.pvv)
    )

    tau245 = (
        einsum("wij->ijw", h.l.poo)
        + einsum("aj,wia->ijw", a.t1, h.l.pov)
    )

    tau246 = (
        einsum("aj,jiw->iaw", a.t1, tau245)
    )

    tau247 = (
        einsum("pw,wij->pij", tau36, h.l.poo)
    )

    tau248 = (
        einsum("aj,pji->pia", a.t1, tau247)
    )

    tau249 = (
        einsum("bp,wab->paw", a.t2.x1, h.l.pvv)
    )

    tau250 = (
        einsum("pjw,piw->pij", tau169, tau63)
    )

    tau251 = (
        einsum("aj,pij->pia", a.t1, tau250)
    )

    tau252 = (
        einsum("pbw,paw->pab", tau249, tau68)
        + einsum("ai,pib->pab", a.t1, tau251)
    )

    tau253 = (
        2 * einsum("pw,wij->pij", tau113, h.l.poo)
        - einsum("pjw,piw->pij", tau201, tau29)
    )

    tau254 = (
        einsum("aj,pji->pia", a.t1, tau253)
    )

    tau255 = (
        einsum("bp,pia->piab", a.t2.x2, tau248)
        + einsum("ip,pab->piab", a.t2.x4, tau252)
        - einsum("bp,pia->piab", a.t2.x1, tau254)
    )

    tau256 = (
        einsum("pw,wij->pij", tau27, h.l.poo)
    )

    tau257 = (
        einsum("aj,pji->pia", a.t1, tau256)
    )

    tau258 = (
        einsum("bp,wab->paw", a.t2.x2, h.l.pvv)
    )

    tau259 = (
        einsum("aj,pij->pia", a.t1, tau147)
    )

    tau260 = (
        einsum("pbw,paw->pab", tau258, tau40)
        + einsum("ai,pib->pab", a.t1, tau259)
    )

    tau261 = (
        2 * einsum("pw,wij->pij", tau100, h.l.poo)
        - einsum("piw,pjw->pij", tau169, tau228)
    )

    tau262 = (
        einsum("aj,pji->pia", a.t1, tau261)
    )

    tau263 = (
        einsum("bp,pia->piab", a.t2.x1, tau257)
        + einsum("ip,pab->piab", a.t2.x3, tau260)
        - einsum("bp,pia->piab", a.t2.x2, tau262)
    )

    tau264 = (
        einsum("pjw,piw->pij", tau199, tau63)
    )

    tau265 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau8)
    )

    tau266 = (
        einsum("ai,ib->ab", a.t1, tau265)
    )

    tau267 = (
        einsum("ab->ab", h.f.vv)
        + 2 * einsum("ab->ab", tau5)
        - einsum("ab->ab", tau266)
    )

    tau268 = (
        einsum("bp,ab->pa", a.t2.x2, tau267)
    )

    tau269 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau14)
    )

    tau270 = (
        einsum("ai,ja->ij", a.t1, tau269)
    )

    tau271 = (
        einsum("ij->ij", h.f.oo)
        + einsum("ji->ij", tau270)
    )

    tau272 = (
        einsum("jp,ji->pi", a.t2.x3, tau271)
    )

    tau273 = (
        - einsum("aj,pji->pia", a.t1, tau264)
        - einsum("ip,pa->pia", a.t2.x3, tau268)
        + einsum("ap,pi->pia", a.t2.x2, tau272)
    )

    tau274 = (
        einsum("pjw,piw->pij", tau226, tau79)
    )

    tau275 = (
        einsum("bp,ab->pa", a.t2.x1, tau267)
    )

    tau276 = (
        einsum("jp,ji->pi", a.t2.x4, tau271)
    )

    tau277 = (
        - einsum("aj,pji->pia", a.t1, tau274)
        - einsum("ip,pa->pia", a.t2.x4, tau275)
        + einsum("ap,pi->pia", a.t2.x1, tau276)
    )

    rt2 = (
        einsum("p,ap,ip,jp,pb->abij",
               a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau0) / 2
        + einsum("p,ap,jp,ip,pb->abij",
                 a.t2.xlam[0, :], a.t2.x2, a.t2.x3, a.t2.x4, tau1) / 2
        - einsum("p,bp,ap,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x3, tau2) / 2
        - einsum("p,ap,bp,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x4, tau3) / 2
        + einsum("p,ap,ip,jp,pb->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau6)
        + einsum("p,ap,jp,ip,pb->abij",
                 a.t2.xlam[0, :], a.t2.x2, a.t2.x3, a.t2.x4, tau7)
        - einsum("p,ap,ip,jp,pb->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau10)
        - einsum("p,ap,jp,ip,pb->abij",
                 a.t2.xlam[0, :], a.t2.x2, a.t2.x3, a.t2.x4, tau12)
        - einsum("p,ap,bp,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x4, tau16)
        - einsum("p,bp,ap,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x3, tau17)
        - einsum("p,ap,ip,jp,pb->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x3, a.t2.x4, tau19) / 2
        - einsum("p,ap,jp,ip,pb->abij",
                 a.t2.xlam[0, :], a.t2.x2, a.t2.x3, a.t2.x4, tau21) / 2
        - einsum("p,ap,bp,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x4, tau23) / 2
        - einsum("p,bp,ap,jp,pi->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x2, a.t2.x3, tau24) / 2
        - einsum("ijba->abij", tau91) / 4
        - einsum("jiab->abij", tau91) / 4
        - einsum("ijab->abij", tau198) / 4
        - einsum("jiba->abij", tau198) / 4
        + einsum("p,ap,pijb->abij", a.t2.xlam[0, :], a.t2.x2, tau225) / 4
        + einsum("p,ap,pjib->abij", a.t2.xlam[0, :], a.t2.x1, tau242) / 4
        + einsum("iaw,jbw->abij", tau243, tau244)
        + einsum("iaw,jbw->abij", tau246, tau89)
        + einsum("p,jp,piab->abij", a.t2.xlam[0, :], a.t2.x3, tau255) / 2
        + einsum("p,jp,piab->abij", a.t2.xlam[0, :], a.t2.x4, tau263) / 2
        - einsum("p,bp,ip,pja->abij",
                 a.t2.xlam[0, :], a.t2.x1, a.t2.x4, tau273) / 2
        - einsum("p,bp,ip,pja->abij",
                 a.t2.xlam[0, :], a.t2.x2, a.t2.x3, tau277) / 2
    )

    return Tensors(t1=rt1, t2=rt2)
