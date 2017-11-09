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


def _rccsd_cpd_ls_t_true_calculate_energy(h, a):
    tau0 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau1 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau3 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau4 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau5 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau6 = (
        einsum("ai,mib->abm", a.t1, h.l.pov)
    )

    tau7 = (
        2 * einsum("ia->ia", h.f.ov)
        - einsum("bam,mib->ia", tau6, h.l.pov)
    )

    energy = (
        2 * einsum("m,m->", tau0, tau1)
        - einsum("qm,qm->", tau2, tau3)
        + 2 * einsum("qm,qm->", tau4, tau5)
        + einsum("ai,ia->", a.t1, tau7)
    )

    return energy


def _rccsd_cpd_ls_t_true_calc_r1(h, a):
    tau0 = (
        einsum("aj,mji->iam", a.t1, h.l.poo)
    )

    tau1 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau2 = (
        einsum("iq,mia->qam", a.t2.x3, h.l.pov)
    )

    tau3 = (
        einsum("aq,mia->qim", a.t2.x2, h.l.pov)
    )

    tau4 = (
        einsum("ai,qim->qam", a.t1, tau3)
    )

    tau5 = (
        einsum("aq,mia->qim", a.t2.x1, h.l.pov)
    )

    tau6 = (
        einsum("ai,qim->qam", a.t1, tau5)
    )

    tau7 = (
        einsum("ai,mib->abm", a.t1, h.l.pov)
    )

    tau8 = (
        einsum("mab->abm", h.l.pvv)
        - einsum("abm->abm", tau7)
    )

    tau9 = (
        einsum("bq,abm->qam", a.t2.x1, tau8)
    )

    tau10 = (
        einsum("bq,abm->qam", a.t2.x2, tau8)
    )

    tau11 = (
        - einsum("bq,qam->qabm", a.t2.x1, tau4)
        + 2 * einsum("bq,qam->qabm", a.t2.x2, tau6)
        + einsum("aq,qbm->qabm", a.t2.x2, tau9)
        - 2 * einsum("aq,qbm->qabm", a.t2.x1, tau10)
    )

    tau12 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau13 = (
        einsum("m,mia->ia", tau12, h.l.pov)
    )

    tau14 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("ia->ia", tau13)
    )

    tau15 = (
        einsum("iq,ia->qa", a.t2.x3, tau14)
    )

    tau16 = (
        2 * einsum("aq,bq->qab", a.t2.x1, a.t2.x2)
        - einsum("bq,aq->qab", a.t2.x1, a.t2.x2)
    )

    tau17 = (
        - einsum("qbm,qbam->qa", tau2, tau11)
        + einsum("qb,qba->qa", tau15, tau16)
    )

    tau18 = (
        einsum("iq,mia->qam", a.t2.x4, h.l.pov)
    )

    tau19 = (
        2 * einsum("bq,qam->qabm", a.t2.x1, tau4)
        - einsum("bq,qam->qabm", a.t2.x2, tau6)
        + einsum("aq,qbm->qabm", a.t2.x1, tau10)
        - 2 * einsum("aq,qbm->qabm", a.t2.x2, tau9)
    )

    tau20 = (
        einsum("iq,ia->qa", a.t2.x4, tau14)
    )

    tau21 = (
        - einsum("aq,bq->qab", a.t2.x1, a.t2.x2)
        + 2 * einsum("bq,aq->qab", a.t2.x1, a.t2.x2)
    )

    tau22 = (
        - einsum("qbm,qbam->qa", tau18, tau19)
        + einsum("qb,qba->qa", tau20, tau21)
    )

    tau23 = (
        einsum("ai,mib->abm", a.t1, h.l.pov)
    )

    tau24 = (
        einsum("ia->ia", h.f.ov)
        + 2 * einsum("m,mia->ia", tau1, h.l.pov)
        - einsum("bam,mib->ia", tau23, h.l.pov)
    )

    tau25 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau26 = (
        einsum("qm,mia->qia", tau25, h.l.pov)
    )

    tau27 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau28 = (
        einsum("qm,mia->qia", tau27, h.l.pov)
    )

    tau29 = (
        - einsum("iq,qia->qa", a.t2.x3, tau26)
        + 2 * einsum("iq,qia->qa", a.t2.x4, tau28)
    )

    tau30 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau31 = (
        einsum("qm,mia->qia", tau30, h.l.pov)
    )

    tau32 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau33 = (
        einsum("qm,mia->qia", tau32, h.l.pov)
    )

    tau34 = (
        2 * einsum("iq,qia->qa", a.t2.x3, tau31)
        - einsum("iq,qia->qa", a.t2.x4, tau33)
    )

    tau35 = (
        - 2 * einsum("ab->ab", h.f.vv)
        + 2 * einsum("cbm,mac->ab", tau23, h.l.pvv)
        - 4 * einsum("m,mab->ab", tau1, h.l.pvv)
        + 2 * einsum("ai,ib->ab", a.t1, tau24)
        + einsum("aq,qb->ab", a.t2.x2, tau29)
        + einsum("aq,qb->ab", a.t2.x1, tau34)
    )

    tau36 = (
        einsum("aj,mji->iam", a.t1, h.l.poo)
    )

    tau37 = (
        einsum("ij->ij", h.f.oo)
        + 2 * einsum("m,mij->ij", tau1, h.l.poo)
        - einsum("jam,mia->ij", tau36, h.l.pov)
    )

    tau38 = (
        einsum("aq,mia->qim", a.t2.x1, h.l.pov)
    )

    tau39 = (
        einsum("jq,mji->qim", a.t2.x4, h.l.poo)
    )

    tau40 = (
        einsum("jq,mji->qim", a.t2.x3, h.l.poo)
    )

    tau41 = (
        2 * einsum("iq,qjm->qijm", a.t2.x3, tau39)
        - einsum("iq,qjm->qijm", a.t2.x4, tau40)
    )

    tau42 = (
        einsum("qjm,qjim->qi", tau38, tau41)
    )

    tau43 = (
        einsum("aq,mia->qim", a.t2.x2, h.l.pov)
    )

    tau44 = (
        - einsum("iq,qjm->qijm", a.t2.x3, tau39)
        + 2 * einsum("iq,qjm->qijm", a.t2.x4, tau40)
    )

    tau45 = (
        einsum("qjm,qjim->qi", tau43, tau44)
    )

    r1 = (
        - einsum("ibm,mab->ai", tau0, h.l.pvv)
        + 2 * einsum("m,mai->ai", tau1, h.l.pvo)
        + einsum("ia->ai", h.f.ov.conj())
        + einsum("iq,qa->ai", a.t2.x4, tau17) / 2
        + einsum("iq,qa->ai", a.t2.x3, tau22) / 2
        - einsum("bi,ab->ai", a.t1, tau35) / 2
        - einsum("aj,ji->ai", a.t1, tau37)
        - einsum("aq,qi->ai", a.t2.x2, tau42) / 2
        - einsum("aq,qi->ai", a.t2.x1, tau45) / 2
    )

    return r1


def _rccsd_cpd_ls_t_true_calc_r21(h, a, b, d):
    tau0 = (
        einsum("iw,ai,iq->wqa", d.t2.d4, a.t1, b.t2.x4)
    )

    tau1 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d2, b.t2.x2, tau0, h.l.pvv)
    )

    tau2 = (
        einsum("iw,ai,iq->wqa", d.t2.d3, a.t1, b.t2.x3)
    )

    tau3 = (
        einsum("wqm,wqa,mia->wqi", tau1, tau2, h.l.pov)
    )

    tau4 = (
        einsum("ai,wqi->wqa", a.t1, tau3)
    )

    tau5 = (
        einsum("aw,ai,aq->wqi", d.t2.d2, a.t1, b.t2.x2)
    )

    tau6 = (
        einsum("wqa,wqi,mia->wqm", tau0, tau5, h.l.pov)
    )

    tau7 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau6, h.l.pvv)
    )

    tau8 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x1, b.t2.x2)
    )

    tau9 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x3, b.t2.x4)
    )

    tau10 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau11 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau10)
    )

    tau12 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau13 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau12, tau8, tau9, tau11)
    )

    tau14 = (
        einsum("ai,wqi->wqa", a.t1, tau13)
    )

    tau15 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x4, b.t2.x3)
    )

    tau16 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau17 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau16)
    )

    tau18 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau19 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau15, tau18, tau8, tau17)
    )

    tau20 = (
        einsum("ai,wqi->wqa", a.t1, tau19)
    )

    tau21 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x4, b.t2.x4)
    )

    tau22 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau23 = (
        einsum("rm,wrq,wrq->wqm", tau22, tau21, tau8)
    )

    tau24 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau23, h.l.poo)
    )

    tau25 = (
        einsum("ai,wqi->wqa", a.t1, tau24)
    )

    tau26 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x2, b.t2.x2)
    )

    tau27 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x3, b.t2.x3)
    )

    tau28 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau29 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau28)
    )

    tau30 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau31 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau26, tau27, tau30, tau29)
    )

    tau32 = (
        einsum("ai,wqi->wqa", a.t1, tau31)
    )

    tau33 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau34 = (
        einsum("rm,wrq,wrq->wqm", tau33, tau26, tau9)
    )

    tau35 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau34, h.l.poo)
    )

    tau36 = (
        einsum("ai,wqi->wqa", a.t1, tau35)
    )

    tau37 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau38 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau37)
    )

    tau39 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau40 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau21, tau26, tau39, tau38)
    )

    tau41 = (
        einsum("ai,wqi->wqa", a.t1, tau40)
    )

    tau42 = (
        einsum("qmi,qmj->qij", tau18, tau28)
    )

    tau43 = (
        einsum("iw,ir,qji,wrj->wqr", d.t2.d4, b.t2.x4, tau42, tau5)
    )

    tau44 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau43)
    )

    tau45 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d4, b.t2.x4, tau5, h.l.poo)
    )

    tau46 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau47 = (
        einsum("qm,wrm->wqr", tau46, tau45)
    )

    tau48 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau47)
    )

    tau49 = (
        einsum("qmj,qmi->qij", tau12, tau37)
    )

    tau50 = (
        einsum("iw,ir,qij,wrj->wqr", d.t2.d3, b.t2.x3, tau49, tau5)
    )

    tau51 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau50)
    )

    tau52 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau53 = (
        einsum("qm,wrm->wqr", tau52, tau45)
    )

    tau54 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau53)
    )

    tau55 = (
        einsum("qmi,qmj->qij", tau10, tau39)
    )

    tau56 = (
        einsum("iw,ir,wrj,qij->wqr", d.t2.d3, b.t2.x3, tau5, tau55)
    )

    tau57 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau56, tau9)
    )

    tau58 = (
        einsum("qmj,qmi->qij", tau16, tau30)
    )

    tau59 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau5, tau58)
    )

    tau60 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau59)
    )

    tau61 = (
        einsum("wqa,mia->wqmi", tau2, h.l.pov)
    )

    tau62 = (
        einsum("jw,jq,mij->wqmi", d.t2.d4, b.t2.x4, h.l.poo)
    )

    tau63 = (
        einsum("wqmi,wqmj->wqij", tau61, tau62)
    )

    tau64 = (
        einsum("iq,jq,wrij->wqr", a.t2.x3, a.t2.x4, tau63)
    )

    tau65 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau26, tau64)
    )

    tau66 = (
        einsum("iq,jq,wrji->wqr", a.t2.x3, a.t2.x4, tau63)
    )

    tau67 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau66, tau8)
    )

    tau68 = (
        einsum("jw,jq,mij->wqmi", d.t2.d3, b.t2.x3, h.l.poo)
    )

    tau69 = (
        einsum("wqa,mia->wqmi", tau0, h.l.pov)
    )

    tau70 = (
        einsum("wqmi,wqmj->wqij", tau68, tau69)
    )

    tau71 = (
        einsum("iq,jq,wrij->wqr", a.t2.x3, a.t2.x4, tau70)
    )

    tau72 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau26, tau71)
    )

    tau73 = (
        einsum("iq,jq,wrji->wqr", a.t2.x3, a.t2.x4, tau70)
    )

    tau74 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau73, tau8)
    )

    tau75 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau76 = (
        einsum("mki,mkj->ij", tau75, h.l.poo)
    )

    tau77 = (
        einsum("jq,ji->qi", a.t2.x4, tau76)
    )

    tau78 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau77)
    )

    tau79 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau78)
    )

    tau80 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau81 = (
        einsum("mkj,mki->ij", tau80, h.l.poo)
    )

    tau82 = (
        einsum("jq,ij->qi", a.t2.x3, tau81)
    )

    tau83 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau82)
    )

    tau84 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau83)
    )

    tau85 = (
        einsum("jq,ij->qi", a.t2.x4, tau81)
    )

    tau86 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau85)
    )

    tau87 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau8, tau86, tau9)
    )

    tau88 = (
        einsum("jq,ji->qi", a.t2.x3, tau76)
    )

    tau89 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau88)
    )

    tau90 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau8, tau89)
    )

    tau91 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau92 = (
        einsum("rm,wrq,wrq->wqm", tau91, tau8, tau9)
    )

    tau93 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau92, h.l.poo)
    )

    tau94 = (
        einsum("ai,wqi->wqa", a.t1, tau93)
    )

    tau95 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau96 = (
        einsum("rm,wrq,wrq->wqm", tau95, tau21, tau26)
    )

    tau97 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau96, h.l.poo)
    )

    tau98 = (
        einsum("ai,wqi->wqa", a.t1, tau97)
    )

    tau99 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau100 = (
        einsum("qm,wrm->wqr", tau99, tau45)
    )

    tau101 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau100, tau27)
    )

    tau102 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau103 = (
        einsum("qm,wrm->wqr", tau102, tau45)
    )

    tau104 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau103, tau15)
    )

    tau105 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau106 = (
        einsum("m,mij->ij", tau105, h.l.poo)
    )

    tau107 = (
        einsum("jq,ji->qi", a.t2.x4, tau106)
    )

    tau108 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau107)
    )

    tau109 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau108, tau26, tau27)
    )

    tau110 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau111 = (
        einsum("m,mij->ij", tau110, h.l.poo)
    )

    tau112 = (
        einsum("jq,ji->qi", a.t2.x3, tau111)
    )

    tau113 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau112)
    )

    tau114 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau113, tau21, tau26)
    )

    tau115 = (
        einsum("jq,ji->qi", a.t2.x4, tau111)
    )

    tau116 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau115)
    )

    tau117 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau116, tau8, tau9)
    )

    tau118 = (
        einsum("jq,ji->qi", a.t2.x3, tau106)
    )

    tau119 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau118)
    )

    tau120 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau119, tau15, tau8)
    )

    tau121 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, h.l.pvo)
    )

    tau122 = (
        einsum("wqm,wqb,mab->wqa", tau121, tau2, h.l.pvv)
    )

    tau123 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d2, b.t2.x2, tau0, h.l.pvv)
    )

    tau124 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau123, h.l.pvo)
    )

    tau125 = (
        einsum("qmj,qmi->qij", tau28, tau37)
    )

    tau126 = (
        einsum("iw,jw,ir,jr,qij->wqr", d.t2.d3,
               d.t2.d4, b.t2.x3, b.t2.x4, tau125)
    )

    tau127 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau126, tau26)
    )

    tau128 = (
        einsum("qmi,qmj->qij", tau10, tau16)
    )

    tau129 = (
        einsum("iw,jw,ir,jr,qij->wqr", d.t2.d3,
               d.t2.d4, b.t2.x3, b.t2.x4, tau128)
    )

    tau130 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau129, tau8)
    )

    tau131 = (
        einsum("wqm,wqb,mab->wqa", tau123, tau2, h.l.pvv)
    )

    tau132 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau121, h.l.pvo)
    )

    tau133 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau6, h.l.pov)
    )

    tau134 = (
        einsum("ai,wqi->wqa", a.t1, tau133)
    )

    tau135 = (
        einsum("bq,ab->qa", a.t2.x2, h.f.vv)
    )

    tau136 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau135)
    )

    tau137 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau136, tau21, tau27)
    )

    tau138 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau135, tau15, tau8, tau9)
    )

    tau139 = (
        einsum("bq,ab->qa", a.t2.x1, h.f.vv)
    )

    tau140 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau139)
    )

    tau141 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau140, tau15, tau9)
    )

    tau142 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau139, tau21, tau26, tau27)
    )

    tau143 = (
        einsum("rm,qm->qr", tau91, tau99)
    )

    tau144 = (
        einsum("qp,wpr,wpr->wqr", tau143, tau8, tau9)
    )

    tau145 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau144, tau27)
    )

    tau146 = (
        einsum("qm,rm->qr", tau95, tau99)
    )

    tau147 = (
        einsum("pq,wpr,wpr->wqr", tau146, tau21, tau26)
    )

    tau148 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau147, tau27)
    )

    tau149 = (
        einsum("qm,rm->qr", tau102, tau91)
    )

    tau150 = (
        einsum("qp,wpr,wpr->wqr", tau149, tau8, tau9)
    )

    tau151 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau150)
    )

    tau152 = (
        einsum("qm,rm->qr", tau102, tau95)
    )

    tau153 = (
        einsum("qp,wpr,wpr->wqr", tau152, tau21, tau26)
    )

    tau154 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau153)
    )

    tau155 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau156 = (
        einsum("qma,qmi->qia", tau155, tau18)
    )

    tau157 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau156)
    )

    tau158 = (
        einsum("pq,wpr,wpr->wqr", tau157, tau8, tau9)
    )

    tau159 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau158, tau27)
    )

    tau160 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau91, h.l.pov)
    )

    tau161 = (
        einsum("ar,qa->qr", a.t2.x2, tau160)
    )

    tau162 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau161, tau15, tau8, tau9)
    )

    tau163 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau162)
    )

    tau164 = (
        einsum("qm,rm->qr", tau22, tau99)
    )

    tau165 = (
        einsum("pq,wpr,wpr->wqr", tau164, tau21, tau8)
    )

    tau166 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau165, tau27)
    )

    tau167 = (
        einsum("pq,wpr->wqr", tau161, tau8)
    )

    tau168 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau167, tau21, tau27)
    )

    tau169 = (
        einsum("qm,rm->qr", tau46, tau91)
    )

    tau170 = (
        einsum("qp,wpr,wpr->wqr", tau169, tau8, tau9)
    )

    tau171 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau170)
    )

    tau172 = (
        einsum("aq,ra->qr", a.t2.x1, tau160)
    )

    tau173 = (
        einsum("pq,wpr,wpr,wpr->wqr", tau172, tau21, tau26, tau27)
    )

    tau174 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau173)
    )

    tau175 = (
        einsum("qm,rm->qr", tau33, tau99)
    )

    tau176 = (
        einsum("pq,wpr,wpr->wqr", tau175, tau26, tau9)
    )

    tau177 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau176, tau27)
    )

    tau178 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau179 = (
        einsum("qmi,qma->qia", tau12, tau178)
    )

    tau180 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau179)
    )

    tau181 = (
        einsum("pq,wpr,wpr->wqr", tau180, tau21, tau26)
    )

    tau182 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau181, tau27)
    )

    tau183 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau102, h.l.pov)
    )

    tau184 = (
        einsum("ar,qa->qr", a.t2.x2, tau183)
    )

    tau185 = (
        einsum("pq,wpr->wqr", tau184, tau26)
    )

    tau186 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau185, tau21, tau27)
    )

    tau187 = (
        einsum("rm,qm->qr", tau46, tau95)
    )

    tau188 = (
        einsum("pq,wpr,wpr->wqr", tau187, tau21, tau26)
    )

    tau189 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau188)
    )

    tau190 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau91, h.l.pov)
    )

    tau191 = (
        einsum("ir,qi->qr", a.t2.x3, tau190)
    )

    tau192 = (
        einsum("pq,wpr->wqr", tau191, tau27)
    )

    tau193 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau192, tau21, tau26)
    )

    tau194 = (
        einsum("ir,qi->qr", a.t2.x4, tau190)
    )

    tau195 = (
        einsum("pq,wpr->wqr", tau194, tau9)
    )

    tau196 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau195, tau26, tau27)
    )

    tau197 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau95, h.l.pov)
    )

    tau198 = (
        einsum("ir,qi->qr", a.t2.x4, tau197)
    )

    tau199 = (
        einsum("pq,wpr->wqr", tau198, tau21)
    )

    tau200 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau199, tau26, tau27)
    )

    tau201 = (
        einsum("ir,qi->qr", a.t2.x3, tau197)
    )

    tau202 = (
        einsum("pq,wpr->wqr", tau201, tau15)
    )

    tau203 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau202, tau21, tau26)
    )

    tau204 = (
        einsum("qm,rm->qr", tau52, tau91)
    )

    tau205 = (
        einsum("qp,wpr,wpr->wqr", tau204, tau8, tau9)
    )

    tau206 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau205, tau27)
    )

    tau207 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau184, tau15, tau8, tau9)
    )

    tau208 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau207)
    )

    tau209 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau156)
    )

    tau210 = (
        einsum("qp,wpr,wpr->wqr", tau209, tau8, tau9)
    )

    tau211 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau210)
    )

    tau212 = (
        einsum("qp,wpr->wqr", tau172, tau8)
    )

    tau213 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau212, tau9)
    )

    tau214 = (
        einsum("qm,rm->qr", tau102, tau22)
    )

    tau215 = (
        einsum("qp,wpr,wpr->wqr", tau214, tau21, tau8)
    )

    tau216 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau215)
    )

    tau217 = (
        einsum("pq,wpr->wqr", tau194, tau27)
    )

    tau218 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau217, tau8, tau9)
    )

    tau219 = (
        einsum("pq,wpr->wqr", tau191, tau9)
    )

    tau220 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau219, tau8)
    )

    tau221 = (
        einsum("pq,wpr->wqr", tau198, tau15)
    )

    tau222 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau221, tau8, tau9)
    )

    tau223 = (
        einsum("pq,wpr->wqr", tau201, tau21)
    )

    tau224 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau223, tau8)
    )

    tau225 = (
        einsum("ar,qa->qr", a.t2.x1, tau183)
    )

    tau226 = (
        einsum("pq,wpr->wqr", tau225, tau26)
    )

    tau227 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau226, tau9)
    )

    tau228 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau229 = (
        einsum("qma,qmi->qia", tau228, tau30)
    )

    tau230 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x3, tau229)
    )

    tau231 = (
        einsum("pq,wpr,wpr->wqr", tau230, tau21, tau26)
    )

    tau232 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau231)
    )

    tau233 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau225, tau21, tau26, tau27)
    )

    tau234 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau233)
    )

    tau235 = (
        einsum("rm,qm->qr", tau52, tau95)
    )

    tau236 = (
        einsum("pq,wpr,wpr->wqr", tau235, tau21, tau26)
    )

    tau237 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau236, tau27)
    )

    tau238 = (
        einsum("qm,rm->qr", tau102, tau33)
    )

    tau239 = (
        einsum("qp,wpr,wpr->wqr", tau238, tau26, tau9)
    )

    tau240 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau239)
    )

    tau241 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau179)
    )

    tau242 = (
        einsum("pq,wpr,wpr->wqr", tau241, tau21, tau8)
    )

    tau243 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau242, tau27)
    )

    tau244 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau46, h.l.pov)
    )

    tau245 = (
        einsum("ar,qa->qr", a.t2.x2, tau244)
    )

    tau246 = (
        einsum("pq,wpr->wqr", tau245, tau8)
    )

    tau247 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau246, tau27)
    )

    tau248 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau156)
    )

    tau249 = (
        einsum("qp,wpr,wpr->wqr", tau248, tau8, tau9)
    )

    tau250 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau249)
    )

    tau251 = (
        einsum("qmi,qma->qia", tau18, tau228)
    )

    tau252 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau251)
    )

    tau253 = (
        einsum("pq,wpr,wpr->wqr", tau252, tau15, tau8)
    )

    tau254 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau253)
    )

    tau255 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau245, tau15, tau8, tau9)
    )

    tau256 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau255)
    )

    tau257 = (
        einsum("rm,qm->qr", tau22, tau46)
    )

    tau258 = (
        einsum("qp,wpr,wpr->wqr", tau257, tau21, tau8)
    )

    tau259 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau258)
    )

    tau260 = (
        einsum("aq,ra->qr", a.t2.x1, tau244)
    )

    tau261 = (
        einsum("pq,wpr,wpr,wpr->wqr", tau260, tau21, tau26, tau27)
    )

    tau262 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau261)
    )

    tau263 = (
        einsum("qma,qmi->qia", tau155, tau30)
    )

    tau264 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau263)
    )

    tau265 = (
        einsum("pq,wpr,wpr->wqr", tau264, tau26, tau27)
    )

    tau266 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau265)
    )

    tau267 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau263)
    )

    tau268 = (
        einsum("pq,wpr,wpr->wqr", tau267, tau26, tau9)
    )

    tau269 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau268, tau27)
    )

    tau270 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau33, h.l.pov)
    )

    tau271 = (
        einsum("ar,qa->qr", a.t2.x2, tau270)
    )

    tau272 = (
        einsum("pq,wpr->wqr", tau271, tau26)
    )

    tau273 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau272)
    )

    tau274 = (
        einsum("qm,rm->qr", tau33, tau46)
    )

    tau275 = (
        einsum("pq,wpr,wpr->wqr", tau274, tau26, tau9)
    )

    tau276 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau275)
    )

    tau277 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau229)
    )

    tau278 = (
        einsum("pq,wpr,wpr->wqr", tau277, tau21, tau26)
    )

    tau279 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau278)
    )

    tau280 = (
        einsum("qmj,qmi->qij", tau12, tau30)
    )

    tau281 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau280)
    )

    tau282 = (
        einsum("pq,wpr,wpr->wqr", tau281, tau21, tau27)
    )

    tau283 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau26, tau282)
    )

    tau284 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau33, h.l.pov)
    )

    tau285 = (
        einsum("ir,qi->qr", a.t2.x3, tau284)
    )

    tau286 = (
        einsum("pq,wpr->wqr", tau285, tau27)
    )

    tau287 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau286)
    )

    tau288 = (
        einsum("ir,qi->qr", a.t2.x4, tau284)
    )

    tau289 = (
        einsum("pq,wpr->wqr", tau288, tau9)
    )

    tau290 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau289)
    )

    tau291 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau22, h.l.pov)
    )

    tau292 = (
        einsum("ir,qi->qr", a.t2.x4, tau291)
    )

    tau293 = (
        einsum("pq,wpr->wqr", tau292, tau21)
    )

    tau294 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau293)
    )

    tau295 = (
        einsum("qmi,qmj->qij", tau18, tau39)
    )

    tau296 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau295)
    )

    tau297 = (
        einsum("pq,wpr,wpr->wqr", tau296, tau15, tau9)
    )

    tau298 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau26, tau297)
    )

    tau299 = (
        einsum("ir,qi->qr", a.t2.x3, tau291)
    )

    tau300 = (
        einsum("pq,wpr->wqr", tau299, tau15)
    )

    tau301 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau300)
    )

    tau302 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x4, tau156)
    )

    tau303 = (
        einsum("qp,wpr,wpr->wqr", tau302, tau8, tau9)
    )

    tau304 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau303)
    )

    tau305 = (
        einsum("rm,qm->qr", tau22, tau52)
    )

    tau306 = (
        einsum("qp,wpr,wpr->wqr", tau305, tau21, tau8)
    )

    tau307 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau306)
    )

    tau308 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau271, tau15, tau8, tau9)
    )

    tau309 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau308)
    )

    tau310 = (
        einsum("qma,qmi->qia", tau178, tau39)
    )

    tau311 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau310)
    )

    tau312 = (
        einsum("qp,wpr,wpr->wqr", tau311, tau15, tau8)
    )

    tau313 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau312, tau9)
    )

    tau314 = (
        einsum("qp,wpr->wqr", tau260, tau8)
    )

    tau315 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau314, tau9)
    )

    tau316 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau251)
    )

    tau317 = (
        einsum("qp,wpr,wpr->wqr", tau316, tau21, tau8)
    )

    tau318 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau317)
    )

    tau319 = (
        einsum("pq,wpr->wqr", tau288, tau27)
    )

    tau320 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau319, tau8, tau9)
    )

    tau321 = (
        einsum("pq,wpr,wpr->wqr", tau296, tau21, tau27)
    )

    tau322 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau321, tau8)
    )

    tau323 = (
        einsum("pq,wpr,wpr->wqr", tau281, tau15, tau9)
    )

    tau324 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau323, tau8)
    )

    tau325 = (
        einsum("pq,wpr->wqr", tau285, tau9)
    )

    tau326 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau325, tau8)
    )

    tau327 = (
        einsum("pq,wpr->wqr", tau292, tau15)
    )

    tau328 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau327, tau8, tau9)
    )

    tau329 = (
        einsum("pq,wpr->wqr", tau299, tau21)
    )

    tau330 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau329, tau8)
    )

    tau331 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x4, tau263)
    )

    tau332 = (
        einsum("pq,wpr,wpr->wqr", tau331, tau26, tau27)
    )

    tau333 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau332, tau9)
    )

    tau334 = (
        einsum("ar,qa->qr", a.t2.x1, tau270)
    )

    tau335 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau334, tau21, tau26, tau27)
    )

    tau336 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau335)
    )

    tau337 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau310)
    )

    tau338 = (
        einsum("pq,wpr,wpr->wqr", tau337, tau21, tau26)
    )

    tau339 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau338)
    )

    tau340 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau263)
    )

    tau341 = (
        einsum("qp,wpr,wpr->wqr", tau340, tau26, tau9)
    )

    tau342 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau341)
    )

    tau343 = (
        einsum("rm,qm->qr", tau33, tau52)
    )

    tau344 = (
        einsum("qp,wpr,wpr->wqr", tau343, tau26, tau9)
    )

    tau345 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau344)
    )

    tau346 = (
        einsum("pq,wpr->wqr", tau334, tau26)
    )

    tau347 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau346, tau9)
    )

    tau348 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau92, h.l.pvv)
    )

    tau349 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau96, h.l.pvv)
    )

    tau350 = (
        einsum("qm,wrm->wqr", tau91, tau1)
    )

    tau351 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau350)
    )

    tau352 = (
        einsum("qm,wrm->wqr", tau95, tau1)
    )

    tau353 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau352)
    )

    tau354 = (
        einsum("m,mab->ab", tau110, h.l.pvv)
    )

    tau355 = (
        einsum("bq,ab->qa", a.t2.x2, tau354)
    )

    tau356 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau355)
    )

    tau357 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau356)
    )

    tau358 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau355, tau15, tau8, tau9)
    )

    tau359 = (
        einsum("bq,ab->qa", a.t2.x1, tau354)
    )

    tau360 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau359)
    )

    tau361 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau360, tau9)
    )

    tau362 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau359, tau21, tau26, tau27)
    )

    tau363 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau364 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau363)
    )

    tau365 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau21, tau27, tau39, tau364)
    )

    tau366 = (
        einsum("ai,wqi->wqa", a.t1, tau365)
    )

    tau367 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau368 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau367)
    )

    tau369 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau12, tau15, tau9, tau368)
    )

    tau370 = (
        einsum("ai,wqi->wqa", a.t1, tau369)
    )

    tau371 = (
        einsum("qmi,wri->wqrm", tau12, tau5)
    )

    tau372 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau21, tau27, tau367, tau371)
    )

    tau373 = (
        einsum("qmi,wri->wqrm", tau39, tau5)
    )

    tau374 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau15, tau363, tau9, tau373)
    )

    tau375 = (
        einsum("qmb,qma->qab", tau228, tau363)
    )

    tau376 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau2, tau375)
    )

    tau377 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau376)
    )

    tau378 = (
        einsum("wqa,mia->wqmi", tau2, h.l.pov)
    )

    tau379 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau378)
    )

    tau380 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau363, tau8, tau9, tau379)
    )

    tau381 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau23, h.l.pvv)
    )

    tau382 = (
        einsum("qmb,qma->qab", tau155, tau367)
    )

    tau383 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau2, tau382)
    )

    tau384 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau383, tau9)
    )

    tau385 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau34, h.l.pvv)
    )

    tau386 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau378)
    )

    tau387 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau21, tau26, tau367, tau386)
    )

    tau388 = (
        einsum("qmb,qma->qab", tau155, tau363)
    )

    tau389 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau0, tau388)
    )

    tau390 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau389)
    )

    tau391 = (
        einsum("qm,wrm->wqr", tau22, tau1)
    )

    tau392 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau391)
    )

    tau393 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau69)
    )

    tau394 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau15, tau363, tau8, tau393)
    )

    tau395 = (
        einsum("qm,wrm->wqr", tau33, tau1)
    )

    tau396 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau395)
    )

    tau397 = (
        einsum("qmb,qma->qab", tau228, tau367)
    )

    tau398 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau0, tau397)
    )

    tau399 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau398)
    )

    tau400 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau69)
    )

    tau401 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau26, tau27, tau367, tau400)
    )

    tau402 = (
        einsum("bi,mab->mia", a.t1, h.l.pvv)
    )

    tau403 = (
        einsum("mia,mib->ab", tau402, h.l.pov)
    )

    tau404 = (
        einsum("bq,ab->qa", a.t2.x2, tau403)
    )

    tau405 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau404)
    )

    tau406 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau405)
    )

    tau407 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau404, tau15, tau8, tau9)
    )

    tau408 = (
        einsum("bq,ab->qa", a.t2.x1, tau403)
    )

    tau409 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau408)
    )

    tau410 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau409, tau9)
    )

    tau411 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau408, tau21, tau26, tau27)
    )

    tau412 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau45, h.l.pov)
    )

    tau413 = (
        einsum("ai,wqi->wqa", a.t1, tau412)
    )

    tau414 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau6, h.l.poo)
    )

    tau415 = (
        einsum("ai,wqi->wqa", a.t1, tau414)
    )

    tau416 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, h.l.pvo)
    )

    tau417 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau416, h.l.pov)
    )

    tau418 = (
        einsum("ai,wqi->wqa", a.t1, tau417)
    )

    tau419 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau1, h.l.poo)
    )

    tau420 = (
        einsum("ai,wqi->wqa", a.t1, tau419)
    )

    tau421 = (
        einsum("wqb,wqm,mab->wqa", tau2, tau45, h.l.pvv)
    )

    tau422 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau6, h.l.pvo)
    )

    tau423 = (
        einsum("jq,ji->qi", a.t2.x4, h.f.oo)
    )

    tau424 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau423)
    )

    tau425 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau424)
    )

    tau426 = (
        einsum("jq,ji->qi", a.t2.x3, h.f.oo)
    )

    tau427 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau426)
    )

    tau428 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau427)
    )

    tau429 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau423)
    )

    tau430 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau429, tau8, tau9)
    )

    tau431 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau426)
    )

    tau432 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau431, tau8)
    )

    tau433 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau21, tau27, tau30, tau371)
    )

    tau434 = (
        einsum("ai,wqi->wqa", a.t1, tau433)
    )

    tau435 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau15, tau18, tau9, tau373)
    )

    tau436 = (
        einsum("ai,wqi->wqa", a.t1, tau435)
    )

    tau437 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau18, tau8, tau9, tau379)
    )

    tau438 = (
        einsum("ai,wqi->wqa", a.t1, tau437)
    )

    tau439 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau23, h.l.pov)
    )

    tau440 = (
        einsum("ai,wqi->wqa", a.t1, tau439)
    )

    tau441 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau34, h.l.pov)
    )

    tau442 = (
        einsum("ai,wqi->wqa", a.t1, tau441)
    )

    tau443 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau21, tau26, tau30, tau386)
    )

    tau444 = (
        einsum("ai,wqi->wqa", a.t1, tau443)
    )

    tau445 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau15, tau18, tau8, tau393)
    )

    tau446 = (
        einsum("ai,wqi->wqa", a.t1, tau445)
    )

    tau447 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau26, tau27, tau30, tau400)
    )

    tau448 = (
        einsum("ai,wqi->wqa", a.t1, tau447)
    )

    tau449 = (
        einsum("mji,mja->ia", tau75, h.l.pov)
    )

    tau450 = (
        einsum("aq,ia->qi", a.t2.x2, tau449)
    )

    tau451 = (
        einsum("ai,qi->qa", a.t1, tau450)
    )

    tau452 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau451, tau15, tau8, tau9)
    )

    tau453 = (
        einsum("aq,ia->qi", a.t2.x1, tau449)
    )

    tau454 = (
        einsum("ai,qi->qa", a.t1, tau453)
    )

    tau455 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau454, tau21, tau26, tau27)
    )

    tau456 = (
        einsum("wra,qia,wri->wqr", tau2, tau251, tau5)
    )

    tau457 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau456)
    )

    tau458 = (
        einsum("wra,qia,wri->wqr", tau2, tau263, tau5)
    )

    tau459 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau458, tau9)
    )

    tau460 = (
        einsum("wra,qia,wri->wqr", tau0, tau156, tau5)
    )

    tau461 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau460)
    )

    tau462 = (
        einsum("wqa,wqi,mia->wqm", tau0, tau5, h.l.pov)
    )

    tau463 = (
        einsum("qm,wrm->wqr", tau22, tau462)
    )

    tau464 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau463)
    )

    tau465 = (
        einsum("qm,wrm->wqr", tau33, tau462)
    )

    tau466 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau465)
    )

    tau467 = (
        einsum("wra,qia,wri->wqr", tau0, tau229, tau5)
    )

    tau468 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau467)
    )

    tau469 = (
        einsum("qi,wri->wqr", tau450, tau5)
    )

    tau470 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau469)
    )

    tau471 = (
        einsum("qi,wri->wqr", tau453, tau5)
    )

    tau472 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau471, tau9)
    )

    tau473 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau474 = (
        einsum("qmb,qma->qab", tau155, tau473)
    )

    tau475 = (
        einsum("wrb,wra,qab->wqr", tau0, tau2, tau474)
    )

    tau476 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau26, tau475)
    )

    tau477 = (
        einsum("mji,mja->ia", tau80, h.l.pov)
    )

    tau478 = (
        einsum("ai,ja->ij", a.t1, tau477)
    )

    tau479 = (
        einsum("jq,ij->qi", a.t2.x3, tau478)
    )

    tau480 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau479)
    )

    tau481 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau480)
    )

    tau482 = (
        einsum("jq,ij->qi", a.t2.x4, tau478)
    )

    tau483 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau482)
    )

    tau484 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau483, tau8, tau9)
    )

    tau485 = (
        einsum("wra,wrb,qab->wqr", tau0, tau2, tau474)
    )

    tau486 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau485, tau8)
    )

    tau487 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau482)
    )

    tau488 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau487)
    )

    tau489 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau479)
    )

    tau490 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau489, tau8)
    )

    tau491 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau92, h.l.pov)
    )

    tau492 = (
        einsum("ai,wqi->wqa", a.t1, tau491)
    )

    tau493 = (
        einsum("wqa,wqm,mia->wqi", tau2, tau96, h.l.pov)
    )

    tau494 = (
        einsum("ai,wqi->wqa", a.t1, tau493)
    )

    tau495 = (
        einsum("m,mia->ia", tau110, h.l.pov)
    )

    tau496 = (
        einsum("aq,ia->qi", a.t2.x2, tau495)
    )

    tau497 = (
        einsum("ai,qi->qa", a.t1, tau496)
    )

    tau498 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau497, tau15, tau8, tau9)
    )

    tau499 = (
        einsum("aq,ia->qi", a.t2.x1, tau495)
    )

    tau500 = (
        einsum("ai,qi->qa", a.t1, tau499)
    )

    tau501 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau500, tau21, tau26, tau27)
    )

    tau502 = (
        einsum("qm,wrm->wqr", tau91, tau462)
    )

    tau503 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau502)
    )

    tau504 = (
        einsum("qm,wrm->wqr", tau95, tau462)
    )

    tau505 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau504)
    )

    tau506 = (
        einsum("qi,wri->wqr", tau496, tau5)
    )

    tau507 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau506)
    )

    tau508 = (
        einsum("qi,wri->wqr", tau499, tau5)
    )

    tau509 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau508, tau9)
    )

    tau510 = (
        einsum("m,mia->ia", tau105, h.l.pov)
    )

    tau511 = (
        einsum("ai,ja->ij", a.t1, tau510)
    )

    tau512 = (
        einsum("jq,ij->qi", a.t2.x3, tau511)
    )

    tau513 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau512)
    )

    tau514 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau513)
    )

    tau515 = (
        einsum("jq,ij->qi", a.t2.x4, tau511)
    )

    tau516 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau515)
    )

    tau517 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau516, tau8, tau9)
    )

    tau518 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau515)
    )

    tau519 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau518)
    )

    tau520 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau512)
    )

    tau521 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau520, tau8)
    )

    tau522 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau523 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau522)
    )

    tau524 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau21, tau27, tau367, tau523)
    )

    tau525 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau526 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau525)
    )

    tau527 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau15, tau363, tau9, tau526)
    )

    tau528 = (
        einsum("aq,ia->qi", a.t2.x2, h.f.ov)
    )

    tau529 = (
        einsum("ai,qi->qa", a.t1, tau528)
    )

    tau530 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau529, tau15, tau8, tau9)
    )

    tau531 = (
        einsum("aq,ia->qi", a.t2.x1, h.f.ov)
    )

    tau532 = (
        einsum("ai,qi->qa", a.t1, tau531)
    )

    tau533 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau532, tau21, tau26, tau27)
    )

    tau534 = (
        einsum("qi,wri->wqr", tau528, tau5)
    )

    tau535 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau27, tau534)
    )

    tau536 = (
        einsum("qi,wri->wqr", tau531, tau5)
    )

    tau537 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau536, tau9)
    )

    tau538 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau539 = (
        einsum("jq,ij->qi", a.t2.x3, tau538)
    )

    tau540 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau539)
    )

    tau541 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau21, tau26, tau540)
    )

    tau542 = (
        einsum("jq,ij->qi", a.t2.x4, tau538)
    )

    tau543 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau542)
    )

    tau544 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau543, tau8, tau9)
    )

    tau545 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau542)
    )

    tau546 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau26, tau27, tau545)
    )

    tau547 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau539)
    )

    tau548 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau15, tau547, tau8)
    )

    tau549 = (
        einsum("qm,wrm->wqr", tau91, tau416)
    )

    tau550 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau549)
    )

    tau551 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau92, h.l.pvo)
    )

    tau552 = (
        einsum("qm,wrm->wqr", tau95, tau416)
    )

    tau553 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau552)
    )

    tau554 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau96, h.l.pvo)
    )

    tau555 = (
        einsum("qmi,qma->qia", tau28, tau363)
    )

    tau556 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, tau555)
    )

    tau557 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau27, tau556)
    )

    tau558 = (
        einsum("qm,wrm->wqr", tau22, tau416)
    )

    tau559 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau15, tau558)
    )

    tau560 = (
        einsum("qmi,qma->qia", tau16, tau363)
    )

    tau561 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d3, b.t2.x2, b.t2.x3, tau560)
    )

    tau562 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau21, tau561)
    )

    tau563 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau28)
    )

    tau564 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau363, tau8, tau9, tau563)
    )

    tau565 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau15, tau363, tau8, tau17)
    )

    tau566 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau23, h.l.pvo)
    )

    tau567 = (
        einsum("qm,wrm->wqr", tau33, tau416)
    )

    tau568 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau27, tau567)
    )

    tau569 = (
        einsum("qmi,qma->qia", tau28, tau367)
    )

    tau570 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d3, b.t2.x2, b.t2.x3, tau569)
    )

    tau571 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau570, tau9)
    )

    tau572 = (
        einsum("qmi,qma->qia", tau16, tau367)
    )

    tau573 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, tau572)
    )

    tau574 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau15, tau573)
    )

    tau575 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau26, tau27, tau367, tau29)
    )

    tau576 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau34, h.l.pvo)
    )

    tau577 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau16)
    )

    tau578 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau21, tau26, tau367, tau577)
    )

    tau579 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau416, h.l.poo)
    )

    tau580 = (
        einsum("ai,wqi->wqa", a.t1, tau579)
    )

    tau581 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d3, b.t2.x3, tau45, h.l.pvo)
    )

    tau582 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d3, b.t2.x3, tau45, h.l.poo)
    )

    tau583 = (
        einsum("ai,wqi->wqa", a.t1, tau582)
    )

    r21 = (
        - einsum("aw,wqa->aq", d.t2.d1, tau4)
        - einsum("aw,wqa->aq", d.t2.d1, tau7)
        + einsum("aw,wqa->aq", d.t2.d1, tau14) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau20) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau25) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau32) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau36) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau41) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau44) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau48) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau51) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau54) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau57) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau60) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau65) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau67) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau72) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau74) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau79) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau84) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau87) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau90) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau94)
        - einsum("aw,wqa->aq", d.t2.d1, tau98)
        - einsum("aw,wqa->aq", d.t2.d1, tau101)
        - einsum("aw,wqa->aq", d.t2.d1, tau104)
        - einsum("aw,wqa->aq", d.t2.d1, tau109)
        - einsum("aw,wqa->aq", d.t2.d1, tau114)
        - einsum("aw,wqa->aq", d.t2.d1, tau117)
        - einsum("aw,wqa->aq", d.t2.d1, tau120)
        + einsum("aw,wqa->aq", d.t2.d1, tau122)
        + einsum("aw,wqa->aq", d.t2.d1, tau124)
        + einsum("aw,wqa->aq", d.t2.d1, tau127) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau130) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau131)
        + einsum("aw,wqa->aq", d.t2.d1, tau132)
        + einsum("aw,wqa->aq", d.t2.d1, tau134)
        + einsum("aw,wqa->aq", d.t2.d1, tau137) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau138) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau141) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau142) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau145)
        + einsum("aw,wqa->aq", d.t2.d1, tau148)
        + einsum("aw,wqa->aq", d.t2.d1, tau151)
        + einsum("aw,wqa->aq", d.t2.d1, tau154)
        - einsum("aw,wqa->aq", d.t2.d1, tau159) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau163) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau166) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau168) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau171) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau174) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau177) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau182) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau186) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau189) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau193) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau196) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau200) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau203) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau206) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau208) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau211) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau213) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau216) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau218) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau220) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau222) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau224) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau227) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau232) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau234) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau237) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau240) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau243) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau247) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau250) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau254) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau256) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau259) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau262) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau266) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau269) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau273) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau276) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau279) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau283) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau287) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau290) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau294) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau298) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau301) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau304) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau307) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau309) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau313) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau315) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau318) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau320) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau322) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau324) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau326) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau328) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau330) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau333) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau336) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau339) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau342) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau345) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau347) / 4
        + einsum("aw,wqa->aq", d.t2.d1, tau348)
        + einsum("aw,wqa->aq", d.t2.d1, tau349)
        + einsum("aw,wqa->aq", d.t2.d1, tau351)
        + einsum("aw,wqa->aq", d.t2.d1, tau353)
        + einsum("aw,wqa->aq", d.t2.d1, tau357)
        + einsum("aw,wqa->aq", d.t2.d1, tau358)
        + einsum("aw,wqa->aq", d.t2.d1, tau361)
        + einsum("aw,wqa->aq", d.t2.d1, tau362)
        - einsum("aw,wqa->aq", d.t2.d1, tau366) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau370) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau372) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau374) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau377) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau380) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau381) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau384) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau385) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau387) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau390) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau392) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau394) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau396) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau399) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau401) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau406) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau407) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau410) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau411) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau413)
        + einsum("aw,wqa->aq", d.t2.d1, tau415)
        - einsum("aw,wqa->aq", d.t2.d1, tau418)
        - einsum("aw,wqa->aq", d.t2.d1, tau420)
        - einsum("aw,wqa->aq", d.t2.d1, tau421)
        - einsum("aw,wqa->aq", d.t2.d1, tau422)
        - einsum("aw,wqa->aq", d.t2.d1, tau425) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau428) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau430) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau432) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau434) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau436) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau438) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau440) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau442) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau444) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau446) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau448) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau452) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau455) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau457) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau459) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau461) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau464) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau466) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau468) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau470) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau472) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau476) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau481) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau484) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau486) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau488) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau490) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau492)
        - einsum("aw,wqa->aq", d.t2.d1, tau494)
        - einsum("aw,wqa->aq", d.t2.d1, tau498)
        - einsum("aw,wqa->aq", d.t2.d1, tau501)
        - einsum("aw,wqa->aq", d.t2.d1, tau503)
        - einsum("aw,wqa->aq", d.t2.d1, tau505)
        - einsum("aw,wqa->aq", d.t2.d1, tau507)
        - einsum("aw,wqa->aq", d.t2.d1, tau509)
        - einsum("aw,wqa->aq", d.t2.d1, tau514)
        - einsum("aw,wqa->aq", d.t2.d1, tau517)
        - einsum("aw,wqa->aq", d.t2.d1, tau519)
        - einsum("aw,wqa->aq", d.t2.d1, tau521)
        + einsum("aw,wqa->aq", d.t2.d1, tau524) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau527) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau530) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau533) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau535) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau537) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau541) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau544) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau546) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau548) / 2
        + einsum("aw,wqa->aq", d.t2.d1, tau550)
        + einsum("aw,wqa->aq", d.t2.d1, tau551)
        + einsum("aw,wqa->aq", d.t2.d1, tau553)
        + einsum("aw,wqa->aq", d.t2.d1, tau554)
        - einsum("aw,wqa->aq", d.t2.d1, tau557) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau559) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau562) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau564) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau565) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau566) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau568) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau571) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau574) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau575) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau576) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau578) / 2
        - einsum("aw,wqa->aq", d.t2.d1, tau580)
        - einsum("aw,wqa->aq", d.t2.d1, tau581)
        + einsum("aw,wqa->aq", d.t2.d1, tau583)
    )
    return r21


def _rccsd_cpd_ls_t_true_calc_r22(h, a, b, d):
    tau0 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x1, b.t2.x1)
    )

    tau1 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x3, b.t2.x3)
    )

    tau2 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau3 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau2)
    )

    tau4 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau5 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau0, tau1, tau4, tau3)
    )

    tau6 = (
        einsum("ai,wqi->wqa", a.t1, tau5)
    )

    tau7 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x4, b.t2.x3)
    )

    tau8 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau9 = (
        einsum("rm,wrq,wrq->wqm", tau8, tau0, tau7)
    )

    tau10 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau9, h.l.poo)
    )

    tau11 = (
        einsum("ai,wqi->wqa", a.t1, tau10)
    )

    tau12 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x4, b.t2.x4)
    )

    tau13 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau14 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau13)
    )

    tau15 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau16 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau0, tau12, tau15, tau14)
    )

    tau17 = (
        einsum("ai,wqi->wqa", a.t1, tau16)
    )

    tau18 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x2, b.t2.x1)
    )

    tau19 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau20 = (
        einsum("rm,wrq,wrq->wqm", tau19, tau1, tau18)
    )

    tau21 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau20, h.l.poo)
    )

    tau22 = (
        einsum("ai,wqi->wqa", a.t1, tau21)
    )

    tau23 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x3, b.t2.x4)
    )

    tau24 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau25 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau24)
    )

    tau26 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau27 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau18, tau23, tau26, tau25)
    )

    tau28 = (
        einsum("ai,wqi->wqa", a.t1, tau27)
    )

    tau29 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau30 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau29)
    )

    tau31 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau32 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau18, tau31, tau7, tau30)
    )

    tau33 = (
        einsum("ai,wqi->wqa", a.t1, tau32)
    )

    tau34 = (
        einsum("aw,ai,aq->wqi", d.t2.d1, a.t1, b.t2.x1)
    )

    tau35 = (
        einsum("qmj,qmi->qij", tau15, tau24)
    )

    tau36 = (
        einsum("iw,ir,wrj,qij->wqr", d.t2.d3, b.t2.x3, tau34, tau35)
    )

    tau37 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau36)
    )

    tau38 = (
        einsum("qmj,qmi->qij", tau29, tau4)
    )

    tau39 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau34, tau38)
    )

    tau40 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau39, tau7)
    )

    tau41 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d3, b.t2.x3, tau34, h.l.poo)
    )

    tau42 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau43 = (
        einsum("qm,wrm->wqr", tau42, tau41)
    )

    tau44 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau43)
    )

    tau45 = (
        einsum("qmj,qmi->qij", tau2, tau31)
    )

    tau46 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau34, tau45)
    )

    tau47 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau46)
    )

    tau48 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau49 = (
        einsum("qm,wrm->wqr", tau48, tau41)
    )

    tau50 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau49)
    )

    tau51 = (
        einsum("qmi,qmj->qij", tau13, tau26)
    )

    tau52 = (
        einsum("iw,ir,wrj,qij->wqr", d.t2.d3, b.t2.x3, tau34, tau51)
    )

    tau53 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau52)
    )

    tau54 = (
        einsum("iw,ai,iq->wqa", d.t2.d3, a.t1, b.t2.x3)
    )

    tau55 = (
        einsum("wqa,mia->wqmi", tau54, h.l.pov)
    )

    tau56 = (
        einsum("jw,jq,mij->wqmi", d.t2.d4, b.t2.x4, h.l.poo)
    )

    tau57 = (
        einsum("wqmi,wqmj->wqij", tau55, tau56)
    )

    tau58 = (
        einsum("iq,jq,wrji->wqr", a.t2.x3, a.t2.x4, tau57)
    )

    tau59 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau18, tau58)
    )

    tau60 = (
        einsum("iq,jq,wrij->wqr", a.t2.x3, a.t2.x4, tau57)
    )

    tau61 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau60)
    )

    tau62 = (
        einsum("jw,jq,mij->wqmi", d.t2.d3, b.t2.x3, h.l.poo)
    )

    tau63 = (
        einsum("iw,ai,iq->wqa", d.t2.d4, a.t1, b.t2.x4)
    )

    tau64 = (
        einsum("wqa,mia->wqmi", tau63, h.l.pov)
    )

    tau65 = (
        einsum("wqmi,wqmj->wqij", tau62, tau64)
    )

    tau66 = (
        einsum("iq,jq,wrji->wqr", a.t2.x3, a.t2.x4, tau65)
    )

    tau67 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau18, tau66)
    )

    tau68 = (
        einsum("iq,jq,wrij->wqr", a.t2.x3, a.t2.x4, tau65)
    )

    tau69 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau68)
    )

    tau70 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau71 = (
        einsum("mkj,mki->ij", tau70, h.l.poo)
    )

    tau72 = (
        einsum("jq,ij->qi", a.t2.x4, tau71)
    )

    tau73 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau72)
    )

    tau74 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau73)
    )

    tau75 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau76 = (
        einsum("mki,mkj->ij", tau75, h.l.poo)
    )

    tau77 = (
        einsum("jq,ji->qi", a.t2.x3, tau76)
    )

    tau78 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau77)
    )

    tau79 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau7, tau78)
    )

    tau80 = (
        einsum("jq,ji->qi", a.t2.x4, tau76)
    )

    tau81 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau80)
    )

    tau82 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau81)
    )

    tau83 = (
        einsum("jq,ij->qi", a.t2.x3, tau71)
    )

    tau84 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau83)
    )

    tau85 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau84)
    )

    tau86 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau87 = (
        einsum("rm,wrq,wrq->wqm", tau86, tau0, tau1)
    )

    tau88 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau87, h.l.poo)
    )

    tau89 = (
        einsum("ai,wqi->wqa", a.t1, tau88)
    )

    tau90 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau91 = (
        einsum("rm,wrq,wrq->wqm", tau90, tau18, tau7)
    )

    tau92 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau91, h.l.poo)
    )

    tau93 = (
        einsum("ai,wqi->wqa", a.t1, tau92)
    )

    tau94 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau95 = (
        einsum("qm,wrm->wqr", tau94, tau41)
    )

    tau96 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau95)
    )

    tau97 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau98 = (
        einsum("qm,wrm->wqr", tau97, tau41)
    )

    tau99 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau98)
    )

    tau100 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau101 = (
        einsum("m,mij->ij", tau100, h.l.poo)
    )

    tau102 = (
        einsum("jq,ji->qi", a.t2.x4, tau101)
    )

    tau103 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau102)
    )

    tau104 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau103, tau18, tau23)
    )

    tau105 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau106 = (
        einsum("m,mij->ij", tau105, h.l.poo)
    )

    tau107 = (
        einsum("jq,ji->qi", a.t2.x3, tau106)
    )

    tau108 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau107)
    )

    tau109 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau108, tau18, tau7)
    )

    tau110 = (
        einsum("jq,ji->qi", a.t2.x4, tau106)
    )

    tau111 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau110)
    )

    tau112 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau111)
    )

    tau113 = (
        einsum("jq,ji->qi", a.t2.x3, tau101)
    )

    tau114 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau113)
    )

    tau115 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau114, tau12)
    )

    tau116 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d1, b.t2.x1, tau54, h.l.pvv)
    )

    tau117 = (
        einsum("wqm,wqa,mia->wqi", tau116, tau63, h.l.pov)
    )

    tau118 = (
        einsum("ai,wqi->wqa", a.t1, tau117)
    )

    tau119 = (
        einsum("wqi,wqa,mia->wqm", tau34, tau54, h.l.pov)
    )

    tau120 = (
        einsum("wqm,wqb,mab->wqa", tau119, tau63, h.l.pvv)
    )

    tau121 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau116, h.l.pvo)
    )

    tau122 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, h.l.pvo)
    )

    tau123 = (
        einsum("wqm,wqb,mab->wqa", tau122, tau63, h.l.pvv)
    )

    tau124 = (
        einsum("qmi,qmj->qij", tau24, tau29)
    )

    tau125 = (
        einsum("iw,jw,ir,jr,qij->wqr", d.t2.d3,
               d.t2.d4, b.t2.x3, b.t2.x4, tau124)
    )

    tau126 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau125, tau18)
    )

    tau127 = (
        einsum("qmi,qmj->qij", tau13, tau2)
    )

    tau128 = (
        einsum("iw,jw,ir,jr,qij->wqr", d.t2.d3,
               d.t2.d4, b.t2.x3, b.t2.x4, tau127)
    )

    tau129 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau128)
    )

    tau130 = (
        einsum("wqm,wqb,mab->wqa", tau116, tau63, h.l.pvv)
    )

    tau131 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau122, h.l.pvo)
    )

    tau132 = (
        einsum("wqi,wqa,mia->wqm", tau34, tau54, h.l.pov)
    )

    tau133 = (
        einsum("wqm,wqa,mia->wqi", tau132, tau63, h.l.pov)
    )

    tau134 = (
        einsum("ai,wqi->wqa", a.t1, tau133)
    )

    tau135 = (
        einsum("qm,rm->qr", tau86, tau94)
    )

    tau136 = (
        einsum("pq,wpr,wpr->wqr", tau135, tau0, tau1)
    )

    tau137 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau136, tau23)
    )

    tau138 = (
        einsum("qm,rm->qr", tau90, tau94)
    )

    tau139 = (
        einsum("pq,wpr,wpr->wqr", tau138, tau18, tau7)
    )

    tau140 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau139, tau23)
    )

    tau141 = (
        einsum("rm,qm->qr", tau86, tau97)
    )

    tau142 = (
        einsum("qp,wpr,wpr->wqr", tau141, tau0, tau1)
    )

    tau143 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau142)
    )

    tau144 = (
        einsum("qm,rm->qr", tau90, tau97)
    )

    tau145 = (
        einsum("pq,wpr,wpr->wqr", tau144, tau18, tau7)
    )

    tau146 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau145)
    )

    tau147 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau94, h.l.pov)
    )

    tau148 = (
        einsum("ar,qa->qr", a.t2.x2, tau147)
    )

    tau149 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau148, tau0, tau1, tau12)
    )

    tau150 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau149)
    )

    tau151 = (
        einsum("qm,rm->qr", tau42, tau86)
    )

    tau152 = (
        einsum("qp,wpr,wpr->wqr", tau151, tau0, tau1)
    )

    tau153 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau152)
    )

    tau154 = (
        einsum("qm,rm->qr", tau8, tau94)
    )

    tau155 = (
        einsum("pq,wpr,wpr->wqr", tau154, tau0, tau7)
    )

    tau156 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau155, tau23)
    )

    tau157 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau158 = (
        einsum("qma,qmi->qia", tau157, tau4)
    )

    tau159 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau158)
    )

    tau160 = (
        einsum("qp,wpr,wpr->wqr", tau159, tau0, tau1)
    )

    tau161 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau160, tau23)
    )

    tau162 = (
        einsum("pq,wpr->wqr", tau148, tau0)
    )

    tau163 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau162, tau23, tau7)
    )

    tau164 = (
        einsum("qm,rm->qr", tau19, tau94)
    )

    tau165 = (
        einsum("pq,wpr,wpr->wqr", tau164, tau1, tau18)
    )

    tau166 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau165, tau23)
    )

    tau167 = (
        einsum("aq,ra->qr", a.t2.x1, tau147)
    )

    tau168 = (
        einsum("pq,wpr,wpr,wpr->wqr", tau167, tau18, tau23, tau7)
    )

    tau169 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau168)
    )

    tau170 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau158)
    )

    tau171 = (
        einsum("pq,wpr,wpr->wqr", tau170, tau18, tau7)
    )

    tau172 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau171, tau23)
    )

    tau173 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau90, h.l.pov)
    )

    tau174 = (
        einsum("ar,qa->qr", a.t2.x2, tau173)
    )

    tau175 = (
        einsum("pq,wpr->wqr", tau174, tau18)
    )

    tau176 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau175, tau23, tau7)
    )

    tau177 = (
        einsum("rm,qm->qr", tau42, tau90)
    )

    tau178 = (
        einsum("pq,wpr,wpr->wqr", tau177, tau18, tau7)
    )

    tau179 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau178)
    )

    tau180 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau94, h.l.pov)
    )

    tau181 = (
        einsum("ir,qi->qr", a.t2.x4, tau180)
    )

    tau182 = (
        einsum("pq,wpr->wqr", tau181, tau1)
    )

    tau183 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau182, tau23)
    )

    tau184 = (
        einsum("ir,qi->qr", a.t2.x3, tau180)
    )

    tau185 = (
        einsum("pq,wpr->wqr", tau184, tau23)
    )

    tau186 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau185, tau7)
    )

    tau187 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau97, h.l.pov)
    )

    tau188 = (
        einsum("ir,qi->qr", a.t2.x4, tau187)
    )

    tau189 = (
        einsum("pq,wpr->wqr", tau188, tau7)
    )

    tau190 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau189, tau23)
    )

    tau191 = (
        einsum("ir,qi->qr", a.t2.x3, tau187)
    )

    tau192 = (
        einsum("pq,wpr->wqr", tau191, tau12)
    )

    tau193 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau192, tau7)
    )

    tau194 = (
        einsum("qm,rm->qr", tau48, tau86)
    )

    tau195 = (
        einsum("qp,wpr,wpr->wqr", tau194, tau0, tau1)
    )

    tau196 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau195, tau23)
    )

    tau197 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau174, tau0, tau1, tau12)
    )

    tau198 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau197)
    )

    tau199 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau200 = (
        einsum("qmi,qma->qia", tau15, tau199)
    )

    tau201 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau200)
    )

    tau202 = (
        einsum("qp,wpr,wpr->wqr", tau201, tau0, tau1)
    )

    tau203 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau202)
    )

    tau204 = (
        einsum("qp,wpr->wqr", tau167, tau0)
    )

    tau205 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau204)
    )

    tau206 = (
        einsum("rm,qm->qr", tau8, tau97)
    )

    tau207 = (
        einsum("qp,wpr,wpr->wqr", tau206, tau0, tau7)
    )

    tau208 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau207)
    )

    tau209 = (
        einsum("pq,wpr->wqr", tau184, tau1)
    )

    tau210 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau209)
    )

    tau211 = (
        einsum("pq,wpr->wqr", tau181, tau23)
    )

    tau212 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau211)
    )

    tau213 = (
        einsum("pq,wpr->wqr", tau188, tau12)
    )

    tau214 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau213)
    )

    tau215 = (
        einsum("pq,wpr->wqr", tau191, tau7)
    )

    tau216 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau215)
    )

    tau217 = (
        einsum("rm,qm->qr", tau19, tau97)
    )

    tau218 = (
        einsum("qp,wpr,wpr->wqr", tau217, tau1, tau18)
    )

    tau219 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau218)
    )

    tau220 = (
        einsum("ar,qa->qr", a.t2.x1, tau173)
    )

    tau221 = (
        einsum("pq,wpr->wqr", tau220, tau18)
    )

    tau222 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau221)
    )

    tau223 = (
        einsum("rm,qm->qr", tau48, tau90)
    )

    tau224 = (
        einsum("pq,wpr,wpr->wqr", tau223, tau18, tau7)
    )

    tau225 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau224, tau23)
    )

    tau226 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau220, tau18, tau23, tau7)
    )

    tau227 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau226)
    )

    tau228 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau229 = (
        einsum("qma,qmi->qia", tau228, tau31)
    )

    tau230 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x3, tau229)
    )

    tau231 = (
        einsum("qp,wpr,wpr->wqr", tau230, tau18, tau7)
    )

    tau232 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau231)
    )

    tau233 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau8, h.l.pov)
    )

    tau234 = (
        einsum("ar,qa->qr", a.t2.x2, tau233)
    )

    tau235 = (
        einsum("pq,wpr->wqr", tau234, tau0)
    )

    tau236 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau235, tau7)
    )

    tau237 = (
        einsum("rm,qm->qr", tau42, tau8)
    )

    tau238 = (
        einsum("pq,wpr,wpr->wqr", tau237, tau0, tau7)
    )

    tau239 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau238)
    )

    tau240 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau234, tau0, tau1, tau12)
    )

    tau241 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau240)
    )

    tau242 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau200)
    )

    tau243 = (
        einsum("qp,wpr,wpr->wqr", tau242, tau0, tau1)
    )

    tau244 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau243)
    )

    tau245 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau158)
    )

    tau246 = (
        einsum("pq,wpr,wpr->wqr", tau245, tau0, tau7)
    )

    tau247 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau246)
    )

    tau248 = (
        einsum("qma,qmi->qia", tau228, tau4)
    )

    tau249 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau248)
    )

    tau250 = (
        einsum("qp,wpr,wpr->wqr", tau249, tau0, tau12)
    )

    tau251 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau250, tau7)
    )

    tau252 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x4, tau158)
    )

    tau253 = (
        einsum("pq,wpr,wpr->wqr", tau252, tau1, tau18)
    )

    tau254 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau253)
    )

    tau255 = (
        einsum("qm,rm->qr", tau19, tau42)
    )

    tau256 = (
        einsum("pq,wpr,wpr->wqr", tau255, tau1, tau18)
    )

    tau257 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau256)
    )

    tau258 = (
        einsum("aq,ra->qr", a.t2.x1, tau233)
    )

    tau259 = (
        einsum("pq,wpr,wpr,wpr->wqr", tau258, tau18, tau23, tau7)
    )

    tau260 = (
        einsum("ar,wrq->wqa", a.t2.x1, tau259)
    )

    tau261 = (
        einsum("qma,qmi->qia", tau199, tau26)
    )

    tau262 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau261)
    )

    tau263 = (
        einsum("pq,wpr,wpr->wqr", tau262, tau18, tau23)
    )

    tau264 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau263, tau7)
    )

    tau265 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau48, h.l.pov)
    )

    tau266 = (
        einsum("ar,qa->qr", a.t2.x2, tau265)
    )

    tau267 = (
        einsum("pq,wpr->wqr", tau266, tau18)
    )

    tau268 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau267, tau7)
    )

    tau269 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau248)
    )

    tau270 = (
        einsum("pq,wpr,wpr->wqr", tau269, tau18, tau7)
    )

    tau271 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau270)
    )

    tau272 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau48, h.l.pov)
    )

    tau273 = (
        einsum("ir,qi->qr", a.t2.x4, tau272)
    )

    tau274 = (
        einsum("pq,wpr->wqr", tau273, tau1)
    )

    tau275 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau274)
    )

    tau276 = (
        einsum("qmj,qmi->qij", tau26, tau4)
    )

    tau277 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau276)
    )

    tau278 = (
        einsum("pq,wpr,wpr->wqr", tau277, tau1, tau12)
    )

    tau279 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau18, tau278)
    )

    tau280 = (
        einsum("qmj,qmi->qij", tau15, tau31)
    )

    tau281 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau280)
    )

    tau282 = (
        einsum("pq,wpr,wpr->wqr", tau281, tau23, tau7)
    )

    tau283 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau18, tau282)
    )

    tau284 = (
        einsum("ir,qi->qr", a.t2.x3, tau272)
    )

    tau285 = (
        einsum("pq,wpr->wqr", tau284, tau23)
    )

    tau286 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau285, tau7)
    )

    tau287 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau42, h.l.pov)
    )

    tau288 = (
        einsum("ir,qi->qr", a.t2.x4, tau287)
    )

    tau289 = (
        einsum("pq,wpr->wqr", tau288, tau7)
    )

    tau290 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau289)
    )

    tau291 = (
        einsum("ir,qi->qr", a.t2.x3, tau287)
    )

    tau292 = (
        einsum("pq,wpr->wqr", tau291, tau12)
    )

    tau293 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau292, tau7)
    )

    tau294 = (
        einsum("qma,qmi->qia", tau157, tau31)
    )

    tau295 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau294)
    )

    tau296 = (
        einsum("qp,wpr,wpr->wqr", tau295, tau0, tau1)
    )

    tau297 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau296)
    )

    tau298 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau266, tau0, tau1, tau12)
    )

    tau299 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau298)
    )

    tau300 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau294)
    )

    tau301 = (
        einsum("qp,wpr,wpr->wqr", tau300, tau0, tau12)
    )

    tau302 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau301)
    )

    tau303 = (
        einsum("qp,wpr->wqr", tau258, tau0)
    )

    tau304 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau303)
    )

    tau305 = (
        einsum("qm,rm->qr", tau48, tau8)
    )

    tau306 = (
        einsum("qp,wpr,wpr->wqr", tau305, tau0, tau7)
    )

    tau307 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau306)
    )

    tau308 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau229)
    )

    tau309 = (
        einsum("qp,wpr,wpr->wqr", tau308, tau0, tau7)
    )

    tau310 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau309)
    )

    tau311 = (
        einsum("pq,wpr,wpr->wqr", tau281, tau1, tau12)
    )

    tau312 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau311)
    )

    tau313 = (
        einsum("pq,wpr->wqr", tau284, tau1)
    )

    tau314 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau313)
    )

    tau315 = (
        einsum("pq,wpr->wqr", tau273, tau23)
    )

    tau316 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau315)
    )

    tau317 = (
        einsum("pq,wpr->wqr", tau288, tau12)
    )

    tau318 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau317)
    )

    tau319 = (
        einsum("pq,wpr,wpr->wqr", tau277, tau23, tau7)
    )

    tau320 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau319)
    )

    tau321 = (
        einsum("pq,wpr->wqr", tau291, tau7)
    )

    tau322 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau321)
    )

    tau323 = (
        einsum("qm,rm->qr", tau19, tau48)
    )

    tau324 = (
        einsum("pq,wpr,wpr->wqr", tau323, tau1, tau18)
    )

    tau325 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau324)
    )

    tau326 = (
        einsum("ar,qa->qr", a.t2.x1, tau265)
    )

    tau327 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau326, tau18, tau23, tau7)
    )

    tau328 = (
        einsum("ar,wrq->wqa", a.t2.x2, tau327)
    )

    tau329 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau261)
    )

    tau330 = (
        einsum("qp,wpr,wpr->wqr", tau329, tau1, tau18)
    )

    tau331 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau330)
    )

    tau332 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x4, tau294)
    )

    tau333 = (
        einsum("qp,wpr,wpr->wqr", tau332, tau18, tau23)
    )

    tau334 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau333)
    )

    tau335 = (
        einsum("pq,wpr->wqr", tau326, tau18)
    )

    tau336 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau335)
    )

    tau337 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau294)
    )

    tau338 = (
        einsum("pq,wpr,wpr->wqr", tau337, tau18, tau7)
    )

    tau339 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau338)
    )

    tau340 = (
        einsum("bq,ab->qa", a.t2.x2, h.f.vv)
    )

    tau341 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau340)
    )

    tau342 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau341, tau7)
    )

    tau343 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau340, tau0, tau1, tau12)
    )

    tau344 = (
        einsum("bq,ab->qa", a.t2.x1, h.f.vv)
    )

    tau345 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau344)
    )

    tau346 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau345)
    )

    tau347 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau344, tau18, tau23, tau7)
    )

    tau348 = (
        einsum("qm,wrm->wqr", tau94, tau116)
    )

    tau349 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau348)
    )

    tau350 = (
        einsum("qm,wrm->wqr", tau97, tau116)
    )

    tau351 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau350)
    )

    tau352 = (
        einsum("rm,wrq,wrq->wqm", tau94, tau0, tau1)
    )

    tau353 = (
        einsum("wqm,wqb,mab->wqa", tau352, tau63, h.l.pvv)
    )

    tau354 = (
        einsum("rm,wrq,wrq->wqm", tau97, tau18, tau7)
    )

    tau355 = (
        einsum("wqm,wqb,mab->wqa", tau354, tau63, h.l.pvv)
    )

    tau356 = (
        einsum("m,mab->ab", tau100, h.l.pvv)
    )

    tau357 = (
        einsum("bq,ab->qa", a.t2.x2, tau356)
    )

    tau358 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau357)
    )

    tau359 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau358, tau7)
    )

    tau360 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau357, tau0, tau1, tau12)
    )

    tau361 = (
        einsum("bq,ab->qa", a.t2.x1, tau356)
    )

    tau362 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau361)
    )

    tau363 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau362)
    )

    tau364 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau361, tau18, tau23, tau7)
    )

    tau365 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau366 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau365)
    )

    tau367 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau1, tau12, tau15, tau366)
    )

    tau368 = (
        einsum("ai,wqi->wqa", a.t1, tau367)
    )

    tau369 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau370 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau369)
    )

    tau371 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau23, tau26, tau7, tau370)
    )

    tau372 = (
        einsum("ai,wqi->wqa", a.t1, tau371)
    )

    tau373 = (
        einsum("qmi,wri->wqrm", tau26, tau34)
    )

    tau374 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau1, tau12, tau369, tau373)
    )

    tau375 = (
        einsum("qmi,wri->wqrm", tau15, tau34)
    )

    tau376 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau23, tau365, tau7, tau375)
    )

    tau377 = (
        einsum("qmb,qma->qab", tau157, tau369)
    )

    tau378 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau377, tau54)
    )

    tau379 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau378)
    )

    tau380 = (
        einsum("qm,wrm->wqr", tau42, tau116)
    )

    tau381 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau380)
    )

    tau382 = (
        einsum("wqa,mia->wqmi", tau54, h.l.pov)
    )

    tau383 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau382)
    )

    tau384 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau0, tau12, tau369, tau383)
    )

    tau385 = (
        einsum("qm,wrm->wqr", tau48, tau116)
    )

    tau386 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau385)
    )

    tau387 = (
        einsum("qmb,qma->qab", tau228, tau365)
    )

    tau388 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau387, tau54)
    )

    tau389 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau388)
    )

    tau390 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau382)
    )

    tau391 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau18, tau23, tau365, tau390)
    )

    tau392 = (
        einsum("qmb,qma->qab", tau228, tau369)
    )

    tau393 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau392, tau63)
    )

    tau394 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau393, tau7)
    )

    tau395 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau64)
    )

    tau396 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau0, tau1, tau369, tau395)
    )

    tau397 = (
        einsum("rm,wrq,wrq->wqm", tau42, tau0, tau7)
    )

    tau398 = (
        einsum("wqm,wqb,mab->wqa", tau397, tau63, h.l.pvv)
    )

    tau399 = (
        einsum("qmb,qma->qab", tau157, tau365)
    )

    tau400 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau399, tau63)
    )

    tau401 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau400)
    )

    tau402 = (
        einsum("rm,wrq,wrq->wqm", tau48, tau1, tau18)
    )

    tau403 = (
        einsum("wqm,wqb,mab->wqa", tau402, tau63, h.l.pvv)
    )

    tau404 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau64)
    )

    tau405 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau18, tau365, tau7, tau404)
    )

    tau406 = (
        einsum("bi,mab->mia", a.t1, h.l.pvv)
    )

    tau407 = (
        einsum("mia,mib->ab", tau406, h.l.pov)
    )

    tau408 = (
        einsum("bq,ab->qa", a.t2.x2, tau407)
    )

    tau409 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau408)
    )

    tau410 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau409, tau7)
    )

    tau411 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau408, tau0, tau1, tau12)
    )

    tau412 = (
        einsum("bq,ab->qa", a.t2.x1, tau407)
    )

    tau413 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau412)
    )

    tau414 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau413)
    )

    tau415 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau412, tau18, tau23, tau7)
    )

    tau416 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau132, h.l.poo)
    )

    tau417 = (
        einsum("ai,wqi->wqa", a.t1, tau416)
    )

    tau418 = (
        einsum("wqm,wqa,mia->wqi", tau41, tau63, h.l.pov)
    )

    tau419 = (
        einsum("ai,wqi->wqa", a.t1, tau418)
    )

    tau420 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau116, h.l.poo)
    )

    tau421 = (
        einsum("ai,wqi->wqa", a.t1, tau420)
    )

    tau422 = (
        einsum("wqm,wqa,mia->wqi", tau122, tau63, h.l.pov)
    )

    tau423 = (
        einsum("ai,wqi->wqa", a.t1, tau422)
    )

    tau424 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau119, h.l.pvo)
    )

    tau425 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d3, b.t2.x3, tau34, h.l.poo)
    )

    tau426 = (
        einsum("wqm,wqb,mab->wqa", tau425, tau63, h.l.pvv)
    )

    tau427 = (
        einsum("jq,ji->qi", a.t2.x4, h.f.oo)
    )

    tau428 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau427)
    )

    tau429 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau428)
    )

    tau430 = (
        einsum("jq,ji->qi", a.t2.x3, h.f.oo)
    )

    tau431 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau430)
    )

    tau432 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau431, tau7)
    )

    tau433 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau427)
    )

    tau434 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau433)
    )

    tau435 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau430)
    )

    tau436 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau435)
    )

    tau437 = (
        einsum("qmi,wri->wqrm", tau31, tau34)
    )

    tau438 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau1, tau12, tau15, tau437)
    )

    tau439 = (
        einsum("ai,wqi->wqa", a.t1, tau438)
    )

    tau440 = (
        einsum("wri,qmi->wqrm", tau34, tau4)
    )

    tau441 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau23, tau26, tau7, tau440)
    )

    tau442 = (
        einsum("ai,wqi->wqa", a.t1, tau441)
    )

    tau443 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau0, tau12, tau4, tau383)
    )

    tau444 = (
        einsum("ai,wqi->wqa", a.t1, tau443)
    )

    tau445 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau18, tau23, tau31, tau390)
    )

    tau446 = (
        einsum("ai,wqi->wqa", a.t1, tau445)
    )

    tau447 = (
        einsum("wrq,wrq,rmi,wrqm->wqi", tau0, tau1, tau4, tau395)
    )

    tau448 = (
        einsum("ai,wqi->wqa", a.t1, tau447)
    )

    tau449 = (
        einsum("wqm,wqa,mia->wqi", tau397, tau63, h.l.pov)
    )

    tau450 = (
        einsum("ai,wqi->wqa", a.t1, tau449)
    )

    tau451 = (
        einsum("wqm,wqa,mia->wqi", tau402, tau63, h.l.pov)
    )

    tau452 = (
        einsum("ai,wqi->wqa", a.t1, tau451)
    )

    tau453 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau18, tau31, tau7, tau404)
    )

    tau454 = (
        einsum("ai,wqi->wqa", a.t1, tau453)
    )

    tau455 = (
        einsum("mji,mja->ia", tau75, h.l.pov)
    )

    tau456 = (
        einsum("aq,ia->qi", a.t2.x2, tau455)
    )

    tau457 = (
        einsum("ai,qi->qa", a.t1, tau456)
    )

    tau458 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau457, tau0, tau1, tau12)
    )

    tau459 = (
        einsum("aq,ia->qi", a.t2.x1, tau455)
    )

    tau460 = (
        einsum("ai,qi->qa", a.t1, tau459)
    )

    tau461 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau460, tau18, tau23, tau7)
    )

    tau462 = (
        einsum("qia,wri,wra->wqr", tau158, tau34, tau54)
    )

    tau463 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau462)
    )

    tau464 = (
        einsum("qm,wrm->wqr", tau42, tau132)
    )

    tau465 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau464)
    )

    tau466 = (
        einsum("qm,wrm->wqr", tau48, tau132)
    )

    tau467 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau466)
    )

    tau468 = (
        einsum("qia,wri,wra->wqr", tau229, tau34, tau54)
    )

    tau469 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau468)
    )

    tau470 = (
        einsum("qia,wri,wra->wqr", tau248, tau34, tau63)
    )

    tau471 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau470, tau7)
    )

    tau472 = (
        einsum("qia,wri,wra->wqr", tau294, tau34, tau63)
    )

    tau473 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau472)
    )

    tau474 = (
        einsum("qi,wri->wqr", tau456, tau34)
    )

    tau475 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau474, tau7)
    )

    tau476 = (
        einsum("qi,wri->wqr", tau459, tau34)
    )

    tau477 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau476)
    )

    tau478 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau479 = (
        einsum("qmb,qma->qab", tau157, tau478)
    )

    tau480 = (
        einsum("qab,wra,wrb->wqr", tau479, tau54, tau63)
    )

    tau481 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau0, tau480)
    )

    tau482 = (
        einsum("mji,mja->ia", tau70, h.l.pov)
    )

    tau483 = (
        einsum("ai,ja->ij", a.t1, tau482)
    )

    tau484 = (
        einsum("jq,ij->qi", a.t2.x4, tau483)
    )

    tau485 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau484)
    )

    tau486 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau485)
    )

    tau487 = (
        einsum("jq,ij->qi", a.t2.x3, tau483)
    )

    tau488 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau487)
    )

    tau489 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau488)
    )

    tau490 = (
        einsum("qab,wrb,wra->wqr", tau479, tau54, tau63)
    )

    tau491 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau18, tau490)
    )

    tau492 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau487)
    )

    tau493 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau492, tau7)
    )

    tau494 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau484)
    )

    tau495 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau494)
    )

    tau496 = (
        einsum("wqm,wqa,mia->wqi", tau352, tau63, h.l.pov)
    )

    tau497 = (
        einsum("ai,wqi->wqa", a.t1, tau496)
    )

    tau498 = (
        einsum("wqm,wqa,mia->wqi", tau354, tau63, h.l.pov)
    )

    tau499 = (
        einsum("ai,wqi->wqa", a.t1, tau498)
    )

    tau500 = (
        einsum("m,mia->ia", tau100, h.l.pov)
    )

    tau501 = (
        einsum("aq,ia->qi", a.t2.x2, tau500)
    )

    tau502 = (
        einsum("ai,qi->qa", a.t1, tau501)
    )

    tau503 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau502, tau0, tau1, tau12)
    )

    tau504 = (
        einsum("aq,ia->qi", a.t2.x1, tau500)
    )

    tau505 = (
        einsum("ai,qi->qa", a.t1, tau504)
    )

    tau506 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau505, tau18, tau23, tau7)
    )

    tau507 = (
        einsum("qm,wrm->wqr", tau94, tau132)
    )

    tau508 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau507)
    )

    tau509 = (
        einsum("qm,wrm->wqr", tau97, tau132)
    )

    tau510 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau509)
    )

    tau511 = (
        einsum("qi,wri->wqr", tau501, tau34)
    )

    tau512 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau511, tau7)
    )

    tau513 = (
        einsum("qi,wri->wqr", tau504, tau34)
    )

    tau514 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau513)
    )

    tau515 = (
        einsum("m,mia->ia", tau105, h.l.pov)
    )

    tau516 = (
        einsum("ai,ja->ij", a.t1, tau515)
    )

    tau517 = (
        einsum("jq,ij->qi", a.t2.x4, tau516)
    )

    tau518 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau517)
    )

    tau519 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau518)
    )

    tau520 = (
        einsum("jq,ij->qi", a.t2.x3, tau516)
    )

    tau521 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau520)
    )

    tau522 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau521)
    )

    tau523 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau520)
    )

    tau524 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau523, tau7)
    )

    tau525 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau517)
    )

    tau526 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau525)
    )

    tau527 = (
        einsum("aq,ia->qi", a.t2.x2, h.f.ov)
    )

    tau528 = (
        einsum("ai,qi->qa", a.t1, tau527)
    )

    tau529 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau528, tau0, tau1, tau12)
    )

    tau530 = (
        einsum("aq,ia->qi", a.t2.x1, h.f.ov)
    )

    tau531 = (
        einsum("ai,qi->qa", a.t1, tau530)
    )

    tau532 = (
        einsum("ra,wrq,wrq,wrq->wqa", tau531, tau18, tau23, tau7)
    )

    tau533 = (
        einsum("qi,wri->wqr", tau527, tau34)
    )

    tau534 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau23, tau533, tau7)
    )

    tau535 = (
        einsum("qi,wri->wqr", tau530, tau34)
    )

    tau536 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau1, tau12, tau535)
    )

    tau537 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau538 = (
        einsum("jq,ij->qi", a.t2.x4, tau537)
    )

    tau539 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau538)
    )

    tau540 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau23, tau539)
    )

    tau541 = (
        einsum("jq,ij->qi", a.t2.x3, tau537)
    )

    tau542 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau541)
    )

    tau543 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau12, tau542)
    )

    tau544 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau541)
    )

    tau545 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x1, tau18, tau544, tau7)
    )

    tau546 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau538)
    )

    tau547 = (
        einsum("ar,wrq,wrq,wrq->wqa", a.t2.x2, tau0, tau1, tau546)
    )

    tau548 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau549 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau1, tau12, tau548, tau366)
    )

    tau550 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau551 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau23, tau550, tau7, tau370)
    )

    tau552 = (
        einsum("qm,wrm->wqr", tau94, tau122)
    )

    tau553 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau552)
    )

    tau554 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau352, h.l.pvo)
    )

    tau555 = (
        einsum("qm,wrm->wqr", tau97, tau122)
    )

    tau556 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau555)
    )

    tau557 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau354, h.l.pvo)
    )

    tau558 = (
        einsum("qmi,qma->qia", tau2, tau369)
    )

    tau559 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, tau558)
    )

    tau560 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau23, tau559)
    )

    tau561 = (
        einsum("qmi,qma->qia", tau29, tau369)
    )

    tau562 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d4, b.t2.x1, b.t2.x4, tau561)
    )

    tau563 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau562, tau7)
    )

    tau564 = (
        einsum("qm,wrm->wqr", tau42, tau122)
    )

    tau565 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x1, tau12, tau564)
    )

    tau566 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau0, tau1, tau369, tau3)
    )

    tau567 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau397, h.l.pvo)
    )

    tau568 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau29)
    )

    tau569 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau0, tau12, tau369, tau568)
    )

    tau570 = (
        einsum("qmi,qma->qia", tau2, tau365)
    )

    tau571 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d4, b.t2.x1, b.t2.x4, tau570)
    )

    tau572 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau1, tau571)
    )

    tau573 = (
        einsum("qm,wrm->wqr", tau48, tau122)
    )

    tau574 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau23, tau573)
    )

    tau575 = (
        einsum("qmi,qma->qia", tau29, tau365)
    )

    tau576 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, tau575)
    )

    tau577 = (
        einsum("ar,wrq,wrq->wqa", a.t2.x2, tau12, tau576)
    )

    tau578 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau402, h.l.pvo)
    )

    tau579 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau2)
    )

    tau580 = (
        einsum("wrq,wrq,rma,wrqm->wqa", tau18, tau23, tau365, tau579)
    )

    tau581 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau18, tau365, tau7, tau30)
    )

    tau582 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau122, h.l.poo)
    )

    tau583 = (
        einsum("ai,wqi->wqa", a.t1, tau582)
    )

    tau584 = (
        einsum("iw,iq,wqm,mai->wqa", d.t2.d4, b.t2.x4, tau425, h.l.pvo)
    )

    tau585 = (
        einsum("jw,jq,wqm,mij->wqi", d.t2.d4, b.t2.x4, tau41, h.l.poo)
    )

    tau586 = (
        einsum("ai,wqi->wqa", a.t1, tau585)
    )

    r22 = (
        einsum("aw,wqa->aq", d.t2.d2, tau6) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau11) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau17) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau22) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau28) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau33) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau37) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau40) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau44) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau47) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau50) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau53) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau59) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau61) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau67) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau69) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau74) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau79) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau82) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau85) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau89)
        - einsum("aw,wqa->aq", d.t2.d2, tau93)
        - einsum("aw,wqa->aq", d.t2.d2, tau96)
        - einsum("aw,wqa->aq", d.t2.d2, tau99)
        - einsum("aw,wqa->aq", d.t2.d2, tau104)
        - einsum("aw,wqa->aq", d.t2.d2, tau109)
        - einsum("aw,wqa->aq", d.t2.d2, tau112)
        - einsum("aw,wqa->aq", d.t2.d2, tau115)
        - einsum("aw,wqa->aq", d.t2.d2, tau118)
        - einsum("aw,wqa->aq", d.t2.d2, tau120)
        + einsum("aw,wqa->aq", d.t2.d2, tau121)
        + einsum("aw,wqa->aq", d.t2.d2, tau123)
        + einsum("aw,wqa->aq", d.t2.d2, tau126) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau129) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau130)
        + einsum("aw,wqa->aq", d.t2.d2, tau131)
        + einsum("aw,wqa->aq", d.t2.d2, tau134)
        + einsum("aw,wqa->aq", d.t2.d2, tau137)
        + einsum("aw,wqa->aq", d.t2.d2, tau140)
        + einsum("aw,wqa->aq", d.t2.d2, tau143)
        + einsum("aw,wqa->aq", d.t2.d2, tau146)
        - einsum("aw,wqa->aq", d.t2.d2, tau150) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau153) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau156) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau161) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau163) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau166) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau169) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau172) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau176) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau179) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau183) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau186) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau190) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau193) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau196) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau198) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau203) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau205) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau208) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau210) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau212) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau214) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau216) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau219) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau222) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau225) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau227) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau232) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau236) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau239) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau241) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau244) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau247) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau251) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau254) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau257) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau260) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau264) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau268) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau271) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau275) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau279) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau283) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau286) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau290) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau293) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau297) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau299) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau302) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau304) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau307) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau310) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau312) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau314) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau316) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau318) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau320) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau322) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau325) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau328) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau331) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau334) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau336) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau339) / 4
        + einsum("aw,wqa->aq", d.t2.d2, tau342) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau343) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau346) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau347) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau349)
        + einsum("aw,wqa->aq", d.t2.d2, tau351)
        + einsum("aw,wqa->aq", d.t2.d2, tau353)
        + einsum("aw,wqa->aq", d.t2.d2, tau355)
        + einsum("aw,wqa->aq", d.t2.d2, tau359)
        + einsum("aw,wqa->aq", d.t2.d2, tau360)
        + einsum("aw,wqa->aq", d.t2.d2, tau363)
        + einsum("aw,wqa->aq", d.t2.d2, tau364)
        - einsum("aw,wqa->aq", d.t2.d2, tau368) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau372) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau374) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau376) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau379) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau381) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau384) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau386) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau389) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau391) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau394) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau396) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau398) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau401) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau403) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau405) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau410) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau411) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau414) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau415) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau417)
        + einsum("aw,wqa->aq", d.t2.d2, tau419)
        - einsum("aw,wqa->aq", d.t2.d2, tau421)
        - einsum("aw,wqa->aq", d.t2.d2, tau423)
        - einsum("aw,wqa->aq", d.t2.d2, tau424)
        - einsum("aw,wqa->aq", d.t2.d2, tau426)
        - einsum("aw,wqa->aq", d.t2.d2, tau429) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau432) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau434) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau436) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau439) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau442) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau444) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau446) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau448) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau450) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau452) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau454) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau458) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau461) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau463) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau465) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau467) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau469) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau471) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau473) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau475) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau477) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau481) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau486) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau489) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau491) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau493) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau495) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau497)
        - einsum("aw,wqa->aq", d.t2.d2, tau499)
        - einsum("aw,wqa->aq", d.t2.d2, tau503)
        - einsum("aw,wqa->aq", d.t2.d2, tau506)
        - einsum("aw,wqa->aq", d.t2.d2, tau508)
        - einsum("aw,wqa->aq", d.t2.d2, tau510)
        - einsum("aw,wqa->aq", d.t2.d2, tau512)
        - einsum("aw,wqa->aq", d.t2.d2, tau514)
        - einsum("aw,wqa->aq", d.t2.d2, tau519)
        - einsum("aw,wqa->aq", d.t2.d2, tau522)
        - einsum("aw,wqa->aq", d.t2.d2, tau524)
        - einsum("aw,wqa->aq", d.t2.d2, tau526)
        - einsum("aw,wqa->aq", d.t2.d2, tau529) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau532) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau534) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau536) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau540) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau543) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau545) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau547) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau549) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau551) / 2
        + einsum("aw,wqa->aq", d.t2.d2, tau553)
        + einsum("aw,wqa->aq", d.t2.d2, tau554)
        + einsum("aw,wqa->aq", d.t2.d2, tau556)
        + einsum("aw,wqa->aq", d.t2.d2, tau557)
        - einsum("aw,wqa->aq", d.t2.d2, tau560) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau563) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau565) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau566) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau567) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau569) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau572) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau574) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau577) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau578) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau580) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau581) / 2
        - einsum("aw,wqa->aq", d.t2.d2, tau583)
        - einsum("aw,wqa->aq", d.t2.d2, tau584)
        + einsum("aw,wqa->aq", d.t2.d2, tau586)
    )

    return r22


def _rccsd_cpd_ls_t_true_calc_r23(h, a, b, d):
    tau0 = (
        einsum("aw,ai,aq->wqi", d.t2.d1, a.t1, b.t2.x1)
    )

    tau1 = (
        einsum("wqi,mia->wqma", tau0, h.l.pov)
    )

    tau2 = (
        einsum("aw,ai,aq->wqi", d.t2.d2, a.t1, b.t2.x2)
    )

    tau3 = (
        einsum("wqi,mia->wqma", tau2, h.l.pov)
    )

    tau4 = (
        einsum("wqma,wqmb->wqab", tau1, tau3)
    )

    tau5 = (
        einsum("aq,bq,wrab->wqr", a.t2.x1, a.t2.x2, tau4)
    )

    tau6 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x4, b.t2.x4)
    )

    tau7 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau5, tau6)
    )

    tau8 = (
        einsum("aq,bq,wrba->wqr", a.t2.x1, a.t2.x2, tau4)
    )

    tau9 = (
        einsum("iw,iq,ir->wqr", d.t2.d4, a.t2.x3, b.t2.x4)
    )

    tau10 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau8, tau9)
    )

    tau11 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau12 = (
        einsum("wri,qmi->wqrm", tau0, tau11)
    )

    tau13 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x1, b.t2.x2)
    )

    tau14 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau15 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau9, tau12)
    )

    tau16 = (
        einsum("ai,wqa->wqi", a.t1, tau15)
    )

    tau17 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau18 = (
        einsum("rm,wrq,wrq->wqm", tau17, tau13, tau6)
    )

    tau19 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau18, h.l.pov)
    )

    tau20 = (
        einsum("ai,wqa->wqi", a.t1, tau19)
    )

    tau21 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x2, b.t2.x2)
    )

    tau22 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau23 = (
        einsum("rm,wrq,wrq->wqm", tau22, tau21, tau9)
    )

    tau24 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau23, h.l.pov)
    )

    tau25 = (
        einsum("ai,wqa->wqi", a.t1, tau24)
    )

    tau26 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau27 = (
        einsum("wri,qmi->wqrm", tau0, tau26)
    )

    tau28 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau29 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau21, tau28, tau6, tau27)
    )

    tau30 = (
        einsum("ai,wqa->wqi", a.t1, tau29)
    )

    tau31 = (
        einsum("iw,ai,iq->wqa", d.t2.d4, a.t1, b.t2.x4)
    )

    tau32 = (
        einsum("qmi,qma->qia", tau11, tau28)
    )

    tau33 = (
        einsum("wri,wra,qia->wqr", tau0, tau31, tau32)
    )

    tau34 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau33)
    )

    tau35 = (
        einsum("qma,qmi->qia", tau14, tau26)
    )

    tau36 = (
        einsum("wri,wra,qia->wqr", tau0, tau31, tau35)
    )

    tau37 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau36)
    )

    tau38 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau39 = (
        einsum("mji,mja->ia", tau38, h.l.pov)
    )

    tau40 = (
        einsum("aq,ia->qi", a.t2.x2, tau39)
    )

    tau41 = (
        einsum("qi,wri->wqr", tau40, tau0)
    )

    tau42 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau41, tau9)
    )

    tau43 = (
        einsum("aq,ia->qi", a.t2.x1, tau39)
    )

    tau44 = (
        einsum("qi,wri->wqr", tau43, tau0)
    )

    tau45 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau44, tau6)
    )

    tau46 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x1, b.t2.x1)
    )

    tau47 = (
        einsum("qmi,wri->wqrm", tau11, tau2)
    )

    tau48 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau28, tau46, tau6, tau47)
    )

    tau49 = (
        einsum("ai,wqa->wqi", a.t1, tau48)
    )

    tau50 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x2, b.t2.x1)
    )

    tau51 = (
        einsum("wri,qmi->wqrm", tau2, tau26)
    )

    tau52 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau14, tau50, tau9, tau51)
    )

    tau53 = (
        einsum("ai,wqa->wqi", a.t1, tau52)
    )

    tau54 = (
        einsum("qmi,qma->qia", tau11, tau14)
    )

    tau55 = (
        einsum("wri,wra,qia->wqr", tau2, tau31, tau54)
    )

    tau56 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau46, tau55)
    )

    tau57 = (
        einsum("wqi,wqa,mia->wqm", tau2, tau31, h.l.pov)
    )

    tau58 = (
        einsum("qm,wrm->wqr", tau17, tau57)
    )

    tau59 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau58)
    )

    tau60 = (
        einsum("qm,wrm->wqr", tau22, tau57)
    )

    tau61 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau60)
    )

    tau62 = (
        einsum("qmi,qma->qia", tau26, tau28)
    )

    tau63 = (
        einsum("wri,wra,qia->wqr", tau2, tau31, tau62)
    )

    tau64 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau50, tau63)
    )

    tau65 = (
        einsum("qi,wri->wqr", tau40, tau2)
    )

    tau66 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau46, tau6, tau65)
    )

    tau67 = (
        einsum("qi,wri->wqr", tau43, tau2)
    )

    tau68 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau50, tau67, tau9)
    )

    tau69 = (
        einsum("wqa,mia->wqmi", tau31, h.l.pov)
    )

    tau70 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau69)
    )

    tau71 = (
        einsum("ir,wrq,wrq,wrqm->wqmi", a.t2.x3, tau21, tau46, tau70)
    )

    tau72 = (
        einsum("mia,wqmi->wqa", h.l.pov, tau71)
    )

    tau73 = (
        einsum("ai,wqa->wqi", a.t1, tau72)
    )

    tau74 = (
        einsum("wqa,mia->wqmi", tau31, h.l.pov)
    )

    tau75 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau74)
    )

    tau76 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau50, tau75)
    )

    tau77 = (
        einsum("ai,wqa->wqi", a.t1, tau76)
    )

    tau78 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau79 = (
        einsum("mji,mja->ia", tau78, h.l.pov)
    )

    tau80 = (
        einsum("ai,ja->ij", a.t1, tau79)
    )

    tau81 = (
        einsum("jq,ij->qi", a.t2.x3, tau80)
    )

    tau82 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau81, tau21, tau46, tau6)
    )

    tau83 = (
        einsum("jq,ij->qi", a.t2.x4, tau80)
    )

    tau84 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau83, tau13, tau50, tau9)
    )

    tau85 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau83)
    )

    tau86 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau46, tau85)
    )

    tau87 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau81)
    )

    tau88 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau50, tau87)
    )

    tau89 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau90 = (
        einsum("rm,wrq,wrq->wqm", tau89, tau13, tau9)
    )

    tau91 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau90, h.l.pov)
    )

    tau92 = (
        einsum("ai,wqa->wqi", a.t1, tau91)
    )

    tau93 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau94 = (
        einsum("rm,wrq,wrq->wqm", tau93, tau21, tau6)
    )

    tau95 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau94, h.l.pov)
    )

    tau96 = (
        einsum("ai,wqa->wqi", a.t1, tau95)
    )

    tau97 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau98 = (
        einsum("m,mia->ia", tau97, h.l.pov)
    )

    tau99 = (
        einsum("aq,ia->qi", a.t2.x2, tau98)
    )

    tau100 = (
        einsum("qi,wri->wqr", tau99, tau0)
    )

    tau101 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau100, tau13, tau9)
    )

    tau102 = (
        einsum("aq,ia->qi", a.t2.x1, tau98)
    )

    tau103 = (
        einsum("qi,wri->wqr", tau102, tau0)
    )

    tau104 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau103, tau21, tau6)
    )

    tau105 = (
        einsum("qm,wrm->wqr", tau89, tau57)
    )

    tau106 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau105, tau46)
    )

    tau107 = (
        einsum("qm,wrm->wqr", tau93, tau57)
    )

    tau108 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau107, tau50)
    )

    tau109 = (
        einsum("qi,wri->wqr", tau99, tau2)
    )

    tau110 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau109, tau46, tau6)
    )

    tau111 = (
        einsum("qi,wri->wqr", tau102, tau2)
    )

    tau112 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau111, tau50, tau9)
    )

    tau113 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau114 = (
        einsum("m,mia->ia", tau113, h.l.pov)
    )

    tau115 = (
        einsum("ai,ja->ij", a.t1, tau114)
    )

    tau116 = (
        einsum("jq,ij->qi", a.t2.x3, tau115)
    )

    tau117 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau116, tau21, tau46, tau6)
    )

    tau118 = (
        einsum("jq,ij->qi", a.t2.x4, tau115)
    )

    tau119 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau118, tau13, tau50, tau9)
    )

    tau120 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau118)
    )

    tau121 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau120, tau21, tau46)
    )

    tau122 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau116)
    )

    tau123 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau122, tau13, tau50)
    )

    tau124 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d4, b.t2.x4, tau2, h.l.poo)
    )

    tau125 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau124, h.l.pov)
    )

    tau126 = (
        einsum("ai,wqa->wqi", a.t1, tau125)
    )

    tau127 = (
        einsum("wqi,wqa,mia->wqm", tau2, tau31, h.l.pov)
    )

    tau128 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau127, h.l.poo)
    )

    tau129 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d2, b.t2.x2, tau31, h.l.pvv)
    )

    tau130 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau129, h.l.pvv)
    )

    tau131 = (
        einsum("ai,wqa->wqi", a.t1, tau130)
    )

    tau132 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, h.l.pvo)
    )

    tau133 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau132, h.l.poo)
    )

    tau134 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau124, h.l.pvo)
    )

    tau135 = (
        einsum("bq,ab->qa", a.t2.x2, h.f.vv)
    )

    tau136 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau135)
    )

    tau137 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau136, tau46, tau6)
    )

    tau138 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau135)
    )

    tau139 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau138, tau9)
    )

    tau140 = (
        einsum("bq,ab->qa", a.t2.x1, h.f.vv)
    )

    tau141 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau140)
    )

    tau142 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau141, tau50, tau9)
    )

    tau143 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau140)
    )

    tau144 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau143, tau21, tau6)
    )

    tau145 = (
        einsum("jq,ji->qi", a.t2.x4, h.f.oo)
    )

    tau146 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau145)
    )

    tau147 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau146, tau21, tau46)
    )

    tau148 = (
        einsum("jq,ji->qi", a.t2.x3, h.f.oo)
    )

    tau149 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau148, tau21, tau46, tau6)
    )

    tau150 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau145, tau13, tau50, tau9)
    )

    tau151 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau148)
    )

    tau152 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau151, tau50)
    )

    tau153 = (
        einsum("aq,ia->qi", a.t2.x2, h.f.ov)
    )

    tau154 = (
        einsum("qi,wri->wqr", tau153, tau0)
    )

    tau155 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau154, tau9)
    )

    tau156 = (
        einsum("aq,ia->qi", a.t2.x1, h.f.ov)
    )

    tau157 = (
        einsum("qi,wri->wqr", tau156, tau0)
    )

    tau158 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau157, tau21, tau6)
    )

    tau159 = (
        einsum("qi,wri->wqr", tau153, tau2)
    )

    tau160 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau159, tau46, tau6)
    )

    tau161 = (
        einsum("qi,wri->wqr", tau156, tau2)
    )

    tau162 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau161, tau50, tau9)
    )

    tau163 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau164 = (
        einsum("jq,ij->qi", a.t2.x3, tau163)
    )

    tau165 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau164, tau21, tau46, tau6)
    )

    tau166 = (
        einsum("jq,ij->qi", a.t2.x4, tau163)
    )

    tau167 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau166, tau13, tau50, tau9)
    )

    tau168 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau166)
    )

    tau169 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau168, tau21, tau46)
    )

    tau170 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau164)
    )

    tau171 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau170, tau50)
    )

    tau172 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau90, h.l.pvv)
    )

    tau173 = (
        einsum("ai,wqa->wqi", a.t1, tau172)
    )

    tau174 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau94, h.l.pvv)
    )

    tau175 = (
        einsum("ai,wqa->wqi", a.t1, tau174)
    )

    tau176 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d2, b.t2.x2, tau31, h.l.pvv)
    )

    tau177 = (
        einsum("qm,wrm->wqr", tau89, tau176)
    )

    tau178 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau177, tau46)
    )

    tau179 = (
        einsum("qm,wrm->wqr", tau93, tau176)
    )

    tau180 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau179, tau50)
    )

    tau181 = (
        einsum("m,mab->ab", tau97, h.l.pvv)
    )

    tau182 = (
        einsum("bq,ab->qa", a.t2.x2, tau181)
    )

    tau183 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau182)
    )

    tau184 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau183, tau46, tau6)
    )

    tau185 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau182)
    )

    tau186 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau185, tau9)
    )

    tau187 = (
        einsum("bq,ab->qa", a.t2.x1, tau181)
    )

    tau188 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau187)
    )

    tau189 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau188, tau50, tau9)
    )

    tau190 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau187)
    )

    tau191 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau190, tau21, tau6)
    )

    tau192 = (
        einsum("wqi,mia->wqma", tau0, h.l.pov)
    )

    tau193 = (
        einsum("bw,bq,mba->wqma", d.t2.d2, b.t2.x2, h.l.pvv)
    )

    tau194 = (
        einsum("wqmb,wqma->wqab", tau192, tau193)
    )

    tau195 = (
        einsum("aq,bq,wrba->wqr", a.t2.x1, a.t2.x2, tau194)
    )

    tau196 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau195, tau6)
    )

    tau197 = (
        einsum("aq,bq,wrab->wqr", a.t2.x1, a.t2.x2, tau194)
    )

    tau198 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau197, tau9)
    )

    tau199 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau200 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau201 = (
        einsum("qma,qmi->qia", tau199, tau200)
    )

    tau202 = (
        einsum("aw,ar,wri,qia->wqr", d.t2.d1, b.t2.x1, tau2, tau201)
    )

    tau203 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau202, tau6)
    )

    tau204 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau205 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau206 = (
        einsum("qmi,qma->qia", tau204, tau205)
    )

    tau207 = (
        einsum("aw,ar,wri,qia->wqr", d.t2.d1, b.t2.x1, tau2, tau206)
    )

    tau208 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau207, tau9)
    )

    tau209 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau205)
    )

    tau210 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau28, tau46, tau6, tau209)
    )

    tau211 = (
        einsum("ai,wqa->wqi", a.t1, tau210)
    )

    tau212 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau205)
    )

    tau213 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau9, tau212)
    )

    tau214 = (
        einsum("ai,wqa->wqi", a.t1, tau213)
    )

    tau215 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau18, h.l.pvv)
    )

    tau216 = (
        einsum("ai,wqa->wqi", a.t1, tau215)
    )

    tau217 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau199)
    )

    tau218 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau14, tau50, tau9, tau217)
    )

    tau219 = (
        einsum("ai,wqa->wqi", a.t1, tau218)
    )

    tau220 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau23, h.l.pvv)
    )

    tau221 = (
        einsum("ai,wqa->wqi", a.t1, tau220)
    )

    tau222 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau199)
    )

    tau223 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau21, tau28, tau6, tau222)
    )

    tau224 = (
        einsum("ai,wqa->wqi", a.t1, tau223)
    )

    tau225 = (
        einsum("qmb,qma->qab", tau14, tau205)
    )

    tau226 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d2, b.t2.x2, tau225, tau31)
    )

    tau227 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau226, tau46)
    )

    tau228 = (
        einsum("qm,wrm->wqr", tau17, tau176)
    )

    tau229 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau228, tau46)
    )

    tau230 = (
        einsum("qma,qmb->qab", tau205, tau28)
    )

    tau231 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau230, tau31)
    )

    tau232 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau231)
    )

    tau233 = (
        einsum("qm,wrm->wqr", tau22, tau176)
    )

    tau234 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau233, tau50)
    )

    tau235 = (
        einsum("qma,qmb->qab", tau199, tau28)
    )

    tau236 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d2, b.t2.x2, tau235, tau31)
    )

    tau237 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau236, tau50)
    )

    tau238 = (
        einsum("qmb,qma->qab", tau14, tau199)
    )

    tau239 = (
        einsum("aw,ar,qab,wrb->wqr", d.t2.d1, b.t2.x1, tau238, tau31)
    )

    tau240 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau239)
    )

    tau241 = (
        einsum("bi,mab->mia", a.t1, h.l.pvv)
    )

    tau242 = (
        einsum("mia,mib->ab", tau241, h.l.pov)
    )

    tau243 = (
        einsum("bq,ab->qa", a.t2.x2, tau242)
    )

    tau244 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau243)
    )

    tau245 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau244, tau46, tau6)
    )

    tau246 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau243)
    )

    tau247 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau246, tau9)
    )

    tau248 = (
        einsum("bq,ab->qa", a.t2.x1, tau242)
    )

    tau249 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau248)
    )

    tau250 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau249, tau50, tau9)
    )

    tau251 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau248)
    )

    tau252 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau251, tau6)
    )

    tau253 = (
        einsum("wri,qmi->wqrm", tau0, tau200)
    )

    tau254 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau255 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau254, tau9, tau253)
    )

    tau256 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau257 = (
        einsum("qmi,qmj->qij", tau11, tau256)
    )

    tau258 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau0, tau257)
    )

    tau259 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau258)
    )

    tau260 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau18, h.l.poo)
    )

    tau261 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau262 = (
        einsum("qmi,qmj->qij", tau26, tau261)
    )

    tau263 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau0, tau262)
    )

    tau264 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau263)
    )

    tau265 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau23, h.l.poo)
    )

    tau266 = (
        einsum("wri,qmi->wqrm", tau0, tau204)
    )

    tau267 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau268 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau21, tau267, tau6, tau266)
    )

    tau269 = (
        einsum("qmi,qmj->qij", tau11, tau261)
    )

    tau270 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau2, tau269)
    )

    tau271 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau270, tau46)
    )

    tau272 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau273 = (
        einsum("qm,wrm->wqr", tau272, tau124)
    )

    tau274 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau273, tau46)
    )

    tau275 = (
        einsum("wri,qmi->wqrm", tau2, tau200)
    )

    tau276 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau267, tau46, tau6, tau275)
    )

    tau277 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau278 = (
        einsum("qm,wrm->wqr", tau277, tau124)
    )

    tau279 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau278, tau50)
    )

    tau280 = (
        einsum("wri,qmi->wqrm", tau2, tau204)
    )

    tau281 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau254, tau50, tau9, tau280)
    )

    tau282 = (
        einsum("qmj,qmi->qij", tau256, tau26)
    )

    tau283 = (
        einsum("iw,ir,wrj,qji->wqr", d.t2.d4, b.t2.x4, tau2, tau282)
    )

    tau284 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau283, tau50)
    )

    tau285 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau261)
    )

    tau286 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau287 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau21, tau286, tau46, tau285)
    )

    tau288 = (
        einsum("ai,wqa->wqi", a.t1, tau287)
    )

    tau289 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d4, b.t2.x4, tau256)
    )

    tau290 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau291 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau290, tau50, tau289)
    )

    tau292 = (
        einsum("ai,wqa->wqi", a.t1, tau291)
    )

    tau293 = (
        einsum("mji,wqmj->wqi", h.l.poo, tau71)
    )

    tau294 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau69)
    )

    tau295 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau254, tau50, tau294)
    )

    tau296 = (
        einsum("mki,mkj->ij", tau38, h.l.poo)
    )

    tau297 = (
        einsum("jq,ji->qi", a.t2.x4, tau296)
    )

    tau298 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau297)
    )

    tau299 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau298, tau46)
    )

    tau300 = (
        einsum("mkj,mki->ij", tau78, h.l.poo)
    )

    tau301 = (
        einsum("jq,ij->qi", a.t2.x3, tau300)
    )

    tau302 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau301, tau21, tau46, tau6)
    )

    tau303 = (
        einsum("jq,ij->qi", a.t2.x4, tau300)
    )

    tau304 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau303, tau13, tau50, tau9)
    )

    tau305 = (
        einsum("jq,ji->qi", a.t2.x3, tau296)
    )

    tau306 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau305)
    )

    tau307 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau306, tau50)
    )

    tau308 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau90, h.l.poo)
    )

    tau309 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau94, h.l.poo)
    )

    tau310 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau311 = (
        einsum("qm,wrm->wqr", tau310, tau124)
    )

    tau312 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau311, tau46)
    )

    tau313 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau314 = (
        einsum("qm,wrm->wqr", tau313, tau124)
    )

    tau315 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau314, tau50)
    )

    tau316 = (
        einsum("m,mij->ij", tau113, h.l.poo)
    )

    tau317 = (
        einsum("jq,ji->qi", a.t2.x4, tau316)
    )

    tau318 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau317)
    )

    tau319 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau318, tau46)
    )

    tau320 = (
        einsum("m,mij->ij", tau97, h.l.poo)
    )

    tau321 = (
        einsum("jq,ji->qi", a.t2.x3, tau320)
    )

    tau322 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau321, tau21, tau46, tau6)
    )

    tau323 = (
        einsum("jq,ji->qi", a.t2.x4, tau320)
    )

    tau324 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau323, tau13, tau50, tau9)
    )

    tau325 = (
        einsum("jq,ji->qi", a.t2.x3, tau316)
    )

    tau326 = (
        einsum("iw,ir,qi->wqr", d.t2.d4, b.t2.x4, tau325)
    )

    tau327 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau326, tau50)
    )

    tau328 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau176, h.l.pov)
    )

    tau329 = (
        einsum("ai,wqa->wqi", a.t1, tau328)
    )

    tau330 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau127, h.l.pvv)
    )

    tau331 = (
        einsum("ai,wqa->wqi", a.t1, tau330)
    )

    tau332 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau127, h.l.pov)
    )

    tau333 = (
        einsum("ai,wqa->wqi", a.t1, tau332)
    )

    tau334 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau21, tau267, tau46, tau285)
    )

    tau335 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau254, tau50, tau289)
    )

    tau336 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau124, h.l.poo)
    )

    tau337 = (
        einsum("wqi,wqm,mia->wqa", tau0, tau132, h.l.pov)
    )

    tau338 = (
        einsum("ai,wqa->wqi", a.t1, tau337)
    )

    tau339 = (
        einsum("wqj,wqm,mji->wqi", tau0, tau176, h.l.poo)
    )

    tau340 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau124, h.l.pvv)
    )

    tau341 = (
        einsum("ai,wqa->wqi", a.t1, tau340)
    )

    tau342 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau127, h.l.pvo)
    )

    tau343 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, h.l.pvo)
    )

    tau344 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d1, b.t2.x1, tau343, h.l.pvv)
    )

    tau345 = (
        einsum("ai,wqa->wqi", a.t1, tau344)
    )

    tau346 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau129, h.l.pvo)
    )

    tau347 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau343, h.l.pvo)
    )

    tau348 = (
        einsum("qm,rm->qr", tau310, tau89)
    )

    tau349 = (
        einsum("qp,wpr,wpr->wqr", tau348, tau13, tau9)
    )

    tau350 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau349, tau46)
    )

    tau351 = (
        einsum("rm,qm->qr", tau310, tau93)
    )

    tau352 = (
        einsum("pq,wpr,wpr->wqr", tau351, tau21, tau6)
    )

    tau353 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau352, tau46)
    )

    tau354 = (
        einsum("qm,rm->qr", tau313, tau89)
    )

    tau355 = (
        einsum("qp,wpr,wpr->wqr", tau354, tau13, tau9)
    )

    tau356 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau355, tau50)
    )

    tau357 = (
        einsum("qm,rm->qr", tau313, tau93)
    )

    tau358 = (
        einsum("qp,wpr,wpr->wqr", tau357, tau21, tau6)
    )

    tau359 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau358, tau50)
    )

    tau360 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau89, h.l.pov)
    )

    tau361 = (
        einsum("ar,qa->qr", a.t2.x2, tau360)
    )

    tau362 = (
        einsum("pq,wpr->wqr", tau361, tau46)
    )

    tau363 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau362, tau9)
    )

    tau364 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau89, h.l.pov)
    )

    tau365 = (
        einsum("ir,qi->qr", a.t2.x3, tau364)
    )

    tau366 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau365, tau21, tau46, tau6)
    )

    tau367 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau366)
    )

    tau368 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau93, h.l.pov)
    )

    tau369 = (
        einsum("ir,qi->qr", a.t2.x3, tau368)
    )

    tau370 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau369, tau21, tau46, tau6)
    )

    tau371 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau370)
    )

    tau372 = (
        einsum("aq,ra->qr", a.t2.x1, tau360)
    )

    tau373 = (
        einsum("qp,wpr->wqr", tau372, tau46)
    )

    tau374 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau373, tau6)
    )

    tau375 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau54)
    )

    tau376 = (
        einsum("pq,wpr,wpr->wqr", tau375, tau13, tau9)
    )

    tau377 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau376, tau46)
    )

    tau378 = (
        einsum("qm,rm->qr", tau17, tau310)
    )

    tau379 = (
        einsum("pq,wpr,wpr->wqr", tau378, tau13, tau6)
    )

    tau380 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau379, tau46)
    )

    tau381 = (
        einsum("pq,wpr->wqr", tau361, tau13)
    )

    tau382 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau381, tau46, tau6)
    )

    tau383 = (
        einsum("qm,rm->qr", tau272, tau89)
    )

    tau384 = (
        einsum("qp,wpr,wpr->wqr", tau383, tau13, tau9)
    )

    tau385 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau384, tau46)
    )

    tau386 = (
        einsum("qm,rm->qr", tau22, tau310)
    )

    tau387 = (
        einsum("pq,wpr,wpr->wqr", tau386, tau21, tau9)
    )

    tau388 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau387, tau46)
    )

    tau389 = (
        einsum("qmi,qma->qia", tau200, tau290)
    )

    tau390 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau389)
    )

    tau391 = (
        einsum("pq,wpr,wpr->wqr", tau390, tau21, tau6)
    )

    tau392 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau391, tau46)
    )

    tau393 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau313, h.l.pov)
    )

    tau394 = (
        einsum("ar,qa->qr", a.t2.x2, tau393)
    )

    tau395 = (
        einsum("pq,wpr->wqr", tau394, tau21)
    )

    tau396 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau395, tau46, tau6)
    )

    tau397 = (
        einsum("rm,qm->qr", tau272, tau93)
    )

    tau398 = (
        einsum("pq,wpr,wpr->wqr", tau397, tau21, tau6)
    )

    tau399 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau398, tau46)
    )

    tau400 = (
        einsum("ir,qi->qr", a.t2.x4, tau364)
    )

    tau401 = (
        einsum("pq,wpr->wqr", tau400, tau9)
    )

    tau402 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau401, tau46)
    )

    tau403 = (
        einsum("ir,qi->qr", a.t2.x4, tau368)
    )

    tau404 = (
        einsum("pq,wpr->wqr", tau403, tau6)
    )

    tau405 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau404, tau46)
    )

    tau406 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau400, tau13, tau50, tau9)
    )

    tau407 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau406)
    )

    tau408 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau403, tau13, tau50, tau9)
    )

    tau409 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau408)
    )

    tau410 = (
        einsum("qm,rm->qr", tau277, tau89)
    )

    tau411 = (
        einsum("qp,wpr,wpr->wqr", tau410, tau13, tau9)
    )

    tau412 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau411, tau50)
    )

    tau413 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau54)
    )

    tau414 = (
        einsum("qp,wpr,wpr->wqr", tau413, tau13, tau9)
    )

    tau415 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau414, tau50)
    )

    tau416 = (
        einsum("qp,wpr->wqr", tau372, tau13)
    )

    tau417 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau416, tau50, tau9)
    )

    tau418 = (
        einsum("rm,qm->qr", tau17, tau313)
    )

    tau419 = (
        einsum("qp,wpr,wpr->wqr", tau418, tau13, tau6)
    )

    tau420 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau419, tau50)
    )

    tau421 = (
        einsum("pq,wpr->wqr", tau394, tau50)
    )

    tau422 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau421, tau9)
    )

    tau423 = (
        einsum("pq,wpr->wqr", tau365, tau9)
    )

    tau424 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau423, tau50)
    )

    tau425 = (
        einsum("pq,wpr->wqr", tau369, tau6)
    )

    tau426 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau425, tau50)
    )

    tau427 = (
        einsum("ar,qa->qr", a.t2.x1, tau393)
    )

    tau428 = (
        einsum("pq,wpr->wqr", tau427, tau50)
    )

    tau429 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau428, tau6)
    )

    tau430 = (
        einsum("pq,wpr->wqr", tau427, tau21)
    )

    tau431 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau430, tau50, tau9)
    )

    tau432 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x3, tau62)
    )

    tau433 = (
        einsum("pq,wpr,wpr->wqr", tau432, tau21, tau6)
    )

    tau434 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau433, tau50)
    )

    tau435 = (
        einsum("rm,qm->qr", tau277, tau93)
    )

    tau436 = (
        einsum("pq,wpr,wpr->wqr", tau435, tau21, tau6)
    )

    tau437 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau436, tau50)
    )

    tau438 = (
        einsum("rm,qm->qr", tau22, tau313)
    )

    tau439 = (
        einsum("qp,wpr,wpr->wqr", tau438, tau21, tau9)
    )

    tau440 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau439, tau50)
    )

    tau441 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau272, h.l.pov)
    )

    tau442 = (
        einsum("ar,qa->qr", a.t2.x2, tau441)
    )

    tau443 = (
        einsum("pq,wpr->wqr", tau442, tau46)
    )

    tau444 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau443, tau9)
    )

    tau445 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau32)
    )

    tau446 = (
        einsum("qp,wpr,wpr->wqr", tau445, tau46, tau6)
    )

    tau447 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau446)
    )

    tau448 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau22, h.l.pov)
    )

    tau449 = (
        einsum("ir,qi->qr", a.t2.x3, tau448)
    )

    tau450 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau449, tau21, tau46, tau6)
    )

    tau451 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau450)
    )

    tau452 = (
        einsum("qmj,qmi->qij", tau200, tau26)
    )

    tau453 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau452)
    )

    tau454 = (
        einsum("qp,wpr,wpr->wqr", tau453, tau21, tau46)
    )

    tau455 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau454, tau6)
    )

    tau456 = (
        einsum("qmi,qmj->qij", tau11, tau204)
    )

    tau457 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau456)
    )

    tau458 = (
        einsum("qp,wpr,wpr->wqr", tau457, tau21, tau46)
    )

    tau459 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau458, tau9)
    )

    tau460 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau17, h.l.pov)
    )

    tau461 = (
        einsum("ir,qi->qr", a.t2.x3, tau460)
    )

    tau462 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau461, tau21, tau46, tau6)
    )

    tau463 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau462)
    )

    tau464 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau35)
    )

    tau465 = (
        einsum("qp,wpr,wpr->wqr", tau464, tau46, tau6)
    )

    tau466 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau465)
    )

    tau467 = (
        einsum("aq,ra->qr", a.t2.x1, tau441)
    )

    tau468 = (
        einsum("qp,wpr->wqr", tau467, tau46)
    )

    tau469 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau468, tau6)
    )

    tau470 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau389)
    )

    tau471 = (
        einsum("pq,wpr,wpr->wqr", tau470, tau13, tau6)
    )

    tau472 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau46, tau471)
    )

    tau473 = (
        einsum("pq,wpr->wqr", tau442, tau13)
    )

    tau474 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau46, tau473, tau6)
    )

    tau475 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau54)
    )

    tau476 = (
        einsum("qp,wpr,wpr->wqr", tau475, tau13, tau9)
    )

    tau477 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau476)
    )

    tau478 = (
        einsum("rm,qm->qr", tau17, tau272)
    )

    tau479 = (
        einsum("qp,wpr,wpr->wqr", tau478, tau13, tau6)
    )

    tau480 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau479)
    )

    tau481 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau35)
    )

    tau482 = (
        einsum("pq,wpr,wpr->wqr", tau481, tau21, tau9)
    )

    tau483 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau46, tau482)
    )

    tau484 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau22, h.l.pov)
    )

    tau485 = (
        einsum("ar,qa->qr", a.t2.x2, tau484)
    )

    tau486 = (
        einsum("pq,wpr->wqr", tau485, tau21)
    )

    tau487 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau46, tau486, tau6)
    )

    tau488 = (
        einsum("qm,rm->qr", tau22, tau272)
    )

    tau489 = (
        einsum("pq,wpr,wpr->wqr", tau488, tau21, tau9)
    )

    tau490 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau489)
    )

    tau491 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau62)
    )

    tau492 = (
        einsum("pq,wpr,wpr->wqr", tau491, tau21, tau6)
    )

    tau493 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau492)
    )

    tau494 = (
        einsum("ir,qi->qr", a.t2.x4, tau448)
    )

    tau495 = (
        einsum("pq,wpr->wqr", tau494, tau9)
    )

    tau496 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau46, tau495)
    )

    tau497 = (
        einsum("ir,qi->qr", a.t2.x4, tau460)
    )

    tau498 = (
        einsum("pq,wpr->wqr", tau497, tau6)
    )

    tau499 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau46, tau498)
    )

    tau500 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau494, tau13, tau50, tau9)
    )

    tau501 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau500)
    )

    tau502 = (
        einsum("qp,wpr,wpr->wqr", tau457, tau13, tau50)
    )

    tau503 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau502, tau6)
    )

    tau504 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau497, tau13, tau50, tau9)
    )

    tau505 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau504)
    )

    tau506 = (
        einsum("qp,wpr,wpr->wqr", tau453, tau13, tau50)
    )

    tau507 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau506, tau9)
    )

    tau508 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x4, tau54)
    )

    tau509 = (
        einsum("qp,wpr,wpr->wqr", tau508, tau13, tau9)
    )

    tau510 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau509)
    )

    tau511 = (
        einsum("rm,qm->qr", tau17, tau277)
    )

    tau512 = (
        einsum("qp,wpr,wpr->wqr", tau511, tau13, tau6)
    )

    tau513 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau512)
    )

    tau514 = (
        einsum("qp,wpr->wqr", tau467, tau13)
    )

    tau515 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau50, tau514, tau9)
    )

    tau516 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau32)
    )

    tau517 = (
        einsum("qp,wpr,wpr->wqr", tau516, tau13, tau6)
    )

    tau518 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau50, tau517)
    )

    tau519 = (
        einsum("qmi,qma->qia", tau204, tau290)
    )

    tau520 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau519)
    )

    tau521 = (
        einsum("pq,wpr,wpr->wqr", tau520, tau50, tau9)
    )

    tau522 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau521)
    )

    tau523 = (
        einsum("pq,wpr->wqr", tau485, tau50)
    )

    tau524 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau523, tau9)
    )

    tau525 = (
        einsum("pq,wpr->wqr", tau449, tau9)
    )

    tau526 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau50, tau525)
    )

    tau527 = (
        einsum("pq,wpr->wqr", tau461, tau6)
    )

    tau528 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau13, tau50, tau527)
    )

    tau529 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau519)
    )

    tau530 = (
        einsum("pq,wpr,wpr->wqr", tau529, tau21, tau6)
    )

    tau531 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau530)
    )

    tau532 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau35)
    )

    tau533 = (
        einsum("qp,wpr,wpr->wqr", tau532, tau21, tau9)
    )

    tau534 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau50, tau533)
    )

    tau535 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x4, tau35)
    )

    tau536 = (
        einsum("qp,wpr,wpr->wqr", tau535, tau50, tau9)
    )

    tau537 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau536)
    )

    tau538 = (
        einsum("ar,qa->qr", a.t2.x1, tau484)
    )

    tau539 = (
        einsum("pq,wpr->wqr", tau538, tau50)
    )

    tau540 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau21, tau539, tau6)
    )

    tau541 = (
        einsum("rm,qm->qr", tau22, tau277)
    )

    tau542 = (
        einsum("qp,wpr,wpr->wqr", tau541, tau21, tau9)
    )

    tau543 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau542)
    )

    tau544 = (
        einsum("pq,wpr->wqr", tau538, tau21)
    )

    tau545 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau50, tau544, tau9)
    )

    tau546 = (
        einsum("qm,wrm->wqr", tau89, tau132)
    )

    tau547 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau46, tau546)
    )

    tau548 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau90, h.l.pvo)
    )

    tau549 = (
        einsum("qm,wrm->wqr", tau93, tau132)
    )

    tau550 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau50, tau549)
    )

    tau551 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau94, h.l.pvo)
    )

    tau552 = (
        einsum("qma,qmi->qia", tau205, tau261)
    )

    tau553 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, tau552)
    )

    tau554 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau46, tau553)
    )

    tau555 = (
        einsum("qm,wrm->wqr", tau17, tau132)
    )

    tau556 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau46, tau555)
    )

    tau557 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau256, tau46, tau6, tau209)
    )

    tau558 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau261, tau9, tau212)
    )

    tau559 = (
        einsum("qma,qmi->qia", tau205, tau256)
    )

    tau560 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d4, b.t2.x1, b.t2.x4, tau559)
    )

    tau561 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau560)
    )

    tau562 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau18, h.l.pvo)
    )

    tau563 = (
        einsum("qm,wrm->wqr", tau22, tau132)
    )

    tau564 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau50, tau563)
    )

    tau565 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau261, tau50, tau9, tau217)
    )

    tau566 = (
        einsum("qma,qmi->qia", tau199, tau256)
    )

    tau567 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d4, b.t2.x2, b.t2.x4, tau566)
    )

    tau568 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau50, tau567)
    )

    tau569 = (
        einsum("qma,qmi->qia", tau199, tau261)
    )

    tau570 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d4, b.t2.x1, b.t2.x4, tau569)
    )

    tau571 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau21, tau570)
    )

    tau572 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d1, b.t2.x1, tau23, h.l.pvo)
    )

    tau573 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau21, tau256, tau6, tau222)
    )

    tau574 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau575 = (
        einsum("qma,qmb->qab", tau199, tau574)
    )

    tau576 = (
        einsum("aw,bw,ar,br,qab->wqr", d.t2.d1,
               d.t2.d2, b.t2.x1, b.t2.x2, tau575)
    )

    tau577 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau576, tau6)
    )

    tau578 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau579 = (
        einsum("qma,qmb->qab", tau205, tau578)
    )

    tau580 = (
        einsum("aw,bw,ar,br,qab->wqr", d.t2.d1,
               d.t2.d2, b.t2.x1, b.t2.x2, tau579)
    )

    tau581 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau580, tau9)
    )

    r23 = (
        einsum("iw,wqi->iq", d.t2.d3, tau7) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau10) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau16) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau20) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau25) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau30) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau34) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau37) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau42) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau45) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau49) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau53) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau56) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau59) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau61) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau64) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau66) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau68) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau73) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau77) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau82) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau84) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau86) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau88) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau92)
        - einsum("iw,wqi->iq", d.t2.d3, tau96)
        - einsum("iw,wqi->iq", d.t2.d3, tau101)
        - einsum("iw,wqi->iq", d.t2.d3, tau104)
        - einsum("iw,wqi->iq", d.t2.d3, tau106)
        - einsum("iw,wqi->iq", d.t2.d3, tau108)
        - einsum("iw,wqi->iq", d.t2.d3, tau110)
        - einsum("iw,wqi->iq", d.t2.d3, tau112)
        - einsum("iw,wqi->iq", d.t2.d3, tau117)
        - einsum("iw,wqi->iq", d.t2.d3, tau119)
        - einsum("iw,wqi->iq", d.t2.d3, tau121)
        - einsum("iw,wqi->iq", d.t2.d3, tau123)
        + einsum("iw,wqi->iq", d.t2.d3, tau126)
        + einsum("iw,wqi->iq", d.t2.d3, tau128)
        + einsum("iw,wqi->iq", d.t2.d3, tau131)
        - einsum("iw,wqi->iq", d.t2.d3, tau133)
        - einsum("iw,wqi->iq", d.t2.d3, tau134)
        + einsum("iw,wqi->iq", d.t2.d3, tau137) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau139) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau142) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau144) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau147) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau149) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau150) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau152) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau155) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau158) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau160) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau162) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau165) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau167) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau169) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau171) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau173)
        + einsum("iw,wqi->iq", d.t2.d3, tau175)
        + einsum("iw,wqi->iq", d.t2.d3, tau178)
        + einsum("iw,wqi->iq", d.t2.d3, tau180)
        + einsum("iw,wqi->iq", d.t2.d3, tau184)
        + einsum("iw,wqi->iq", d.t2.d3, tau186)
        + einsum("iw,wqi->iq", d.t2.d3, tau189)
        + einsum("iw,wqi->iq", d.t2.d3, tau191)
        - einsum("iw,wqi->iq", d.t2.d3, tau196) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau198) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau203) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau208) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau211) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau214) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau216) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau219) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau221) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau224) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau227) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau229) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau232) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau234) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau237) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau240) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau245) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau247) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau250) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau252) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau255) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau259) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau260) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau264) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau265) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau268) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau271) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau274) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau276) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau279) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau281) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau284) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau288) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau292) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau293) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau295) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau299) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau302) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau304) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau307) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau308)
        - einsum("iw,wqi->iq", d.t2.d3, tau309)
        - einsum("iw,wqi->iq", d.t2.d3, tau312)
        - einsum("iw,wqi->iq", d.t2.d3, tau315)
        - einsum("iw,wqi->iq", d.t2.d3, tau319)
        - einsum("iw,wqi->iq", d.t2.d3, tau322)
        - einsum("iw,wqi->iq", d.t2.d3, tau324)
        - einsum("iw,wqi->iq", d.t2.d3, tau327)
        - einsum("iw,wqi->iq", d.t2.d3, tau329)
        - einsum("iw,wqi->iq", d.t2.d3, tau331)
        + einsum("iw,wqi->iq", d.t2.d3, tau333)
        + einsum("iw,wqi->iq", d.t2.d3, tau334) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau335) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau336)
        - einsum("iw,wqi->iq", d.t2.d3, tau338)
        - einsum("iw,wqi->iq", d.t2.d3, tau339)
        - einsum("iw,wqi->iq", d.t2.d3, tau341)
        - einsum("iw,wqi->iq", d.t2.d3, tau342)
        + einsum("iw,wqi->iq", d.t2.d3, tau345)
        + einsum("iw,wqi->iq", d.t2.d3, tau346)
        + einsum("iw,wqi->iq", d.t2.d3, tau347)
        + einsum("iw,wqi->iq", d.t2.d3, tau350)
        + einsum("iw,wqi->iq", d.t2.d3, tau353)
        + einsum("iw,wqi->iq", d.t2.d3, tau356)
        + einsum("iw,wqi->iq", d.t2.d3, tau359)
        - einsum("iw,wqi->iq", d.t2.d3, tau363) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau367) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau371) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau374) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau377) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau380) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau382) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau385) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau388) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau392) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau396) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau399) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau402) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau405) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau407) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau409) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau412) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau415) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau417) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau420) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau422) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau424) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau426) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau429) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau431) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau434) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau437) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau440) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau444) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau447) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau451) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau455) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau459) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau463) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau466) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau469) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau472) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau474) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau477) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau480) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau483) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau487) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau490) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau493) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau496) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau499) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau501) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau503) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau505) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau507) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau510) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau513) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau515) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau518) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau522) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau524) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau526) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau528) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau531) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau534) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau537) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau540) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau543) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau545) / 4
        + einsum("iw,wqi->iq", d.t2.d3, tau547)
        + einsum("iw,wqi->iq", d.t2.d3, tau548)
        + einsum("iw,wqi->iq", d.t2.d3, tau550)
        + einsum("iw,wqi->iq", d.t2.d3, tau551)
        - einsum("iw,wqi->iq", d.t2.d3, tau554) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau556) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau557) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau558) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau561) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau562) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau564) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau565) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau568) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau571) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau572) / 2
        - einsum("iw,wqi->iq", d.t2.d3, tau573) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau577) / 2
        + einsum("iw,wqi->iq", d.t2.d3, tau581) / 2
    )

    return r23


def _rccsd_cpd_ls_t_true_calc_r24(h, a, b, d):
    tau0 = (
        einsum("aw,ai,aq->wqi", d.t2.d1, a.t1, b.t2.x1)
    )

    tau1 = (
        einsum("wqi,mia->wqma", tau0, h.l.pov)
    )

    tau2 = (
        einsum("aw,ai,aq->wqi", d.t2.d2, a.t1, b.t2.x2)
    )

    tau3 = (
        einsum("wqi,mia->wqma", tau2, h.l.pov)
    )

    tau4 = (
        einsum("wqma,wqmb->wqab", tau1, tau3)
    )

    tau5 = (
        einsum("aq,bq,wrba->wqr", a.t2.x1, a.t2.x2, tau4)
    )

    tau6 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x4, b.t2.x3)
    )

    tau7 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau5, tau6)
    )

    tau8 = (
        einsum("aq,bq,wrab->wqr", a.t2.x1, a.t2.x2, tau4)
    )

    tau9 = (
        einsum("iw,iq,ir->wqr", d.t2.d3, a.t2.x3, b.t2.x3)
    )

    tau10 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau8, tau9)
    )

    tau11 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau12 = (
        einsum("wri,qmi->wqrm", tau0, tau11)
    )

    tau13 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x1, b.t2.x2)
    )

    tau14 = (
        einsum("iq,mia->qma", a.t2.x3, h.l.pov)
    )

    tau15 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau6, tau12)
    )

    tau16 = (
        einsum("ai,wqa->wqi", a.t1, tau15)
    )

    tau17 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau18 = (
        einsum("wri,qmi->wqrm", tau0, tau17)
    )

    tau19 = (
        einsum("aw,aq,ar->wqr", d.t2.d2, a.t2.x2, b.t2.x2)
    )

    tau20 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau21 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau19, tau20, tau9, tau18)
    )

    tau22 = (
        einsum("ai,wqa->wqi", a.t1, tau21)
    )

    tau23 = (
        einsum("iw,ai,iq->wqa", d.t2.d3, a.t1, b.t2.x3)
    )

    tau24 = (
        einsum("qmi,qma->qia", tau11, tau20)
    )

    tau25 = (
        einsum("wri,wra,qia->wqr", tau0, tau23, tau24)
    )

    tau26 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau25)
    )

    tau27 = (
        einsum("wqi,wqa,mia->wqm", tau0, tau23, h.l.pov)
    )

    tau28 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau29 = (
        einsum("qm,wrm->wqr", tau28, tau27)
    )

    tau30 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau29)
    )

    tau31 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau32 = (
        einsum("qm,wrm->wqr", tau31, tau27)
    )

    tau33 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau32)
    )

    tau34 = (
        einsum("qma,qmi->qia", tau14, tau17)
    )

    tau35 = (
        einsum("wri,wra,qia->wqr", tau0, tau23, tau34)
    )

    tau36 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau35)
    )

    tau37 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau38 = (
        einsum("mji,mja->ia", tau37, h.l.pov)
    )

    tau39 = (
        einsum("aq,ia->qi", a.t2.x2, tau38)
    )

    tau40 = (
        einsum("qi,wri->wqr", tau39, tau0)
    )

    tau41 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau40, tau6)
    )

    tau42 = (
        einsum("aq,ia->qi", a.t2.x1, tau38)
    )

    tau43 = (
        einsum("qi,wri->wqr", tau42, tau0)
    )

    tau44 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau43, tau9)
    )

    tau45 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x1, b.t2.x1)
    )

    tau46 = (
        einsum("qmi,wri->wqrm", tau11, tau2)
    )

    tau47 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau20, tau45, tau9, tau46)
    )

    tau48 = (
        einsum("ai,wqa->wqi", a.t1, tau47)
    )

    tau49 = (
        einsum("rm,wrq,wrq->wqm", tau28, tau45, tau6)
    )

    tau50 = (
        einsum("wqi,wqm,mia->wqa", tau2, tau49, h.l.pov)
    )

    tau51 = (
        einsum("ai,wqa->wqi", a.t1, tau50)
    )

    tau52 = (
        einsum("aw,aq,ar->wqr", d.t2.d1, a.t2.x2, b.t2.x1)
    )

    tau53 = (
        einsum("rm,wrq,wrq->wqm", tau31, tau52, tau9)
    )

    tau54 = (
        einsum("wqi,wqm,mia->wqa", tau2, tau53, h.l.pov)
    )

    tau55 = (
        einsum("ai,wqa->wqi", a.t1, tau54)
    )

    tau56 = (
        einsum("qmi,wri->wqrm", tau17, tau2)
    )

    tau57 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau14, tau52, tau6, tau56)
    )

    tau58 = (
        einsum("ai,wqa->wqi", a.t1, tau57)
    )

    tau59 = (
        einsum("qmi,qma->qia", tau11, tau14)
    )

    tau60 = (
        einsum("wri,wra,qia->wqr", tau2, tau23, tau59)
    )

    tau61 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau45, tau60)
    )

    tau62 = (
        einsum("qmi,qma->qia", tau17, tau20)
    )

    tau63 = (
        einsum("wri,wra,qia->wqr", tau2, tau23, tau62)
    )

    tau64 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau52, tau63)
    )

    tau65 = (
        einsum("qi,wri->wqr", tau39, tau2)
    )

    tau66 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau45, tau65, tau9)
    )

    tau67 = (
        einsum("qi,wri->wqr", tau42, tau2)
    )

    tau68 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau52, tau6, tau67)
    )

    tau69 = (
        einsum("wqa,mia->wqmi", tau23, h.l.pov)
    )

    tau70 = (
        einsum("iq,wrmi->wqrm", a.t2.x3, tau69)
    )

    tau71 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau19, tau20, tau45, tau70)
    )

    tau72 = (
        einsum("ai,wqa->wqi", a.t1, tau71)
    )

    tau73 = (
        einsum("wqa,mia->wqmi", tau23, h.l.pov)
    )

    tau74 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau73)
    )

    tau75 = (
        einsum("ir,wrq,wrq,wrqm->wqmi", a.t2.x3, tau13, tau52, tau74)
    )

    tau76 = (
        einsum("mia,wqmi->wqa", h.l.pov, tau75)
    )

    tau77 = (
        einsum("ai,wqa->wqi", a.t1, tau76)
    )

    tau78 = (
        einsum("ai,mja->mij", a.t1, h.l.pov)
    )

    tau79 = (
        einsum("mji,mja->ia", tau78, h.l.pov)
    )

    tau80 = (
        einsum("ai,ja->ij", a.t1, tau79)
    )

    tau81 = (
        einsum("jq,ij->qi", a.t2.x4, tau80)
    )

    tau82 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau81, tau19, tau45, tau9)
    )

    tau83 = (
        einsum("jq,ij->qi", a.t2.x3, tau80)
    )

    tau84 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau83, tau13, tau52, tau6)
    )

    tau85 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau83)
    )

    tau86 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau45, tau85)
    )

    tau87 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau81)
    )

    tau88 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau52, tau87)
    )

    tau89 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau90 = (
        einsum("qm,wrm->wqr", tau89, tau27)
    )

    tau91 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau90)
    )

    tau92 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau93 = (
        einsum("qm,wrm->wqr", tau92, tau27)
    )

    tau94 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau93)
    )

    tau95 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau96 = (
        einsum("m,mia->ia", tau95, h.l.pov)
    )

    tau97 = (
        einsum("aq,ia->qi", a.t2.x2, tau96)
    )

    tau98 = (
        einsum("qi,wri->wqr", tau97, tau0)
    )

    tau99 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau6, tau98)
    )

    tau100 = (
        einsum("aq,ia->qi", a.t2.x1, tau96)
    )

    tau101 = (
        einsum("qi,wri->wqr", tau100, tau0)
    )

    tau102 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau101, tau19, tau9)
    )

    tau103 = (
        einsum("rm,wrq,wrq->wqm", tau89, tau45, tau9)
    )

    tau104 = (
        einsum("wqm,wqi,mia->wqa", tau103, tau2, h.l.pov)
    )

    tau105 = (
        einsum("ai,wqa->wqi", a.t1, tau104)
    )

    tau106 = (
        einsum("rm,wrq,wrq->wqm", tau92, tau52, tau6)
    )

    tau107 = (
        einsum("wqm,wqi,mia->wqa", tau106, tau2, h.l.pov)
    )

    tau108 = (
        einsum("ai,wqa->wqi", a.t1, tau107)
    )

    tau109 = (
        einsum("qi,wri->wqr", tau97, tau2)
    )

    tau110 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau109, tau45, tau9)
    )

    tau111 = (
        einsum("qi,wri->wqr", tau100, tau2)
    )

    tau112 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau111, tau52, tau6)
    )

    tau113 = (
        einsum("ai,mia->m", a.t1, h.l.pov)
    )

    tau114 = (
        einsum("m,mia->ia", tau113, h.l.pov)
    )

    tau115 = (
        einsum("ai,ja->ij", a.t1, tau114)
    )

    tau116 = (
        einsum("jq,ij->qi", a.t2.x4, tau115)
    )

    tau117 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau116, tau19, tau45, tau9)
    )

    tau118 = (
        einsum("jq,ij->qi", a.t2.x3, tau115)
    )

    tau119 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau118, tau13, tau52, tau6)
    )

    tau120 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau118)
    )

    tau121 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau120, tau19, tau45)
    )

    tau122 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau116)
    )

    tau123 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau122, tau13, tau52)
    )

    tau124 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d3, b.t2.x3, tau0, h.l.poo)
    )

    tau125 = (
        einsum("wqm,wqi,mia->wqa", tau124, tau2, h.l.pov)
    )

    tau126 = (
        einsum("ai,wqa->wqi", a.t1, tau125)
    )

    tau127 = (
        einsum("wqj,wqm,mji->wqi", tau2, tau27, h.l.poo)
    )

    tau128 = (
        einsum("aw,aq,wqb,mab->wqm", d.t2.d1, b.t2.x1, tau23, h.l.pvv)
    )

    tau129 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau128, h.l.pvv)
    )

    tau130 = (
        einsum("ai,wqa->wqi", a.t1, tau129)
    )

    tau131 = (
        einsum("iw,iq,wqj,mji->wqm", d.t2.d3, b.t2.x3, tau0, h.l.poo)
    )

    tau132 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau131, h.l.pvo)
    )

    tau133 = (
        einsum("aw,iw,aq,iq,mai->wqm", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, h.l.pvo)
    )

    tau134 = (
        einsum("wqm,wqj,mji->wqi", tau133, tau2, h.l.poo)
    )

    tau135 = (
        einsum("bq,ab->qa", a.t2.x2, h.f.vv)
    )

    tau136 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau135)
    )

    tau137 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau136, tau45, tau9)
    )

    tau138 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau135)
    )

    tau139 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau138, tau6)
    )

    tau140 = (
        einsum("bq,ab->qa", a.t2.x1, h.f.vv)
    )

    tau141 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau140)
    )

    tau142 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau141, tau52, tau6)
    )

    tau143 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau140)
    )

    tau144 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau143, tau19, tau9)
    )

    tau145 = (
        einsum("jq,ji->qi", a.t2.x4, h.f.oo)
    )

    tau146 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau145, tau19, tau45, tau9)
    )

    tau147 = (
        einsum("jq,ji->qi", a.t2.x3, h.f.oo)
    )

    tau148 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau147)
    )

    tau149 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau148, tau19, tau45)
    )

    tau150 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau145)
    )

    tau151 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau150, tau52)
    )

    tau152 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau147, tau13, tau52, tau6)
    )

    tau153 = (
        einsum("aq,ia->qi", a.t2.x2, h.f.ov)
    )

    tau154 = (
        einsum("qi,wri->wqr", tau153, tau0)
    )

    tau155 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau154, tau6)
    )

    tau156 = (
        einsum("aq,ia->qi", a.t2.x1, h.f.ov)
    )

    tau157 = (
        einsum("qi,wri->wqr", tau156, tau0)
    )

    tau158 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau157, tau19, tau9)
    )

    tau159 = (
        einsum("qi,wri->wqr", tau153, tau2)
    )

    tau160 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau159, tau45, tau9)
    )

    tau161 = (
        einsum("qi,wri->wqr", tau156, tau2)
    )

    tau162 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau161, tau52, tau6)
    )

    tau163 = (
        einsum("ai,ja->ij", a.t1, h.f.ov)
    )

    tau164 = (
        einsum("jq,ij->qi", a.t2.x4, tau163)
    )

    tau165 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau164, tau19, tau45, tau9)
    )

    tau166 = (
        einsum("jq,ij->qi", a.t2.x3, tau163)
    )

    tau167 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau166, tau13, tau52, tau6)
    )

    tau168 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau166)
    )

    tau169 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau168, tau19, tau45)
    )

    tau170 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau164)
    )

    tau171 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau170, tau52)
    )

    tau172 = (
        einsum("aq,mia->qmi", a.t2.x2, h.l.pov)
    )

    tau173 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau174 = (
        einsum("qmj,qmi->qij", tau172, tau173)
    )

    tau175 = (
        einsum("iw,ir,wrj,qij->wqr", d.t2.d3, b.t2.x3, tau0, tau174)
    )

    tau176 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau175)
    )

    tau177 = (
        einsum("qm,wrm->wqr", tau28, tau124)
    )

    tau178 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau177)
    )

    tau179 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau180 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau179, tau6, tau12)
    )

    tau181 = (
        einsum("qm,wrm->wqr", tau31, tau124)
    )

    tau182 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau181, tau19)
    )

    tau183 = (
        einsum("jq,mji->qmi", a.t2.x4, h.l.poo)
    )

    tau184 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau19, tau9, tau18)
    )

    tau185 = (
        einsum("aq,mia->qmi", a.t2.x1, h.l.pov)
    )

    tau186 = (
        einsum("jq,mji->qmi", a.t2.x3, h.l.poo)
    )

    tau187 = (
        einsum("qmj,qmi->qij", tau185, tau186)
    )

    tau188 = (
        einsum("iw,ir,wrj,qij->wqr", d.t2.d3, b.t2.x3, tau0, tau187)
    )

    tau189 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau188, tau19)
    )

    tau190 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau45, tau9, tau46)
    )

    tau191 = (
        einsum("qmj,qmi->qij", tau172, tau186)
    )

    tau192 = (
        einsum("iw,ir,qij,wrj->wqr", d.t2.d3, b.t2.x3, tau191, tau2)
    )

    tau193 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau192, tau45)
    )

    tau194 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x3, h.l.pov)
    )

    tau195 = (
        einsum("rm,wrq,wrq->wqm", tau194, tau45, tau6)
    )

    tau196 = (
        einsum("wqm,wqj,mji->wqi", tau195, tau2, h.l.poo)
    )

    tau197 = (
        einsum("qmi,qmj->qij", tau173, tau185)
    )

    tau198 = (
        einsum("iw,ir,qij,wrj->wqr", d.t2.d3, b.t2.x3, tau197, tau2)
    )

    tau199 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau198, tau52)
    )

    tau200 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x4, h.l.pov)
    )

    tau201 = (
        einsum("rm,wrq,wrq->wqm", tau200, tau52, tau9)
    )

    tau202 = (
        einsum("wqj,wqm,mji->wqi", tau2, tau201, h.l.poo)
    )

    tau203 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau179, tau52, tau6, tau56)
    )

    tau204 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau186)
    )

    tau205 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau19, tau20, tau45, tau204)
    )

    tau206 = (
        einsum("ai,wqa->wqi", a.t1, tau205)
    )

    tau207 = (
        einsum("iw,ir,qmi->wqrm", d.t2.d3, b.t2.x3, tau173)
    )

    tau208 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau52, tau207)
    )

    tau209 = (
        einsum("ai,wqa->wqi", a.t1, tau208)
    )

    tau210 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau19, tau45, tau70)
    )

    tau211 = (
        einsum("iq,wrmi->wqrm", a.t2.x4, tau69)
    )

    tau212 = (
        einsum("ir,wrq,wrq,wrqm->wqmi", a.t2.x3, tau13, tau52, tau211)
    )

    tau213 = (
        einsum("mji,wqmj->wqi", h.l.poo, tau212)
    )

    tau214 = (
        einsum("mki,mkj->ij", tau37, h.l.poo)
    )

    tau215 = (
        einsum("jq,ji->qi", a.t2.x4, tau214)
    )

    tau216 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau215, tau19, tau45, tau9)
    )

    tau217 = (
        einsum("mkj,mki->ij", tau78, h.l.poo)
    )

    tau218 = (
        einsum("jq,ij->qi", a.t2.x3, tau217)
    )

    tau219 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau218)
    )

    tau220 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau219, tau45)
    )

    tau221 = (
        einsum("jq,ij->qi", a.t2.x4, tau217)
    )

    tau222 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau221)
    )

    tau223 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau222, tau52)
    )

    tau224 = (
        einsum("jq,ji->qi", a.t2.x3, tau214)
    )

    tau225 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau224, tau13, tau52, tau6)
    )

    tau226 = (
        einsum("qm,wrm->wqr", tau89, tau124)
    )

    tau227 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau226)
    )

    tau228 = (
        einsum("qm,wrm->wqr", tau92, tau124)
    )

    tau229 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau228)
    )

    tau230 = (
        einsum("aq,iq,mia->qm", a.t2.x2, a.t2.x4, h.l.pov)
    )

    tau231 = (
        einsum("rm,wrq,wrq->wqm", tau230, tau45, tau9)
    )

    tau232 = (
        einsum("wqj,wqm,mji->wqi", tau2, tau231, h.l.poo)
    )

    tau233 = (
        einsum("aq,iq,mia->qm", a.t2.x1, a.t2.x3, h.l.pov)
    )

    tau234 = (
        einsum("rm,wrq,wrq->wqm", tau233, tau52, tau6)
    )

    tau235 = (
        einsum("wqj,wqm,mji->wqi", tau2, tau234, h.l.poo)
    )

    tau236 = (
        einsum("m,mij->ij", tau113, h.l.poo)
    )

    tau237 = (
        einsum("jq,ji->qi", a.t2.x4, tau236)
    )

    tau238 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau237, tau19, tau45, tau9)
    )

    tau239 = (
        einsum("m,mij->ij", tau95, h.l.poo)
    )

    tau240 = (
        einsum("jq,ji->qi", a.t2.x3, tau239)
    )

    tau241 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau240)
    )

    tau242 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau241, tau45)
    )

    tau243 = (
        einsum("jq,ji->qi", a.t2.x4, tau239)
    )

    tau244 = (
        einsum("iw,ir,qi->wqr", d.t2.d3, b.t2.x3, tau243)
    )

    tau245 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau244, tau52)
    )

    tau246 = (
        einsum("jq,ji->qi", a.t2.x3, tau236)
    )

    tau247 = (
        einsum("ri,wrq,wrq,wrq->wqi", tau246, tau13, tau52, tau6)
    )

    tau248 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau103, h.l.pvv)
    )

    tau249 = (
        einsum("ai,wqa->wqi", a.t1, tau248)
    )

    tau250 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau106, h.l.pvv)
    )

    tau251 = (
        einsum("ai,wqa->wqi", a.t1, tau250)
    )

    tau252 = (
        einsum("qm,wrm->wqr", tau89, tau128)
    )

    tau253 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau252)
    )

    tau254 = (
        einsum("qm,wrm->wqr", tau92, tau128)
    )

    tau255 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau254)
    )

    tau256 = (
        einsum("m,mab->ab", tau95, h.l.pvv)
    )

    tau257 = (
        einsum("bq,ab->qa", a.t2.x2, tau256)
    )

    tau258 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau257)
    )

    tau259 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau258, tau45, tau9)
    )

    tau260 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau257)
    )

    tau261 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau260, tau6)
    )

    tau262 = (
        einsum("bq,ab->qa", a.t2.x1, tau256)
    )

    tau263 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau262)
    )

    tau264 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau263, tau52, tau6)
    )

    tau265 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau262)
    )

    tau266 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau265, tau9)
    )

    tau267 = (
        einsum("wqi,mia->wqma", tau0, h.l.pov)
    )

    tau268 = (
        einsum("bw,bq,mba->wqma", d.t2.d2, b.t2.x2, h.l.pvv)
    )

    tau269 = (
        einsum("wqmb,wqma->wqab", tau267, tau268)
    )

    tau270 = (
        einsum("aq,bq,wrab->wqr", a.t2.x1, a.t2.x2, tau269)
    )

    tau271 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau270, tau6)
    )

    tau272 = (
        einsum("aq,bq,wrba->wqr", a.t2.x1, a.t2.x2, tau269)
    )

    tau273 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau272, tau9)
    )

    tau274 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau275 = (
        einsum("qmi,qma->qia", tau185, tau274)
    )

    tau276 = (
        einsum("aw,ar,wri,qia->wqr", d.t2.d1, b.t2.x1, tau2, tau275)
    )

    tau277 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau276, tau6)
    )

    tau278 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau279 = (
        einsum("qmi,qma->qia", tau172, tau278)
    )

    tau280 = (
        einsum("aw,ar,wri,qia->wqr", d.t2.d1, b.t2.x1, tau2, tau279)
    )

    tau281 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau280, tau9)
    )

    tau282 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau274)
    )

    tau283 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau20, tau45, tau9, tau282)
    )

    tau284 = (
        einsum("ai,wqa->wqi", a.t1, tau283)
    )

    tau285 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau49, h.l.pvv)
    )

    tau286 = (
        einsum("ai,wqa->wqi", a.t1, tau285)
    )

    tau287 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau274)
    )

    tau288 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau13, tau14, tau6, tau287)
    )

    tau289 = (
        einsum("ai,wqa->wqi", a.t1, tau288)
    )

    tau290 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau53, h.l.pvv)
    )

    tau291 = (
        einsum("ai,wqa->wqi", a.t1, tau290)
    )

    tau292 = (
        einsum("aw,ar,qma->wqrm", d.t2.d2, b.t2.x2, tau278)
    )

    tau293 = (
        einsum("rma,wrq,wrq,wrqm->wqa", tau14, tau52, tau6, tau292)
    )

    tau294 = (
        einsum("ai,wqa->wqi", a.t1, tau293)
    )

    tau295 = (
        einsum("aw,ar,qma->wqrm", d.t2.d1, b.t2.x1, tau278)
    )

    tau296 = (
        einsum("wrq,rma,wrq,wrqm->wqa", tau19, tau20, tau9, tau295)
    )

    tau297 = (
        einsum("ai,wqa->wqi", a.t1, tau296)
    )

    tau298 = (
        einsum("qmb,qma->qab", tau14, tau274)
    )

    tau299 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau23, tau298)
    )

    tau300 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau299, tau45)
    )

    tau301 = (
        einsum("qmb,qma->qab", tau20, tau274)
    )

    tau302 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d1, b.t2.x1, tau23, tau301)
    )

    tau303 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau302)
    )

    tau304 = (
        einsum("qm,wrm->wqr", tau28, tau128)
    )

    tau305 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau304)
    )

    tau306 = (
        einsum("qmb,qma->qab", tau20, tau278)
    )

    tau307 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d2, b.t2.x2, tau23, tau306)
    )

    tau308 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau307, tau52)
    )

    tau309 = (
        einsum("qm,wrm->wqr", tau31, tau128)
    )

    tau310 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau309)
    )

    tau311 = (
        einsum("qmb,qma->qab", tau14, tau278)
    )

    tau312 = (
        einsum("aw,ar,wrb,qab->wqr", d.t2.d1, b.t2.x1, tau23, tau311)
    )

    tau313 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau312)
    )

    tau314 = (
        einsum("bi,mab->mia", a.t1, h.l.pvv)
    )

    tau315 = (
        einsum("mia,mib->ab", tau314, h.l.pov)
    )

    tau316 = (
        einsum("bq,ab->qa", a.t2.x2, tau315)
    )

    tau317 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau316)
    )

    tau318 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau317, tau45, tau9)
    )

    tau319 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau316)
    )

    tau320 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau319, tau6)
    )

    tau321 = (
        einsum("bq,ab->qa", a.t2.x1, tau315)
    )

    tau322 = (
        einsum("aw,ar,qa->wqr", d.t2.d2, b.t2.x2, tau321)
    )

    tau323 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau322, tau52, tau6)
    )

    tau324 = (
        einsum("aw,ar,qa->wqr", d.t2.d1, b.t2.x1, tau321)
    )

    tau325 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau324, tau9)
    )

    tau326 = (
        einsum("wqi,wqa,mia->wqm", tau0, tau23, h.l.pov)
    )

    tau327 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau326, h.l.pvv)
    )

    tau328 = (
        einsum("ai,wqa->wqi", a.t1, tau327)
    )

    tau329 = (
        einsum("wqm,wqi,mia->wqa", tau128, tau2, h.l.pov)
    )

    tau330 = (
        einsum("ai,wqa->wqi", a.t1, tau329)
    )

    tau331 = (
        einsum("wqi,wqm,mia->wqa", tau2, tau27, h.l.pov)
    )

    tau332 = (
        einsum("ai,wqa->wqi", a.t1, tau331)
    )

    tau333 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau19, tau45, tau204)
    )

    tau334 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau179, tau52, tau207)
    )

    tau335 = (
        einsum("wqm,wqj,mji->wqi", tau124, tau2, h.l.poo)
    )

    tau336 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau131, h.l.pvv)
    )

    tau337 = (
        einsum("ai,wqa->wqi", a.t1, tau336)
    )

    tau338 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau326, h.l.pvo)
    )

    tau339 = (
        einsum("wqm,wqi,mia->wqa", tau133, tau2, h.l.pov)
    )

    tau340 = (
        einsum("ai,wqa->wqi", a.t1, tau339)
    )

    tau341 = (
        einsum("wqm,wqj,mji->wqi", tau128, tau2, h.l.poo)
    )

    tau342 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau133, h.l.pvo)
    )

    tau343 = (
        einsum("bw,bq,wqm,mba->wqa", d.t2.d2, b.t2.x2, tau133, h.l.pvv)
    )

    tau344 = (
        einsum("ai,wqa->wqi", a.t1, tau343)
    )

    tau345 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau128, h.l.pvo)
    )

    tau346 = (
        einsum("qm,rm->qr", tau230, tau89)
    )

    tau347 = (
        einsum("pq,wpr,wpr->wqr", tau346, tau45, tau9)
    )

    tau348 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau347)
    )

    tau349 = (
        einsum("rm,qm->qr", tau230, tau92)
    )

    tau350 = (
        einsum("qp,wpr,wpr->wqr", tau349, tau45, tau9)
    )

    tau351 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau350)
    )

    tau352 = (
        einsum("qm,rm->qr", tau233, tau89)
    )

    tau353 = (
        einsum("pq,wpr,wpr->wqr", tau352, tau52, tau6)
    )

    tau354 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau353)
    )

    tau355 = (
        einsum("qm,rm->qr", tau233, tau92)
    )

    tau356 = (
        einsum("pq,wpr,wpr->wqr", tau355, tau52, tau6)
    )

    tau357 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau356)
    )

    tau358 = (
        einsum("qm,rm->qr", tau194, tau89)
    )

    tau359 = (
        einsum("pq,wpr,wpr->wqr", tau358, tau45, tau6)
    )

    tau360 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau359)
    )

    tau361 = (
        einsum("rm,qm->qr", tau230, tau28)
    )

    tau362 = (
        einsum("qp,wpr,wpr->wqr", tau361, tau45, tau9)
    )

    tau363 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau362)
    )

    tau364 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau24)
    )

    tau365 = (
        einsum("qp,wpr,wpr->wqr", tau364, tau45, tau9)
    )

    tau366 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau365)
    )

    tau367 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau89, h.l.pov)
    )

    tau368 = (
        einsum("ar,qa->qr", a.t2.x2, tau367)
    )

    tau369 = (
        einsum("pq,wpr->wqr", tau368, tau45)
    )

    tau370 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau369, tau6)
    )

    tau371 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau89, h.l.pov)
    )

    tau372 = (
        einsum("ir,qi->qr", a.t2.x4, tau371)
    )

    tau373 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau372, tau19, tau45, tau9)
    )

    tau374 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau373)
    )

    tau375 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau92, h.l.pov)
    )

    tau376 = (
        einsum("ir,qi->qr", a.t2.x4, tau375)
    )

    tau377 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau376, tau19, tau45, tau9)
    )

    tau378 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau377)
    )

    tau379 = (
        einsum("rm,qm->qr", tau230, tau31)
    )

    tau380 = (
        einsum("qp,wpr,wpr->wqr", tau379, tau45, tau9)
    )

    tau381 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau380)
    )

    tau382 = (
        einsum("iq,mia->qma", a.t2.x4, h.l.pov)
    )

    tau383 = (
        einsum("qmi,qma->qia", tau172, tau382)
    )

    tau384 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau383)
    )

    tau385 = (
        einsum("qp,wpr,wpr->wqr", tau384, tau45, tau9)
    )

    tau386 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau385)
    )

    tau387 = (
        einsum("aq,ra->qr", a.t2.x1, tau367)
    )

    tau388 = (
        einsum("qp,wpr->wqr", tau387, tau45)
    )

    tau389 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau388, tau9)
    )

    tau390 = (
        einsum("rm,qm->qr", tau194, tau92)
    )

    tau391 = (
        einsum("qp,wpr,wpr->wqr", tau390, tau45, tau6)
    )

    tau392 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau391)
    )

    tau393 = (
        einsum("pq,wpr->wqr", tau368, tau13)
    )

    tau394 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau393, tau45, tau9)
    )

    tau395 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau233, h.l.pov)
    )

    tau396 = (
        einsum("ar,qa->qr", a.t2.x2, tau395)
    )

    tau397 = (
        einsum("pq,wpr->wqr", tau396, tau19)
    )

    tau398 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau397, tau45, tau9)
    )

    tau399 = (
        einsum("ir,qi->qr", a.t2.x3, tau371)
    )

    tau400 = (
        einsum("pq,wpr->wqr", tau399, tau9)
    )

    tau401 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau400, tau45)
    )

    tau402 = (
        einsum("ir,qi->qr", a.t2.x3, tau375)
    )

    tau403 = (
        einsum("pq,wpr->wqr", tau402, tau6)
    )

    tau404 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau403, tau45)
    )

    tau405 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau399, tau13, tau52, tau6)
    )

    tau406 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau405)
    )

    tau407 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau402, tau13, tau52, tau6)
    )

    tau408 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau407)
    )

    tau409 = (
        einsum("qp,wpr->wqr", tau387, tau13)
    )

    tau410 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau409, tau52, tau6)
    )

    tau411 = (
        einsum("qm,rm->qr", tau200, tau89)
    )

    tau412 = (
        einsum("pq,wpr,wpr->wqr", tau411, tau52, tau9)
    )

    tau413 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau412)
    )

    tau414 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau24)
    )

    tau415 = (
        einsum("pq,wpr,wpr->wqr", tau414, tau52, tau6)
    )

    tau416 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau415)
    )

    tau417 = (
        einsum("pq,wpr->wqr", tau396, tau52)
    )

    tau418 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau417, tau6)
    )

    tau419 = (
        einsum("qm,rm->qr", tau233, tau28)
    )

    tau420 = (
        einsum("pq,wpr,wpr->wqr", tau419, tau52, tau6)
    )

    tau421 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau420)
    )

    tau422 = (
        einsum("pq,wpr->wqr", tau372, tau9)
    )

    tau423 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau422, tau52)
    )

    tau424 = (
        einsum("pq,wpr->wqr", tau376, tau6)
    )

    tau425 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau424, tau52)
    )

    tau426 = (
        einsum("qm,rm->qr", tau233, tau31)
    )

    tau427 = (
        einsum("pq,wpr,wpr->wqr", tau426, tau52, tau6)
    )

    tau428 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau427)
    )

    tau429 = (
        einsum("rm,qm->qr", tau200, tau92)
    )

    tau430 = (
        einsum("qp,wpr,wpr->wqr", tau429, tau52, tau9)
    )

    tau431 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau430)
    )

    tau432 = (
        einsum("ar,qa->qr", a.t2.x1, tau395)
    )

    tau433 = (
        einsum("pq,wpr->wqr", tau432, tau52)
    )

    tau434 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau433, tau9)
    )

    tau435 = (
        einsum("pq,wpr->wqr", tau432, tau19)
    )

    tau436 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau435, tau52, tau6)
    )

    tau437 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x3, tau34)
    )

    tau438 = (
        einsum("qp,wpr,wpr->wqr", tau437, tau52, tau6)
    )

    tau439 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau438)
    )

    tau440 = (
        einsum("iq,qm,mia->qa", a.t2.x4, tau194, h.l.pov)
    )

    tau441 = (
        einsum("ar,qa->qr", a.t2.x2, tau440)
    )

    tau442 = (
        einsum("pq,wpr->wqr", tau441, tau45)
    )

    tau443 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau442, tau6)
    )

    tau444 = (
        einsum("qm,rm->qr", tau194, tau28)
    )

    tau445 = (
        einsum("pq,wpr,wpr->wqr", tau444, tau45, tau6)
    )

    tau446 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau445)
    )

    tau447 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau24)
    )

    tau448 = (
        einsum("pq,wpr,wpr->wqr", tau447, tau45, tau6)
    )

    tau449 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau448)
    )

    tau450 = (
        einsum("aq,iq,ria->qr", a.t2.x2, a.t2.x3, tau383)
    )

    tau451 = (
        einsum("qp,wpr,wpr->wqr", tau450, tau45, tau9)
    )

    tau452 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau451)
    )

    tau453 = (
        einsum("aq,qm,mia->qi", a.t2.x2, tau31, h.l.pov)
    )

    tau454 = (
        einsum("ir,qi->qr", a.t2.x4, tau453)
    )

    tau455 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau454, tau19, tau45, tau9)
    )

    tau456 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau455)
    )

    tau457 = (
        einsum("qmi,qmj->qij", tau11, tau185)
    )

    tau458 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau457)
    )

    tau459 = (
        einsum("qp,wpr,wpr->wqr", tau458, tau19, tau45)
    )

    tau460 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau459, tau6)
    )

    tau461 = (
        einsum("aq,qm,mia->qi", a.t2.x1, tau28, h.l.pov)
    )

    tau462 = (
        einsum("ir,qi->qr", a.t2.x4, tau461)
    )

    tau463 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau462, tau19, tau45, tau9)
    )

    tau464 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau463)
    )

    tau465 = (
        einsum("qmi,qmj->qij", tau17, tau172)
    )

    tau466 = (
        einsum("ir,jr,qij->qr", a.t2.x3, a.t2.x4, tau465)
    )

    tau467 = (
        einsum("qp,wpr,wpr->wqr", tau466, tau19, tau45)
    )

    tau468 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau467, tau9)
    )

    tau469 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x4, tau62)
    )

    tau470 = (
        einsum("qp,wpr,wpr->wqr", tau469, tau45, tau9)
    )

    tau471 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau470)
    )

    tau472 = (
        einsum("rm,qm->qr", tau194, tau31)
    )

    tau473 = (
        einsum("qp,wpr,wpr->wqr", tau472, tau45, tau6)
    )

    tau474 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau473)
    )

    tau475 = (
        einsum("aq,ra->qr", a.t2.x1, tau440)
    )

    tau476 = (
        einsum("qp,wpr->wqr", tau475, tau45)
    )

    tau477 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau476, tau9)
    )

    tau478 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau34)
    )

    tau479 = (
        einsum("qp,wpr,wpr->wqr", tau478, tau45, tau6)
    )

    tau480 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau479)
    )

    tau481 = (
        einsum("pq,wpr->wqr", tau441, tau13)
    )

    tau482 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau45, tau481, tau9)
    )

    tau483 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau59)
    )

    tau484 = (
        einsum("pq,wpr,wpr->wqr", tau483, tau13, tau6)
    )

    tau485 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau45, tau484)
    )

    tau486 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau62)
    )

    tau487 = (
        einsum("pq,wpr,wpr->wqr", tau486, tau19, tau9)
    )

    tau488 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau45, tau487)
    )

    tau489 = (
        einsum("iq,qm,mia->qa", a.t2.x3, tau31, h.l.pov)
    )

    tau490 = (
        einsum("ar,qa->qr", a.t2.x2, tau489)
    )

    tau491 = (
        einsum("pq,wpr->wqr", tau490, tau19)
    )

    tau492 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau45, tau491, tau9)
    )

    tau493 = (
        einsum("ir,qi->qr", a.t2.x3, tau453)
    )

    tau494 = (
        einsum("pq,wpr->wqr", tau493, tau9)
    )

    tau495 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau45, tau494)
    )

    tau496 = (
        einsum("ir,qi->qr", a.t2.x3, tau461)
    )

    tau497 = (
        einsum("pq,wpr->wqr", tau496, tau6)
    )

    tau498 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau45, tau497)
    )

    tau499 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau493, tau13, tau52, tau6)
    )

    tau500 = (
        einsum("ir,wrq->wqi", a.t2.x3, tau499)
    )

    tau501 = (
        einsum("qp,wpr,wpr->wqr", tau466, tau13, tau52)
    )

    tau502 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau501, tau6)
    )

    tau503 = (
        einsum("qp,wpr,wpr->wqr", tau458, tau13, tau52)
    )

    tau504 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau503, tau9)
    )

    tau505 = (
        einsum("qp,wpr,wpr,wpr->wqr", tau496, tau13, tau52, tau6)
    )

    tau506 = (
        einsum("ir,wrq->wqi", a.t2.x4, tau505)
    )

    tau507 = (
        einsum("qmi,qma->qia", tau185, tau382)
    )

    tau508 = (
        einsum("ar,ir,qia->qr", a.t2.x2, a.t2.x3, tau507)
    )

    tau509 = (
        einsum("qp,wpr,wpr->wqr", tau508, tau13, tau6)
    )

    tau510 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau509, tau52)
    )

    tau511 = (
        einsum("qp,wpr->wqr", tau475, tau13)
    )

    tau512 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau511, tau52, tau6)
    )

    tau513 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x4, tau24)
    )

    tau514 = (
        einsum("pq,wpr,wpr->wqr", tau513, tau52, tau9)
    )

    tau515 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau514)
    )

    tau516 = (
        einsum("pq,wpr->wqr", tau490, tau52)
    )

    tau517 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau516, tau6)
    )

    tau518 = (
        einsum("qm,rm->qr", tau200, tau28)
    )

    tau519 = (
        einsum("pq,wpr,wpr->wqr", tau518, tau52, tau9)
    )

    tau520 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau519)
    )

    tau521 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau59)
    )

    tau522 = (
        einsum("pq,wpr,wpr->wqr", tau521, tau52, tau6)
    )

    tau523 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau522)
    )

    tau524 = (
        einsum("pq,wpr->wqr", tau454, tau9)
    )

    tau525 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau52, tau524)
    )

    tau526 = (
        einsum("pq,wpr->wqr", tau462, tau6)
    )

    tau527 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau13, tau52, tau526)
    )

    tau528 = (
        einsum("qm,rm->qr", tau200, tau31)
    )

    tau529 = (
        einsum("pq,wpr,wpr->wqr", tau528, tau52, tau9)
    )

    tau530 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau529)
    )

    tau531 = (
        einsum("ar,ir,qia->qr", a.t2.x1, a.t2.x4, tau62)
    )

    tau532 = (
        einsum("pq,wpr,wpr->wqr", tau531, tau19, tau9)
    )

    tau533 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau52, tau532)
    )

    tau534 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau62)
    )

    tau535 = (
        einsum("pq,wpr,wpr->wqr", tau534, tau52, tau6)
    )

    tau536 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau535)
    )

    tau537 = (
        einsum("aq,iq,ria->qr", a.t2.x1, a.t2.x3, tau507)
    )

    tau538 = (
        einsum("qp,wpr,wpr->wqr", tau537, tau52, tau9)
    )

    tau539 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau538)
    )

    tau540 = (
        einsum("ar,qa->qr", a.t2.x1, tau489)
    )

    tau541 = (
        einsum("pq,wpr->wqr", tau540, tau52)
    )

    tau542 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x4, tau19, tau541, tau9)
    )

    tau543 = (
        einsum("pq,wpr->wqr", tau540, tau19)
    )

    tau544 = (
        einsum("ir,wrq,wrq,wrq->wqi", a.t2.x3, tau52, tau543, tau6)
    )

    tau545 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau103, h.l.pvo)
    )

    tau546 = (
        einsum("qm,wrm->wqr", tau89, tau133)
    )

    tau547 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau546)
    )

    tau548 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau106, h.l.pvo)
    )

    tau549 = (
        einsum("qm,wrm->wqr", tau92, tau133)
    )

    tau550 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau549)
    )

    tau551 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau45, tau9, tau282)
    )

    tau552 = (
        einsum("qmi,qma->qia", tau179, tau274)
    )

    tau553 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d3, b.t2.x2, b.t2.x3, tau552)
    )

    tau554 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau45, tau553)
    )

    tau555 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau49, h.l.pvo)
    )

    tau556 = (
        einsum("qmi,qma->qia", tau183, tau274)
    )

    tau557 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, tau556)
    )

    tau558 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau13, tau557)
    )

    tau559 = (
        einsum("qm,wrm->wqr", tau28, tau133)
    )

    tau560 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau13, tau559)
    )

    tau561 = (
        einsum("wrq,rmi,wrq,wrqm->wqi", tau13, tau179, tau6, tau287)
    )

    tau562 = (
        einsum("qmi,qma->qia", tau183, tau278)
    )

    tau563 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d2,
               d.t2.d3, b.t2.x2, b.t2.x3, tau562)
    )

    tau564 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau52, tau563)
    )

    tau565 = (
        einsum("aw,aq,wqm,mai->wqi", d.t2.d2, b.t2.x2, tau53, h.l.pvo)
    )

    tau566 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau179, tau52, tau6, tau292)
    )

    tau567 = (
        einsum("qm,wrm->wqr", tau31, tau133)
    )

    tau568 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau19, tau567)
    )

    tau569 = (
        einsum("rmi,wrq,wrq,wrqm->wqi", tau183, tau19, tau9, tau295)
    )

    tau570 = (
        einsum("qmi,qma->qia", tau179, tau278)
    )

    tau571 = (
        einsum("aw,iw,ar,ir,qia->wqr", d.t2.d1,
               d.t2.d3, b.t2.x1, b.t2.x3, tau570)
    )

    tau572 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau19, tau571)
    )

    tau573 = (
        einsum("bq,mab->qma", a.t2.x1, h.l.pvv)
    )

    tau574 = (
        einsum("qma,qmb->qab", tau274, tau573)
    )

    tau575 = (
        einsum("aw,bw,ar,br,qab->wqr", d.t2.d1,
               d.t2.d2, b.t2.x1, b.t2.x2, tau574)
    )

    tau576 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x3, tau575, tau6)
    )

    tau577 = (
        einsum("bq,mab->qma", a.t2.x2, h.l.pvv)
    )

    tau578 = (
        einsum("qma,qmb->qab", tau278, tau577)
    )

    tau579 = (
        einsum("aw,bw,ar,br,qab->wqr", d.t2.d1,
               d.t2.d2, b.t2.x1, b.t2.x2, tau578)
    )

    tau580 = (
        einsum("ir,wrq,wrq->wqi", a.t2.x4, tau579, tau9)
    )

    r24 = (
        einsum("iw,wqi->iq", d.t2.d4, tau7) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau10) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau16) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau22) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau26) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau30) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau33) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau36) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau41) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau44) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau48) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau51) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau55) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau58) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau61) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau64) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau66) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau68) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau72) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau77) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau82) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau84) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau86) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau88) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau91)
        - einsum("iw,wqi->iq", d.t2.d4, tau94)
        - einsum("iw,wqi->iq", d.t2.d4, tau99)
        - einsum("iw,wqi->iq", d.t2.d4, tau102)
        - einsum("iw,wqi->iq", d.t2.d4, tau105)
        - einsum("iw,wqi->iq", d.t2.d4, tau108)
        - einsum("iw,wqi->iq", d.t2.d4, tau110)
        - einsum("iw,wqi->iq", d.t2.d4, tau112)
        - einsum("iw,wqi->iq", d.t2.d4, tau117)
        - einsum("iw,wqi->iq", d.t2.d4, tau119)
        - einsum("iw,wqi->iq", d.t2.d4, tau121)
        - einsum("iw,wqi->iq", d.t2.d4, tau123)
        + einsum("iw,wqi->iq", d.t2.d4, tau126)
        + einsum("iw,wqi->iq", d.t2.d4, tau127)
        + einsum("iw,wqi->iq", d.t2.d4, tau130)
        - einsum("iw,wqi->iq", d.t2.d4, tau132)
        - einsum("iw,wqi->iq", d.t2.d4, tau134)
        + einsum("iw,wqi->iq", d.t2.d4, tau137) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau139) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau142) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau144) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau146) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau149) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau151) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau152) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau155) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau158) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau160) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau162) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau165) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau167) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau169) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau171) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau176) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau178) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau180) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau182) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau184) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau189) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau190) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau193) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau196) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau199) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau202) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau203) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau206) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau209) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau210) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau213) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau216) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau220) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau223) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau225) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau227)
        - einsum("iw,wqi->iq", d.t2.d4, tau229)
        - einsum("iw,wqi->iq", d.t2.d4, tau232)
        - einsum("iw,wqi->iq", d.t2.d4, tau235)
        - einsum("iw,wqi->iq", d.t2.d4, tau238)
        - einsum("iw,wqi->iq", d.t2.d4, tau242)
        - einsum("iw,wqi->iq", d.t2.d4, tau245)
        - einsum("iw,wqi->iq", d.t2.d4, tau247)
        + einsum("iw,wqi->iq", d.t2.d4, tau249)
        + einsum("iw,wqi->iq", d.t2.d4, tau251)
        + einsum("iw,wqi->iq", d.t2.d4, tau253)
        + einsum("iw,wqi->iq", d.t2.d4, tau255)
        + einsum("iw,wqi->iq", d.t2.d4, tau259)
        + einsum("iw,wqi->iq", d.t2.d4, tau261)
        + einsum("iw,wqi->iq", d.t2.d4, tau264)
        + einsum("iw,wqi->iq", d.t2.d4, tau266)
        - einsum("iw,wqi->iq", d.t2.d4, tau271) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau273) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau277) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau281) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau284) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau286) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau289) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau291) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau294) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau297) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau300) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau303) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau305) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau308) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau310) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau313) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau318) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau320) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau323) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau325) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau328)
        - einsum("iw,wqi->iq", d.t2.d4, tau330)
        + einsum("iw,wqi->iq", d.t2.d4, tau332)
        + einsum("iw,wqi->iq", d.t2.d4, tau333) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau334) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau335)
        - einsum("iw,wqi->iq", d.t2.d4, tau337)
        - einsum("iw,wqi->iq", d.t2.d4, tau338)
        - einsum("iw,wqi->iq", d.t2.d4, tau340)
        - einsum("iw,wqi->iq", d.t2.d4, tau341)
        + einsum("iw,wqi->iq", d.t2.d4, tau342)
        + einsum("iw,wqi->iq", d.t2.d4, tau344)
        + einsum("iw,wqi->iq", d.t2.d4, tau345)
        + einsum("iw,wqi->iq", d.t2.d4, tau348)
        + einsum("iw,wqi->iq", d.t2.d4, tau351)
        + einsum("iw,wqi->iq", d.t2.d4, tau354)
        + einsum("iw,wqi->iq", d.t2.d4, tau357)
        - einsum("iw,wqi->iq", d.t2.d4, tau360) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau363) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau366) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau370) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau374) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau378) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau381) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau386) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau389) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau392) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau394) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau398) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau401) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau404) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau406) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau408) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau410) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau413) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau416) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau418) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau421) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau423) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau425) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau428) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau431) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau434) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau436) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau439) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau443) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau446) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau449) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau452) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau456) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau460) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau464) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau468) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau471) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau474) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau477) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau480) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau482) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau485) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau488) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau492) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau495) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau498) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau500) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau502) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau504) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau506) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau510) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau512) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau515) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau517) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau520) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau523) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau525) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau527) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau530) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau533) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau536) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau539) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau542) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau544) / 4
        + einsum("iw,wqi->iq", d.t2.d4, tau545)
        + einsum("iw,wqi->iq", d.t2.d4, tau547)
        + einsum("iw,wqi->iq", d.t2.d4, tau548)
        + einsum("iw,wqi->iq", d.t2.d4, tau550)
        - einsum("iw,wqi->iq", d.t2.d4, tau551) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau554) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau555) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau558) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau560) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau561) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau564) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau565) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau566) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau568) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau569) / 2
        - einsum("iw,wqi->iq", d.t2.d4, tau572) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau576) / 2
        + einsum("iw,wqi->iq", d.t2.d4, tau580) / 2
    )

    return r24
