from numpy import einsum

def calculate_energy_cpd(h, a):
    tau0 = (
        einsum("ap,ip,oia->po", a.x1, a.x4, h.l.pov)
    )

    tau1 = (
        einsum("ap,ip,oia->po", a.x2, a.x3, h.l.pov)
    )

    tau2 = (
        einsum("ap,ip,oia->po", a.x2, a.x4, h.l.pov)
    )

    tau3 = (
        einsum("ap,ip,oia->po", a.x1, a.x3, h.l.pov)
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

def calc_residuals_cpd(h, a):
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
        einsum("ap,ip,oia->po", a.x1, a.x4, h.l.pov)
    )

    tau9 = (
        einsum("po,oia->pia", tau8, h.l.pov)
    )

    tau10 = (
        einsum("ap,oia->poi", a.x1, h.l.pov)
    )

    tau11 = (
        einsum("ip,oia->poa", a.x4, h.l.pov)
    )

    tau12 = (
        einsum("poi,poa->pia", tau10, tau11)
    )

    tau13 = (
        - einsum("pia->pia", tau9)
        + 2 * einsum("pia->pia", tau12)
    )

    tau14 = (
        einsum("ap,pia->pi", a.x2, tau13)
    )

    tau15 = (
        einsum("ap,ip,oia->po", a.x1, a.x3, h.l.pov)
    )

    tau16 = (
        einsum("po,oia->pia", tau15, h.l.pov)
    )

    tau17 = (
        einsum("ap,ip,oia->po", a.x2, a.x3, h.l.pov)
    )

    tau18 = (
        einsum("po,oia->pia", tau17, h.l.pov)
    )

    tau19 = (
        einsum("ap,pia->pi", a.x1, tau18)
    )

    tau20 = (
        2 * einsum("ap,pia->pi", a.x2, tau16)
        - einsum("pi->pi", tau19)
    )

    tau21 = (
        4 * einsum("ij->ij", tau1)
        + 2 * einsum("ij->ij", h.f.oo)
        - 2 * einsum("ji->ij", tau3)
        + 2 * einsum("aj,ia->ij", a.t1, tau7)
        + einsum("pi,jp->ij", tau14, a.x3)
        + einsum("pi,jp->ij", tau20, a.x4)
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
        einsum("ia,ap->pi", tau26, a.x1)
    )

    tau28 = (
        einsum("pi,ip->p", tau27, a.x3)
    )

    tau29 = (
        einsum("pi,ip->p", tau27, a.x4)
    )

    tau30 = (
        einsum("ap,ip,oia->po", a.x1, a.x3, h.l.pov)
    )

    tau31 = (
        einsum("jp,oji->poi", a.x3, h.l.poo)
    )

    tau32 = (
        einsum("ap,oia->poi", a.x1, h.l.pov)
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
        - einsum("jp,pij->pi", a.x3, tau22)
        - 2 * einsum("p,ip->pi", tau28, a.x4)
        + einsum("p,ip->pi", tau29, a.x3)
        + einsum("jp,pji->pi", a.x4, tau35)
    )

    tau37 = (
        einsum("ap,ip,oia->po", a.x2, a.x3, h.l.pov)
    )

    tau38 = (
        einsum("po,oij->pij", tau17, h.l.poo)
    )

    tau39 = (
        einsum("ia,ap->pi", tau24, a.x2)
    )

    tau40 = (
        einsum("po,oji->pij", tau37, tau2)
        + einsum("pij->pij", tau38)
        + 4 * einsum("pi,jp->pij", tau39, a.x3)
    )

    tau41 = (
        einsum("ia,ap->pi", h.f.ov, a.x2)
    )

    tau42 = (
        einsum("ap,ip,oia->po", a.x2, a.x4, h.l.pov)
    )

    tau43 = (
        einsum("oij->oij", h.l.poo)
        + einsum("oji->oij", tau5)
    )

    tau44 = (
        einsum("pi,jp->pij", tau41, a.x4)
        + 2 * einsum("po,oij->pij", tau42, tau43)
    )

    tau45 = (
        2 * einsum("ia->ia", tau24)
        - einsum("ia->ia", tau25)
    )

    tau46 = (
        einsum("ia,ap->pi", tau45, a.x2)
    )

    tau47 = (
        einsum("pi,ip->p", tau46, a.x3)
    )

    tau48 = (
        - einsum("ia->ia", tau25)
        + einsum("ia->ia", h.f.ov)
    )

    tau49 = (
        einsum("ia,ap->pi", tau48, a.x2)
    )

    tau50 = (
        einsum("pi,ip->p", tau49, a.x4)
    )

    tau51 = (
        - einsum("jp,pji->pi", a.x4, tau40)
        + einsum("jp,pji->pi", a.x3, tau44)
        + einsum("p,ip->pi", tau47, a.x4)
        - 2 * einsum("p,ip->pi", tau50, a.x3)
    )

    tau52 = (
        einsum("oij->oij", h.l.poo)
        + einsum("oji->oij", tau2)
    )

    tau53 = (
        einsum("po,ap,ip->oia", tau17, a.x1, a.x4)
        + einsum("po,ap,ip->oia", tau8, a.x2, a.x3)
        - 2 * einsum("po,ap,ip->oia", tau42, a.x1, a.x3)
        - 2 * einsum("po,ap,ip->oia", tau15, a.x2, a.x4)
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
        einsum("ap,oia->poi", a.x2, h.l.pov)
    )

    tau59 = (
        einsum("poj,poi->pij", tau32, tau58)
    )

    tau60 = (
        einsum("kp,lp,pij->ijkl", a.x3, a.x4, tau59)
    )

    tau61 = (
        einsum("al,iljk->ijka", a.t1, tau60)
    )

    tau62 = (
        einsum("ak,kijb->ijab", a.t1, tau61)
    )

    tau63 = (
        einsum("ap,oia->poi", a.x2, h.l.pov)
    )

    tau64 = (
        einsum("iq,poi->pqo", a.x3, tau63)
    )

    tau65 = (
        einsum("poi,qpo->pqi", tau58, tau64)
    )

    tau66 = (
        einsum("iq,pqi->pq", a.x3, tau65)
    )

    tau67 = (
        einsum("qp,aq,iq->pia", tau66, a.x1, a.x4)
    )

    tau68 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x4, tau67)
    )

    tau69 = (
        einsum("bp,oab->poa", a.x2, h.l.pvv)
    )

    tau70 = (
        einsum("bp,oab->poa", a.x1, h.l.pvv)
    )

    tau71 = (
        einsum("poa,pob->pab", tau69, tau70)
    )

    tau72 = (
        einsum("ip,jp,pab->ijab", a.x3, a.x4, tau71)
    )

    tau73 = (
        einsum("iq,poi->pqo", a.x4, tau32)
    )

    tau74 = (
        einsum("poi,qpo->pqi", tau10, tau73)
    )

    tau75 = (
        einsum("iq,pqi->pq", a.x4, tau74)
    )

    tau76 = (
        einsum("qp,aq,iq->pia", tau75, a.x2, a.x3)
    )

    tau77 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x3, tau76)
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
        einsum("jq,iq,pij->pq", a.x3, a.x4, tau79)
    )

    tau81 = (
        einsum("qp,iq,jq->pij", tau80, a.x3, a.x4)
    )

    tau82 = (
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau81)
    )

    tau83 = (
        einsum("ap,ip,oia->po", a.x1, a.x4, h.l.pov)
    )

    tau84 = (
        einsum("qo,po->pq", tau17, tau83)
    )

    tau85 = (
        einsum("pq,aq,iq->pia", tau84, a.x1, a.x4)
    )

    tau86 = (
        einsum("bp,ip,pja->ijab", a.x2, a.x3, tau85)
    )

    tau87 = (
        einsum("jp,oij->poi", a.x4, tau2)
    )

    tau88 = (
        einsum("poj,poi->pij", tau10, tau87)
    )

    tau89 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau88)
    )

    tau90 = (
        einsum("jp,oij->poi", a.x3, tau2)
    )

    tau91 = (
        einsum("poj,poi->pij", tau58, tau90)
    )

    tau92 = (
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau91)
    )

    tau93 = (
        einsum("bi,oab->oia", a.t1, h.l.pvv)
    )

    tau94 = (
        einsum("ojk,oia->ijka", tau2, tau93)
    )

    tau95 = (
        einsum("bp,oab->poa", a.x1, h.l.pvv)
    )

    tau96 = (
        einsum("poi,poa->pia", tau63, tau95)
    )

    tau97 = (
        einsum("ia,ap->pi", tau7, a.x2)
    )

    tau98 = (
        einsum("pia->pia", tau96)
        + einsum("pi,ap->pia", tau97, a.x1)
    )

    tau99 = (
        einsum("ip,jp,pka->ijka", a.x3, a.x4, tau98)
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
        einsum("ji,jp->pi", tau105, a.x4)
    )

    tau107 = (
        einsum("jp,oji->poi", a.x4, tau52)
    )

    tau108 = (
        einsum("poi,poa->pia", tau107, tau95)
    )

    tau109 = (
        einsum("pi,ap->pia", tau106, a.x1)
        + einsum("pia->pia", tau108)
    )

    tau110 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x3, tau109)
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
        einsum("ab,bp->pa", tau112, a.x2)
    )

    tau114 = (
        einsum("jp,oji->poi", a.x3, tau52)
    )

    tau115 = (
        einsum("poi,poa->pia", tau114, tau69)
    )

    tau116 = (
        - einsum("pa,ip->pia", tau113, a.x3)
        + einsum("pia->pia", tau115)
    )

    tau117 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x4, tau116)
    )

    tau118 = (
        einsum("ip,pia->pa", a.x3, tau13)
    )

    tau119 = (
        einsum("qa,ap->pq", tau118, a.x1)
    )

    tau120 = (
        einsum("pq,aq->pa", tau119, a.x2)
    )

    tau121 = (
        einsum("pb,ap,jp,ip->ijab", tau120, a.x2, a.x3, a.x4)
    )

    tau122 = (
        einsum("po,oia->pia", tau30, h.l.pov)
    )

    tau123 = (
        einsum("ap,pia->pi", a.x2, tau122)
    )

    tau124 = (
        2 * einsum("pi->pi", tau123)
        - einsum("pi->pi", tau19)
    )

    tau125 = (
        einsum("pj,ip->ij", tau124, a.x4)
    )

    tau126 = (
        einsum("ij,jp->pi", tau125, a.x3)
    )

    tau127 = (
        einsum("pj,bp,ap,ip->ijab", tau126, a.x1, a.x2, a.x4)
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
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau129)
    )

    tau131 = (
        einsum("oil,ojk->ijkl", tau2, h.l.poo)
    )

    tau132 = (
        einsum("al,ijkl->ijka", a.t1, tau131)
    )

    tau133 = (
        einsum("jp,oji->poi", a.x4, h.l.poo)
    )

    tau134 = (
        einsum("poi,poj->pij", tau10, tau133)
    )

    tau135 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau134)
    )

    tau136 = (
        einsum("jp,oji->poi", a.x3, h.l.poo)
    )

    tau137 = (
        einsum("poj,poi->pij", tau136, tau58)
    )

    tau138 = (
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau137)
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
        einsum("ji,jp->pi", tau143, a.x4)
    )

    tau145 = (
        einsum("pj,bp,ap,ip->ijab", tau144, a.x1, a.x2, a.x3)
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
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau147)
    )

    tau149 = (
        einsum("jp,oji->poi", a.x4, h.l.poo)
    )

    tau150 = (
        einsum("poi,poj->pij", tau149, tau32)
    )

    tau151 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau150)
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
        einsum("ji,jp->pi", tau154, a.x4)
    )

    tau156 = (
        einsum("pj,bp,ap,ip->ijab", tau155, a.x1, a.x2, a.x3)
    )

    tau157 = (
        einsum("jp,oji->poi", a.x4, tau43)
    )

    tau158 = (
        einsum("poj,poi->pij", tau136, tau157)
    )

    tau159 = (
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau158)
    )

    tau160 = (
        einsum("ijab->ijab", tau153)
        - einsum("jiba->ijab", tau156)
        + einsum("ijab->ijab", tau159)
    )

    tau161 = (
        einsum("bp,oab->poa", a.x2, h.l.pvv)
    )

    tau162 = (
        einsum("pob,poa->pab", tau161, tau95)
    )

    tau163 = (
        einsum("ip,jp,pab->ijab", a.x3, a.x4, tau162)
    )

    tau164 = (
        einsum("oai,obj->ijab", h.l.pvo, h.l.pvo)
    )

    tau165 = (
        einsum("qo,po->pq", tau8, tau83)
    )

    tau166 = (
        einsum("pq,aq,iq->pia", tau165, a.x2, a.x3)
    )

    tau167 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x3, tau166)
    )

    tau168 = (
        einsum("qo,po->pq", tau17, tau37)
    )

    tau169 = (
        einsum("pq,aq,iq->pia", tau168, a.x1, a.x4)
    )

    tau170 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x4, tau169)
    )

    tau171 = (
        einsum("oai->oia", h.l.pvo)
        + einsum("oia->oia", tau56)
    )

    tau172 = (
        einsum("oia,ojb->ijab", tau171, tau93)
    )

    tau173 = (
        einsum("kp,lp,pij->ijkl", a.x3, a.x4, tau79)
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
        einsum("iq,poi->pqo", a.x4, tau63)
    )

    tau179 = (
        einsum("qpo,poi->pqi", tau178, tau58)
    )

    tau180 = (
        einsum("iq,pqi->pq", a.x4, tau179)
    )

    tau181 = (
        einsum("ap,ip,oia->po", a.x2, a.x4, h.l.pov)
    )

    tau182 = (
        einsum("po,qo->pq", tau181, tau42)
    )

    tau183 = (
        - einsum("qp->pq", tau180)
        + 2 * einsum("pq->pq", tau182)
    )

    tau184 = (
        einsum("pq,aq,iq->pia", tau183, a.x1, a.x3)
    )

    tau185 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x3, tau184)
    )

    tau186 = (
        einsum("ip,oia->poa", a.x3, h.l.pov)
    )

    tau187 = (
        einsum("poi,poa->pia", tau10, tau186)
    )

    tau188 = (
        - einsum("pia->pia", tau187)
        + 2 * einsum("pia->pia", tau16)
    )

    tau189 = (
        einsum("ap,qia->pqi", a.x1, tau188)
    )

    tau190 = (
        einsum("ip,pqi->pq", a.x3, tau189)
    )

    tau191 = (
        einsum("qp,aq,iq->pia", tau190, a.x2, a.x4)
    )

    tau192 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x4, tau191)
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
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau194)
    )

    tau196 = (
        einsum("po,oij->pij", tau37, h.l.poo)
    )

    tau197 = (
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau196)
    )

    tau198 = (
        einsum("po,oij->pij", tau83, h.l.poo)
    )

    tau199 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau198)
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
        einsum("ap,ip,pjk->ijka", a.x1, a.x3, tau202)
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
        einsum("ap,ip,pjk->ijka", a.x2, a.x4, tau206)
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
        einsum("ji,jp->pi", tau143, a.x3)
    )

    tau211 = (
        einsum("pj,ap,bp,ip->ijab", tau210, a.x1, a.x2, a.x4)
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
        einsum("iq,pqi->pq", a.x3, tau213)
    )

    tau215 = (
        einsum("pq,aq,iq->pia", tau214, a.x1, a.x4)
    )

    tau216 = (
        einsum("bp,ip,pja->ijab", a.x2, a.x3, tau215)
    )

    tau217 = (
        einsum("po,oia->pia", tau83, h.l.pov)
    )

    tau218 = (
        2 * einsum("pia->pia", tau12)
        - einsum("pia->pia", tau217)
    )

    tau219 = (
        einsum("ap,pia->pi", a.x2, tau218)
    )

    tau220 = (
        einsum("pj,ip->ij", tau219, a.x3)
    )

    tau221 = (
        2 * einsum("ij->ij", h.f.oo)
        + 2 * einsum("ji->ij", tau104)
        + einsum("ji->ij", tau220)
    )

    tau222 = (
        einsum("ji,jp->pi", tau221, a.x3)
    )

    tau223 = (
        einsum("ab,bp->pa", tau112, a.x1)
    )

    tau224 = (
        einsum("ip,oia->poa", a.x3, h.l.pov)
    )

    tau225 = (
        einsum("poa,poi->pia", tau224, tau63)
    )

    tau226 = (
        - einsum("pia->pia", tau18)
        + 2 * einsum("pia->pia", tau225)
    )

    tau227 = (
        einsum("ip,pia->pa", a.x4, tau226)
    )

    tau228 = (
        einsum("qa,ap->pq", tau227, a.x1)
    )

    tau229 = (
        einsum("pq,aq->pa", tau228, a.x1)
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
        einsum("iq,pqi->pq", a.x3, tau232)
    )

    tau234 = (
        2 * einsum("pq->pq", tau231)
        - einsum("pq->pq", tau233)
    )

    tau235 = (
        einsum("aq,pia->pqi", a.x2, tau122)
    )

    tau236 = (
        2 * einsum("pqi->pqi", tau235)
        - einsum("pqi->pqi", tau232)
    )

    tau237 = (
        einsum("ip,qpi->pq", a.x4, tau236)
    )

    tau238 = (
        einsum("pq,iq->pqi", tau234, a.x4)
        - 2 * einsum("qp,iq->pqi", tau237, a.x3)
    )

    tau239 = (
        einsum("aq,pqi->pia", a.x1, tau238)
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
        einsum("ap,qia->pqi", a.x1, tau242)
    )

    tau244 = (
        einsum("ip,pqi->pq", a.x4, tau243)
    )

    tau245 = (
        einsum("qp,aq,iq->pia", tau244, a.x2, a.x3)
    )

    tau246 = (
        einsum("poi,poa->pia", tau114, tau95)
    )

    tau247 = (
        einsum("pi,ap->pia", tau222, a.x1)
        - einsum("pa,ip->pia", tau230, a.x3)
        + einsum("pia->pia", tau239)
        - 4 * einsum("pia->pia", tau240)
        + einsum("pia->pia", tau245)
        + 2 * einsum("pia->pia", tau246)
    )

    tau248 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x4, tau247)
    )

    tau249 = (
        einsum("ji->ij", tau220)
        + einsum("ji->ij", tau125)
    )

    tau250 = (
        einsum("ji,jp->pi", tau249, a.x4)
    )

    tau251 = (
        einsum("ip,pia->pa", a.x3, tau218)
    )

    tau252 = (
        einsum("qa,ap->pq", tau251, a.x2)
    )

    tau253 = (
        einsum("pq,aq->pa", tau252, a.x2)
    )

    tau254 = (
        einsum("po,oia->pia", tau37, h.l.pov)
    )

    tau255 = (
        2 * einsum("pia->pia", tau225)
        - einsum("pia->pia", tau254)
    )

    tau256 = (
        einsum("ip,pia->pa", a.x4, tau255)
    )

    tau257 = (
        einsum("qa,ap->pq", tau256, a.x2)
    )

    tau258 = (
        einsum("pq,aq->pa", tau257, a.x1)
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
        einsum("ap,qia->pqi", a.x2, tau262)
    )

    tau264 = (
        einsum("ip,pqi->pq", a.x3, tau263)
    )

    tau265 = (
        einsum("qp,aq,iq->pia", tau264, a.x1, a.x4)
    )

    tau266 = (
        einsum("iq,pqi->pq", a.x4, tau213)
    )

    tau267 = (
        einsum("qo,po->pq", tau42, tau83)
    )

    tau268 = (
        - einsum("pq->pq", tau266)
        + 2 * einsum("pq->pq", tau267)
    )

    tau269 = (
        einsum("qp,aq,iq->pia", tau268, a.x2, a.x3)
    )

    tau270 = (
        einsum("pi,ap->pia", tau250, a.x2)
        + einsum("pa,ip->pia", tau259, a.x4)
        + einsum("pia->pia", tau265)
        + einsum("pia->pia", tau269)
    )

    tau271 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x3, tau270)
    )

    tau272 = (
        einsum("po,oij->pij", tau17, tau5)
    )

    tau273 = (
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau272)
    )

    tau274 = (
        einsum("oik,oaj->ijka", tau2, h.l.pvo)
    )

    tau275 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau22)
    )

    tau276 = (
        einsum("poi,poa->pia", tau32, tau69)
    )

    tau277 = (
        einsum("jp,kp,pia->ijka", a.x3, a.x4, tau276)
    )

    tau278 = (
        einsum("poj,poi->pij", tau10, tau90)
    )

    tau279 = (
        einsum("po,oij->pij", tau15, tau5)
    )

    tau280 = (
        einsum("ia,ap->pi", tau7, a.x1)
    )

    tau281 = (
        - einsum("pji->pij", tau278)
        + 2 * einsum("pji->pij", tau279)
        + einsum("pi,jp->pij", tau280, a.x3)
    )

    tau282 = (
        einsum("ap,ip,pjk->ijka", a.x2, a.x4, tau281)
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
        einsum("ap,ip,pjk->ijka", a.x1, a.x3, tau285)
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
        einsum("ap,ip,pjb->ijab", a.x1, a.x3, tau291)
    )

    tau293 = (
        einsum("po,oia->pia", tau17, tau100)
    )

    tau294 = (
        einsum("ap,ip,pjb->ijab", a.x1, a.x4, tau293)
    )

    tau295 = (
        einsum("po,oia->pia", tau8, tau100)
    )

    tau296 = (
        einsum("ap,ip,pjb->ijab", a.x2, a.x3, tau295)
    )

    tau297 = (
        einsum("iq,jq,pij->pq", a.x3, a.x4, tau79)
    )

    tau298 = (
        einsum("qp,iq,jq->pij", tau297, a.x3, a.x4)
    )

    tau299 = (
        einsum("jp,oij->poi", a.x3, tau5)
    )

    tau300 = (
        einsum("poi,poj->pij", tau299, tau87)
    )

    tau301 = (
        einsum("pij->pij", tau298)
        + 2 * einsum("pij->pij", tau300)
    )

    tau302 = (
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau301)
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
        einsum("ij,jp->pi", tau3, a.x3)
    )

    tau305 = (
        einsum("pi,ap,bp,jp->ijab", tau304, a.x1, a.x2, a.x4)
    )

    tau306 = (
        einsum("po,oij->pij", tau8, h.l.poo)
    )

    tau307 = (
        einsum("ap,kp,pij->ijka", a.x2, a.x3, tau306)
    )

    tau308 = (
        einsum("ap,kp,pij->ijka", a.x1, a.x4, tau38)
    )

    tau309 = (
        - einsum("pji->pij", tau33)
        + 2 * einsum("pij->pij", tau34)
    )

    tau310 = (
        einsum("ap,ip,pjk->ijka", a.x2, a.x4, tau309)
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
        einsum("ap,ip,pjk->ijka", a.x1, a.x3, tau313)
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
        einsum("ji,jp->pi", tau1, a.x3)
    )

    tau320 = (
        einsum("jp,oji->poi", a.x3, tau43)
    )

    tau321 = (
        einsum("poj,poi->pij", tau133, tau320)
    )

    tau322 = (
        - 2 * einsum("pi,jp->pij", tau319, a.x4)
        + einsum("pij->pij", tau321)
    )

    tau323 = (
        einsum("ap,bp,pij->ijab", a.x1, a.x2, tau322)
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
        - einsum("pi,ap->ai", tau36, a.x2)
        - einsum("pi,ap->ai", tau51, a.x1)
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

    return rt1, rt2

def calc_r2dr2dx(h, a, rt2):
    tau0 = (
        einsum("bp,jp,abji->pia", a.x1, a.x3, rt2)
    )

    tau1 = (
        einsum("ap,pia->pi", a.x2, tau0)
    )

    tau2 = (
        einsum("pi,iq->pq", tau1, a.x3)
    )

    tau3 = (
        einsum("qp,iq->pi", tau2, a.x4)
    )

    tau4 = (
        einsum("ap,ip,oia->po", a.x2, a.x4, h.l.pov)
    )

    tau5 = (
        einsum("po,oia->pia", tau4, h.l.pov)
    )

    tau6 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau7 = (
        einsum("o,oia->ia", tau6, h.l.pov)
    )

    tau8 = (
        einsum("ia,ap->pi", tau7, a.x2)
    )

    tau9 = (
        einsum("ai,pi->pa", a.t1, tau8)
    )

    tau10 = (
        einsum("pb,jp,baij->pia", tau9, a.x3, rt2)
    )

    tau11 = (
        einsum("ap,ip,oia->po", a.x2, a.x3, h.l.pov)
    )

    tau12 = (
        einsum("po,oia->pia", tau11, h.l.pov)
    )

    tau13 = (
        einsum("ap,pia->pi", a.x1, tau12)
    )

    tau14 = (
        einsum("pi,iq->pq", tau13, a.x4)
    )

    tau15 = (
        einsum("qp,iq->pi", tau14, a.x4)
    )

    tau16 = (
        einsum("bp,jp,baij->pia", a.x2, a.x3, rt2)
    )

    tau17 = (
        einsum("aq,pia->pqi", a.x2, tau16)
    )

    tau18 = (
        einsum("iq,pqi->pq", a.x4, tau17)
    )

    tau19 = (
        einsum("ap,oia->poi", a.x1, h.l.pov)
    )

    tau20 = (
        einsum("iq,poi->pqo", a.x3, tau19)
    )

    tau21 = (
        einsum("qp,iq,qpo->poi", tau18, a.x4, tau20)
    )

    tau22 = (
        einsum("bp,jp,abij->pia", a.x2, a.x4, rt2)
    )

    tau23 = (
        einsum("ip,pia->pa", a.x3, tau22)
    )

    tau24 = (
        einsum("pa,aq->pq", tau23, a.x2)
    )

    tau25 = (
        einsum("qp,aq->pa", tau24, a.x1)
    )

    tau26 = (
        einsum("pa,ip,oia->po", tau25, a.x4, h.l.pov)
    )

    tau27 = (
        einsum("po,oia->pia", tau26, h.l.pov)
    )

    tau28 = (
        einsum("ap,ip,oia->po", a.x1, a.x3, h.l.pov)
    )

    tau29 = (
        einsum("bp,jp,baij->pia", a.x2, a.x4, rt2)
    )

    tau30 = (
        einsum("aq,pia->pqi", a.x2, tau29)
    )

    tau31 = (
        einsum("iq,pqi->pq", a.x4, tau30)
    )

    tau32 = (
        einsum("qo,qp->po", tau28, tau31)
    )

    tau33 = (
        einsum("po,oia->pia", tau32, h.l.pov)
    )

    tau34 = (
        einsum("bp,jp,abji->pia", a.x2, a.x4, rt2)
    )

    tau35 = (
        einsum("ap,pia->pi", a.x1, tau34)
    )

    tau36 = (
        einsum("pi,iq->pq", tau35, a.x4)
    )

    tau37 = (
        einsum("qp,iq->pi", tau36, a.x3)
    )

    tau38 = (
        einsum("pi,ap,oia->po", tau37, a.x2, h.l.pov)
    )

    tau39 = (
        einsum("po,oia->pia", tau38, h.l.pov)
    )

    tau40 = (
        einsum("ai,oja->oij", a.t1, h.l.pov)
    )

    tau41 = (
        einsum("oji,oja->ia", tau40, h.l.pov)
    )

    tau42 = (
        einsum("ai,ja->ij", a.t1, tau41)
    )

    tau43 = (
        einsum("ij,jp->pi", tau42, a.x4)
    )

    tau44 = (
        einsum("o,oab->ab", tau6, h.l.pvv)
    )

    tau45 = (
        einsum("ab,bp->pa", tau44, a.x2)
    )

    tau46 = (
        einsum("pb,jp,baij->pia", tau45, a.x3, rt2)
    )

    tau47 = (
        einsum("bp,jp,abij->pia", a.x2, a.x3, rt2)
    )

    tau48 = (
        einsum("ip,pia->pa", a.x4, tau47)
    )

    tau49 = (
        einsum("qa,ap->pq", tau48, a.x1)
    )

    tau50 = (
        einsum("ip,pia->pa", a.x4, tau12)
    )

    tau51 = (
        einsum("aj,pia->pij", a.t1, tau22)
    )

    tau52 = (
        einsum("aj,pji->pia", a.t1, tau51)
    )

    tau53 = (
        einsum("pia,oia->po", tau52, h.l.pov)
    )

    tau54 = (
        einsum("po,oia->pia", tau53, h.l.pov)
    )

    tau55 = (
        einsum("ip,pia->pa", a.x4, tau16)
    )

    tau56 = (
        einsum("pb,oba->poa", tau55, h.l.pvv)
    )

    tau57 = (
        einsum("ai,poa->poi", a.t1, tau56)
    )

    tau58 = (
        einsum("bp,oab->poa", a.x2, h.l.pvv)
    )

    tau59 = (
        einsum("jp,oij->poi", a.x4, tau40)
    )

    tau60 = (
        einsum("poa,poi->pia", tau58, tau59)
    )

    tau61 = (
        einsum("pjb,abij->pia", tau60, rt2)
    )

    tau62 = (
        einsum("jp,oij->poi", a.x3, tau40)
    )

    tau63 = (
        einsum("bp,jp,baji->pia", a.x2, a.x4, rt2)
    )

    tau64 = (
        einsum("poi,pia->poa", tau62, tau63)
    )

    tau65 = (
        einsum("po,oia->pia", tau28, h.l.pov)
    )

    tau66 = (
        einsum("ap,pia->pi", a.x2, tau65)
    )

    tau67 = (
        einsum("pi,iq->pq", tau66, a.x4)
    )

    tau68 = (
        einsum("qp,iq->pi", tau67, a.x4)
    )

    tau69 = (
        einsum("bp,oab->poa", a.x2, h.l.pvv)
    )

    tau70 = (
        einsum("jp,ip,abij->pab", a.x3, a.x4, rt2)
    )

    tau71 = (
        einsum("pob,pab->poa", tau69, tau70)
    )

    tau72 = (
        einsum("jp,oji->poi", a.x4, h.l.poo)
    )

    tau73 = (
        einsum("aj,pia->pij", a.t1, tau47)
    )

    tau74 = (
        einsum("poj,pji->poi", tau72, tau73)
    )

    tau75 = (
        einsum("ai,oia->o", a.t1, h.l.pov)
    )

    tau76 = (
        einsum("o,oij->ij", tau75, h.l.poo)
    )

    tau77 = (
        einsum("ji,jp->pi", tau76, a.x3)
    )

    tau78 = (
        einsum("pj,bp,baij->pia", tau77, a.x2, rt2)
    )

    tau79 = (
        einsum("ai,oja->oij", a.t1, h.l.pov)
    )

    tau80 = (
        einsum("oki,okj->ij", tau79, h.l.poo)
    )

    tau81 = (
        einsum("ji,jp->pi", tau80, a.x3)
    )

    tau82 = (
        einsum("pj,bp,baij->pia", tau81, a.x2, rt2)
    )

    tau83 = (
        einsum("aj,pia->pij", a.t1, tau16)
    )

    tau84 = (
        einsum("aj,pji->pia", a.t1, tau83)
    )

    tau85 = (
        einsum("pia,oia->po", tau84, h.l.pov)
    )

    tau86 = (
        einsum("po,oia->pia", tau85, h.l.pov)
    )

    tau87 = (
        einsum("bp,jp,abij->pia", a.x1, a.x3, rt2)
    )

    tau88 = (
        einsum("ap,oia->poi", a.x2, h.l.pov)
    )

    tau89 = (
        einsum("ap,oia->poi", a.x2, h.l.pov)
    )

    tau90 = (
        einsum("iq,poi->pqo", a.x4, tau89)
    )

    tau91 = (
        einsum("poi,qpo->pqi", tau88, tau90)
    )

    tau92 = (
        einsum("ip,qpi->pq", a.x3, tau91)
    )

    tau93 = (
        einsum("pij,oji->po", tau83, h.l.poo)
    )

    tau94 = (
        einsum("po,oia->pia", tau93, h.l.pov)
    )

    tau95 = (
        einsum("ap,oia->poi", a.x1, h.l.pov)
    )

    tau96 = (
        einsum("iq,poi->pqo", a.x3, tau95)
    )

    tau97 = (
        einsum("qp,iq,qpo->poi", tau31, a.x3, tau96)
    )

    tau98 = (
        einsum("iq,pqi->pq", a.x4, tau91)
    )

    tau99 = (
        einsum("ab,bp->pa", h.f.vv, a.x2)
    )

    tau100 = (
        einsum("pb,jp,abij->pia", tau99, a.x4, rt2)
    )

    tau101 = (
        einsum("bp,jp,baij->pia", a.x1, a.x3, rt2)
    )

    tau102 = (
        einsum("ip,pia->pa", a.x4, tau101)
    )

    tau103 = (
        einsum("ip,pia->pa", a.x3, tau5)
    )

    tau104 = (
        einsum("pa,aq->pq", tau103, a.x2)
    )

    tau105 = (
        einsum("ap,ip,oia->po", a.x1, a.x4, h.l.pov)
    )

    tau106 = (
        einsum("po,qo->pq", tau105, tau4)
    )

    tau107 = (
        einsum("ap,pia->pi", a.x1, tau5)
    )

    tau108 = (
        einsum("pi,iq->pq", tau107, a.x3)
    )

    tau109 = (
        einsum("qp,iq->pi", tau108, a.x3)
    )

    tau110 = (
        einsum("aq,pia->pqi", a.x2, tau47)
    )

    tau111 = (
        einsum("ip,qpi->pq", a.x3, tau110)
    )

    tau112 = (
        einsum("qo,qp->po", tau105, tau111)
    )

    tau113 = (
        einsum("po,oia->pia", tau112, h.l.pov)
    )

    tau114 = (
        einsum("ap,ip,oia->po", a.x2, a.x3, h.l.pov)
    )

    tau115 = (
        einsum("po,oia->pia", tau114, h.l.pov)
    )

    tau116 = (
        einsum("ip,pia->pa", a.x4, tau115)
    )

    tau117 = (
        einsum("pa,aq->pq", tau116, a.x2)
    )

    tau118 = (
        einsum("pj,bp,baij->pia", tau109, a.x2, rt2)
    )

    tau119 = (
        einsum("oji,oja->ia", tau79, h.l.pov)
    )

    tau120 = (
        einsum("ia,ap->pi", tau119, a.x2)
    )

    tau121 = (
        einsum("ai,pi->pa", a.t1, tau120)
    )

    tau122 = (
        einsum("pb,jp,abij->pia", tau121, a.x3, rt2)
    )

    tau123 = (
        einsum("bp,jp,baij->pia", a.x1, a.x4, rt2)
    )

    tau124 = (
        einsum("ip,pia->pa", a.x3, tau123)
    )

    tau125 = (
        einsum("pa,aq->pq", tau124, a.x2)
    )

    tau126 = (
        einsum("qp,aq->pa", tau125, a.x2)
    )

    tau127 = (
        einsum("pa,ip,oia->po", tau126, a.x4, h.l.pov)
    )

    tau128 = (
        einsum("po,oia->pia", tau127, h.l.pov)
    )

    tau129 = (
        einsum("iq,pqi->pq", a.x4, tau110)
    )

    tau130 = (
        einsum("pq,qo->po", tau129, tau28)
    )

    tau131 = (
        einsum("po,oia->pia", tau130, h.l.pov)
    )

    tau132 = (
        einsum("poi,poj->pij", tau19, tau89)
    )

    tau133 = (
        einsum("iq,jq,pij->pq", a.x3, a.x4, tau132)
    )

    tau134 = (
        einsum("qp,iq,jq->pij", tau133, a.x3, a.x4)
    )

    tau135 = (
        einsum("pji,abij->pab", tau134, rt2)
    )

    tau136 = (
        einsum("bp,jp,abji->pia", a.x2, a.x3, rt2)
    )

    tau137 = (
        einsum("ap,pia->pi", a.x1, tau136)
    )

    tau138 = (
        einsum("pi,iq->pq", tau137, a.x4)
    )

    tau139 = (
        einsum("qp,iq->pi", tau138, a.x4)
    )

    tau140 = (
        einsum("pi,ap,oia->po", tau139, a.x2, h.l.pov)
    )

    tau141 = (
        einsum("po,oia->pia", tau140, h.l.pov)
    )

    tau142 = (
        einsum("po,oij->pij", tau11, h.l.poo)
    )

    tau143 = (
        einsum("aj,pji->pia", a.t1, tau142)
    )

    tau144 = (
        einsum("pjb,abji->pia", tau143, rt2)
    )

    tau145 = (
        einsum("bp,jp,abji->pia", a.x1, a.x4, rt2)
    )

    tau146 = (
        einsum("ap,pia->pi", a.x2, tau145)
    )

    tau147 = (
        einsum("pi,iq->pq", tau146, a.x3)
    )

    tau148 = (
        einsum("qp,iq->pi", tau147, a.x3)
    )

    tau149 = (
        einsum("bp,jp,abij->pia", a.x1, a.x4, rt2)
    )

    tau150 = (
        einsum("ap,pia->pi", a.x2, tau149)
    )

    tau151 = (
        einsum("pi,iq->pq", tau150, a.x4)
    )

    tau152 = (
        einsum("qp,iq->pi", tau151, a.x3)
    )

    tau153 = (
        einsum("ap,pia->pi", a.x1, tau22)
    )

    tau154 = (
        einsum("pi,iq->pq", tau153, a.x4)
    )

    tau155 = (
        einsum("qp,iq->pi", tau154, a.x3)
    )

    tau156 = (
        einsum("jp,oij->poi", a.x4, tau79)
    )

    tau157 = (
        einsum("jp,oji->poi", a.x3, h.l.poo)
    )

    tau158 = (
        einsum("poi,poj->pij", tau156, tau157)
    )

    tau159 = (
        einsum("pij,abij->pab", tau158, rt2)
    )

    tau160 = (
        einsum("pi,iq->pq", tau1, a.x4)
    )

    tau161 = (
        einsum("qp,iq->pi", tau160, a.x4)
    )

    tau162 = (
        einsum("ap,qia->pqi", a.x1, tau22)
    )

    tau163 = (
        einsum("ip,pqi->pq", a.x4, tau162)
    )

    tau164 = (
        einsum("qo,qp->po", tau11, tau163)
    )

    tau165 = (
        einsum("po,oia->pia", tau164, h.l.pov)
    )

    tau166 = (
        einsum("poi,poj->pij", tau59, tau88)
    )

    tau167 = (
        einsum("aj,pij->pia", a.t1, tau166)
    )

    tau168 = (
        einsum("pjb,abij->pia", tau167, rt2)
    )

    tau169 = (
        einsum("po,oia->pia", tau105, h.l.pov)
    )

    tau170 = (
        einsum("ap,pia->pi", a.x2, tau169)
    )

    tau171 = (
        einsum("pi,iq->pq", tau170, a.x4)
    )

    tau172 = (
        einsum("qp,iq->pi", tau171, a.x3)
    )

    tau173 = (
        einsum("bi,pba->pia", a.t1, tau70)
    )

    tau174 = (
        einsum("pia,poa->poi", tau173, tau58)
    )

    tau175 = (
        einsum("aq,pia->pqi", a.x2, tau101)
    )

    tau176 = (
        einsum("iq,pqi->pq", a.x4, tau175)
    )

    tau177 = (
        einsum("qp,qo->po", tau176, tau4)
    )

    tau178 = (
        einsum("po,oia->pia", tau177, h.l.pov)
    )

    tau179 = (
        einsum("poj,poi->pij", tau157, tau88)
    )

    tau180 = (
        einsum("aj,pji->pia", a.t1, tau179)
    )

    tau181 = (
        einsum("pjb,abij->pia", tau180, rt2)
    )

    tau182 = (
        einsum("poi,qpo->pqi", tau19, tau90)
    )

    tau183 = (
        einsum("iq,pqi->pq", a.x3, tau182)
    )

    tau184 = (
        einsum("ip,pia->pa", a.x3, tau29)
    )

    tau185 = (
        einsum("pa,aq->pq", tau184, a.x2)
    )

    tau186 = (
        einsum("ip,pia->pa", a.x4, tau65)
    )

    tau187 = (
        einsum("ip,pia->pa", a.x3, tau169)
    )

    tau188 = (
        einsum("pa,aq->pq", tau187, a.x2)
    )

    tau189 = (
        einsum("qp,aq->pa", tau188, a.x2)
    )

    tau190 = (
        einsum("pb,jp,abij->pia", tau189, a.x4, rt2)
    )

    tau191 = (
        einsum("ip,qpi->pq", a.x3, tau30)
    )

    tau192 = (
        einsum("qp,iq,qpo->poi", tau191, a.x4, tau20)
    )

    tau193 = (
        einsum("pia,poi->poa", tau16, tau59)
    )

    tau194 = (
        einsum("qa,ap->pq", tau50, a.x1)
    )

    tau195 = (
        einsum("ap,ip,oia->po", a.x1, a.x4, h.l.pov)
    )

    tau196 = (
        einsum("aq,pia->pqi", a.x2, tau22)
    )

    tau197 = (
        einsum("ip,qpi->pq", a.x3, tau196)
    )

    tau198 = (
        einsum("qo,qp->po", tau195, tau197)
    )

    tau199 = (
        einsum("po,oia->pia", tau198, h.l.pov)
    )

    tau200 = (
        einsum("jp,oji->poi", a.x4, h.l.poo)
    )

    tau201 = (
        einsum("bp,jp,baji->pia", a.x2, a.x3, rt2)
    )

    tau202 = (
        einsum("poi,pia->poa", tau200, tau201)
    )

    tau203 = (
        einsum("ip,pia->pa", a.x3, tau149)
    )

    tau204 = (
        einsum("ai,pja->pij", a.t1, tau5)
    )

    tau205 = (
        einsum("aj,pij->pia", a.t1, tau204)
    )

    tau206 = (
        einsum("pjb,abji->pia", tau205, rt2)
    )

    tau207 = (
        einsum("pjb,abij->pia", tau205, rt2)
    )

    tau208 = (
        einsum("po,qo->pq", tau114, tau4)
    )

    tau209 = (
        einsum("bi,oab->oia", a.t1, h.l.pvv)
    )

    tau210 = (
        einsum("oia,poi->pa", tau209, tau89)
    )

    tau211 = (
        einsum("pb,jp,abij->pia", tau210, a.x4, rt2)
    )

    tau212 = (
        einsum("ap,bp,abij->pij", a.x1, a.x2, rt2)
    )

    tau213 = (
        einsum("jq,iq,pij->pq", a.x3, a.x4, tau212)
    )

    tau214 = (
        einsum("qp,iq,jq->pij", tau213, a.x3, a.x4)
    )

    tau215 = (
        einsum("pij,poj->poi", tau214, tau89)
    )

    tau216 = (
        einsum("poj,poi->pij", tau157, tau72)
    )

    tau217 = (
        einsum("pij,abij->pab", tau216, rt2)
    )

    tau218 = (
        einsum("iq,poi->pqo", a.x3, tau89)
    )

    tau219 = (
        einsum("qpo,poi->pqi", tau218, tau88)
    )

    tau220 = (
        einsum("iq,pqi->pq", a.x3, tau219)
    )

    tau221 = (
        einsum("po,qo->pq", tau105, tau11)
    )

    tau222 = (
        einsum("bp,jp,baji->pia", a.x1, a.x4, rt2)
    )

    tau223 = (
        einsum("pj,bp,abij->pia", tau68, a.x2, rt2)
    )

    tau224 = (
        einsum("pa,ip,oia->po", tau126, a.x3, h.l.pov)
    )

    tau225 = (
        einsum("po,oia->pia", tau224, h.l.pov)
    )

    tau226 = (
        einsum("pia,oai->po", tau136, h.l.pvo)
    )

    tau227 = (
        einsum("po,oia->pia", tau226, h.l.pov)
    )

    tau228 = (
        einsum("bi,pab->pia", a.t1, tau70)
    )

    tau229 = (
        einsum("pia,poa->poi", tau228, tau58)
    )

    tau230 = (
        einsum("poi,qpo->pqi", tau19, tau218)
    )

    tau231 = (
        einsum("iq,pqi->pq", a.x3, tau230)
    )

    tau232 = (
        einsum("aj,pia->pij", a.t1, tau34)
    )

    tau233 = (
        einsum("aj,pji->pia", a.t1, tau232)
    )

    tau234 = (
        einsum("pia,oia->po", tau233, h.l.pov)
    )

    tau235 = (
        einsum("po,oia->pia", tau234, h.l.pov)
    )

    tau236 = (
        einsum("ap,qia->pqi", a.x1, tau29)
    )

    tau237 = (
        einsum("ip,pqi->pq", a.x3, tau236)
    )

    tau238 = (
        einsum("qp,qo->po", tau237, tau4)
    )

    tau239 = (
        einsum("po,oia->pia", tau238, h.l.pov)
    )

    tau240 = (
        einsum("o,oia->ia", tau75, h.l.pov)
    )

    tau241 = (
        einsum("ai,ja->ij", a.t1, tau240)
    )

    tau242 = (
        einsum("ij,jp->pi", tau241, a.x3)
    )

    tau243 = (
        einsum("pij,oji->po", tau73, h.l.poo)
    )

    tau244 = (
        einsum("po,oia->pia", tau243, h.l.pov)
    )

    tau245 = (
        einsum("pa,aq->pq", tau186, a.x2)
    )

    tau246 = (
        einsum("qp,aq->pa", tau245, a.x2)
    )

    tau247 = (
        einsum("pb,jp,abij->pia", tau246, a.x3, rt2)
    )

    tau248 = (
        einsum("ip,jp,abij->pab", a.x3, a.x4, rt2)
    )

    tau249 = (
        einsum("bi,pab->pia", a.t1, tau248)
    )

    tau250 = (
        einsum("pia,poi->poa", tau249, tau89)
    )

    tau251 = (
        einsum("qp,aq->pa", tau185, a.x1)
    )

    tau252 = (
        einsum("pa,ip,oia->po", tau251, a.x4, h.l.pov)
    )

    tau253 = (
        einsum("po,oia->pia", tau252, h.l.pov)
    )

    tau254 = (
        einsum("ap,ip,oia->po", a.x2, a.x4, h.l.pov)
    )

    tau255 = (
        einsum("po,oij->pij", tau254, h.l.poo)
    )

    tau256 = (
        einsum("aj,pji->pia", a.t1, tau255)
    )

    tau257 = (
        einsum("pjb,abij->pia", tau256, rt2)
    )

    tau258 = (
        einsum("aj,pia->pij", a.t1, tau29)
    )

    tau259 = (
        einsum("pji,poj->poi", tau258, tau62)
    )

    tau260 = (
        einsum("qa,ap->pq", tau23, a.x1)
    )

    tau261 = (
        einsum("pj,bp,baij->pia", tau172, a.x2, rt2)
    )

    tau262 = (
        einsum("poa,poi->pia", tau58, tau62)
    )

    tau263 = (
        einsum("pjb,abij->pia", tau262, rt2)
    )

    tau264 = (
        einsum("bp,jp,baji->pia", a.x1, a.x3, rt2)
    )

    tau265 = (
        einsum("pia,poi->poa", tau16, tau200)
    )

    tau266 = (
        einsum("bi,pia->pab", a.t1, tau29)
    )

    tau267 = (
        einsum("pab,oab->po", tau266, h.l.pvv)
    )

    tau268 = (
        einsum("po,oia->pia", tau267, h.l.pov)
    )

    tau269 = (
        einsum("ji,jp->pi", tau80, a.x4)
    )

    tau270 = (
        einsum("pj,bp,baij->pia", tau269, a.x2, rt2)
    )

    tau271 = (
        einsum("pab,pob->poa", tau248, tau69)
    )

    tau272 = (
        einsum("ia,aj->ij", h.f.ov, a.t1)
    )

    tau273 = (
        einsum("ji,jp->pi", tau272, a.x4)
    )

    tau274 = (
        einsum("po,oai->pia", tau11, h.l.pvo)
    )

    tau275 = (
        einsum("pjb,abji->pia", tau274, rt2)
    )

    tau276 = (
        einsum("poi,poa->pia", tau157, tau58)
    )

    tau277 = (
        einsum("pjb,baji->pia", tau276, rt2)
    )

    tau278 = (
        einsum("aj,pji->pia", a.t1, tau258)
    )

    tau279 = (
        einsum("pia,oia->po", tau278, h.l.pov)
    )

    tau280 = (
        einsum("po,oia->pia", tau279, h.l.pov)
    )

    tau281 = (
        einsum("bi,pia->pab", a.t1, tau201)
    )

    tau282 = (
        einsum("pab,oab->po", tau281, h.l.pvv)
    )

    tau283 = (
        einsum("po,oia->pia", tau282, h.l.pov)
    )

    tau284 = (
        einsum("po,oab->pab", tau4, h.l.pvv)
    )

    tau285 = (
        einsum("bi,pab->pia", a.t1, tau284)
    )

    tau286 = (
        einsum("pjb,abji->pia", tau285, rt2)
    )

    tau287 = (
        einsum("iq,poi->pqo", a.x4, tau95)
    )

    tau288 = (
        einsum("pq,iq,qpo->poi", tau18, a.x3, tau287)
    )

    tau289 = (
        einsum("ip,pqi->pq", a.x3, tau162)
    )

    tau290 = (
        einsum("qp,iq,qpo->poi", tau289, a.x4, tau218)
    )

    tau291 = (
        einsum("ia,ap->pi", h.f.ov, a.x2)
    )

    tau292 = (
        einsum("ai,pi->pa", a.t1, tau291)
    )

    tau293 = (
        einsum("pb,jp,baji->pia", tau292, a.x3, rt2)
    )

    tau294 = (
        einsum("pi,iq->pq", tau170, a.x3)
    )

    tau295 = (
        einsum("qp,iq->pi", tau294, a.x3)
    )

    tau296 = (
        einsum("pi,iq->pq", tau13, a.x3)
    )

    tau297 = (
        einsum("qp,iq->pi", tau296, a.x4)
    )

    tau298 = (
        einsum("pa,aq->pq", tau55, a.x2)
    )

    tau299 = (
        einsum("po,oia->pia", tau195, h.l.pov)
    )

    tau300 = (
        einsum("ip,pia->pa", a.x3, tau299)
    )

    tau301 = (
        einsum("ap,qia->pqi", a.x1, tau47)
    )

    tau302 = (
        einsum("ip,pqi->pq", a.x3, tau301)
    )

    tau303 = (
        einsum("qp,iq,qpo->poi", tau302, a.x4, tau90)
    )

    tau304 = (
        einsum("pb,jp,abij->pia", tau246, a.x4, rt2)
    )

    tau305 = (
        einsum("pa,ip,oia->po", tau25, a.x3, h.l.pov)
    )

    tau306 = (
        einsum("po,oia->pia", tau305, h.l.pov)
    )

    tau307 = (
        einsum("pjb,abij->pia", tau276, rt2)
    )

    tau308 = (
        einsum("pj,bp,abij->pia", tau172, a.x2, rt2)
    )

    tau309 = (
        einsum("pb,jp,baij->pia", tau99, a.x4, rt2)
    )

    tau310 = (
        einsum("qo,po->pq", tau11, tau28)
    )

    tau311 = (
        einsum("qp,qo->po", tau191, tau195)
    )

    tau312 = (
        einsum("po,oia->pia", tau311, h.l.pov)
    )

    tau313 = (
        einsum("jp,oji->poi", a.x3, h.l.poo)
    )

    tau314 = (
        einsum("pji,poj->poi", tau258, tau313)
    )

    tau315 = (
        einsum("pj,bp,abij->pia", tau43, a.x2, rt2)
    )

    tau316 = (
        einsum("pb,jp,baji->pia", tau292, a.x4, rt2)
    )

    tau317 = (
        einsum("pia,poi->poa", tau29, tau62)
    )

    tau318 = (
        einsum("ap,pia->pi", a.x2, tau87)
    )

    tau319 = (
        einsum("pi,iq->pq", tau318, a.x3)
    )

    tau320 = (
        einsum("qp,iq->pi", tau319, a.x4)
    )

    tau321 = (
        einsum("pjb,baij->pia", tau205, rt2)
    )

    tau322 = (
        einsum("aq,pia->pqi", a.x2, tau123)
    )

    tau323 = (
        einsum("iq,pqi->pq", a.x4, tau322)
    )

    tau324 = (
        einsum("qo,qp->po", tau11, tau323)
    )

    tau325 = (
        einsum("po,oia->pia", tau324, h.l.pov)
    )

    tau326 = (
        einsum("pj,bp,baij->pia", tau43, a.x2, rt2)
    )

    tau327 = (
        einsum("ap,ip,oia->po", a.x1, a.x3, h.l.pov)
    )

    tau328 = (
        einsum("ip,qpi->pq", a.x4, tau196)
    )

    tau329 = (
        einsum("qo,pq->po", tau327, tau328)
    )

    tau330 = (
        einsum("po,oia->pia", tau329, h.l.pov)
    )

    tau331 = (
        einsum("pb,jp,abji->pia", tau292, a.x4, rt2)
    )

    tau332 = (
        einsum("qo,po->pq", tau11, tau114)
    )

    tau333 = (
        einsum("iq,pqi->pq", a.x4, tau230)
    )

    tau334 = (
        einsum("pb,jp,baij->pia", tau99, a.x3, rt2)
    )

    tau335 = (
        einsum("pia,oai->po", tau34, h.l.pvo)
    )

    tau336 = (
        einsum("po,oia->pia", tau335, h.l.pov)
    )

    tau337 = (
        einsum("ji,jp->pi", h.f.oo, a.x4)
    )

    tau338 = (
        einsum("pj,bp,baij->pia", tau337, a.x2, rt2)
    )

    tau339 = (
        einsum("iq,pqi->pq", a.x3, tau322)
    )

    tau340 = (
        einsum("qp,iq,qpo->poi", tau339, a.x3, tau90)
    )

    tau341 = (
        einsum("pi,iq->pq", tau153, a.x3)
    )

    tau342 = (
        einsum("qp,iq->pi", tau341, a.x3)
    )

    tau343 = (
        einsum("pi,ap,oia->po", tau342, a.x2, h.l.pov)
    )

    tau344 = (
        einsum("po,oia->pia", tau343, h.l.pov)
    )

    tau345 = (
        einsum("ip,pia->pa", a.x4, tau87)
    )

    tau346 = (
        einsum("pa,aq->pq", tau345, a.x2)
    )

    tau347 = (
        einsum("qp,aq->pa", tau346, a.x2)
    )

    tau348 = (
        einsum("pa,ip,oia->po", tau347, a.x3, h.l.pov)
    )

    tau349 = (
        einsum("po,oia->pia", tau348, h.l.pov)
    )

    tau350 = (
        einsum("iq,pqi->pq", a.x3, tau17)
    )

    tau351 = (
        einsum("qp,iq,qpo->poi", tau350, a.x4, tau287)
    )

    tau352 = (
        einsum("poi,pia->poa", tau157, tau22)
    )

    tau353 = (
        einsum("pjb,baij->pia", tau256, rt2)
    )

    tau354 = (
        einsum("ai,pa->pi", a.t1, tau55)
    )

    tau355 = (
        einsum("ai,pa->pi", a.t1, tau23)
    )

    tau356 = (
        einsum("ap,qia->pqi", a.x1, tau16)
    )

    tau357 = (
        einsum("ip,pqi->pq", a.x3, tau356)
    )

    tau358 = (
        einsum("qp,qo->po", tau357, tau4)
    )

    tau359 = (
        einsum("po,oia->pia", tau358, h.l.pov)
    )

    tau360 = (
        einsum("ji,jp->pi", h.f.oo, a.x3)
    )

    tau361 = (
        einsum("pj,bp,baij->pia", tau360, a.x2, rt2)
    )

    tau362 = (
        einsum("pb,jp,baij->pia", tau189, a.x3, rt2)
    )

    tau363 = (
        einsum("pji,poj->poi", tau214, tau89)
    )

    tau364 = (
        einsum("bi,pba->pia", a.t1, tau248)
    )

    tau365 = (
        einsum("aj,pia->pij", a.t1, tau364)
    )

    tau366 = (
        einsum("pji,poj->poi", tau365, tau88)
    )

    tau367 = (
        einsum("bp,ap,abij->pij", a.x1, a.x2, rt2)
    )

    tau368 = (
        einsum("jq,iq,pij->pq", a.x3, a.x4, tau367)
    )

    tau369 = (
        einsum("qp,iq,jq->pij", tau368, a.x3, a.x4)
    )

    tau370 = (
        einsum("pji,poj->poi", tau369, tau89)
    )

    tau371 = (
        einsum("po,qo->pq", tau254, tau4)
    )

    tau372 = (
        einsum("qp,iq,qpo->poi", tau163, a.x3, tau218)
    )

    tau373 = (
        einsum("iq,jq,pij->pq", a.x3, a.x4, tau212)
    )

    tau374 = (
        einsum("qp,iq,jq->pij", tau373, a.x3, a.x4)
    )

    tau375 = (
        einsum("pij,poj->poi", tau374, tau89)
    )

    tau376 = (
        einsum("poi,poj->pij", tau72, tau89)
    )

    tau377 = (
        einsum("aj,pij->pia", a.t1, tau376)
    )

    tau378 = (
        einsum("pjb,abji->pia", tau377, rt2)
    )

    tau379 = (
        einsum("pjb,baji->pia", tau262, rt2)
    )

    tau380 = (
        einsum("iq,pqi->pq", a.x3, tau175)
    )

    tau381 = (
        einsum("qp,qo->po", tau380, tau4)
    )

    tau382 = (
        einsum("po,oia->pia", tau381, h.l.pov)
    )

    tau383 = (
        einsum("aj,pia->pij", a.t1, tau201)
    )

    tau384 = (
        einsum("pji,poj->poi", tau383, tau59)
    )

    tau385 = (
        einsum("pb,jp,abij->pia", tau210, a.x3, rt2)
    )

    tau386 = (
        einsum("o,oij->ij", tau6, h.l.poo)
    )

    tau387 = (
        einsum("ji,jp->pi", tau386, a.x3)
    )

    tau388 = (
        einsum("jq,iq,pij->pq", a.x3, a.x4, tau132)
    )

    tau389 = (
        einsum("qp,iq,jq->pij", tau388, a.x3, a.x4)
    )

    tau390 = (
        einsum("pji,abij->pab", tau389, rt2)
    )

    tau391 = (
        einsum("pa,aq->pq", tau102, a.x2)
    )

    tau392 = (
        einsum("qp,aq->pa", tau391, a.x2)
    )

    tau393 = (
        einsum("pa,ip,oia->po", tau392, a.x3, h.l.pov)
    )

    tau394 = (
        einsum("po,oia->pia", tau393, h.l.pov)
    )

    tau395 = (
        einsum("pia,poa->poi", tau364, tau58)
    )

    tau396 = (
        einsum("pjb,baji->pia", tau60, rt2)
    )

    tau397 = (
        einsum("ji,jp->pi", tau272, a.x3)
    )

    tau398 = (
        einsum("pj,bp,baji->pia", tau397, a.x2, rt2)
    )

    tau399 = (
        einsum("pj,bp,baij->pia", tau15, a.x2, rt2)
    )

    tau400 = (
        einsum("pj,bp,baij->pia", tau68, a.x2, rt2)
    )

    tau401 = (
        einsum("pj,bp,abij->pia", tau295, a.x2, rt2)
    )

    tau402 = (
        einsum("pi,ap,oia->po", tau320, a.x2, h.l.pov)
    )

    tau403 = (
        einsum("po,oia->pia", tau402, h.l.pov)
    )

    tau404 = (
        einsum("qp,aq->pa", tau104, a.x1)
    )

    tau405 = (
        einsum("pb,jp,baij->pia", tau404, a.x3, rt2)
    )

    tau406 = (
        einsum("pj,bp,baij->pia", tau297, a.x2, rt2)
    )

    tau407 = (
        einsum("jp,oij->poi", a.x3, tau79)
    )

    tau408 = (
        einsum("poj,poi->pij", tau200, tau407)
    )

    tau409 = (
        einsum("pij,abij->pab", tau408, rt2)
    )

    tau410 = (
        einsum("pia,oai->po", tau201, h.l.pvo)
    )

    tau411 = (
        einsum("po,oia->pia", tau410, h.l.pov)
    )

    tau412 = (
        einsum("poi,poa->pia", tau200, tau58)
    )

    tau413 = (
        einsum("pjb,abji->pia", tau412, rt2)
    )

    tau414 = (
        einsum("pq,iq,qpo->poi", tau31, a.x3, tau20)
    )

    tau415 = (
        einsum("poi,poj->pij", tau407, tau59)
    )

    tau416 = (
        einsum("pij,abij->pab", tau415, rt2)
    )

    tau417 = (
        einsum("pi,iq->pq", tau66, a.x3)
    )

    tau418 = (
        einsum("qp,iq->pi", tau417, a.x4)
    )

    tau419 = (
        einsum("pb,jp,abji->pia", tau292, a.x3, rt2)
    )

    tau420 = (
        einsum("pia,poi->poa", tau136, tau200)
    )

    tau421 = (
        einsum("pjb,baij->pia", tau285, rt2)
    )

    tau422 = (
        einsum("po,oij->pij", tau4, h.l.poo)
    )

    tau423 = (
        einsum("aj,pji->pia", a.t1, tau422)
    )

    tau424 = (
        einsum("pjb,abji->pia", tau423, rt2)
    )

    tau425 = (
        einsum("qa,ap->pq", tau55, a.x1)
    )

    tau426 = (
        einsum("poj,poi->pij", tau313, tau59)
    )

    tau427 = (
        einsum("pji,abij->pab", tau426, rt2)
    )

    tau428 = (
        einsum("ip,pqi->pq", a.x4, tau301)
    )

    tau429 = (
        einsum("qp,iq,qpo->poi", tau428, a.x3, tau90)
    )

    tau430 = (
        einsum("ai,pa->pi", a.t1, tau48)
    )

    tau431 = (
        einsum("pjb,abji->pia", tau60, rt2)
    )

    tau432 = (
        einsum("aj,pia->pij", a.t1, tau63)
    )

    tau433 = (
        einsum("pij,oji->po", tau432, h.l.poo)
    )

    tau434 = (
        einsum("po,oia->pia", tau433, h.l.pov)
    )

    tau435 = (
        einsum("pb,jp,baij->pia", tau246, a.x3, rt2)
    )

    tau436 = (
        einsum("aq,pia->pqi", a.x2, tau87)
    )

    tau437 = (
        einsum("iq,pqi->pq", a.x3, tau436)
    )

    tau438 = (
        einsum("qo,qp->po", tau4, tau437)
    )

    tau439 = (
        einsum("po,oia->pia", tau438, h.l.pov)
    )

    tau440 = (
        einsum("pji,poj->poi", tau232, tau62)
    )

    tau441 = (
        einsum("pia,poi->poa", tau364, tau89)
    )

    tau442 = (
        einsum("ij,jp->pi", tau241, a.x4)
    )

    tau443 = (
        einsum("pi,iq->pq", tau150, a.x3)
    )

    tau444 = (
        einsum("qp,iq->pi", tau443, a.x3)
    )

    tau445 = (
        einsum("pi,ap,oia->po", tau444, a.x2, h.l.pov)
    )

    tau446 = (
        einsum("po,oia->pia", tau445, h.l.pov)
    )

    tau447 = (
        einsum("bi,pia->pab", a.t1, tau22)
    )

    tau448 = (
        einsum("pab,oab->po", tau447, h.l.pvv)
    )

    tau449 = (
        einsum("po,oia->pia", tau448, h.l.pov)
    )

    tau450 = (
        einsum("iq,jq,pij->pq", a.x3, a.x4, tau367)
    )

    tau451 = (
        einsum("qp,iq,jq->pij", tau450, a.x3, a.x4)
    )

    tau452 = (
        einsum("pji,poj->poi", tau451, tau89)
    )

    tau453 = (
        einsum("ap,pia->pi", a.x1, tau47)
    )

    tau454 = (
        einsum("pi,iq->pq", tau453, a.x4)
    )

    tau455 = (
        einsum("qp,iq->pi", tau454, a.x4)
    )

    tau456 = (
        einsum("pjb,baji->pia", tau412, rt2)
    )

    tau457 = (
        einsum("pi,iq->pq", tau146, a.x4)
    )

    tau458 = (
        einsum("qp,iq->pi", tau457, a.x3)
    )

    tau459 = (
        einsum("qo,qp->po", tau105, tau350)
    )

    tau460 = (
        einsum("po,oia->pia", tau459, h.l.pov)
    )

    tau461 = (
        einsum("pij,oji->po", tau383, h.l.poo)
    )

    tau462 = (
        einsum("po,oia->pia", tau461, h.l.pov)
    )

    tau463 = (
        einsum("pij,poj->poi", tau365, tau89)
    )

    tau464 = (
        einsum("qp,qo->po", tau129, tau195)
    )

    tau465 = (
        einsum("po,oia->pia", tau464, h.l.pov)
    )

    tau466 = (
        einsum("pj,bp,abij->pia", tau77, a.x2, rt2)
    )

    tau467 = (
        einsum("pjb,abji->pia", tau276, rt2)
    )

    tau468 = (
        einsum("pi,ap,oia->po", tau155, a.x2, h.l.pov)
    )

    tau469 = (
        einsum("po,oia->pia", tau468, h.l.pov)
    )

    tau470 = (
        einsum("pa,oia->poi", tau48, tau209)
    )

    tau471 = (
        einsum("qp,aq->pa", tau117, a.x1)
    )

    tau472 = (
        einsum("pb,jp,abij->pia", tau471, a.x3, rt2)
    )

    tau473 = (
        einsum("qa,ap->pq", tau103, a.x1)
    )

    tau474 = (
        einsum("poj,pji->poi", tau200, tau383)
    )

    tau475 = (
        einsum("aj,pia->pij", a.t1, tau136)
    )

    tau476 = (
        einsum("pij,oji->po", tau475, h.l.poo)
    )

    tau477 = (
        einsum("po,oia->pia", tau476, h.l.pov)
    )

    tau478 = (
        einsum("pi,iq->pq", tau318, a.x4)
    )

    tau479 = (
        einsum("qp,iq->pi", tau478, a.x4)
    )

    tau480 = (
        einsum("pb,jp,baij->pia", tau9, a.x4, rt2)
    )

    tau481 = (
        einsum("pb,jp,baij->pia", tau121, a.x3, rt2)
    )

    tau482 = (
        einsum("pia,poi->poa", tau47, tau59)
    )

    tau483 = (
        einsum("pi,iq->pq", tau35, a.x3)
    )

    tau484 = (
        einsum("qp,iq->pi", tau483, a.x3)
    )

    tau485 = (
        einsum("pj,bp,baij->pia", tau295, a.x2, rt2)
    )

    tau486 = (
        einsum("pj,bp,abji->pia", tau360, a.x2, rt2)
    )

    tau487 = (
        einsum("qo,qp->po", tau11, tau339)
    )

    tau488 = (
        einsum("po,oia->pia", tau487, h.l.pov)
    )

    tau489 = (
        einsum("pia,poi->poa", tau34, tau62)
    )

    tau490 = (
        einsum("pb,jp,abij->pia", tau9, a.x3, rt2)
    )

    tau491 = (
        einsum("pb,jp,abij->pia", tau404, a.x3, rt2)
    )

    tau492 = (
        einsum("qp,qo->po", tau302, tau4)
    )

    tau493 = (
        einsum("po,oia->pia", tau492, h.l.pov)
    )

    tau494 = (
        einsum("pi,iq->pq", tau453, a.x3)
    )

    tau495 = (
        einsum("qp,iq->pi", tau494, a.x4)
    )

    tau496 = (
        einsum("pj,bp,abij->pia", tau242, a.x2, rt2)
    )

    tau497 = (
        einsum("poi,pia->poa", tau200, tau47)
    )

    tau498 = (
        einsum("bi,pia->pab", a.t1, tau47)
    )

    tau499 = (
        einsum("pab,oab->po", tau498, h.l.pvv)
    )

    tau500 = (
        einsum("po,oia->pia", tau499, h.l.pov)
    )

    tau501 = (
        einsum("qp,aq->pa", tau298, a.x1)
    )

    tau502 = (
        einsum("pa,ip,oia->po", tau501, a.x4, h.l.pov)
    )

    tau503 = (
        einsum("po,oia->pia", tau502, h.l.pov)
    )

    tau504 = (
        einsum("aq,pia->pqi", a.x2, tau149)
    )

    tau505 = (
        einsum("iq,pqi->pq", a.x4, tau504)
    )

    tau506 = (
        einsum("qo,qp->po", tau11, tau505)
    )

    tau507 = (
        einsum("po,oia->pia", tau506, h.l.pov)
    )

    tau508 = (
        einsum("bi,pia->pab", a.t1, tau136)
    )

    tau509 = (
        einsum("pab,oab->po", tau508, h.l.pvv)
    )

    tau510 = (
        einsum("po,oia->pia", tau509, h.l.pov)
    )

    tau511 = (
        einsum("ji,jp->pi", tau386, a.x4)
    )

    tau512 = (
        einsum("pj,bp,baij->pia", tau242, a.x2, rt2)
    )

    tau513 = (
        einsum("pij,poj->poi", tau451, tau89)
    )

    tau514 = (
        einsum("pj,bp,baji->pia", tau337, a.x2, rt2)
    )

    tau515 = (
        einsum("pia,poi->poa", tau22, tau62)
    )

    tau516 = (
        einsum("pij,abij->pab", tau134, rt2)
    )

    tau517 = (
        einsum("poj,poi->pij", tau200, tau313)
    )

    tau518 = (
        einsum("pij,abij->pab", tau517, rt2)
    )

    tau519 = (
        einsum("iq,pqi->pq", a.x4, tau436)
    )

    tau520 = (
        einsum("qp,iq,qpo->poi", tau519, a.x4, tau218)
    )

    tau521 = (
        einsum("ji,jp->pi", tau76, a.x4)
    )

    tau522 = (
        einsum("pj,bp,baij->pia", tau521, a.x2, rt2)
    )

    tau523 = (
        einsum("po,oai->pia", tau4, h.l.pvo)
    )

    tau524 = (
        einsum("pjb,baij->pia", tau523, rt2)
    )

    tau525 = (
        einsum("pij,oji->po", tau232, h.l.poo)
    )

    tau526 = (
        einsum("po,oia->pia", tau525, h.l.pov)
    )

    tau527 = (
        einsum("pa,ip,oia->po", tau392, a.x4, h.l.pov)
    )

    tau528 = (
        einsum("po,oia->pia", tau527, h.l.pov)
    )

    tau529 = (
        einsum("pi,iq->pq", tau137, a.x3)
    )

    tau530 = (
        einsum("qp,iq->pi", tau529, a.x4)
    )

    tau531 = (
        einsum("pb,oba->poa", tau184, h.l.pvv)
    )

    tau532 = (
        einsum("ai,poa->poi", a.t1, tau531)
    )

    tau533 = (
        einsum("pb,jp,abij->pia", tau9, a.x4, rt2)
    )

    tau534 = (
        einsum("poi,poj->pij", tau62, tau72)
    )

    tau535 = (
        einsum("pji,abij->pab", tau534, rt2)
    )

    tau536 = (
        einsum("qp,iq,qpo->poi", tau176, a.x4, tau218)
    )

    tau537 = (
        einsum("pa,aq->pq", tau48, a.x2)
    )

    tau538 = (
        einsum("pji,poj->poi", tau432, tau62)
    )

    tau539 = (
        einsum("iq,pqi->pq", a.x3, tau504)
    )

    tau540 = (
        einsum("qo,qp->po", tau11, tau539)
    )

    tau541 = (
        einsum("po,oia->pia", tau540, h.l.pov)
    )

    tau542 = (
        einsum("iq,pqi->pq", a.x4, tau182)
    )

    tau543 = (
        einsum("pjb,baji->pia", tau377, rt2)
    )

    tau544 = (
        einsum("ij,jp->pi", tau42, a.x3)
    )

    tau545 = (
        einsum("pj,bp,abij->pia", tau544, a.x2, rt2)
    )

    tau546 = (
        einsum("pa,aq->pq", tau203, a.x2)
    )

    tau547 = (
        einsum("qp,aq->pa", tau546, a.x2)
    )

    tau548 = (
        einsum("pa,ip,oia->po", tau547, a.x3, h.l.pov)
    )

    tau549 = (
        einsum("po,oia->pia", tau548, h.l.pov)
    )

    tau550 = (
        einsum("pb,jp,abij->pia", tau45, a.x3, rt2)
    )

    tau551 = (
        einsum("poj,poi->pij", tau200, tau88)
    )

    tau552 = (
        einsum("aj,pji->pia", a.t1, tau551)
    )

    tau553 = (
        einsum("pjb,abij->pia", tau552, rt2)
    )

    tau554 = (
        einsum("pb,jp,abij->pia", tau121, a.x4, rt2)
    )

    tau555 = (
        einsum("pi,ap,oia->po", tau455, a.x2, h.l.pov)
    )

    tau556 = (
        einsum("po,oia->pia", tau555, h.l.pov)
    )

    tau557 = (
        einsum("pi,ap,oia->po", tau530, a.x2, h.l.pov)
    )

    tau558 = (
        einsum("po,oia->pia", tau557, h.l.pov)
    )

    tau559 = (
        einsum("ai,pja->pij", a.t1, tau12)
    )

    tau560 = (
        einsum("aj,pij->pia", a.t1, tau559)
    )

    tau561 = (
        einsum("pjb,abji->pia", tau560, rt2)
    )

    tau562 = (
        einsum("pjb,baij->pia", tau60, rt2)
    )

    tau563 = (
        einsum("aj,pji->pia", a.t1, tau73)
    )

    tau564 = (
        einsum("pia,oia->po", tau563, h.l.pov)
    )

    tau565 = (
        einsum("po,oia->pia", tau564, h.l.pov)
    )

    tau566 = (
        einsum("po,oab->pab", tau11, h.l.pvv)
    )

    tau567 = (
        einsum("bi,pab->pia", a.t1, tau566)
    )

    tau568 = (
        einsum("pjb,abij->pia", tau567, rt2)
    )

    tau569 = (
        einsum("ip,pqi->pq", a.x4, tau236)
    )

    tau570 = (
        einsum("qo,qp->po", tau11, tau569)
    )

    tau571 = (
        einsum("po,oia->pia", tau570, h.l.pov)
    )

    tau572 = (
        einsum("okj,oki->ij", tau40, h.l.poo)
    )

    tau573 = (
        einsum("ij,jp->pi", tau572, a.x3)
    )

    tau574 = (
        einsum("pia,poa->poi", tau249, tau58)
    )

    tau575 = (
        einsum("pa,oia->poi", tau23, tau209)
    )

    tau576 = (
        einsum("pij,abij->pab", tau389, rt2)
    )

    tau577 = (
        einsum("qp,iq,qpo->poi", tau129, a.x4, tau20)
    )

    tau578 = (
        einsum("pb,jp,abij->pia", tau45, a.x4, rt2)
    )

    tau579 = (
        einsum("pj,bp,abij->pia", tau521, a.x2, rt2)
    )

    tau580 = (
        einsum("poi,poj->pij", tau313, tau89)
    )

    tau581 = (
        einsum("aj,pij->pia", a.t1, tau580)
    )

    tau582 = (
        einsum("pjb,baji->pia", tau581, rt2)
    )

    tau583 = (
        einsum("pj,bp,abij->pia", tau442, a.x2, rt2)
    )

    tau584 = (
        einsum("pa,ip,oia->po", tau501, a.x3, h.l.pov)
    )

    tau585 = (
        einsum("po,oia->pia", tau584, h.l.pov)
    )

    tau586 = (
        einsum("pj,bp,abij->pia", tau15, a.x2, rt2)
    )

    tau587 = (
        einsum("qa,ap->pq", tau184, a.x1)
    )

    tau588 = (
        einsum("qo,qp->po", tau28, tau328)
    )

    tau589 = (
        einsum("po,oia->pia", tau588, h.l.pov)
    )

    tau590 = (
        einsum("pj,bp,abij->pia", tau81, a.x2, rt2)
    )

    tau591 = (
        einsum("pq,qo->po", tau18, tau28)
    )

    tau592 = (
        einsum("po,oia->pia", tau591, h.l.pov)
    )

    tau593 = (
        einsum("pia,poi->poa", tau228, tau89)
    )

    tau594 = (
        einsum("pia,oai->po", tau22, h.l.pvo)
    )

    tau595 = (
        einsum("po,oia->pia", tau594, h.l.pov)
    )

    tau596 = (
        einsum("pq,iq,qpo->poi", tau328, a.x3, tau20)
    )

    tau597 = (
        einsum("qp,iq,qpo->poi", tau539, a.x3, tau90)
    )

    tau598 = (
        einsum("pj,bp,baji->pia", tau360, a.x2, rt2)
    )

    tau599 = (
        einsum("pi,ap,oia->po", tau479, a.x2, h.l.pov)
    )

    tau600 = (
        einsum("po,oia->pia", tau599, h.l.pov)
    )

    tau601 = (
        einsum("po,qo->pq", tau28, tau4)
    )

    tau602 = (
        einsum("pjb,baij->pia", tau167, rt2)
    )

    tau603 = (
        einsum("pjb,abij->pia", tau412, rt2)
    )

    tau604 = (
        einsum("pjb,abij->pia", tau523, rt2)
    )

    tau605 = (
        einsum("pj,bp,baij->pia", tau544, a.x2, rt2)
    )

    tau606 = (
        einsum("ip,pqi->pq", a.x4, tau356)
    )

    tau607 = (
        einsum("qp,iq,qpo->poi", tau606, a.x3, tau90)
    )

    tau608 = (
        einsum("aj,pji->pia", a.t1, tau475)
    )

    tau609 = (
        einsum("pia,oia->po", tau608, h.l.pov)
    )

    tau610 = (
        einsum("po,oia->pia", tau609, h.l.pov)
    )

    tau611 = (
        einsum("pq,iq,qpo->poi", tau197, a.x3, tau287)
    )

    tau612 = (
        einsum("pi,iq->pq", tau107, a.x4)
    )

    tau613 = (
        einsum("qp,iq->pi", tau612, a.x3)
    )

    tau614 = (
        einsum("pj,bp,baij->pia", tau613, a.x2, rt2)
    )

    tau615 = (
        einsum("poj,pji->poi", tau313, tau51)
    )

    tau616 = (
        einsum("pj,bp,abij->pia", tau613, a.x2, rt2)
    )

    tau617 = (
        einsum("pi,ap,oia->po", tau458, a.x2, h.l.pov)
    )

    tau618 = (
        einsum("po,oia->pia", tau617, h.l.pov)
    )

    tau619 = (
        einsum("pjb,abji->pia", tau523, rt2)
    )

    tau620 = (
        einsum("aj,pji->pia", a.t1, tau383)
    )

    tau621 = (
        einsum("pia,oia->po", tau620, h.l.pov)
    )

    tau622 = (
        einsum("po,oia->pia", tau621, h.l.pov)
    )

    tau623 = (
        einsum("pij,oji->po", tau51, h.l.poo)
    )

    tau624 = (
        einsum("po,oia->pia", tau623, h.l.pov)
    )

    tau625 = (
        einsum("qp,iq,qpo->poi", tau357, a.x4, tau90)
    )

    tau626 = (
        einsum("pjb,baij->pia", tau180, rt2)
    )

    tau627 = (
        einsum("pb,jp,baij->pia", tau189, a.x4, rt2)
    )

    tau628 = (
        einsum("ij,jp->pi", tau572, a.x4)
    )

    tau629 = (
        einsum("pia,oai->po", tau47, h.l.pvo)
    )

    tau630 = (
        einsum("po,oia->pia", tau629, h.l.pov)
    )

    tau631 = (
        einsum("ai,pa->pi", a.t1, tau184)
    )

    tau632 = (
        einsum("pjb,baij->pia", tau560, rt2)
    )

    tau633 = (
        einsum("iq,poi->pqo", a.x4, tau19)
    )

    tau634 = (
        einsum("pq,iq,qpo->poi", tau350, a.x4, tau633)
    )

    tau635 = (
        einsum("poi,poj->pij", tau62, tau88)
    )

    tau636 = (
        einsum("aj,pij->pia", a.t1, tau635)
    )

    tau637 = (
        einsum("pjb,baij->pia", tau636, rt2)
    )

    tau638 = (
        einsum("pjb,abji->pia", tau567, rt2)
    )

    tau639 = (
        einsum("pjb,baji->pia", tau167, rt2)
    )

    tau640 = (
        einsum("pjb,baji->pia", tau274, rt2)
    )

    tau641 = (
        einsum("pij,oji->po", tau258, h.l.poo)
    )

    tau642 = (
        einsum("po,oia->pia", tau641, h.l.pov)
    )

    tau643 = (
        einsum("pji,abij->pab", tau415, rt2)
    )

    tau644 = (
        einsum("pq,iq,qpo->poi", tau111, a.x4, tau633)
    )

    tau645 = (
        einsum("pia,oai->po", tau63, h.l.pvo)
    )

    tau646 = (
        einsum("po,oia->pia", tau645, h.l.pov)
    )

    tau647 = (
        einsum("pj,bp,abij->pia", tau337, a.x2, rt2)
    )

    tau648 = (
        einsum("poi,pia->poa", tau157, tau29)
    )

    tau649 = (
        einsum("pia,oai->po", tau29, h.l.pvo)
    )

    tau650 = (
        einsum("po,oia->pia", tau649, h.l.pov)
    )

    tau651 = (
        einsum("qp,iq,qpo->poi", tau569, a.x3, tau218)
    )

    tau652 = (
        einsum("po,oij->pij", tau114, h.l.poo)
    )

    tau653 = (
        einsum("aj,pji->pia", a.t1, tau652)
    )

    tau654 = (
        einsum("pjb,abij->pia", tau653, rt2)
    )

    tau655 = (
        einsum("pia,poi->poa", tau136, tau59)
    )

    tau656 = (
        einsum("pji,poj->poi", tau475, tau59)
    )

    tau657 = (
        einsum("qp,iq,qpo->poi", tau437, a.x4, tau90)
    )

    tau658 = (
        einsum("qo,qp->po", tau11, tau606)
    )

    tau659 = (
        einsum("po,oia->pia", tau658, h.l.pov)
    )

    tau660 = (
        einsum("poj,pji->poi", tau59, tau73)
    )

    tau661 = (
        einsum("pjb,baji->pia", tau143, rt2)
    )

    tau662 = (
        einsum("aj,pia->pij", a.t1, tau173)
    )

    tau663 = (
        einsum("pij,poj->poi", tau662, tau89)
    )

    tau664 = (
        einsum("poj,pji->poi", tau157, tau232)
    )

    tau665 = (
        einsum("pj,bp,baij->pia", tau442, a.x2, rt2)
    )

    tau666 = (
        einsum("qo,pq->po", tau195, tau350)
    )

    tau667 = (
        einsum("po,oia->pia", tau666, h.l.pov)
    )

    tau668 = (
        einsum("pq,qo->po", tau111, tau195)
    )

    tau669 = (
        einsum("po,oia->pia", tau668, h.l.pov)
    )

    tau670 = (
        einsum("qp,iq,qpo->poi", tau380, a.x4, tau90)
    )

    tau671 = (
        einsum("pb,jp,baij->pia", tau121, a.x4, rt2)
    )

    tau672 = (
        einsum("pjb,abij->pia", tau560, rt2)
    )

    tau673 = (
        einsum("pjb,baji->pia", tau523, rt2)
    )

    tau674 = (
        einsum("pi,ap,oia->po", tau484, a.x2, h.l.pov)
    )

    tau675 = (
        einsum("po,oia->pia", tau674, h.l.pov)
    )

    tau676 = (
        einsum("pb,jp,abij->pia", tau471, a.x4, rt2)
    )

    tau677 = (
        einsum("pb,jp,baij->pia", tau45, a.x4, rt2)
    )

    tau678 = (
        einsum("pjb,baij->pia", tau276, rt2)
    )

    tau679 = (
        einsum("bi,pia->pab", a.t1, tau34)
    )

    tau680 = (
        einsum("pab,oab->po", tau679, h.l.pvv)
    )

    tau681 = (
        einsum("po,oia->pia", tau680, h.l.pov)
    )

    tau682 = (
        einsum("pjb,abij->pia", tau636, rt2)
    )

    tau683 = (
        einsum("pjb,baji->pia", tau285, rt2)
    )

    tau684 = (
        einsum("pjb,baij->pia", tau412, rt2)
    )

    tau685 = (
        einsum("pi,ap,oia->po", tau161, a.x2, h.l.pov)
    )

    tau686 = (
        einsum("po,oia->pia", tau685, h.l.pov)
    )

    tau687 = (
        einsum("pi,ap,oia->po", tau3, a.x2, h.l.pov)
    )

    tau688 = (
        einsum("po,oia->pia", tau687, h.l.pov)
    )

    tau689 = (
        einsum("pj,bp,abij->pia", tau418, a.x2, rt2)
    )

    tau690 = (
        einsum("poi,pia->poa", tau157, tau34)
    )

    tau691 = (
        einsum("pjb,baji->pia", tau560, rt2)
    )

    tau692 = (
        einsum("pj,bp,abij->pia", tau269, a.x2, rt2)
    )

    tau693 = (
        einsum("pj,bp,abij->pia", tau297, a.x2, rt2)
    )

    tau694 = (
        einsum("pi,ap,oia->po", tau148, a.x2, h.l.pov)
    )

    tau695 = (
        einsum("po,oia->pia", tau694, h.l.pov)
    )

    tau696 = (
        einsum("pj,bp,abji->pia", tau273, a.x2, rt2)
    )

    tau697 = (
        einsum("pb,jp,abij->pia", tau189, a.x3, rt2)
    )

    tau698 = (
        einsum("pi,ap,oia->po", tau152, a.x2, h.l.pov)
    )

    tau699 = (
        einsum("po,oia->pia", tau698, h.l.pov)
    )

    tau700 = (
        einsum("pjb,baij->pia", tau567, rt2)
    )

    tau701 = (
        einsum("qp,aq->pa", tau537, a.x1)
    )

    tau702 = (
        einsum("pa,ip,oia->po", tau701, a.x4, h.l.pov)
    )

    tau703 = (
        einsum("po,oia->pia", tau702, h.l.pov)
    )

    tau704 = (
        einsum("pji,poj->poi", tau374, tau89)
    )

    tau705 = (
        einsum("pa,ip,oia->po", tau701, a.x3, h.l.pov)
    )

    tau706 = (
        einsum("po,oia->pia", tau705, h.l.pov)
    )

    tau707 = (
        einsum("bi,pia->pab", a.t1, tau63)
    )

    tau708 = (
        einsum("pab,oab->po", tau707, h.l.pvv)
    )

    tau709 = (
        einsum("po,oia->pia", tau708, h.l.pov)
    )

    tau710 = (
        einsum("pq,qo->po", tau31, tau327)
    )

    tau711 = (
        einsum("po,oia->pia", tau710, h.l.pov)
    )

    tau712 = (
        einsum("poj,pji->poi", tau59, tau83)
    )

    tau713 = (
        einsum("pa,ip,oia->po", tau547, a.x4, h.l.pov)
    )

    tau714 = (
        einsum("po,oia->pia", tau713, h.l.pov)
    )

    tau715 = (
        einsum("qo,qp->po", tau11, tau428)
    )

    tau716 = (
        einsum("po,oia->pia", tau715, h.l.pov)
    )

    tau717 = (
        einsum("pjb,abji->pia", tau167, rt2)
    )

    tau718 = (
        einsum("pjb,baji->pia", tau567, rt2)
    )

    tau719 = (
        einsum("qp,iq,qpo->poi", tau505, a.x3, tau218)
    )

    tau720 = (
        einsum("aj,pji->pia", a.t1, tau432)
    )

    tau721 = (
        einsum("pia,oia->po", tau720, h.l.pov)
    )

    tau722 = (
        einsum("po,oia->pia", tau721, h.l.pov)
    )

    tau723 = (
        einsum("pjb,abij->pia", tau285, rt2)
    )

    tau724 = (
        einsum("pq,iq,qpo->poi", tau191, a.x3, tau287)
    )

    tau725 = (
        einsum("pb,jp,baij->pia", tau471, a.x3, rt2)
    )

    tau726 = (
        einsum("pia,poi->poa", tau201, tau59)
    )

    tau727 = (
        einsum("pjb,baji->pia", tau423, rt2)
    )

    tau728 = (
        einsum("pb,jp,baij->pia", tau471, a.x4, rt2)
    )

    tau729 = (
        einsum("pb,jp,baij->pia", tau246, a.x4, rt2)
    )

    tau730 = (
        einsum("pj,bp,abji->pia", tau337, a.x2, rt2)
    )

    tau731 = (
        einsum("pb,jp,baij->pia", tau210, a.x3, rt2)
    )

    tau732 = (
        einsum("pb,jp,baij->pia", tau210, a.x4, rt2)
    )

    tau733 = (
        einsum("pj,bp,baij->pia", tau418, a.x2, rt2)
    )

    tau734 = (
        einsum("pj,bp,baji->pia", tau273, a.x2, rt2)
    )

    tau735 = (
        einsum("pq,qo->po", tau191, tau28)
    )

    tau736 = (
        einsum("po,oia->pia", tau735, h.l.pov)
    )

    tau737 = (
        einsum("qp,qo->po", tau289, tau4)
    )

    tau738 = (
        einsum("po,oia->pia", tau737, h.l.pov)
    )

    tau739 = (
        einsum("pj,bp,abji->pia", tau397, a.x2, rt2)
    )

    tau740 = (
        einsum("pjb,baji->pia", tau636, rt2)
    )

    tau741 = (
        einsum("qo,qp->po", tau4, tau519)
    )

    tau742 = (
        einsum("po,oia->pia", tau741, h.l.pov)
    )

    tau743 = (
        einsum("pia,poi->poa", tau173, tau89)
    )

    tau744 = (
        einsum("bi,pia->pab", a.t1, tau16)
    )

    tau745 = (
        einsum("pab,oab->po", tau744, h.l.pvv)
    )

    tau746 = (
        einsum("po,oia->pia", tau745, h.l.pov)
    )

    tau747 = (
        einsum("pjb,baij->pia", tau653, rt2)
    )

    tau748 = (
        einsum("pjb,baij->pia", tau552, rt2)
    )

    tau749 = (
        einsum("poj,pji->poi", tau157, tau432)
    )

    tau750 = (
        einsum("pq,iq,qpo->poi", tau129, a.x3, tau287)
    )

    tau751 = (
        einsum("qp,iq,qpo->poi", tau237, a.x4, tau218)
    )

    tau752 = (
        einsum("pb,jp,abij->pia", tau99, a.x3, rt2)
    )

    tau753 = (
        einsum("poi,pia->poa", tau157, tau63)
    )

    tau754 = (
        einsum("pba,pob->poa", tau248, tau58)
    )

    tau755 = (
        einsum("pj,bp,abij->pia", tau109, a.x2, rt2)
    )

    tau756 = (
        einsum("pjb,abji->pia", tau262, rt2)
    )

    tau757 = (
        einsum("qp,iq,qpo->poi", tau197, a.x4, tau20)
    )

    tau758 = (
        einsum("pa,ip,oia->po", tau251, a.x3, h.l.pov)
    )

    tau759 = (
        einsum("po,oia->pia", tau758, h.l.pov)
    )

    tau760 = (
        einsum("pjb,abij->pia", tau274, rt2)
    )

    tau761 = (
        einsum("pob,pba->poa", tau58, tau70)
    )

    tau762 = (
        einsum("pjb,abji->pia", tau636, rt2)
    )

    tau763 = (
        einsum("poj,pji->poi", tau72, tau83)
    )

    tau764 = (
        einsum("pb,jp,abij->pia", tau404, a.x4, rt2)
    )

    tau765 = (
        einsum("pia,oai->po", tau16, h.l.pvo)
    )

    tau766 = (
        einsum("po,oia->pia", tau765, h.l.pov)
    )

    tau767 = (
        einsum("pij,poj->poi", tau369, tau89)
    )

    tau768 = (
        einsum("pji,poj->poi", tau662, tau88)
    )

    tau769 = (
        einsum("pjb,baij->pia", tau274, rt2)
    )

    tau770 = (
        einsum("poj,pji->poi", tau200, tau475)
    )

    tau771 = (
        einsum("pi,ap,oia->po", tau495, a.x2, h.l.pov)
    )

    tau772 = (
        einsum("po,oia->pia", tau771, h.l.pov)
    )

    tau773 = (
        einsum("pji,poj->poi", tau51, tau62)
    )

    tau774 = (
        einsum("qp,iq,qpo->poi", tau328, a.x3, tau96)
    )

    tau775 = (
        einsum("pjb,abji->pia", tau581, rt2)
    )

    tau776 = (
        einsum("pjb,baji->pia", tau205, rt2)
    )

    tau777 = (
        einsum("qp,qo->po", tau18, tau195)
    )

    tau778 = (
        einsum("po,oia->pia", tau777, h.l.pov)
    )

    tau779 = (
        einsum("pj,bp,abij->pia", tau360, a.x2, rt2)
    )

    tau780 = (
        einsum("pq,qo->po", tau197, tau28)
    )

    tau781 = (
        einsum("po,oia->pia", tau780, h.l.pov)
    )

    tau782 = (
        einsum("pjb,baij->pia", tau262, rt2)
    )

    tau783 = (
        einsum("qp,iq,qpo->poi", tau111, a.x4, tau287)
    )

    tau784 = (
        einsum("pb,jp,baij->pia", tau404, a.x4, rt2)
    )

    tau785 = (
        einsum("pa,ip,oia->po", tau347, a.x4, h.l.pov)
    )

    tau786 = (
        einsum("po,oia->pia", tau785, h.l.pov)
    )

    tau787 = (
        einsum("qp,iq,qpo->poi", tau323, a.x3, tau218)
    )

    tau788 = (
        einsum("pia,poi->poa", tau145, tau62)
    )

    tau789 = (
        einsum("po,oia->pia", tau327, h.l.pov)
    )

    tau790 = (
        einsum("ai,pja->pij", a.t1, tau789)
    )

    tau791 = (
        einsum("aj,pij->pia", a.t1, tau790)
    )

    tau792 = (
        einsum("pjb,abij->pia", tau791, rt2)
    )

    tau793 = (
        einsum("pa,aq->pq", tau186, a.x1)
    )

    tau794 = (
        einsum("pia,poi->poa", tau101, tau200)
    )

    tau795 = (
        einsum("pia,oai->po", tau123, h.l.pvo)
    )

    tau796 = (
        einsum("po,oia->pia", tau795, h.l.pov)
    )

    tau797 = (
        einsum("poj,pji->poi", tau19, tau369)
    )

    tau798 = (
        einsum("pq,aq->pa", tau194, a.x1)
    )

    tau799 = (
        einsum("pb,jp,baij->pia", tau798, a.x3, rt2)
    )

    tau800 = (
        einsum("poj,pij->poi", tau19, tau374)
    )

    tau801 = (
        einsum("poi,qpo->pqi", tau19, tau287)
    )

    tau802 = (
        einsum("ip,qpi->pq", a.x3, tau801)
    )

    tau803 = (
        einsum("qo,po->pq", tau195, tau28)
    )

    tau804 = (
        einsum("poi,qpo->pqi", tau19, tau96)
    )

    tau805 = (
        einsum("iq,pqi->pq", a.x3, tau804)
    )

    tau806 = (
        einsum("ab,bp->pa", tau44, a.x1)
    )

    tau807 = (
        einsum("pb,jp,baij->pia", tau806, a.x4, rt2)
    )

    tau808 = (
        einsum("poj,pij->poi", tau19, tau369)
    )

    tau809 = (
        einsum("aj,pia->pij", a.t1, tau149)
    )

    tau810 = (
        einsum("poj,pji->poi", tau62, tau809)
    )

    tau811 = (
        einsum("ia,ap->pi", tau7, a.x1)
    )

    tau812 = (
        einsum("ai,pi->pa", a.t1, tau811)
    )

    tau813 = (
        einsum("pb,jp,abij->pia", tau812, a.x3, rt2)
    )

    tau814 = (
        einsum("aj,pia->pij", a.t1, tau101)
    )

    tau815 = (
        einsum("poj,pji->poi", tau59, tau814)
    )

    tau816 = (
        einsum("pq,iq,qpo->poi", tau539, a.x4, tau20)
    )

    tau817 = (
        einsum("bi,pia->pab", a.t1, tau264)
    )

    tau818 = (
        einsum("pab,oab->po", tau817, h.l.pvv)
    )

    tau819 = (
        einsum("po,oia->pia", tau818, h.l.pov)
    )

    tau820 = (
        einsum("pi,ap,oia->po", tau479, a.x1, h.l.pov)
    )

    tau821 = (
        einsum("po,oia->pia", tau820, h.l.pov)
    )

    tau822 = (
        einsum("pi,ap,oia->po", tau148, a.x1, h.l.pov)
    )

    tau823 = (
        einsum("po,oia->pia", tau822, h.l.pov)
    )

    tau824 = (
        einsum("pa,aq->pq", tau345, a.x1)
    )

    tau825 = (
        einsum("qp,aq->pa", tau824, a.x2)
    )

    tau826 = (
        einsum("pa,ip,oia->po", tau825, a.x4, h.l.pov)
    )

    tau827 = (
        einsum("po,oia->pia", tau826, h.l.pov)
    )

    tau828 = (
        einsum("ai,pa->pi", a.t1, tau124)
    )

    tau829 = (
        einsum("pj,bp,baij->pia", tau242, a.x1, rt2)
    )

    tau830 = (
        einsum("aj,pia->pij", a.t1, tau264)
    )

    tau831 = (
        einsum("pij,oji->po", tau830, h.l.poo)
    )

    tau832 = (
        einsum("po,oia->pia", tau831, h.l.pov)
    )

    tau833 = (
        einsum("poi,poj->pij", tau313, tau95)
    )

    tau834 = (
        einsum("aj,pij->pia", a.t1, tau833)
    )

    tau835 = (
        einsum("pjb,abji->pia", tau834, rt2)
    )

    tau836 = (
        einsum("pj,bp,baij->pia", tau418, a.x1, rt2)
    )

    tau837 = (
        einsum("pa,aq->pq", tau300, a.x1)
    )

    tau838 = (
        einsum("poj,pij->poi", tau19, tau214)
    )

    tau839 = (
        einsum("pia,poi->poa", tau123, tau157)
    )

    tau840 = (
        einsum("pb,oba->poa", tau124, h.l.pvv)
    )

    tau841 = (
        einsum("ai,poa->poi", a.t1, tau840)
    )

    tau842 = (
        einsum("iq,pqi->pq", a.x4, tau801)
    )

    tau843 = (
        einsum("po,oij->pij", tau327, h.l.poo)
    )

    tau844 = (
        einsum("aj,pji->pia", a.t1, tau843)
    )

    tau845 = (
        einsum("pjb,abji->pia", tau844, rt2)
    )

    tau846 = (
        einsum("pq,iq,qpo->poi", tau289, a.x3, tau633)
    )

    tau847 = (
        einsum("pi,ap,oia->po", tau342, a.x1, h.l.pov)
    )

    tau848 = (
        einsum("po,oia->pia", tau847, h.l.pov)
    )

    tau849 = (
        einsum("aj,pia->pij", a.t1, tau222)
    )

    tau850 = (
        einsum("aj,pji->pia", a.t1, tau849)
    )

    tau851 = (
        einsum("pia,oia->po", tau850, h.l.pov)
    )

    tau852 = (
        einsum("po,oia->pia", tau851, h.l.pov)
    )

    tau853 = (
        einsum("pq,iq,qpo->poi", tau357, a.x4, tau633)
    )

    tau854 = (
        einsum("aq,pia->pqi", a.x1, tau123)
    )

    tau855 = (
        einsum("iq,pqi->pq", a.x4, tau854)
    )

    tau856 = (
        einsum("iq,poi->pqo", a.x3, tau88)
    )

    tau857 = (
        einsum("pq,iq,qpo->poi", tau855, a.x3, tau856)
    )

    tau858 = (
        einsum("bp,oab->poa", a.x1, h.l.pvv)
    )

    tau859 = (
        einsum("pia,poa->poi", tau249, tau858)
    )

    tau860 = (
        einsum("aq,pia->pqi", a.x1, tau149)
    )

    tau861 = (
        einsum("ip,qpi->pq", a.x4, tau860)
    )

    tau862 = (
        einsum("qp,iq,qpo->poi", tau861, a.x3, tau218)
    )

    tau863 = (
        einsum("pi,ap,oia->po", tau152, a.x1, h.l.pov)
    )

    tau864 = (
        einsum("po,oia->pia", tau863, h.l.pov)
    )

    tau865 = (
        einsum("qp,aq->pa", tau793, a.x2)
    )

    tau866 = (
        einsum("pb,jp,baij->pia", tau865, a.x4, rt2)
    )

    tau867 = (
        einsum("ab,bp->pa", h.f.vv, a.x1)
    )

    tau868 = (
        einsum("pb,jp,baij->pia", tau867, a.x4, rt2)
    )

    tau869 = (
        einsum("po,qo->pq", tau28, tau327)
    )

    tau870 = (
        einsum("po,oab->pab", tau327, h.l.pvv)
    )

    tau871 = (
        einsum("bi,pab->pia", a.t1, tau870)
    )

    tau872 = (
        einsum("pjb,abij->pia", tau871, rt2)
    )

    tau873 = (
        einsum("ip,qpi->pq", a.x3, tau854)
    )

    tau874 = (
        einsum("qo,qp->po", tau4, tau873)
    )

    tau875 = (
        einsum("po,oia->pia", tau874, h.l.pov)
    )

    tau876 = (
        einsum("po,qo->pq", tau105, tau195)
    )

    tau877 = (
        einsum("pia,poi->poa", tau123, tau62)
    )

    tau878 = (
        einsum("ip,qpi->pq", a.x3, tau860)
    )

    tau879 = (
        einsum("qo,pq->po", tau114, tau878)
    )

    tau880 = (
        einsum("po,oia->pia", tau879, h.l.pov)
    )

    tau881 = (
        einsum("aq,pia->pqi", a.x1, tau101)
    )

    tau882 = (
        einsum("iq,pqi->pq", a.x3, tau881)
    )

    tau883 = (
        einsum("qp,iq,qpo->poi", tau882, a.x4, tau90)
    )

    tau884 = (
        einsum("pia,poi->poa", tau149, tau62)
    )

    tau885 = (
        einsum("pa,aq->pq", tau124, a.x1)
    )

    tau886 = (
        einsum("qp,aq->pa", tau885, a.x2)
    )

    tau887 = (
        einsum("pa,ip,oia->po", tau886, a.x3, h.l.pov)
    )

    tau888 = (
        einsum("po,oia->pia", tau887, h.l.pov)
    )

    tau889 = (
        einsum("pa,aq->pq", tau102, a.x1)
    )

    tau890 = (
        einsum("qp,aq->pa", tau889, a.x2)
    )

    tau891 = (
        einsum("pa,ip,oia->po", tau890, a.x3, h.l.pov)
    )

    tau892 = (
        einsum("po,oia->pia", tau891, h.l.pov)
    )

    tau893 = (
        einsum("pia,poa->poi", tau173, tau858)
    )

    tau894 = (
        einsum("pa,aq->pq", tau203, a.x1)
    )

    tau895 = (
        einsum("qp,aq->pa", tau894, a.x2)
    )

    tau896 = (
        einsum("pa,ip,oia->po", tau895, a.x3, h.l.pov)
    )

    tau897 = (
        einsum("po,oia->pia", tau896, h.l.pov)
    )

    tau898 = (
        einsum("poj,pji->poi", tau59, tau830)
    )

    tau899 = (
        einsum("aj,pia->pij", a.t1, tau123)
    )

    tau900 = (
        einsum("aj,pji->pia", a.t1, tau899)
    )

    tau901 = (
        einsum("pia,oia->po", tau900, h.l.pov)
    )

    tau902 = (
        einsum("po,oia->pia", tau901, h.l.pov)
    )

    tau903 = (
        einsum("ai,pa->pi", a.t1, tau203)
    )

    tau904 = (
        einsum("pj,bp,abji->pia", tau273, a.x1, rt2)
    )

    tau905 = (
        einsum("poj,poi->pij", tau157, tau19)
    )

    tau906 = (
        einsum("aj,pji->pia", a.t1, tau905)
    )

    tau907 = (
        einsum("pjb,abij->pia", tau906, rt2)
    )

    tau908 = (
        einsum("ai,pa->pi", a.t1, tau102)
    )

    tau909 = (
        einsum("pa,ip,oia->po", tau886, a.x4, h.l.pov)
    )

    tau910 = (
        einsum("po,oia->pia", tau909, h.l.pov)
    )

    tau911 = (
        einsum("aj,pia->pij", a.t1, tau145)
    )

    tau912 = (
        einsum("pij,oji->po", tau911, h.l.poo)
    )

    tau913 = (
        einsum("po,oia->pia", tau912, h.l.pov)
    )

    tau914 = (
        einsum("poj,poi->pij", tau19, tau59)
    )

    tau915 = (
        einsum("aj,pij->pia", a.t1, tau914)
    )

    tau916 = (
        einsum("pjb,baij->pia", tau915, rt2)
    )

    tau917 = (
        einsum("poi,poa->pia", tau62, tau858)
    )

    tau918 = (
        einsum("pjb,abij->pia", tau917, rt2)
    )

    tau919 = (
        einsum("pj,bp,baij->pia", tau337, a.x1, rt2)
    )

    tau920 = (
        einsum("aq,pia->pqi", a.x1, tau87)
    )

    tau921 = (
        einsum("iq,pqi->pq", a.x4, tau920)
    )

    tau922 = (
        einsum("pq,iq,qpo->poi", tau921, a.x3, tau90)
    )

    tau923 = (
        einsum("qp,iq,qpo->poi", tau873, a.x4, tau856)
    )

    tau924 = (
        einsum("ai,pa->pi", a.t1, tau345)
    )

    tau925 = (
        einsum("pb,jp,baij->pia", tau812, a.x4, rt2)
    )

    tau926 = (
        einsum("oia,poi->pa", tau209, tau95)
    )

    tau927 = (
        einsum("pb,jp,abij->pia", tau926, a.x3, rt2)
    )

    tau928 = (
        einsum("pjb,baji->pia", tau917, rt2)
    )

    tau929 = (
        einsum("poj,pji->poi", tau62, tau899)
    )

    tau930 = (
        einsum("pia,poi->poa", tau264, tau59)
    )

    tau931 = (
        einsum("poi,poj->pij", tau72, tau95)
    )

    tau932 = (
        einsum("aj,pij->pia", a.t1, tau931)
    )

    tau933 = (
        einsum("pjb,baji->pia", tau932, rt2)
    )

    tau934 = (
        einsum("qo,pq->po", tau105, tau357)
    )

    tau935 = (
        einsum("po,oia->pia", tau934, h.l.pov)
    )

    tau936 = (
        einsum("pq,iq,qpo->poi", tau505, a.x3, tau20)
    )

    tau937 = (
        einsum("pij,oji->po", tau849, h.l.poo)
    )

    tau938 = (
        einsum("po,oia->pia", tau937, h.l.pov)
    )

    tau939 = (
        einsum("pia,poi->poa", tau145, tau157)
    )

    tau940 = (
        einsum("pq,iq,qpo->poi", tau380, a.x4, tau633)
    )

    tau941 = (
        einsum("ia,ap->pi", h.f.ov, a.x1)
    )

    tau942 = (
        einsum("ai,pi->pa", a.t1, tau941)
    )

    tau943 = (
        einsum("pb,jp,baji->pia", tau942, a.x4, rt2)
    )

    tau944 = (
        einsum("poj,pji->poi", tau313, tau899)
    )

    tau945 = (
        einsum("qo,pq->po", tau105, tau539)
    )

    tau946 = (
        einsum("po,oia->pia", tau945, h.l.pov)
    )

    tau947 = (
        einsum("qo,pq->po", tau105, tau428)
    )

    tau948 = (
        einsum("po,oia->pia", tau947, h.l.pov)
    )

    tau949 = (
        einsum("ip,qpi->pq", a.x3, tau920)
    )

    tau950 = (
        einsum("qo,qp->po", tau254, tau949)
    )

    tau951 = (
        einsum("po,oia->pia", tau950, h.l.pov)
    )

    tau952 = (
        einsum("pi,ap,oia->po", tau320, a.x1, h.l.pov)
    )

    tau953 = (
        einsum("po,oia->pia", tau952, h.l.pov)
    )

    tau954 = (
        einsum("poj,pji->poi", tau62, tau849)
    )

    tau955 = (
        einsum("pq,aq->pa", tau473, a.x1)
    )

    tau956 = (
        einsum("pb,jp,abij->pia", tau955, a.x4, rt2)
    )

    tau957 = (
        einsum("qp,aq->pa", tau837, a.x2)
    )

    tau958 = (
        einsum("pb,jp,baij->pia", tau957, a.x3, rt2)
    )

    tau959 = (
        einsum("qo,pq->po", tau105, tau339)
    )

    tau960 = (
        einsum("po,oia->pia", tau959, h.l.pov)
    )

    tau961 = (
        einsum("pj,bp,baij->pia", tau109, a.x1, rt2)
    )

    tau962 = (
        einsum("qo,pq->po", tau28, tau505)
    )

    tau963 = (
        einsum("po,oia->pia", tau962, h.l.pov)
    )

    tau964 = (
        einsum("pj,bp,abij->pia", tau68, a.x1, rt2)
    )

    tau965 = (
        einsum("qo,pq->po", tau114, tau921)
    )

    tau966 = (
        einsum("po,oia->pia", tau965, h.l.pov)
    )

    tau967 = (
        einsum("pq,qo->po", tau237, tau28)
    )

    tau968 = (
        einsum("po,oia->pia", tau967, h.l.pov)
    )

    tau969 = (
        einsum("iq,pqi->pq", a.x4, tau881)
    )

    tau970 = (
        einsum("qo,qp->po", tau4, tau969)
    )

    tau971 = (
        einsum("po,oia->pia", tau970, h.l.pov)
    )

    tau972 = (
        einsum("poj,pji->poi", tau157, tau911)
    )

    tau973 = (
        einsum("poj,poi->pij", tau19, tau62)
    )

    tau974 = (
        einsum("aj,pij->pia", a.t1, tau973)
    )

    tau975 = (
        einsum("pjb,baji->pia", tau974, rt2)
    )

    tau976 = (
        einsum("pia,poi->poa", tau0, tau200)
    )

    tau977 = (
        einsum("po,oai->pia", tau327, h.l.pvo)
    )

    tau978 = (
        einsum("pjb,baij->pia", tau977, rt2)
    )

    tau979 = (
        einsum("pia,poi->poa", tau222, tau62)
    )

    tau980 = (
        einsum("qo,pq->po", tau105, tau606)
    )

    tau981 = (
        einsum("po,oia->pia", tau980, h.l.pov)
    )

    tau982 = (
        einsum("bi,pia->pab", a.t1, tau0)
    )

    tau983 = (
        einsum("pab,oab->po", tau982, h.l.pvv)
    )

    tau984 = (
        einsum("po,oia->pia", tau983, h.l.pov)
    )

    tau985 = (
        einsum("pb,jp,baji->pia", tau942, a.x3, rt2)
    )

    tau986 = (
        einsum("poj,pji->poi", tau72, tau814)
    )

    tau987 = (
        einsum("ai,pja->pij", a.t1, tau299)
    )

    tau988 = (
        einsum("aj,pij->pia", a.t1, tau987)
    )

    tau989 = (
        einsum("pjb,baij->pia", tau988, rt2)
    )

    tau990 = (
        einsum("pq,iq,qpo->poi", tau873, a.x3, tau90)
    )

    tau991 = (
        einsum("bi,pia->pab", a.t1, tau101)
    )

    tau992 = (
        einsum("pab,oab->po", tau991, h.l.pvv)
    )

    tau993 = (
        einsum("po,oia->pia", tau992, h.l.pov)
    )

    tau994 = (
        einsum("pj,bp,abij->pia", tau43, a.x1, rt2)
    )

    tau995 = (
        einsum("qp,iq,qpo->poi", tau969, a.x4, tau856)
    )

    tau996 = (
        einsum("po,oij->pij", tau195, h.l.poo)
    )

    tau997 = (
        einsum("aj,pji->pia", a.t1, tau996)
    )

    tau998 = (
        einsum("pjb,baji->pia", tau997, rt2)
    )

    tau999 = (
        einsum("qo,pq->po", tau114, tau873)
    )

    tau1000 = (
        einsum("po,oia->pia", tau999, h.l.pov)
    )

    tau1001 = (
        einsum("pb,jp,abij->pia", tau806, a.x3, rt2)
    )

    tau1002 = (
        einsum("pjb,abji->pia", tau791, rt2)
    )

    tau1003 = (
        einsum("pj,bp,baij->pia", tau68, a.x1, rt2)
    )

    tau1004 = (
        einsum("pb,jp,abji->pia", tau942, a.x4, rt2)
    )

    tau1005 = (
        einsum("qo,pq->po", tau28, tau569)
    )

    tau1006 = (
        einsum("po,oia->pia", tau1005, h.l.pov)
    )

    tau1007 = (
        einsum("pij,poj->poi", tau662, tau95)
    )

    tau1008 = (
        einsum("qo,pq->po", tau11, tau861)
    )

    tau1009 = (
        einsum("po,oia->pia", tau1008, h.l.pov)
    )

    tau1010 = (
        einsum("pb,jp,abij->pia", tau865, a.x4, rt2)
    )

    tau1011 = (
        einsum("pi,ap,oia->po", tau495, a.x1, h.l.pov)
    )

    tau1012 = (
        einsum("po,oia->pia", tau1011, h.l.pov)
    )

    tau1013 = (
        einsum("pj,bp,baij->pia", tau172, a.x1, rt2)
    )

    tau1014 = (
        einsum("pjb,baji->pia", tau915, rt2)
    )

    tau1015 = (
        einsum("pj,bp,baji->pia", tau273, a.x1, rt2)
    )

    tau1016 = (
        einsum("pi,ap,oia->po", tau37, a.x1, h.l.pov)
    )

    tau1017 = (
        einsum("po,oia->pia", tau1016, h.l.pov)
    )

    tau1018 = (
        einsum("pjb,baij->pia", tau791, rt2)
    )

    tau1019 = (
        einsum("pjb,baji->pia", tau871, rt2)
    )

    tau1020 = (
        einsum("aj,pji->pia", a.t1, tau814)
    )

    tau1021 = (
        einsum("pia,oia->po", tau1020, h.l.pov)
    )

    tau1022 = (
        einsum("po,oia->pia", tau1021, h.l.pov)
    )

    tau1023 = (
        einsum("aj,pia->pij", a.t1, tau87)
    )

    tau1024 = (
        einsum("pji,poj->poi", tau1023, tau59)
    )

    tau1025 = (
        einsum("pia,oai->po", tau149, h.l.pvo)
    )

    tau1026 = (
        einsum("po,oia->pia", tau1025, h.l.pov)
    )

    tau1027 = (
        einsum("po,oai->pia", tau195, h.l.pvo)
    )

    tau1028 = (
        einsum("pjb,baji->pia", tau1027, rt2)
    )

    tau1029 = (
        einsum("pa,oia->poi", tau345, tau209)
    )

    tau1030 = (
        einsum("pj,bp,abij->pia", tau418, a.x1, rt2)
    )

    tau1031 = (
        einsum("poj,pji->poi", tau19, tau662)
    )

    tau1032 = (
        einsum("ia,ap->pi", tau119, a.x1)
    )

    tau1033 = (
        einsum("ai,pi->pa", a.t1, tau1032)
    )

    tau1034 = (
        einsum("pb,jp,abij->pia", tau1033, a.x3, rt2)
    )

    tau1035 = (
        einsum("pia,poi->poa", tau173, tau95)
    )

    tau1036 = (
        einsum("pij,oji->po", tau809, h.l.poo)
    )

    tau1037 = (
        einsum("po,oia->pia", tau1036, h.l.pov)
    )

    tau1038 = (
        einsum("pq,aq->pa", tau260, a.x1)
    )

    tau1039 = (
        einsum("pa,ip,oia->po", tau1038, a.x4, h.l.pov)
    )

    tau1040 = (
        einsum("po,oia->pia", tau1039, h.l.pov)
    )

    tau1041 = (
        einsum("pj,bp,abij->pia", tau172, a.x1, rt2)
    )

    tau1042 = (
        einsum("iq,poi->pqo", a.x4, tau88)
    )

    tau1043 = (
        einsum("pq,iq,qpo->poi", tau949, a.x4, tau1042)
    )

    tau1044 = (
        einsum("poj,pji->poi", tau200, tau830)
    )

    tau1045 = (
        einsum("po,oab->pab", tau195, h.l.pvv)
    )

    tau1046 = (
        einsum("bi,pab->pia", a.t1, tau1045)
    )

    tau1047 = (
        einsum("pjb,abij->pia", tau1046, rt2)
    )

    tau1048 = (
        einsum("bp,oab->poa", a.x1, h.l.pvv)
    )

    tau1049 = (
        einsum("pob,pab->poa", tau1048, tau248)
    )

    tau1050 = (
        einsum("poi,poa->pia", tau59, tau858)
    )

    tau1051 = (
        einsum("pjb,abij->pia", tau1050, rt2)
    )

    tau1052 = (
        einsum("pjb,baij->pia", tau1027, rt2)
    )

    tau1053 = (
        einsum("po,oij->pij", tau28, h.l.poo)
    )

    tau1054 = (
        einsum("aj,pji->pia", a.t1, tau1053)
    )

    tau1055 = (
        einsum("pjb,abij->pia", tau1054, rt2)
    )

    tau1056 = (
        einsum("pq,aq->pa", tau587, a.x1)
    )

    tau1057 = (
        einsum("pa,ip,oia->po", tau1056, a.x3, h.l.pov)
    )

    tau1058 = (
        einsum("po,oia->pia", tau1057, h.l.pov)
    )

    tau1059 = (
        einsum("bi,pia->pab", a.t1, tau149)
    )

    tau1060 = (
        einsum("pab,oab->po", tau1059, h.l.pvv)
    )

    tau1061 = (
        einsum("po,oia->pia", tau1060, h.l.pov)
    )

    tau1062 = (
        einsum("pij,oji->po", tau814, h.l.poo)
    )

    tau1063 = (
        einsum("po,oia->pia", tau1062, h.l.pov)
    )

    tau1064 = (
        einsum("qo,qp->po", tau114, tau855)
    )

    tau1065 = (
        einsum("po,oia->pia", tau1064, h.l.pov)
    )

    tau1066 = (
        einsum("pj,bp,baji->pia", tau397, a.x1, rt2)
    )

    tau1067 = (
        einsum("pj,bp,abij->pia", tau242, a.x1, rt2)
    )

    tau1068 = (
        einsum("pjb,abji->pia", tau932, rt2)
    )

    tau1069 = (
        einsum("pi,ap,oia->po", tau139, a.x1, h.l.pov)
    )

    tau1070 = (
        einsum("po,oia->pia", tau1069, h.l.pov)
    )

    tau1071 = (
        einsum("pa,ip,oia->po", tau825, a.x3, h.l.pov)
    )

    tau1072 = (
        einsum("po,oia->pia", tau1071, h.l.pov)
    )

    tau1073 = (
        einsum("pj,bp,abij->pia", tau269, a.x1, rt2)
    )

    tau1074 = (
        einsum("pb,jp,abij->pia", tau798, a.x4, rt2)
    )

    tau1075 = (
        einsum("pb,jp,abij->pia", tau1033, a.x4, rt2)
    )

    tau1076 = (
        einsum("pjb,abij->pia", tau977, rt2)
    )

    tau1077 = (
        einsum("pba,pob->poa", tau248, tau858)
    )

    tau1078 = (
        einsum("pjb,baji->pia", tau791, rt2)
    )

    tau1079 = (
        einsum("pb,jp,baij->pia", tau957, a.x4, rt2)
    )

    tau1080 = (
        einsum("pb,jp,baij->pia", tau926, a.x4, rt2)
    )

    tau1081 = (
        einsum("po,oij->pij", tau105, h.l.poo)
    )

    tau1082 = (
        einsum("aj,pji->pia", a.t1, tau1081)
    )

    tau1083 = (
        einsum("pjb,baij->pia", tau1082, rt2)
    )

    tau1084 = (
        einsum("poj,pji->poi", tau313, tau809)
    )

    tau1085 = (
        einsum("poi,pia->poa", tau200, tau87)
    )

    tau1086 = (
        einsum("pb,jp,baij->pia", tau926, a.x3, rt2)
    )

    tau1087 = (
        einsum("pjb,abij->pia", tau988, rt2)
    )

    tau1088 = (
        einsum("pj,bp,baij->pia", tau43, a.x1, rt2)
    )

    tau1089 = (
        einsum("qo,qp->po", tau4, tau878)
    )

    tau1090 = (
        einsum("po,oia->pia", tau1089, h.l.pov)
    )

    tau1091 = (
        einsum("qo,qp->po", tau114, tau861)
    )

    tau1092 = (
        einsum("po,oia->pia", tau1091, h.l.pov)
    )

    tau1093 = (
        einsum("pq,iq,qpo->poi", tau339, a.x4, tau20)
    )

    tau1094 = (
        einsum("aj,pji->pia", a.t1, tau809)
    )

    tau1095 = (
        einsum("pia,oia->po", tau1094, h.l.pov)
    )

    tau1096 = (
        einsum("po,oia->pia", tau1095, h.l.pov)
    )

    tau1097 = (
        einsum("pb,jp,baij->pia", tau865, a.x3, rt2)
    )

    tau1098 = (
        einsum("qo,pq->po", tau11, tau855)
    )

    tau1099 = (
        einsum("po,oia->pia", tau1098, h.l.pov)
    )

    tau1100 = (
        einsum("pjb,abji->pia", tau915, rt2)
    )

    tau1101 = (
        einsum("poi,poa->pia", tau200, tau858)
    )

    tau1102 = (
        einsum("pjb,baji->pia", tau1101, rt2)
    )

    tau1103 = (
        einsum("pia,oai->po", tau0, h.l.pvo)
    )

    tau1104 = (
        einsum("po,oia->pia", tau1103, h.l.pov)
    )

    tau1105 = (
        einsum("pji,poj->poi", tau1023, tau72)
    )

    tau1106 = (
        einsum("pjb,abji->pia", tau1027, rt2)
    )

    tau1107 = (
        einsum("pq,aq->pa", tau49, a.x1)
    )

    tau1108 = (
        einsum("pa,ip,oia->po", tau1107, a.x4, h.l.pov)
    )

    tau1109 = (
        einsum("po,oia->pia", tau1108, h.l.pov)
    )

    tau1110 = (
        einsum("qp,iq,qpo->poi", tau949, a.x4, tau90)
    )

    tau1111 = (
        einsum("pia,poi->poa", tau149, tau157)
    )

    tau1112 = (
        einsum("pb,jp,baij->pia", tau812, a.x3, rt2)
    )

    tau1113 = (
        einsum("pq,iq,qpo->poi", tau606, a.x4, tau20)
    )

    tau1114 = (
        einsum("pj,bp,abij->pia", tau337, a.x1, rt2)
    )

    tau1115 = (
        einsum("pia,oai->po", tau145, h.l.pvo)
    )

    tau1116 = (
        einsum("po,oia->pia", tau1115, h.l.pov)
    )

    tau1117 = (
        einsum("poi,poa->pia", tau157, tau858)
    )

    tau1118 = (
        einsum("pjb,abji->pia", tau1117, rt2)
    )

    tau1119 = (
        einsum("pj,bp,baij->pia", tau360, a.x1, rt2)
    )

    tau1120 = (
        einsum("pi,ap,oia->po", tau155, a.x1, h.l.pov)
    )

    tau1121 = (
        einsum("po,oia->pia", tau1120, h.l.pov)
    )

    tau1122 = (
        einsum("pb,jp,abij->pia", tau867, a.x4, rt2)
    )

    tau1123 = (
        einsum("pj,bp,baji->pia", tau337, a.x1, rt2)
    )

    tau1124 = (
        einsum("pba,pob->poa", tau70, tau858)
    )

    tau1125 = (
        einsum("qo,pq->po", tau28, tau289)
    )

    tau1126 = (
        einsum("po,oia->pia", tau1125, h.l.pov)
    )

    tau1127 = (
        einsum("pq,iq,qpo->poi", tau569, a.x3, tau20)
    )

    tau1128 = (
        einsum("aj,pia->pij", a.t1, tau0)
    )

    tau1129 = (
        einsum("pij,oji->po", tau1128, h.l.poo)
    )

    tau1130 = (
        einsum("po,oia->pia", tau1129, h.l.pov)
    )

    tau1131 = (
        einsum("pb,jp,baij->pia", tau867, a.x3, rt2)
    )

    tau1132 = (
        einsum("pia,poi->poa", tau101, tau59)
    )

    tau1133 = (
        einsum("poi,pia->poa", tau157, tau222)
    )

    tau1134 = (
        einsum("pj,bp,baij->pia", tau81, a.x1, rt2)
    )

    tau1135 = (
        einsum("qp,iq,qpo->poi", tau921, a.x4, tau856)
    )

    tau1136 = (
        einsum("pia,oai->po", tau101, h.l.pvo)
    )

    tau1137 = (
        einsum("po,oia->pia", tau1136, h.l.pov)
    )

    tau1138 = (
        einsum("pjb,abij->pia", tau1027, rt2)
    )

    tau1139 = (
        einsum("pa,oia->poi", tau203, tau209)
    )

    tau1140 = (
        einsum("pjb,baji->pia", tau1046, rt2)
    )

    tau1141 = (
        einsum("qo,pq->po", tau114, tau969)
    )

    tau1142 = (
        einsum("po,oia->pia", tau1141, h.l.pov)
    )

    tau1143 = (
        einsum("pjb,abij->pia", tau1117, rt2)
    )

    tau1144 = (
        einsum("pia,poi->poa", tau0, tau59)
    )

    tau1145 = (
        einsum("pq,iq,qpo->poi", tau882, a.x4, tau1042)
    )

    tau1146 = (
        einsum("pi,ap,oia->po", tau484, a.x1, h.l.pov)
    )

    tau1147 = (
        einsum("po,oia->pia", tau1146, h.l.pov)
    )

    tau1148 = (
        einsum("pij,oji->po", tau899, h.l.poo)
    )

    tau1149 = (
        einsum("po,oia->pia", tau1148, h.l.pov)
    )

    tau1150 = (
        einsum("pjb,baij->pia", tau1050, rt2)
    )

    tau1151 = (
        einsum("pjb,baij->pia", tau906, rt2)
    )

    tau1152 = (
        einsum("pb,jp,baij->pia", tau806, a.x3, rt2)
    )

    tau1153 = (
        einsum("pq,aq->pa", tau425, a.x1)
    )

    tau1154 = (
        einsum("pa,ip,oia->po", tau1153, a.x4, h.l.pov)
    )

    tau1155 = (
        einsum("po,oia->pia", tau1154, h.l.pov)
    )

    tau1156 = (
        einsum("poi,poj->pij", tau19, tau200)
    )

    tau1157 = (
        einsum("aj,pji->pia", a.t1, tau1156)
    )

    tau1158 = (
        einsum("pjb,abij->pia", tau1157, rt2)
    )

    tau1159 = (
        einsum("bi,pia->pab", a.t1, tau123)
    )

    tau1160 = (
        einsum("pab,oab->po", tau1159, h.l.pvv)
    )

    tau1161 = (
        einsum("po,oia->pia", tau1160, h.l.pov)
    )

    tau1162 = (
        einsum("pb,jp,abij->pia", tau867, a.x3, rt2)
    )

    tau1163 = (
        einsum("qo,pq->po", tau4, tau882)
    )

    tau1164 = (
        einsum("po,oia->pia", tau1163, h.l.pov)
    )

    tau1165 = (
        einsum("pjb,abji->pia", tau997, rt2)
    )

    tau1166 = (
        einsum("poi,pia->poa", tau200, tau264)
    )

    tau1167 = (
        einsum("pj,bp,abij->pia", tau360, a.x1, rt2)
    )

    tau1168 = (
        einsum("pj,bp,baij->pia", tau295, a.x1, rt2)
    )

    tau1169 = (
        einsum("pq,iq,qpo->poi", tau437, a.x4, tau633)
    )

    tau1170 = (
        einsum("poi,pia->poa", tau59, tau87)
    )

    tau1171 = (
        einsum("pa,ip,oia->po", tau895, a.x4, h.l.pov)
    )

    tau1172 = (
        einsum("po,oia->pia", tau1171, h.l.pov)
    )

    tau1173 = (
        einsum("pj,bp,abij->pia", tau521, a.x1, rt2)
    )

    tau1174 = (
        einsum("pi,ap,oia->po", tau458, a.x1, h.l.pov)
    )

    tau1175 = (
        einsum("po,oia->pia", tau1174, h.l.pov)
    )

    tau1176 = (
        einsum("pj,bp,abij->pia", tau77, a.x1, rt2)
    )

    tau1177 = (
        einsum("pjb,abji->pia", tau1050, rt2)
    )

    tau1178 = (
        einsum("pia,oai->po", tau264, h.l.pvo)
    )

    tau1179 = (
        einsum("po,oia->pia", tau1178, h.l.pov)
    )

    tau1180 = (
        einsum("pj,bp,baij->pia", tau544, a.x1, rt2)
    )

    tau1181 = (
        einsum("pq,iq,qpo->poi", tau176, a.x3, tau633)
    )

    tau1182 = (
        einsum("pjb,baij->pia", tau1054, rt2)
    )

    tau1183 = (
        einsum("pi,ap,oia->po", tau444, a.x1, h.l.pov)
    )

    tau1184 = (
        einsum("po,oia->pia", tau1183, h.l.pov)
    )

    tau1185 = (
        einsum("pb,jp,abij->pia", tau955, a.x3, rt2)
    )

    tau1186 = (
        einsum("pq,iq,qpo->poi", tau237, a.x3, tau633)
    )

    tau1187 = (
        einsum("pia,poi->poa", tau228, tau95)
    )

    tau1188 = (
        einsum("pjb,baij->pia", tau1117, rt2)
    )

    tau1189 = (
        einsum("pq,qo->po", tau176, tau28)
    )

    tau1190 = (
        einsum("po,oia->pia", tau1189, h.l.pov)
    )

    tau1191 = (
        einsum("pb,jp,abij->pia", tau926, a.x4, rt2)
    )

    tau1192 = (
        einsum("poj,pji->poi", tau157, tau849)
    )

    tau1193 = (
        einsum("pjb,abji->pia", tau1046, rt2)
    )

    tau1194 = (
        einsum("pq,iq,qpo->poi", tau969, a.x3, tau90)
    )

    tau1195 = (
        einsum("qp,iq,qpo->poi", tau855, a.x3, tau218)
    )

    tau1196 = (
        einsum("qo,pq->po", tau105, tau380)
    )

    tau1197 = (
        einsum("po,oia->pia", tau1196, h.l.pov)
    )

    tau1198 = (
        einsum("pb,jp,baij->pia", tau955, a.x4, rt2)
    )

    tau1199 = (
        einsum("pjb,baji->pia", tau977, rt2)
    )

    tau1200 = (
        einsum("pq,iq,qpo->poi", tau861, a.x3, tau856)
    )

    tau1201 = (
        einsum("pjb,abji->pia", tau1101, rt2)
    )

    tau1202 = (
        einsum("pj,bp,abij->pia", tau109, a.x1, rt2)
    )

    tau1203 = (
        einsum("pq,iq,qpo->poi", tau163, a.x3, tau20)
    )

    tau1204 = (
        einsum("pa,ip,oia->po", tau1038, a.x3, h.l.pov)
    )

    tau1205 = (
        einsum("po,oia->pia", tau1204, h.l.pov)
    )

    tau1206 = (
        einsum("pj,bp,abji->pia", tau360, a.x1, rt2)
    )

    tau1207 = (
        einsum("qo,pq->po", tau28, tau519)
    )

    tau1208 = (
        einsum("po,oia->pia", tau1207, h.l.pov)
    )

    tau1209 = (
        einsum("pb,jp,baij->pia", tau1033, a.x4, rt2)
    )

    tau1210 = (
        einsum("pj,bp,baij->pia", tau613, a.x1, rt2)
    )

    tau1211 = (
        einsum("pa,ip,oia->po", tau1056, a.x4, h.l.pov)
    )

    tau1212 = (
        einsum("po,oia->pia", tau1211, h.l.pov)
    )

    tau1213 = (
        einsum("pjb,abij->pia", tau1082, rt2)
    )

    tau1214 = (
        einsum("pj,bp,abij->pia", tau544, a.x1, rt2)
    )

    tau1215 = (
        einsum("qo,pq->po", tau28, tau323)
    )

    tau1216 = (
        einsum("po,oia->pia", tau1215, h.l.pov)
    )

    tau1217 = (
        einsum("pq,qo->po", tau163, tau28)
    )

    tau1218 = (
        einsum("po,oia->pia", tau1217, h.l.pov)
    )

    tau1219 = (
        einsum("pa,ip,oia->po", tau1153, a.x3, h.l.pov)
    )

    tau1220 = (
        einsum("po,oia->pia", tau1219, h.l.pov)
    )

    tau1221 = (
        einsum("pia,oai->po", tau87, h.l.pvo)
    )

    tau1222 = (
        einsum("po,oia->pia", tau1221, h.l.pov)
    )

    tau1223 = (
        einsum("pb,jp,abij->pia", tau798, a.x3, rt2)
    )

    tau1224 = (
        einsum("poj,pji->poi", tau62, tau911)
    )

    tau1225 = (
        einsum("pjb,baji->pia", tau1117, rt2)
    )

    tau1226 = (
        einsum("aj,pji->pia", a.t1, tau1023)
    )

    tau1227 = (
        einsum("pia,oia->po", tau1226, h.l.pov)
    )

    tau1228 = (
        einsum("po,oia->pia", tau1227, h.l.pov)
    )

    tau1229 = (
        einsum("pq,iq,qpo->poi", tau519, a.x3, tau633)
    )

    tau1230 = (
        einsum("pjb,abji->pia", tau974, rt2)
    )

    tau1231 = (
        einsum("pb,jp,abij->pia", tau957, a.x4, rt2)
    )

    tau1232 = (
        einsum("pjb,baij->pia", tau974, rt2)
    )

    tau1233 = (
        einsum("bi,pia->pab", a.t1, tau87)
    )

    tau1234 = (
        einsum("pab,oab->po", tau1233, h.l.pvv)
    )

    tau1235 = (
        einsum("po,oia->pia", tau1234, h.l.pov)
    )

    tau1236 = (
        einsum("pjb,abji->pia", tau977, rt2)
    )

    tau1237 = (
        einsum("pb,jp,abij->pia", tau957, a.x3, rt2)
    )

    tau1238 = (
        einsum("pb,jp,abij->pia", tau806, a.x4, rt2)
    )

    tau1239 = (
        einsum("pjb,abji->pia", tau871, rt2)
    )

    tau1240 = (
        einsum("pa,ip,oia->po", tau1107, a.x3, h.l.pov)
    )

    tau1241 = (
        einsum("po,oia->pia", tau1240, h.l.pov)
    )

    tau1242 = (
        einsum("pjb,abji->pia", tau917, rt2)
    )

    tau1243 = (
        einsum("pq,iq,qpo->poi", tau878, a.x3, tau90)
    )

    tau1244 = (
        einsum("pj,bp,abij->pia", tau613, a.x1, rt2)
    )

    tau1245 = (
        einsum("pji,poj->poi", tau1128, tau200)
    )

    tau1246 = (
        einsum("pj,bp,abij->pia", tau295, a.x1, rt2)
    )

    tau1247 = (
        einsum("poj,pij->poi", tau19, tau451)
    )

    tau1248 = (
        einsum("poj,pji->poi", tau19, tau365)
    )

    tau1249 = (
        einsum("pjb,baij->pia", tau917, rt2)
    )

    tau1250 = (
        einsum("pob,pab->poa", tau1048, tau70)
    )

    tau1251 = (
        einsum("pb,jp,abij->pia", tau865, a.x3, rt2)
    )

    tau1252 = (
        einsum("pi,ap,oia->po", tau530, a.x1, h.l.pov)
    )

    tau1253 = (
        einsum("po,oia->pia", tau1252, h.l.pov)
    )

    tau1254 = (
        einsum("pj,bp,abji->pia", tau397, a.x1, rt2)
    )

    tau1255 = (
        einsum("pia,poa->poi", tau364, tau858)
    )

    tau1256 = (
        einsum("pj,bp,abji->pia", tau337, a.x1, rt2)
    )

    tau1257 = (
        einsum("pjb,baji->pia", tau844, rt2)
    )

    tau1258 = (
        einsum("qo,pq->po", tau105, tau302)
    )

    tau1259 = (
        einsum("po,oia->pia", tau1258, h.l.pov)
    )

    tau1260 = (
        einsum("pia,poi->poa", tau364, tau95)
    )

    tau1261 = (
        einsum("pjb,baij->pia", tau1101, rt2)
    )

    tau1262 = (
        einsum("pj,bp,abij->pia", tau81, a.x1, rt2)
    )

    tau1263 = (
        einsum("pb,jp,baij->pia", tau1033, a.x3, rt2)
    )

    tau1264 = (
        einsum("aj,pji->pia", a.t1, tau830)
    )

    tau1265 = (
        einsum("pia,oia->po", tau1264, h.l.pov)
    )

    tau1266 = (
        einsum("po,oia->pia", tau1265, h.l.pov)
    )

    tau1267 = (
        einsum("pjb,abij->pia", tau1101, rt2)
    )

    tau1268 = (
        einsum("pj,bp,baij->pia", tau297, a.x1, rt2)
    )

    tau1269 = (
        einsum("bi,pia->pab", a.t1, tau222)
    )

    tau1270 = (
        einsum("pab,oab->po", tau1269, h.l.pvv)
    )

    tau1271 = (
        einsum("po,oia->pia", tau1270, h.l.pov)
    )

    tau1272 = (
        einsum("qp,iq,qpo->poi", tau878, a.x4, tau856)
    )

    tau1273 = (
        einsum("pj,bp,abij->pia", tau442, a.x1, rt2)
    )

    tau1274 = (
        einsum("pia,poa->poi", tau228, tau858)
    )

    tau1275 = (
        einsum("qo,qp->po", tau4, tau921)
    )

    tau1276 = (
        einsum("po,oia->pia", tau1275, h.l.pov)
    )

    tau1277 = (
        einsum("pb,jp,baij->pia", tau798, a.x4, rt2)
    )

    tau1278 = (
        einsum("qo,pq->po", tau4, tau949)
    )

    tau1279 = (
        einsum("po,oia->pia", tau1278, h.l.pov)
    )

    tau1280 = (
        einsum("pb,oba->poa", tau102, h.l.pvv)
    )

    tau1281 = (
        einsum("ai,poa->poi", a.t1, tau1280)
    )

    tau1282 = (
        einsum("pj,bp,abij->pia", tau297, a.x1, rt2)
    )

    tau1283 = (
        einsum("poj,pji->poi", tau19, tau214)
    )

    tau1284 = (
        einsum("pi,ap,oia->po", tau3, a.x1, h.l.pov)
    )

    tau1285 = (
        einsum("po,oia->pia", tau1284, h.l.pov)
    )

    tau1286 = (
        einsum("pj,bp,baji->pia", tau360, a.x1, rt2)
    )

    tau1287 = (
        einsum("qo,pq->po", tau105, tau437)
    )

    tau1288 = (
        einsum("po,oia->pia", tau1287, h.l.pov)
    )

    tau1289 = (
        einsum("pq,iq,qpo->poi", tau323, a.x3, tau20)
    )

    tau1290 = (
        einsum("pi,ap,oia->po", tau455, a.x1, h.l.pov)
    )

    tau1291 = (
        einsum("po,oia->pia", tau1290, h.l.pov)
    )

    tau1292 = (
        einsum("pj,bp,abij->pia", tau15, a.x1, rt2)
    )

    tau1293 = (
        einsum("pjb,baij->pia", tau1046, rt2)
    )

    tau1294 = (
        einsum("pq,iq,qpo->poi", tau302, a.x4, tau633)
    )

    tau1295 = (
        einsum("pi,ap,oia->po", tau161, a.x1, h.l.pov)
    )

    tau1296 = (
        einsum("po,oia->pia", tau1295, h.l.pov)
    )

    tau1297 = (
        einsum("pb,jp,abij->pia", tau812, a.x4, rt2)
    )

    tau1298 = (
        einsum("qo,qp->po", tau254, tau882)
    )

    tau1299 = (
        einsum("po,oia->pia", tau1298, h.l.pov)
    )

    tau1300 = (
        einsum("poj,pji->poi", tau19, tau374)
    )

    tau1301 = (
        einsum("pji,poj->poi", tau1128, tau59)
    )

    tau1302 = (
        einsum("pjb,baji->pia", tau1050, rt2)
    )

    tau1303 = (
        einsum("pjb,baij->pia", tau871, rt2)
    )

    tau1304 = (
        einsum("pij,oji->po", tau1023, h.l.poo)
    )

    tau1305 = (
        einsum("po,oia->pia", tau1304, h.l.pov)
    )

    tau1306 = (
        einsum("pia,oai->po", tau222, h.l.pvo)
    )

    tau1307 = (
        einsum("po,oia->pia", tau1306, h.l.pov)
    )

    tau1308 = (
        einsum("pb,jp,baij->pia", tau955, a.x3, rt2)
    )

    tau1309 = (
        einsum("poj,pji->poi", tau19, tau451)
    )

    tau1310 = (
        einsum("pjb,baji->pia", tau834, rt2)
    )

    tau1311 = (
        einsum("pj,bp,baij->pia", tau521, a.x1, rt2)
    )

    tau1312 = (
        einsum("aj,pji->pia", a.t1, tau911)
    )

    tau1313 = (
        einsum("pia,oia->po", tau1312, h.l.pov)
    )

    tau1314 = (
        einsum("po,oia->pia", tau1313, h.l.pov)
    )

    tau1315 = (
        einsum("pjb,abij->pia", tau915, rt2)
    )

    tau1316 = (
        einsum("pa,ip,oia->po", tau890, a.x4, h.l.pov)
    )

    tau1317 = (
        einsum("po,oia->pia", tau1316, h.l.pov)
    )

    tau1318 = (
        einsum("aj,pji->pia", a.t1, tau1128)
    )

    tau1319 = (
        einsum("pia,oia->po", tau1318, h.l.pov)
    )

    tau1320 = (
        einsum("po,oia->pia", tau1319, h.l.pov)
    )

    tau1321 = (
        einsum("pb,jp,abji->pia", tau942, a.x3, rt2)
    )

    tau1322 = (
        einsum("pjb,abji->pia", tau988, rt2)
    )

    tau1323 = (
        einsum("pj,bp,baij->pia", tau77, a.x1, rt2)
    )

    tau1324 = (
        einsum("pia,poi->poa", tau249, tau95)
    )

    tau1325 = (
        einsum("pj,bp,baij->pia", tau15, a.x1, rt2)
    )

    tau1326 = (
        einsum("pij,poj->poi", tau365, tau95)
    )

    tau1327 = (
        einsum("pjb,baij->pia", tau1157, rt2)
    )

    tau1328 = (
        einsum("pj,bp,baij->pia", tau442, a.x1, rt2)
    )

    tau1329 = (
        einsum("pq,iq,qpo->poi", tau428, a.x4, tau20)
    )

    tau1330 = (
        einsum("pj,bp,baij->pia", tau269, a.x1, rt2)
    )

    tau1331 = (
        einsum("pjb,abij->pia", tau974, rt2)
    )

    tau1332 = (
        einsum("bi,pia->pab", a.t1, tau145)
    )

    tau1333 = (
        einsum("pab,oab->po", tau1332, h.l.pvv)
    )

    tau1334 = (
        einsum("po,oia->pia", tau1333, h.l.pov)
    )

    tau1335 = (
        einsum("pjb,baji->pia", tau988, rt2)
    )

    tau1336 = (
        einsum("pb,jp,abji->pia", tau926, a.x4, rt2)
    )

    tau1337 = (
        einsum("poj,pij->poi", tau19, tau232)
    )

    tau1338 = (
        einsum("pji,poj->poi", tau212, tau59)
    )

    tau1339 = (
        einsum("pj,bp,baji->pia", tau613, a.x1, rt2)
    )

    tau1340 = (
        einsum("pj,bp,baji->pia", tau511, a.x2, rt2)
    )

    tau1341 = (
        einsum("pj,bp,baij->pia", tau273, a.x1, rt2)
    )

    tau1342 = (
        einsum("pj,bp,baji->pia", tau442, a.x2, rt2)
    )

    tau1343 = (
        einsum("qp,iq,poi->pqo", tau31, a.x3, tau19)
    )

    tau1344 = (
        einsum("poa,poi->pia", tau858, tau89)
    )

    tau1345 = (
        einsum("ai,pib->pab", a.t1, tau1344)
    )

    tau1346 = (
        einsum("pba,abij->pij", tau1345, rt2)
    )

    tau1347 = (
        einsum("qp,iq,poi->pqo", tau289, a.x4, tau19)
    )

    tau1348 = (
        einsum("qp,iq,poi->pqo", tau197, a.x4, tau95)
    )

    tau1349 = (
        einsum("pab,abij->pij", tau1345, rt2)
    )

    tau1350 = (
        einsum("pia,poa->poi", tau22, tau858)
    )

    tau1351 = (
        einsum("aq,pia->pqi", a.x2, tau264)
    )

    tau1352 = (
        einsum("poi,poj->pij", tau88, tau95)
    )

    tau1353 = (
        einsum("aj,pij->pia", a.t1, tau1352)
    )

    tau1354 = (
        einsum("ai,pib->pab", a.t1, tau1353)
    )

    tau1355 = (
        einsum("pab,abij->pij", tau1354, rt2)
    )

    tau1356 = (
        einsum("qp,iq,poi->pqo", tau328, a.x3, tau19)
    )

    tau1357 = (
        einsum("pb,jp,baji->pia", tau1033, a.x4, rt2)
    )

    tau1358 = (
        einsum("pia,poa->poi", tau29, tau858)
    )

    tau1359 = (
        einsum("pj,bp,baji->pia", tau15, a.x1, rt2)
    )

    tau1360 = (
        einsum("pij,poj->poi", tau212, tau59)
    )

    tau1361 = (
        einsum("ap,qia->pqi", a.x1, tau201)
    )

    tau1362 = (
        einsum("poj,pij->poi", tau88, tau911)
    )

    tau1363 = (
        einsum("qp,iq,poi->pqo", tau855, a.x3, tau88)
    )

    tau1364 = (
        einsum("aq,pia->pqi", a.x1, tau222)
    )

    tau1365 = (
        einsum("pia,poa->poi", tau34, tau858)
    )

    tau1366 = (
        einsum("pb,jp,abji->pia", tau806, a.x4, rt2)
    )

    tau1367 = (
        einsum("aq,pia->pqi", a.x2, tau34)
    )

    tau1368 = (
        einsum("pj,bp,baji->pia", tau628, a.x1, rt2)
    )

    tau1369 = (
        einsum("aq,pia->pqi", a.x1, tau145)
    )

    tau1370 = (
        einsum("poj,pij->poi", tau88, tau899)
    )

    tau1371 = (
        einsum("pij,poj->poi", tau849, tau88)
    )

    tau1372 = (
        einsum("qp,iq,poi->pqo", tau176, a.x4, tau19)
    )

    tau1373 = (
        einsum("aq,pia->pqi", a.x1, tau0)
    )

    tau1374 = (
        einsum("poj,pij->poi", tau19, tau258)
    )

    tau1375 = (
        einsum("ap,qia->pqi", a.x1, tau34)
    )

    tau1376 = (
        einsum("aq,pia->pqi", a.x2, tau0)
    )

    tau1377 = (
        einsum("pia,poa->poi", tau63, tau858)
    )

    tau1378 = (
        einsum("pj,bp,baji->pia", tau172, a.x2, rt2)
    )

    tau1379 = (
        einsum("pb,jp,baji->pia", tau955, a.x4, rt2)
    )

    tau1380 = (
        einsum("poa,poi->pia", tau58, tau95)
    )

    tau1381 = (
        einsum("ai,pib->pab", a.t1, tau1380)
    )

    tau1382 = (
        einsum("pba,abij->pij", tau1381, rt2)
    )

    tau1383 = (
        einsum("qp,iq,poi->pqo", tau323, a.x3, tau19)
    )

    tau1384 = (
        einsum("pji,poj->poi", tau367, tau59)
    )

    tau1385 = (
        einsum("pia,poa->poi", tau123, tau58)
    )

    tau1386 = (
        einsum("poj,pij->poi", tau19, tau432)
    )

    tau1387 = (
        einsum("pq,iq,poi->pqo", tau539, a.x4, tau89)
    )

    tau1388 = (
        einsum("aq,pia->pqi", a.x1, tau264)
    )

    tau1389 = (
        einsum("poj,pij->poi", tau200, tau367)
    )

    tau1390 = (
        einsum("pq,iq,poi->pqo", tau569, a.x3, tau89)
    )

    tau1391 = (
        einsum("qp,iq,poi->pqo", tau129, a.x4, tau95)
    )

    tau1392 = (
        einsum("pia,poa->poi", tau222, tau58)
    )

    tau1393 = (
        einsum("poj,pij->poi", tau200, tau212)
    )

    tau1394 = (
        einsum("qp,iq,poi->pqo", tau18, a.x4, tau95)
    )

    tau1395 = (
        einsum("aq,pia->pqi", a.x2, tau136)
    )

    tau1396 = (
        einsum("pj,bp,baji->pia", tau43, a.x2, rt2)
    )

    tau1397 = (
        einsum("pj,bp,baji->pia", tau442, a.x1, rt2)
    )

    tau1398 = (
        einsum("qp,iq,poi->pqo", tau237, a.x4, tau19)
    )

    tau1399 = (
        einsum("ap,qia->pqi", a.x1, tau63)
    )

    tau1400 = (
        einsum("aq,pia->pqi", a.x2, tau201)
    )

    tau1401 = (
        einsum("pia,poa->poi", tau149, tau58)
    )

    tau1402 = (
        einsum("pb,jp,baji->pia", tau865, a.x4, rt2)
    )

    tau1403 = (
        einsum("pq,iq,poi->pqo", tau323, a.x3, tau89)
    )

    tau1404 = (
        einsum("aq,pia->pqi", a.x2, tau145)
    )

    tau1405 = (
        einsum("pij,poj->poi", tau367, tau59)
    )

    tau1406 = (
        einsum("pj,bp,baji->pia", tau628, a.x2, rt2)
    )

    tau1407 = (
        einsum("pq,iq,poi->pqo", tau328, a.x3, tau95)
    )

    tau1408 = (
        einsum("pia,poa->poi", tau145, tau58)
    )

    tau1409 = (
        einsum("qp,iq,poi->pqo", tau873, a.x4, tau89)
    )

    tau1410 = (
        einsum("pb,jp,baji->pia", tau404, a.x4, rt2)
    )

    tau1411 = (
        einsum("pb,jp,baji->pia", tau471, a.x4, rt2)
    )

    tau1412 = (
        einsum("pij,poj->poi", tau809, tau89)
    )

    tau1413 = (
        einsum("pj,bp,baji->pia", tau172, a.x1, rt2)
    )

    tau1414 = (
        einsum("qp,iq,poi->pqo", tau505, a.x3, tau19)
    )

    tau1415 = (
        einsum("pji,poj->poi", tau367, tau72)
    )

    tau1416 = (
        einsum("aj,pij->pia", a.t1, tau132)
    )

    tau1417 = (
        einsum("ai,pib->pab", a.t1, tau1416)
    )

    tau1418 = (
        einsum("pab,abij->pij", tau1417, rt2)
    )

    tau1419 = (
        einsum("pb,jp,baji->pia", tau9, a.x4, rt2)
    )

    tau1420 = (
        einsum("poj,pji->poi", tau156, tau367)
    )

    tau1421 = (
        einsum("pb,jp,abji->pia", tau210, a.x4, rt2)
    )

    tau1422 = (
        einsum("pb,jp,baji->pia", tau957, a.x4, rt2)
    )

    tau1423 = (
        einsum("qp,iq,poi->pqo", tau861, a.x3, tau88)
    )

    tau1424 = (
        einsum("pj,bp,baji->pia", tau68, a.x2, rt2)
    )

    tau1425 = (
        einsum("pji,poj->poi", tau212, tau72)
    )

    tau1426 = (
        einsum("qp,iq,poi->pqo", tau191, a.x4, tau95)
    )

    tau1427 = (
        einsum("poj,pji->poi", tau156, tau212)
    )

    tau1428 = (
        einsum("pob,poa->pab", tau69, tau858)
    )

    tau1429 = (
        einsum("pab,abij->pij", tau1428, rt2)
    )

    tau1430 = (
        einsum("pij,poj->poi", tau51, tau95)
    )

    tau1431 = (
        einsum("aq,pia->pqi", a.x2, tau222)
    )

    tau1432 = (
        einsum("qp,iq,poi->pqo", tau519, a.x4, tau19)
    )

    tau1433 = (
        einsum("pj,bp,baji->pia", tau511, a.x1, rt2)
    )

    tau1434 = (
        einsum("pb,jp,abji->pia", tau45, a.x4, rt2)
    )

    tau1435 = (
        einsum("pb,jp,baji->pia", tau798, a.x4, rt2)
    )

    tau1436 = (
        einsum("pq,iq,poi->pqo", tau861, a.x3, tau89)
    )

    tau1437 = (
        einsum("ap,qia->pqi", a.x1, tau136)
    )

    tau1438 = (
        einsum("pb,jp,baji->pia", tau121, a.x4, rt2)
    )

    tau1439 = (
        einsum("pb,jp,baij->pia", tau292, a.x4, rt2)
    )

    tau1440 = (
        einsum("qp,iq,poi->pqo", tau921, a.x4, tau89)
    )

    tau1441 = (
        einsum("pj,bp,baji->pia", tau68, a.x1, rt2)
    )

    tau1442 = (
        einsum("poj,pij->poi", tau19, tau51)
    )

    tau1443 = (
        einsum("pob,poa->pab", tau1048, tau58)
    )

    tau1444 = (
        einsum("pab,abij->pij", tau1443, rt2)
    )

    tau1445 = (
        einsum("pq,iq,poi->pqo", tau855, a.x3, tau89)
    )

    tau1446 = (
        einsum("pq,iq,poi->pqo", tau339, a.x4, tau89)
    )

    tau1447 = (
        einsum("pq,iq,poi->pqo", tau606, a.x4, tau89)
    )

    tau1448 = (
        einsum("poj,pij->poi", tau89, tau899)
    )

    tau1449 = (
        einsum("aq,pia->pqi", a.x2, tau63)
    )

    tau1450 = (
        einsum("pj,bp,baji->pia", tau15, a.x2, rt2)
    )

    tau1451 = (
        einsum("pj,bp,baji->pia", tau613, a.x2, rt2)
    )

    tau1452 = (
        einsum("pb,jp,baji->pia", tau246, a.x4, rt2)
    )

    tau1453 = (
        einsum("qp,iq,poi->pqo", tau163, a.x3, tau19)
    )

    tau1454 = (
        einsum("pq,iq,poi->pqo", tau31, a.x3, tau95)
    )

    tau1455 = (
        einsum("pj,bp,baij->pia", tau273, a.x2, rt2)
    )

    tau1456 = (
        einsum("qp,iq,poi->pqo", tau569, a.x3, tau19)
    )

    tau1457 = (
        einsum("pj,bp,baji->pia", tau43, a.x1, rt2)
    )

    tau1458 = (
        einsum("pb,jp,baji->pia", tau189, a.x4, rt2)
    )

    tau1459 = (
        einsum("pab,abij->pij", tau1381, rt2)
    )

    tau1460 = (
        einsum("pij,poj->poi", tau258, tau95)
    )

    tau1461 = (
        einsum("pij,poj->poi", tau809, tau88)
    )

    tau1462 = (
        einsum("pq,iq,poi->pqo", tau163, a.x3, tau89)
    )

    tau1463 = (
        einsum("pb,jp,baji->pia", tau812, a.x4, rt2)
    )

    tau1464 = (
        einsum("pb,jp,baij->pia", tau942, a.x4, rt2)
    )

    tau1465 = (
        einsum("qp,iq,poi->pqo", tau878, a.x4, tau89)
    )

    tau1466 = (
        einsum("pq,iq,poi->pqo", tau505, a.x3, tau89)
    )

    tau1467 = (
        einsum("pq,iq,poi->pqo", tau428, a.x4, tau89)
    )

    tau1468 = (
        einsum("qp,iq,poi->pqo", tau969, a.x4, tau89)
    )

    tau1469 = (
        einsum("pb,jp,abji->pia", tau210, a.x3, rt2)
    )

    tau1470 = (
        einsum("poj,pij->poi", tau19, tau383)
    )

    tau1471 = (
        einsum("qp,iq,poi->pqo", tau111, a.x4, tau19)
    )

    tau1472 = (
        einsum("pji,poj->poi", tau367, tau407)
    )

    tau1473 = (
        einsum("pij,poj->poi", tau830, tau88)
    )

    tau1474 = (
        einsum("poa,pia->poi", tau58, tau87)
    )

    tau1475 = (
        einsum("pj,bp,baji->pia", tau573, a.x1, rt2)
    )

    tau1476 = (
        einsum("pq,iq,poi->pqo", tau969, a.x3, tau88)
    )

    tau1477 = (
        einsum("pj,bp,baji->pia", tau418, a.x2, rt2)
    )

    tau1478 = (
        einsum("pb,jp,abji->pia", tau926, a.x3, rt2)
    )

    tau1479 = (
        einsum("pj,bp,baji->pia", tau297, a.x1, rt2)
    )

    tau1480 = (
        einsum("poj,pij->poi", tau157, tau212)
    )

    tau1481 = (
        einsum("pij,poj->poi", tau1128, tau88)
    )

    tau1482 = (
        einsum("pij,poj->poi", tau1023, tau89)
    )

    tau1483 = (
        einsum("pb,jp,baji->pia", tau955, a.x3, rt2)
    )

    tau1484 = (
        einsum("pj,bp,baji->pia", tau573, a.x2, rt2)
    )

    tau1485 = (
        einsum("pia,poa->poi", tau201, tau858)
    )

    tau1486 = (
        einsum("pq,iq,poi->pqo", tau302, a.x4, tau89)
    )

    tau1487 = (
        einsum("pij,poj->poi", tau212, tau407)
    )

    tau1488 = (
        einsum("pj,bp,baji->pia", tau387, a.x2, rt2)
    )

    tau1489 = (
        einsum("pq,iq,poi->pqo", tau18, a.x3, tau19)
    )

    tau1490 = (
        einsum("pij,poj->poi", tau367, tau62)
    )

    tau1491 = (
        einsum("pia,poa->poi", tau264, tau58)
    )

    tau1492 = (
        einsum("pia,poa->poi", tau16, tau858)
    )

    tau1493 = (
        einsum("pj,bp,baji->pia", tau297, a.x2, rt2)
    )

    tau1494 = (
        einsum("pq,iq,poi->pqo", tau873, a.x3, tau88)
    )

    tau1495 = (
        einsum("qp,iq,poi->pqo", tau539, a.x3, tau19)
    )

    tau1496 = (
        einsum("pj,bp,baij->pia", tau397, a.x2, rt2)
    )

    tau1497 = (
        einsum("pb,jp,baji->pia", tau957, a.x3, rt2)
    )

    tau1498 = (
        einsum("pj,bp,baji->pia", tau544, a.x2, rt2)
    )

    tau1499 = (
        einsum("pb,jp,baji->pia", tau121, a.x3, rt2)
    )

    tau1500 = (
        einsum("pq,iq,poi->pqo", tau949, a.x4, tau89)
    )

    tau1501 = (
        einsum("qp,iq,poi->pqo", tau437, a.x4, tau19)
    )

    tau1502 = (
        einsum("pq,iq,poi->pqo", tau437, a.x4, tau89)
    )

    tau1503 = (
        einsum("pq,iq,poi->pqo", tau380, a.x4, tau89)
    )

    tau1504 = (
        einsum("pj,bp,baji->pia", tau418, a.x1, rt2)
    )

    tau1505 = (
        einsum("pb,jp,baij->pia", tau292, a.x3, rt2)
    )

    tau1506 = (
        einsum("pj,bp,baji->pia", tau242, a.x1, rt2)
    )

    tau1507 = (
        einsum("poj,pij->poi", tau19, tau475)
    )

    tau1508 = (
        einsum("poj,pji->poi", tau313, tau367)
    )

    tau1509 = (
        einsum("pia,poa->poi", tau47, tau858)
    )

    tau1510 = (
        einsum("pb,jp,baji->pia", tau1033, a.x3, rt2)
    )

    tau1511 = (
        einsum("pj,bp,baji->pia", tau387, a.x1, rt2)
    )

    tau1512 = (
        einsum("pij,poj->poi", tau814, tau88)
    )

    tau1513 = (
        einsum("pb,jp,baji->pia", tau404, a.x3, rt2)
    )

    tau1514 = (
        einsum("pj,bp,baji->pia", tau109, a.x1, rt2)
    )

    tau1515 = (
        einsum("pq,iq,poi->pqo", tau111, a.x4, tau95)
    )

    tau1516 = (
        einsum("pb,jp,baij->pia", tau942, a.x3, rt2)
    )

    tau1517 = (
        einsum("pij,poj->poi", tau814, tau89)
    )

    tau1518 = (
        einsum("pq,iq,poi->pqo", tau357, a.x4, tau89)
    )

    tau1519 = (
        einsum("pj,bp,baji->pia", tau295, a.x1, rt2)
    )

    tau1520 = (
        einsum("pb,jp,baji->pia", tau812, a.x3, rt2)
    )

    tau1521 = (
        einsum("qp,iq,poi->pqo", tau882, a.x4, tau88)
    )

    tau1522 = (
        einsum("pb,jp,baji->pia", tau189, a.x3, rt2)
    )

    tau1523 = (
        einsum("pq,iq,poi->pqo", tau921, a.x3, tau88)
    )

    tau1524 = (
        einsum("pia,poa->poi", tau0, tau58)
    )

    tau1525 = (
        einsum("pia,poa->poi", tau101, tau58)
    )

    tau1526 = (
        einsum("pij,poj->poi", tau212, tau62)
    )

    tau1527 = (
        einsum("qp,iq,poi->pqo", tau350, a.x4, tau19)
    )

    tau1528 = (
        einsum("qp,iq,poi->pqo", tau428, a.x3, tau19)
    )

    tau1529 = (
        einsum("pq,iq,poi->pqo", tau882, a.x4, tau89)
    )

    tau1530 = (
        einsum("pq,iq,poi->pqo", tau176, a.x3, tau89)
    )

    tau1531 = (
        einsum("qp,iq,poi->pqo", tau357, a.x4, tau19)
    )

    tau1532 = (
        einsum("pij,poj->poi", tau83, tau95)
    )

    tau1533 = (
        einsum("pb,jp,abji->pia", tau806, a.x3, rt2)
    )

    tau1534 = (
        einsum("pb,jp,abji->pia", tau45, a.x3, rt2)
    )

    tau1535 = (
        einsum("qp,iq,poi->pqo", tau380, a.x4, tau19)
    )

    tau1536 = (
        einsum("pj,bp,baji->pia", tau295, a.x2, rt2)
    )

    tau1537 = (
        einsum("pb,jp,baji->pia", tau471, a.x3, rt2)
    )

    tau1538 = (
        einsum("pji,poj->poi", tau212, tau313)
    )

    tau1539 = (
        einsum("pb,jp,baji->pia", tau798, a.x3, rt2)
    )

    tau1540 = (
        einsum("pq,iq,poi->pqo", tau289, a.x3, tau89)
    )

    tau1541 = (
        einsum("pj,bp,baji->pia", tau544, a.x1, rt2)
    )

    tau1542 = (
        einsum("pq,iq,poi->pqo", tau878, a.x3, tau88)
    )

    tau1543 = (
        einsum("pji,poj->poi", tau212, tau407)
    )

    tau1544 = (
        einsum("pij,poj->poi", tau367, tau407)
    )

    tau1545 = (
        einsum("pq,iq,poi->pqo", tau350, a.x4, tau95)
    )

    tau1546 = (
        einsum("pq,iq,poi->pqo", tau197, a.x3, tau19)
    )

    tau1547 = (
        einsum("pq,iq,poi->pqo", tau129, a.x3, tau19)
    )

    tau1548 = (
        einsum("qp,iq,poi->pqo", tau339, a.x3, tau19)
    )

    tau1549 = (
        einsum("pij,poj->poi", tau1023, tau88)
    )

    tau1550 = (
        einsum("qp,iq,poi->pqo", tau949, a.x4, tau88)
    )

    tau1551 = (
        einsum("pj,bp,baij->pia", tau397, a.x1, rt2)
    )

    tau1552 = (
        einsum("pij,poj->poi", tau73, tau95)
    )

    tau1553 = (
        einsum("pq,iq,poi->pqo", tau191, a.x3, tau19)
    )

    tau1554 = (
        einsum("pb,jp,baji->pia", tau246, a.x3, rt2)
    )

    tau1555 = (
        einsum("pia,poa->poi", tau136, tau858)
    )

    tau1556 = (
        einsum("poj,pij->poi", tau19, tau73)
    )

    tau1557 = (
        einsum("pb,jp,baji->pia", tau9, a.x3, rt2)
    )

    tau1558 = (
        einsum("pq,iq,poi->pqo", tau519, a.x3, tau89)
    )

    tau1559 = (
        einsum("pj,bp,baji->pia", tau242, a.x2, rt2)
    )

    tau1560 = (
        einsum("pq,iq,poi->pqo", tau237, a.x3, tau89)
    )

    tau1561 = (
        einsum("pb,jp,baji->pia", tau865, a.x3, rt2)
    )

    tau1562 = (
        einsum("qp,iq,poi->pqo", tau606, a.x3, tau19)
    )

    tau1563 = (
        einsum("pj,bp,baji->pia", tau109, a.x2, rt2)
    )

    tau1564 = (
        einsum("poj,pij->poi", tau19, tau83)
    )

    tau1565 = (
        einsum("qp,iq,poi->pqo", tau302, a.x4, tau19)
    )

    tau1566 = (
        einsum("poj,pij->poi", tau157, tau367)
    )

    rx1 = (
        einsum("pi,pia->ap", tau3, tau5)
        - 4 * einsum("ip,pia->ap", a.x4, tau10)
        + einsum("pi,pia->ap", tau15, tau16)
        - einsum("poi,oia->ap", tau21, h.l.pov) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau27)
        - 2 * einsum("ip,pia->ap", a.x3, tau33)
        + einsum("ip,pia->ap", a.x3, tau39)
        + 2 * einsum("pi,pia->ap", tau43, tau16)
        + 4 * einsum("ip,pia->ap", a.x4, tau46)
        - einsum("qp,qa->ap", tau49, tau50) / 2
        - 4 * einsum("ip,pia->ap", a.x3, tau54)
        - 2 * einsum("poi,oia->ap", tau57, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau61)
        - 2 * einsum("pob,oba->ap", tau64, h.l.pvv)
        + einsum("pi,pia->ap", tau68, tau47)
        - einsum("pob,oba->ap", tau71, h.l.pvv)
        - einsum("poi,oia->ap", tau74, h.l.pov)
        - 4 * einsum("ip,pia->ap", a.x4, tau78)
        + 2 * einsum("ip,pia->ap", a.x4, tau82)
        - einsum("ip,pia->ap", a.x4, tau86)
        + einsum("pq,ip,qia->ap", tau92, a.x4, tau87)
        - einsum("ip,pia->ap", a.x4, tau94)
        + einsum("poi,oia->ap", tau97, h.l.pov)
        + einsum("qp,ip,qia->ap", tau98, a.x3, tau0)
        + 2 * einsum("ip,pia->ap", a.x3, tau100)
        + einsum("qa,pq->ap", tau102, tau104)
        + einsum("qp,ip,qia->ap", tau106, a.x3, tau16)
        + einsum("pi,pia->ap", tau109, tau29)
        + einsum("ip,pia->ap", a.x4, tau113)
        - einsum("qa,pq->ap", tau102, tau117) / 2
        - 2 * einsum("ip,pia->ap", a.x4, tau118)
        - einsum("ip,pia->ap", a.x4, tau122)
        - 2 * einsum("ip,pia->ap", a.x3, tau128)
        - 2 * einsum("ip,pia->ap", a.x4, tau131)
        + einsum("bp,pba->ap", a.x2, tau135)
        - 2 * einsum("ip,pia->ap", a.x3, tau141)
        - einsum("ip,pia->ap", a.x4, tau144)
        - 2 * einsum("pi,pia->ap", tau148, tau5)
        - einsum("pi,pia->ap", tau152, tau12) / 2
        + einsum("pi,pia->ap", tau155, tau12)
        - einsum("bp,pab->ap", a.x2, tau159)
        - einsum("pi,pia->ap", tau161, tau12) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau165)
        + 2 * einsum("ip,pia->ap", a.x3, tau168)
        + einsum("pi,pia->ap", tau172, tau16)
        + einsum("poi,oia->ap", tau174, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau178)
        - 2 * einsum("qp,ip,qia->ap", tau98, a.x3, tau87)
        - einsum("ip,pia->ap", a.x4, tau181)
        + einsum("qp,ip,qia->ap", tau183, a.x4, tau136)
        + einsum("pq,qa->ap", tau185, tau186)
        + einsum("ip,pia->ap", a.x3, tau190)
        - einsum("poi,oia->ap", tau192, h.l.pov) / 2
        - 2 * einsum("pob,oba->ap", tau193, h.l.pvv)
        - einsum("qp,qa->ap", tau194, tau48) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau199)
        + einsum("pob,oba->ap", tau202, h.l.pvv)
        + einsum("pq,qa->ap", tau104, tau203)
        + 2 * einsum("ip,pia->ap", a.x3, tau206)
        - 4 * einsum("ip,pia->ap", a.x3, tau207)
        + einsum("qp,ip,qia->ap", tau208, a.x3, tau123)
        - 2 * einsum("ip,pia->ap", a.x3, tau211)
        - einsum("poi,oia->ap", tau215, h.l.pov) / 2
        - einsum("bp,pab->ap", a.x2, tau217)
        - einsum("qp,ip,qia->ap", tau220, a.x4, tau149) / 2
        + einsum("qp,ip,qia->ap", tau221, a.x4, tau201)
        - 2 * einsum("qp,ip,qia->ap", tau208, a.x3, tau222)
        - 2 * einsum("ip,pia->ap", a.x3, tau223)
        + einsum("ip,pia->ap", a.x4, tau225)
        + einsum("ip,pia->ap", a.x4, tau227)
        - 2 * einsum("poi,oia->ap", tau229, h.l.pov)
        + einsum("qp,ip,qia->ap", tau231, a.x4, tau63)
        + 2 * einsum("ip,pia->ap", a.x3, tau235)
        - 2 * einsum("ip,pia->ap", a.x3, tau239)
        + 2 * einsum("pi,pia->ap", tau242, tau29)
        + 2 * einsum("ip,pia->ap", a.x4, tau244)
        - einsum("ba,pb->ap", h.f.vv, tau184)
        + einsum("ip,pia->ap", a.x4, tau247)
        - 2 * einsum("pob,oba->ap", tau250, h.l.pvv)
        + einsum("ip,pia->ap", a.x3, tau253)
        - 2 * einsum("qa,pq->ap", tau186, tau24)
        - 4 * einsum("ip,pia->ap", a.x3, tau257)
        - einsum("poi,oia->ap", tau259, h.l.pov)
        - 2 * einsum("qa,qp->ap", tau103, tau260)
        + 2 * einsum("ba,pb->ap", h.f.vv, tau23)
        - einsum("ip,pia->ap", a.x3, tau261) / 2
        + einsum("ip,pia->ap", a.x4, tau263)
        + einsum("pq,ip,qia->ap", tau92, a.x4, tau264)
        - 2 * einsum("pob,oba->ap", tau265, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x3, tau268)
        + einsum("qp,ip,qia->ap", tau92, a.x3, tau149)
        - einsum("ip,pia->ap", a.x3, tau270)
        + 2 * einsum("pob,oba->ap", tau271, h.l.pvv)
        - 2 * einsum("pi,pia->ap", tau273, tau136)
        + einsum("ip,pia->ap", a.x4, tau275)
        + einsum("ip,pia->ap", a.x4, tau277)
        + 2 * einsum("ip,pia->ap", a.x3, tau280)
        - 2 * einsum("ip,pia->ap", a.x4, tau283)
        - 2 * einsum("ip,pia->ap", a.x3, tau286)
        - einsum("poi,oia->ap", tau288, h.l.pov) / 2
        - 2 * einsum("poi,oia->ap", tau290, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau293)
        - einsum("pi,pia->ap", tau295, tau29) / 2
        - einsum("pi,pia->ap", tau297, tau29) / 2
        + einsum("pq,qa->ap", tau298, tau300)
        + einsum("qp,ip,qia->ap", tau231, a.x4, tau22)
        + einsum("poi,oia->ap", tau303, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau304)
        + einsum("ip,pia->ap", a.x4, tau306)
        + einsum("ip,pia->ap", a.x4, tau307)
        + einsum("ip,pia->ap", a.x3, tau308)
        - einsum("ip,pia->ap", a.x3, tau309)
        + einsum("qp,ip,qia->ap", tau310, a.x4, tau34)
        + einsum("ip,pia->ap", a.x3, tau312)
        - einsum("poi,oia->ap", tau314, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x3, tau315)
        - 2 * einsum("ip,pia->ap", a.x3, tau316)
        + einsum("pob,oba->ap", tau317, h.l.pvv)
        - 2 * einsum("pi,pia->ap", tau320, tau5)
        + 2 * einsum("ip,pia->ap", a.x3, tau321)
        + einsum("ip,pia->ap", a.x3, tau325)
        - einsum("ip,pia->ap", a.x3, tau326)
        + 4 * einsum("ip,pia->ap", a.x3, tau330)
        + einsum("ip,pia->ap", a.x3, tau331)
        - einsum("pq,ip,qia->ap", tau332, a.x4, tau145) / 2
        + einsum("qp,ip,qia->ap", tau333, a.x3, tau34)
        + 2 * einsum("ip,pia->ap", a.x4, tau334)
        - 2 * einsum("ip,pia->ap", a.x3, tau336)
        + einsum("ip,pia->ap", a.x3, tau338)
        + einsum("poi,oia->ap", tau340, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau344)
        + einsum("ip,pia->ap", a.x4, tau349)
        + einsum("poi,oia->ap", tau351, h.l.pov)
        + einsum("qp,qa->ap", tau194, tau55)
        + einsum("qp,ip,qia->ap", tau220, a.x4, tau145)
        + einsum("pq,ip,qia->ap", tau208, a.x4, tau0)
        - 2 * einsum("pob,oba->ap", tau352, h.l.pvv)
        + 2 * einsum("ip,pia->ap", a.x3, tau353)
        + 2 * einsum("ia,pi->ap", tau119, tau354)
        + 2 * einsum("ia,pi->ap", tau119, tau355)
        + einsum("ip,pia->ap", a.x4, tau359)
        - 2 * einsum("ip,pia->ap", a.x4, tau361)
        + einsum("ip,pia->ap", a.x4, tau362)
        + einsum("poi,oia->ap", tau363, h.l.pov)
        - einsum("poi,oia->ap", tau366, h.l.pov)
        - einsum("pi,pia->ap", tau172, tau47) / 2
        - einsum("pi,pia->ap", tau37, tau12) / 2
        - einsum("poi,oia->ap", tau370, h.l.pov) / 2
        + einsum("pq,qa->ap", tau24, tau300)
        + 4 * einsum("pq,ip,qia->ap", tau371, a.x3, tau87)
        + einsum("poi,oia->ap", tau372, h.l.pov)
        + einsum("qp,qa->ap", tau260, tau50)
        + einsum("poi,oia->ap", tau375, h.l.pov)
        - einsum("ip,pia->ap", a.x3, tau378)
        + einsum("ip,pia->ap", a.x4, tau379)
        + einsum("ip,pia->ap", a.x4, tau382)
        - einsum("pq,ip,qia->ap", tau92, a.x4, tau101) / 2
        - einsum("poi,oia->ap", tau384, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau385)
        - 2 * einsum("ia,pi->ap", h.f.ov, tau354)
        - 4 * einsum("pi,pia->ap", tau387, tau22)
        - einsum("bp,pba->ap", a.x2, tau390) / 2
        + 4 * einsum("pb,ba->ap", tau23, tau44)
        - einsum("pq,qa->ap", tau185, tau300) / 2
        - einsum("ip,pia->ap", a.x4, tau394) / 2
        - 2 * einsum("poi,oia->ap", tau395, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau396)
        + 2 * einsum("bp,pba->ap", a.x2, tau217)
        + einsum("ip,pia->ap", a.x4, tau398)
        - einsum("ip,pia->ap", a.x3, tau399) / 2
        + einsum("ip,pia->ap", a.x3, tau400)
        - einsum("ip,pia->ap", a.x4, tau401) / 2
        - 2 * einsum("pq,qa->ap", tau104, tau345)
        + einsum("ip,pia->ap", a.x4, tau403)
        - 2 * einsum("ip,pia->ap", a.x4, tau405)
        + einsum("ip,pia->ap", a.x4, tau406)
        - einsum("bp,pba->ap", a.x2, tau409)
        - 2 * einsum("ip,pia->ap", a.x4, tau411)
        + einsum("ip,pia->ap", a.x3, tau413)
        + einsum("qa,qp->ap", tau103, tau49)
        + einsum("poi,oia->ap", tau414, h.l.pov)
        - einsum("bp,pba->ap", a.x2, tau416)
        - einsum("pi,pia->ap", tau43, tau47)
        - 2 * einsum("pi,pia->ap", tau418, tau22)
        - 2 * einsum("ip,pia->ap", a.x4, tau419)
        - 2 * einsum("pi,pia->ap", tau342, tau5)
        - 2 * einsum("pob,oba->ap", tau420, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x3, tau421)
        + 2 * einsum("ip,pia->ap", a.x3, tau424)
        + einsum("qp,qa->ap", tau425, tau50)
        - einsum("bp,pba->ap", a.x2, tau427)
        - einsum("poi,oia->ap", tau429, h.l.pov) / 2
        + einsum("ia,pi->ap", h.f.ov, tau430)
        + einsum("ip,pia->ap", a.x3, tau431)
        - 4 * einsum("ip,pia->ap", a.x3, tau434)
        - 2 * einsum("qp,ip,qia->ap", tau333, a.x3, tau22)
        - 2 * einsum("ip,pia->ap", a.x4, tau435)
        - 2 * einsum("ip,pia->ap", a.x4, tau439)
        - einsum("poi,oia->ap", tau440, h.l.pov)
        - 2 * einsum("pi,pia->ap", tau68, tau16)
        + einsum("pq,qa->ap", tau117, tau124)
        + einsum("pob,oba->ap", tau441, h.l.pvv)
        - einsum("pi,pia->ap", tau15, tau47) / 2
        + 2 * einsum("pi,pia->ap", tau442, tau47)
        - einsum("ip,pia->ap", a.x4, tau446) / 2
        + 4 * einsum("ip,pia->ap", a.x3, tau449)
        + einsum("qp,qa->ap", tau194, tau23)
        + einsum("poi,oia->ap", tau452, h.l.pov)
        - einsum("pi,pia->ap", tau455, tau12) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau456)
        + einsum("pi,pia->ap", tau458, tau12)
        - einsum("qa,qp->ap", tau184, tau194) / 2
        + 2 * einsum("bp,pab->ap", a.x2, tau409)
        + 4 * einsum("ba,pb->ap", tau44, tau55)
        - einsum("ip,pia->ap", a.x4, tau460) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau462)
        + 2 * einsum("poi,oia->ap", tau463, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau465)
        + 2 * einsum("ip,pia->ap", a.x4, tau466)
        - 2 * einsum("ip,pia->ap", a.x4, tau467)
        - 2 * einsum("ip,pia->ap", a.x3, tau469)
        + einsum("poi,oia->ap", tau470, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau472) / 2
        + einsum("pq,ip,qia->ap", tau98, a.x3, tau101)
        - einsum("pq,ip,qia->ap", tau92, a.x4, tau0) / 2
        - 2 * einsum("qa,qp->ap", tau23, tau473)
        - einsum("poi,oia->ap", tau474, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau477)
        + einsum("pi,pia->ap", tau479, tau12)
        + 2 * einsum("ip,pia->ap", a.x3, tau480)
        - einsum("qp,ip,qia->ap", tau92, a.x3, tau123) / 2
        + einsum("qp,ip,qia->ap", tau92, a.x3, tau222)
        + 2 * einsum("ip,pia->ap", a.x4, tau481)
        - 4 * einsum("pi,pia->ap", tau442, tau16)
        + einsum("pob,oba->ap", tau482, h.l.pvv)
        + einsum("pi,pia->ap", tau484, tau5)
        + einsum("ip,pia->ap", a.x4, tau485)
        - 2 * einsum("ip,pia->ap", a.x4, tau486)
        - einsum("ip,pia->ap", a.x4, tau488) / 2
        + einsum("pob,oba->ap", tau489, h.l.pvv)
        + 2 * einsum("ip,pia->ap", a.x4, tau490)
        + einsum("ip,pia->ap", a.x4, tau491)
        - 2 * einsum("qp,ip,qia->ap", tau106, a.x3, tau47)
        - 2 * einsum("ip,pia->ap", a.x4, tau493)
        + einsum("pi,pia->ap", tau495, tau5)
        + einsum("pi,pia->ap", tau139, tau12)
        + 2 * einsum("ip,pia->ap", a.x4, tau496)
        - einsum("ba,pb->ap", h.f.vv, tau48)
        + einsum("pob,oba->ap", tau497, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau500)
        - 2 * einsum("ip,pia->ap", a.x3, tau503)
        - 2 * einsum("pq,qa->ap", tau104, tau124)
        + einsum("qa,qp->ap", tau184, tau473)
        - 4 * einsum("pi,ia->ap", tau355, tau7)
        - 2 * einsum("ip,pia->ap", a.x3, tau507)
        + einsum("ip,pia->ap", a.x4, tau510)
        - 4 * einsum("pi,pia->ap", tau511, tau16)
        - 4 * einsum("ip,pia->ap", a.x4, tau512)
        - einsum("poi,oia->ap", tau513, h.l.pov) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau514)
        - 2 * einsum("pob,oba->ap", tau515, h.l.pvv)
        + einsum("bp,pab->ap", a.x2, tau516)
        + 2 * einsum("bp,pab->ap", a.x2, tau518)
        - 2 * einsum("poi,oia->ap", tau520, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x3, tau522)
        - 2 * einsum("pq,ip,qia->ap", tau371, a.x3, tau0)
        - 2 * einsum("ip,pia->ap", a.x3, tau524)
        + 2 * einsum("ip,pia->ap", a.x3, tau526)
        + einsum("ip,pia->ap", a.x3, tau528)
        + einsum("pi,pia->ap", tau397, tau34)
        - 2 * einsum("pi,pia->ap", tau530, tau5)
        - einsum("qp,ip,qia->ap", tau231, a.x4, tau29) / 2
        + einsum("poi,oia->ap", tau532, h.l.pov)
        - 4 * einsum("ip,pia->ap", a.x3, tau533)
        - einsum("bp,pab->ap", a.x2, tau535)
        + 2 * einsum("bp,pab->ap", a.x2, tau427)
        + einsum("pi,pia->ap", tau444, tau5)
        - 2 * einsum("pq,ip,qia->ap", tau208, a.x4, tau87)
        + einsum("poi,oia->ap", tau536, h.l.pov)
        + einsum("qa,pq->ap", tau186, tau537)
        - 2 * einsum("ia,pi->ap", h.f.ov, tau355)
        + 2 * einsum("poi,oia->ap", tau538, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau541)
        - 2 * einsum("pq,ip,qia->ap", tau208, a.x4, tau264)
        + einsum("qp,ip,qia->ap", tau332, a.x4, tau222)
        - einsum("qp,ip,qia->ap", tau542, a.x3, tau136) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau543)
        - einsum("ip,pia->ap", a.x4, tau545)
        - einsum("ip,pia->ap", a.x4, tau549) / 2
        - 2 * einsum("ip,pia->ap", a.x4, tau550)
        + 2 * einsum("ip,pia->ap", a.x3, tau553)
        + 2 * einsum("ip,pia->ap", a.x3, tau554)
        + einsum("bp,pab->ap", a.x2, tau390)
        - einsum("ia,pi->ap", tau119, tau430)
        + einsum("ip,pia->ap", a.x3, tau556)
        + einsum("ip,pia->ap", a.x4, tau558)
        - einsum("ip,pia->ap", a.x4, tau561)
        + einsum("ip,pia->ap", a.x3, tau562)
        + 2 * einsum("ip,pia->ap", a.x4, tau565)
        - einsum("pq,qa->ap", tau117, tau203) / 2
        - einsum("qp,ip,qia->ap", tau332, a.x4, tau123) / 2
        + 2 * einsum("ba,pb->ap", h.f.vv, tau55)
        - 2 * einsum("ip,pia->ap", a.x4, tau568)
        + einsum("ip,pia->ap", a.x3, tau571)
        + 2 * einsum("pi,ia->ap", tau430, tau7)
        + einsum("pq,ip,qia->ap", tau332, a.x4, tau149)
        + 2 * einsum("pi,pia->ap", tau573, tau22)
        + einsum("poi,oia->ap", tau574, h.l.pov)
        - 2 * einsum("poi,oia->ap", tau575, h.l.pov)
        - einsum("bp,pab->ap", a.x2, tau576) / 2
        + einsum("poi,oia->ap", tau577, h.l.pov)
        + 4 * einsum("ip,pia->ap", a.x3, tau578)
        + 2 * einsum("pi,pia->ap", tau511, tau47)
        - 4 * einsum("ip,pia->ap", a.x3, tau579)
        - einsum("ip,pia->ap", a.x4, tau582)
        - einsum("pi,pia->ap", tau544, tau29)
        - 4 * einsum("ip,pia->ap", a.x3, tau583)
        + einsum("ip,pia->ap", a.x4, tau585)
        + einsum("ip,pia->ap", a.x3, tau586)
        - einsum("qa,qp->ap", tau50, tau587) / 2
        + 4 * einsum("ip,pia->ap", a.x3, tau589)
        - einsum("ip,pia->ap", a.x4, tau590)
        + einsum("ip,pia->ap", a.x4, tau592)
        + einsum("pob,oba->ap", tau593, h.l.pvv)
        + 4 * einsum("ip,pia->ap", a.x3, tau595)
        - 2 * einsum("poi,oia->ap", tau596, h.l.pov)
        - 2 * einsum("qp,qa->ap", tau473, tau55)
        - einsum("poi,oia->ap", tau597, h.l.pov) / 2
        + einsum("pq,ip,qia->ap", tau220, a.x4, tau123)
        + einsum("ip,pia->ap", a.x4, tau598)
        - 2 * einsum("ip,pia->ap", a.x3, tau600)
        - 2 * einsum("qp,ip,qia->ap", tau601, a.x3, tau34)
        + einsum("qp,ip,qia->ap", tau542, a.x3, tau47)
        - einsum("ip,pia->ap", a.x3, tau602)
        - 2 * einsum("ip,pia->ap", a.x3, tau603)
        + 4 * einsum("ip,pia->ap", a.x3, tau604)
        + 2 * einsum("ip,pia->ap", a.x4, tau605)
        + einsum("poi,oia->ap", tau607, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau610)
        + einsum("pi,pia->ap", tau273, tau201)
        + einsum("qp,ip,qia->ap", tau221, a.x4, tau47)
        + einsum("poi,oia->ap", tau611, h.l.pov)
        + einsum("ip,pia->ap", a.x3, tau614)
        + 2 * einsum("poi,oia->ap", tau615, h.l.pov)
        - einsum("pi,pia->ap", tau573, tau29)
        + 4 * einsum("qp,ip,qia->ap", tau601, a.x3, tau22)
        - 2 * einsum("ip,pia->ap", a.x3, tau616)
        - 2 * einsum("ip,pia->ap", a.x3, tau618)
        - einsum("qp,ip,qia->ap", tau183, a.x4, tau47) / 2
        - 2 * einsum("pi,pia->ap", tau397, tau63)
        - 2 * einsum("ip,pia->ap", a.x3, tau619)
        + 2 * einsum("ip,pia->ap", a.x4, tau622)
        - einsum("bp,pab->ap", a.x2, tau135) / 2
        - 4 * einsum("ip,pia->ap", a.x3, tau624)
        - einsum("poi,oia->ap", tau625, h.l.pov) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau626)
        - einsum("ip,pia->ap", a.x3, tau627) / 2
        + 2 * einsum("bp,pba->ap", a.x2, tau159)
        + einsum("qp,ip,qia->ap", tau542, a.x3, tau201)
        + einsum("qp,ip,qia->ap", tau183, a.x4, tau16)
        + einsum("qa,qp->ap", tau103, tau587)
        - einsum("pi,pia->ap", tau628, tau47)
        - 2 * einsum("ip,pia->ap", a.x4, tau630)
        + einsum("pi,pia->ap", tau613, tau47)
        + einsum("bp,pba->ap", a.x2, tau576)
        + einsum("ia,pi->ap", h.f.ov, tau631)
        - einsum("ip,pia->ap", a.x4, tau632)
        - einsum("bp,pba->ap", a.x2, tau518)
        + einsum("poi,oia->ap", tau634, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x4, tau637)
        - 2 * einsum("qp,ip,qia->ap", tau310, a.x4, tau22)
        + einsum("ip,pia->ap", a.x4, tau638)
        + 2 * einsum("ip,pia->ap", a.x3, tau639)
        - 2 * einsum("ip,pia->ap", a.x4, tau640)
        + 2 * einsum("ip,pia->ap", a.x3, tau642)
        - einsum("bp,pab->ap", a.x2, tau643)
        - einsum("poi,oia->ap", tau644, h.l.pov) / 2
        - einsum("qp,ip,qia->ap", tau92, a.x3, tau145) / 2
        + 4 * einsum("ip,pia->ap", a.x3, tau646)
        - 2 * einsum("ip,pia->ap", a.x3, tau647)
        + einsum("pob,oba->ap", tau648, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x3, tau650)
        - einsum("poi,oia->ap", tau651, h.l.pov) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau654)
        + einsum("qp,ip,qia->ap", tau333, a.x3, tau29)
        - 2 * einsum("pob,oba->ap", tau655, h.l.pvv)
        - einsum("pq,ip,qia->ap", tau220, a.x4, tau222) / 2
        + 2 * einsum("poi,oia->ap", tau656, h.l.pov)
        + einsum("poi,oia->ap", tau657, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau659) / 2
        - 2 * einsum("qa,qp->ap", tau103, tau425)
        + einsum("qp,ip,qia->ap", tau106, a.x3, tau136)
        - einsum("poi,oia->ap", tau660, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x4, tau661)
        - einsum("poi,oia->ap", tau663, h.l.pov)
        - einsum("poi,oia->ap", tau664, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x3, tau665)
        - einsum("ip,pia->ap", a.x4, tau667) / 2
        + einsum("ip,pia->ap", a.x4, tau669)
        - einsum("poi,oia->ap", tau670, h.l.pov) / 2
        - einsum("ip,pia->ap", a.x3, tau671)
        + 2 * einsum("ip,pia->ap", a.x4, tau672)
        + 4 * einsum("ip,pia->ap", a.x3, tau673)
        - 2 * einsum("qp,ip,qia->ap", tau310, a.x4, tau63)
        - einsum("ip,pia->ap", a.x4, tau675) / 2
        + einsum("ip,pia->ap", a.x3, tau676)
        + einsum("qp,ip,qia->ap", tau208, a.x3, tau145)
        - 2 * einsum("ip,pia->ap", a.x3, tau677)
        - 2 * einsum("ip,pia->ap", a.x4, tau678)
        - 2 * einsum("ip,pia->ap", a.x3, tau681)
        - einsum("ip,pia->ap", a.x4, tau682)
        + 2 * einsum("bp,pba->ap", a.x2, tau643)
        + 4 * einsum("ip,pia->ap", a.x3, tau683)
        + einsum("ip,pia->ap", a.x3, tau684)
        + einsum("ip,pia->ap", a.x3, tau686)
        - 2 * einsum("pi,pia->ap", tau109, tau22)
        - einsum("ip,pia->ap", a.x4, tau688) / 2
        + einsum("ip,pia->ap", a.x4, tau689)
        + einsum("pob,oba->ap", tau690, h.l.pvv)
        + 2 * einsum("ip,pia->ap", a.x4, tau691)
        - 2 * einsum("qp,ip,qia->ap", tau208, a.x3, tau149)
        + 2 * einsum("ip,pia->ap", a.x3, tau692)
        - 2 * einsum("pi,pia->ap", tau613, tau16)
        - einsum("ip,pia->ap", a.x4, tau693) / 2
        + einsum("ip,pia->ap", a.x4, tau695)
        + einsum("ip,pia->ap", a.x3, tau696)
        + 2 * einsum("pi,ia->ap", tau631, tau7)
        - einsum("ip,pia->ap", a.x4, tau697) / 2
        + einsum("ip,pia->ap", a.x3, tau699)
        + einsum("ip,pia->ap", a.x4, tau700)
        + einsum("ip,pia->ap", a.x3, tau703)
        + 2 * einsum("bp,pba->ap", a.x2, tau535)
        - 4 * einsum("pi,pia->ap", tau242, tau22)
        - einsum("poi,oia->ap", tau704, h.l.pov) / 2
        - einsum("ip,pia->ap", a.x4, tau706) / 2
        + 4 * einsum("ip,pia->ap", a.x3, tau709)
        - 2 * einsum("ip,pia->ap", a.x3, tau711)
        + 2 * einsum("poi,oia->ap", tau712, h.l.pov)
        + einsum("ip,pia->ap", a.x3, tau714)
        + 2 * einsum("bp,pab->ap", a.x2, tau416)
        - 2 * einsum("qp,ip,qia->ap", tau601, a.x3, tau29)
        + einsum("ip,pia->ap", a.x4, tau716)
        - einsum("ip,pia->ap", a.x3, tau717)
        - 2 * einsum("ip,pia->ap", a.x4, tau718)
        + einsum("poi,oia->ap", tau719, h.l.pov)
        - einsum("qp,ip,qia->ap", tau183, a.x4, tau201) / 2
        - 4 * einsum("ip,pia->ap", a.x3, tau722)
        + einsum("pq,ip,qia->ap", tau208, a.x4, tau101)
        + 4 * einsum("qp,ip,qia->ap", tau601, a.x3, tau63)
        - 2 * einsum("qp,ip,qia->ap", tau333, a.x3, tau63)
        + 4 * einsum("ip,pia->ap", a.x3, tau723)
        - einsum("poi,oia->ap", tau724, h.l.pov) / 2
        + einsum("qp,ip,qia->ap", tau310, a.x4, tau29)
        - 2 * einsum("pb,ba->ap", tau184, tau44)
        + einsum("pq,qa->ap", tau117, tau345)
        - einsum("qp,ip,qia->ap", tau221, a.x4, tau136) / 2
        + einsum("ip,pia->ap", a.x4, tau725)
        + einsum("pob,oba->ap", tau726, h.l.pvv)
        - 4 * einsum("ip,pia->ap", a.x3, tau727)
        - einsum("ip,pia->ap", a.x3, tau728) / 2
        + einsum("ip,pia->ap", a.x3, tau729)
        + einsum("pi,pia->ap", tau295, tau22)
        + einsum("ip,pia->ap", a.x3, tau730)
        - 2 * einsum("ip,pia->ap", a.x4, tau731)
        - einsum("qp,ip,qia->ap", tau221, a.x4, tau16) / 2
        + einsum("ip,pia->ap", a.x3, tau732)
        - 2 * einsum("ip,pia->ap", a.x4, tau733)
        - 2 * einsum("ip,pia->ap", a.x3, tau734)
        + einsum("ip,pia->ap", a.x4, tau736)
        + 4 * einsum("ip,pia->ap", a.x3, tau738)
        - 2 * einsum("ip,pia->ap", a.x4, tau739)
        - einsum("ip,pia->ap", a.x4, tau740)
        + einsum("pi,pia->ap", tau418, tau29)
        + 4 * einsum("ip,pia->ap", a.x3, tau742)
        - 2 * einsum("pob,oba->ap", tau743, h.l.pvv)
        - einsum("ia,pi->ap", tau119, tau631)
        + einsum("ip,pia->ap", a.x4, tau746)
        - 2 * einsum("qp,ip,qia->ap", tau371, a.x3, tau101)
        - einsum("ip,pia->ap", a.x4, tau747)
        - einsum("ip,pia->ap", a.x3, tau748)
        + 2 * einsum("poi,oia->ap", tau749, h.l.pov)
        + einsum("poi,oia->ap", tau750, h.l.pov)
        - 2 * einsum("ba,pb->ap", tau44, tau48)
        + einsum("poi,oia->ap", tau751, h.l.pov)
        - einsum("qp,ip,qia->ap", tau542, a.x3, tau16) / 2
        - einsum("ip,pia->ap", a.x4, tau752)
        - 2 * einsum("pob,oba->ap", tau753, h.l.pvv)
        - einsum("qp,ip,qia->ap", tau231, a.x4, tau34) / 2
        - einsum("pob,oba->ap", tau754, h.l.pvv)
        + einsum("pi,pia->ap", tau297, tau22)
        + einsum("ip,pia->ap", a.x4, tau755)
        + einsum("qp,qa->ap", tau473, tau48)
        - 2 * einsum("ip,pia->ap", a.x4, tau756)
        + einsum("poi,oia->ap", tau757, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau759) / 2
        - 2 * einsum("ip,pia->ap", a.x4, tau760)
        + 2 * einsum("pob,oba->ap", tau761, h.l.pvv)
        + 2 * einsum("ip,pia->ap", a.x4, tau762)
        - einsum("qa,pq->ap", tau300, tau537) / 2
        + 2 * einsum("poi,oia->ap", tau763, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau764)
        + einsum("ip,pia->ap", a.x4, tau766)
        + einsum("poi,oia->ap", tau767, h.l.pov)
        + 2 * einsum("pi,pia->ap", tau544, tau22)
        - 4 * einsum("pi,ia->ap", tau354, tau7)
        + 2 * einsum("poi,oia->ap", tau768, h.l.pov)
        - 2 * einsum("pq,ip,qia->ap", tau98, a.x3, tau264)
        + 2 * einsum("pi,pia->ap", tau387, tau29)
        + einsum("ip,pia->ap", a.x4, tau769)
        + 4 * einsum("qp,ip,qia->ap", tau371, a.x3, tau264)
        + 2 * einsum("poi,oia->ap", tau770, h.l.pov)
        - 2 * einsum("qa,pq->ap", tau186, tau298)
        - einsum("bp,pba->ap", a.x2, tau516) / 2
        - einsum("ip,pia->ap", a.x4, tau772) / 2
        + 2 * einsum("poi,oia->ap", tau773, h.l.pov)
        - 2 * einsum("poi,oia->ap", tau774, h.l.pov)
        - 2 * einsum("qp,ip,qia->ap", tau106, a.x3, tau201)
        + 2 * einsum("pi,pia->ap", tau628, tau16)
        + 2 * einsum("ip,pia->ap", a.x4, tau775)
        - 4 * einsum("ip,pia->ap", a.x3, tau776)
        + einsum("ip,pia->ap", a.x3, tau778)
        + einsum("ip,pia->ap", a.x4, tau779)
        - 2 * einsum("ip,pia->ap", a.x4, tau781)
        - 2 * einsum("ip,pia->ap", a.x4, tau782)
        - einsum("poi,oia->ap", tau783, h.l.pov) / 2
        + einsum("ip,pia->ap", a.x3, tau784)
        - 2 * einsum("ip,pia->ap", a.x3, tau786)
        - einsum("poi,oia->ap", tau787, h.l.pov) / 2
    )

    rx2 = (
        - 2 * einsum("pob,oba->ap", tau788, h.l.pvv)
        - 4 * einsum("ip,pia->ap", a.x4, tau792)
        + einsum("qa,pq->ap", tau186, tau391)
        + 2 * einsum("pi,pia->ap", tau442, tau101)
        - 2 * einsum("qa,pq->ap", tau55, tau793)
        + einsum("pob,oba->ap", tau794, h.l.pvv)
        + einsum("ip,pia->ap", a.x3, tau796)
        + einsum("poi,oia->ap", tau797, h.l.pov)
        - einsum("ip,pia->ap", a.x4, tau799) / 2
        - einsum("poi,oia->ap", tau800, h.l.pov) / 2
        + einsum("pq,ip,qia->ap", tau802, a.x4, tau47)
        + einsum("qp,ip,qia->ap", tau802, a.x3, tau63)
        + einsum("pq,ip,qia->ap", tau310, a.x4, tau145)
        + 4 * einsum("pb,ba->ap", tau124, tau44)
        + einsum("bp,pab->ap", a.x1, tau135)
        - 2 * einsum("qa,pq->ap", tau186, tau346)
        + einsum("qp,ip,qia->ap", tau803, a.x3, tau34)
        - 2 * einsum("pq,ip,qia->ap", tau805, a.x4, tau63)
        + 4 * einsum("ip,pia->ap", a.x3, tau807)
        - einsum("poi,oia->ap", tau808, h.l.pov) / 2
        - einsum("poi,oia->ap", tau810, h.l.pov)
        - 4 * einsum("ip,pia->ap", a.x4, tau813)
        - einsum("poi,oia->ap", tau815, h.l.pov)
        - einsum("poi,oia->ap", tau816, h.l.pov) / 2
        - einsum("qa,pq->ap", tau187, tau546) / 2
        + 4 * einsum("ip,pia->ap", a.x4, tau819)
        - einsum("bp,pab->ap", a.x1, tau416)
        - 2 * einsum("pb,ba->ap", tau203, tau44)
        - 2 * einsum("pi,pia->ap", tau109, tau123)
        + einsum("ip,pia->ap", a.x3, tau821)
        - 2 * einsum("ip,pia->ap", a.x4, tau823)
        + einsum("ip,pia->ap", a.x3, tau827)
        - einsum("bp,pab->ap", a.x1, tau427)
        - 4 * einsum("ia,pi->ap", tau7, tau828)
        - einsum("pq,ip,qia->ap", tau542, a.x3, tau101) / 2
        - 2 * einsum("pq,ip,qia->ap", tau601, a.x4, tau0)
        + 2 * einsum("ip,pia->ap", a.x4, tau829)
        + 2 * einsum("bp,pab->ap", a.x1, tau159)
        - einsum("pi,pia->ap", tau444, tau169) / 2
        - 4 * einsum("ip,pia->ap", a.x4, tau832)
        - einsum("bp,pba->ap", a.x1, tau643)
        - einsum("ip,pia->ap", a.x4, tau835)
        + 2 * einsum("bp,pba->ap", a.x1, tau416)
        + einsum("ip,pia->ap", a.x4, tau836)
        - einsum("bp,pba->ap", a.x1, tau135) / 2
        + einsum("qa,pq->ap", tau187, tau346)
        - einsum("qa,pq->ap", tau48, tau837) / 2
        + einsum("pi,pia->ap", tau109, tau149)
        + einsum("poi,oia->ap", tau838, h.l.pov)
        - 2 * einsum("pob,oba->ap", tau839, h.l.pvv)
        - 2 * einsum("poi,oia->ap", tau841, h.l.pov)
        + einsum("pq,ip,qia->ap", tau842, a.x3, tau16)
        + 2 * einsum("ip,pia->ap", a.x4, tau845)
        - 2 * einsum("poi,oia->ap", tau846, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x4, tau848)
        + 2 * einsum("ip,pia->ap", a.x3, tau852)
        + 2 * einsum("pi,pia->ap", tau573, tau123)
        - einsum("poi,oia->ap", tau853, h.l.pov) / 2
        - 2 * einsum("qp,ip,qia->ap", tau803, a.x3, tau63)
        + einsum("poi,oia->ap", tau857, h.l.pov)
        - 2 * einsum("poi,oia->ap", tau859, h.l.pov)
        - einsum("poi,oia->ap", tau862, h.l.pov) / 2
        - einsum("qa,pq->ap", tau184, tau837) / 2
        - 2 * einsum("qp,ip,qia->ap", tau805, a.x4, tau22)
        - 2 * einsum("pq,ip,qia->ap", tau106, a.x3, tau87)
        + einsum("pq,ip,qia->ap", tau803, a.x4, tau136)
        - einsum("ip,pia->ap", a.x3, tau864) / 2
        - 2 * einsum("pq,ip,qia->ap", tau333, a.x4, tau264)
        - 2 * einsum("ip,pia->ap", a.x3, tau866)
        + 2 * einsum("bp,pba->ap", a.x1, tau409)
        + 2 * einsum("ip,pia->ap", a.x3, tau868)
        - 2 * einsum("pq,ip,qia->ap", tau869, a.x4, tau34)
        + einsum("pq,qa->ap", tau188, tau345)
        + 4 * einsum("ip,pia->ap", a.x4, tau872)
        - einsum("pq,ip,qia->ap", tau183, a.x3, tau149) / 2
        - 2 * einsum("pi,pia->ap", tau418, tau123)
        + einsum("pi,pia->ap", tau297, tau123)
        + einsum("ip,pia->ap", a.x3, tau875)
        + einsum("pi,pia->ap", tau320, tau169)
        + einsum("qp,ip,qia->ap", tau876, a.x3, tau201)
        - einsum("pi,pia->ap", tau544, tau149)
        - 2 * einsum("pob,oba->ap", tau877, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau880)
        + einsum("poi,oia->ap", tau883, h.l.pov)
        + einsum("pob,oba->ap", tau884, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau888)
        + einsum("ip,pia->ap", a.x4, tau892)
        - 2 * einsum("poi,oia->ap", tau893, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau897)
        + 2 * einsum("poi,oia->ap", tau898, h.l.pov)
        + einsum("pi,pia->ap", tau273, tau0)
        - einsum("ip,pia->ap", a.x3, tau902)
        + einsum("ia,pi->ap", h.f.ov, tau903)
        - 2 * einsum("ip,pia->ap", a.x3, tau904)
        + 2 * einsum("ip,pia->ap", a.x4, tau907)
        + 2 * einsum("ia,pi->ap", tau7, tau908)
        + einsum("ip,pia->ap", a.x3, tau910)
        + 2 * einsum("bp,pab->ap", a.x1, tau217)
        - einsum("ip,pia->ap", a.x3, tau913)
        + 2 * einsum("ip,pia->ap", a.x3, tau916)
        - 2 * einsum("ip,pia->ap", a.x4, tau918)
        - einsum("qa,pq->ap", tau187, tau391) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau919)
        + 2 * einsum("ba,pb->ap", h.f.vv, tau124)
        + einsum("poi,oia->ap", tau922, h.l.pov)
        - einsum("poi,oia->ap", tau923, h.l.pov) / 2
        + 2 * einsum("ia,pi->ap", tau119, tau924)
        - 4 * einsum("ip,pia->ap", a.x3, tau925)
        - 2 * einsum("ip,pia->ap", a.x4, tau927)
        - 2 * einsum("ip,pia->ap", a.x4, tau928)
        + 2 * einsum("poi,oia->ap", tau929, h.l.pov)
        - 2 * einsum("pob,oba->ap", tau930, h.l.pvv)
        - einsum("ip,pia->ap", a.x3, tau933)
        + einsum("ip,pia->ap", a.x4, tau935)
        + einsum("poi,oia->ap", tau936, h.l.pov)
        + einsum("qa,pq->ap", tau124, tau188)
        + einsum("qa,pq->ap", tau186, tau546)
        + 2 * einsum("ip,pia->ap", a.x3, tau938)
        - 2 * einsum("pob,oba->ap", tau939, h.l.pvv)
        - einsum("poi,oia->ap", tau940, h.l.pov) / 2
        + einsum("ip,pia->ap", a.x3, tau943)
        + einsum("pq,ip,qia->ap", tau333, a.x4, tau101)
        + 2 * einsum("poi,oia->ap", tau944, h.l.pov)
        + einsum("ip,pia->ap", a.x3, tau946)
        + einsum("ip,pia->ap", a.x3, tau948)
        + 4 * einsum("ip,pia->ap", a.x4, tau951)
        - 2 * einsum("ip,pia->ap", a.x4, tau953)
        + einsum("pq,ip,qia->ap", tau876, a.x3, tau47)
        - einsum("poi,oia->ap", tau954, h.l.pov)
        + einsum("ip,pia->ap", a.x3, tau956)
        + einsum("ia,pi->ap", h.f.ov, tau908)
        - 2 * einsum("ia,pi->ap", h.f.ov, tau924)
        + 2 * einsum("bp,pab->ap", a.x1, tau643)
        + einsum("pi,pia->ap", tau613, tau101)
        - 2 * einsum("pi,pia->ap", tau68, tau87)
        - einsum("ip,pia->ap", a.x4, tau958) / 2
        + einsum("qa,pq->ap", tau48, tau793)
        - einsum("ip,pia->ap", a.x3, tau960) / 2
        + einsum("ip,pia->ap", a.x4, tau961)
        + einsum("qa,pq->ap", tau23, tau837)
        - 2 * einsum("ip,pia->ap", a.x3, tau963)
        + einsum("ip,pia->ap", a.x3, tau964)
        - 2 * einsum("ip,pia->ap", a.x4, tau966)
        - 2 * einsum("ip,pia->ap", a.x4, tau968)
        - einsum("bp,pba->ap", a.x1, tau159)
        + einsum("ip,pia->ap", a.x3, tau971)
        + 2 * einsum("poi,oia->ap", tau972, h.l.pov)
        + einsum("pi,pia->ap", tau342, tau169)
        - 2 * einsum("pq,ip,qia->ap", tau106, a.x3, tau264)
        + 2 * einsum("ip,pia->ap", a.x4, tau975)
        - 2 * einsum("pq,qa->ap", tau245, tau345)
        + 2 * einsum("pi,pia->ap", tau242, tau149)
        + einsum("pob,oba->ap", tau976, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau978)
        + einsum("pob,oba->ap", tau979, h.l.pvv)
        - einsum("ip,pia->ap", a.x3, tau981) / 2
        - einsum("pq,ip,qia->ap", tau183, a.x3, tau222) / 2
        + einsum("pq,ip,qia->ap", tau231, a.x4, tau222)
        - 2 * einsum("ip,pia->ap", a.x4, tau984)
        - 2 * einsum("ip,pia->ap", a.x4, tau985)
        - einsum("poi,oia->ap", tau986, h.l.pov)
        - einsum("ip,pia->ap", a.x3, tau989)
        - einsum("poi,oia->ap", tau990, h.l.pov) / 2
        - 2 * einsum("ip,pia->ap", a.x4, tau993)
        - einsum("pq,ip,qia->ap", tau842, a.x3, tau201) / 2
        - einsum("ip,pia->ap", a.x3, tau994)
        - einsum("poi,oia->ap", tau995, h.l.pov) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau998)
        + einsum("ip,pia->ap", a.x4, tau1000)
        + 4 * einsum("ip,pia->ap", a.x4, tau1001)
        - 2 * einsum("qa,pq->ap", tau103, tau885)
        + 2 * einsum("ip,pia->ap", a.x4, tau1002)
        - 2 * einsum("ip,pia->ap", a.x3, tau1003)
        - 2 * einsum("ip,pia->ap", a.x3, tau1004)
        + einsum("ip,pia->ap", a.x3, tau1006)
        + 2 * einsum("poi,oia->ap", tau1007, h.l.pov)
        + einsum("pq,ip,qia->ap", tau106, a.x3, tau101)
        - einsum("qa,pq->ap", tau116, tau889) / 2
        + einsum("ip,pia->ap", a.x3, tau1009)
        + 2 * einsum("pi,pia->ap", tau387, tau149)
        + einsum("ip,pia->ap", a.x3, tau1010)
        + einsum("ip,pia->ap", a.x4, tau1012)
        + einsum("ip,pia->ap", a.x3, tau1013)
        - einsum("pi,pia->ap", tau628, tau101)
        - einsum("pq,ip,qia->ap", tau802, a.x4, tau16) / 2
        - einsum("qa,pq->ap", tau116, tau894) / 2
        - einsum("ip,pia->ap", a.x3, tau1014)
        + einsum("pq,ip,qia->ap", tau231, a.x4, tau149)
        - einsum("bp,pab->ap", a.x1, tau518)
        - 4 * einsum("pi,pia->ap", tau511, tau87)
        + einsum("ip,pia->ap", a.x3, tau1015)
        - einsum("pi,pia->ap", tau3, tau169) / 2
        - einsum("ia,pi->ap", tau119, tau903)
        - 4 * einsum("pi,pia->ap", tau242, tau123)
        - 2 * einsum("pi,pia->ap", tau273, tau264)
        - einsum("ip,pia->ap", a.x3, tau1017) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau1018)
        + einsum("pq,ip,qia->ap", tau542, a.x3, tau87)
        - 4 * einsum("pi,pia->ap", tau387, tau123)
        + 4 * einsum("ip,pia->ap", a.x4, tau1019)
        + 2 * einsum("ip,pia->ap", a.x4, tau1022)
        + 2 * einsum("poi,oia->ap", tau1024, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau1026)
        - einsum("pq,ip,qia->ap", tau231, a.x4, tau123) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau1028)
        - 2 * einsum("poi,oia->ap", tau1029, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x4, tau1030)
        - einsum("pi,pia->ap", tau295, tau149) / 2
        - einsum("poi,oia->ap", tau1031, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x4, tau1034)
        + einsum("pob,oba->ap", tau1035, h.l.pvv)
        + einsum("qa,pq->ap", tau116, tau885)
        + einsum("bp,pab->ap", a.x1, tau576)
        + 2 * einsum("ip,pia->ap", a.x3, tau1037)
        + einsum("ip,pia->ap", a.x3, tau1040)
        - einsum("ip,pia->ap", a.x3, tau1041) / 2
        + einsum("pi,pia->ap", tau68, tau101)
        - 2 * einsum("poi,oia->ap", tau1043, h.l.pov)
        + 2 * einsum("poi,oia->ap", tau1044, h.l.pov)
        + einsum("qp,ip,qia->ap", tau805, a.x4, tau34)
        - 2 * einsum("ip,pia->ap", a.x3, tau1047)
        - einsum("pob,oba->ap", tau1049, h.l.pvv)
        + 2 * einsum("pi,pia->ap", tau43, tau87)
        + einsum("ip,pia->ap", a.x3, tau1051)
        - 2 * einsum("pq,ip,qia->ap", tau803, a.x4, tau47)
        + einsum("ip,pia->ap", a.x3, tau1052)
        + 4 * einsum("pq,ip,qia->ap", tau601, a.x4, tau87)
        - 2 * einsum("pi,pia->ap", tau139, tau65)
        - 4 * einsum("ip,pia->ap", a.x4, tau1055)
        + einsum("ip,pia->ap", a.x4, tau1058)
        - 2 * einsum("ip,pia->ap", a.x3, tau1061)
        + 2 * einsum("ip,pia->ap", a.x4, tau1063)
        - einsum("ip,pia->ap", a.x3, tau1065) / 2
        - einsum("pq,ip,qia->ap", tau876, a.x3, tau136) / 2
        - 2 * einsum("ip,pia->ap", a.x4, tau1066)
        - 4 * einsum("ip,pia->ap", a.x4, tau1067)
        + 2 * einsum("ip,pia->ap", a.x3, tau1068)
        - einsum("qa,pq->ap", tau102, tau188) / 2
        + einsum("ip,pia->ap", a.x3, tau1070)
        - 2 * einsum("qa,pq->ap", tau103, tau824)
        - 2 * einsum("ip,pia->ap", a.x4, tau1072)
        - einsum("ip,pia->ap", a.x3, tau1073)
        + einsum("pi,pia->ap", tau530, tau169)
        - einsum("ip,pia->ap", a.x3, tau1074) / 2
        - einsum("ip,pia->ap", a.x3, tau1075)
        + 4 * einsum("ip,pia->ap", a.x4, tau1076)
        + 2 * einsum("pob,oba->ap", tau1077, h.l.pvv)
        + 2 * einsum("pi,pia->ap", tau511, tau101)
        - 4 * einsum("ip,pia->ap", a.x4, tau1078)
        + einsum("ip,pia->ap", a.x3, tau1079)
        - 2 * einsum("ip,pia->ap", a.x3, tau1080)
        - einsum("ip,pia->ap", a.x3, tau1083)
        - einsum("poi,oia->ap", tau1084, h.l.pov)
        - 2 * einsum("pob,oba->ap", tau1085, h.l.pvv)
        + einsum("ip,pia->ap", a.x4, tau1086)
        + 2 * einsum("ip,pia->ap", a.x3, tau1087)
        + 2 * einsum("ip,pia->ap", a.x3, tau1088)
        + 4 * einsum("qp,ip,qia->ap", tau869, a.x4, tau63)
        - 2 * einsum("ip,pia->ap", a.x3, tau1090)
        + einsum("pi,pia->ap", tau397, tau222)
        + einsum("ip,pia->ap", a.x3, tau1092)
        + einsum("poi,oia->ap", tau1093, h.l.pov)
        + einsum("qp,ip,qia->ap", tau842, a.x3, tau136)
        + 2 * einsum("ip,pia->ap", a.x3, tau1096)
        + einsum("pi,pia->ap", tau161, tau65)
        + einsum("ip,pia->ap", a.x4, tau1097)
        - einsum("bp,pab->ap", a.x1, tau516) / 2
        - einsum("ip,pia->ap", a.x3, tau1099) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau1100)
        + einsum("ip,pia->ap", a.x3, tau1102)
        - 2 * einsum("ip,pia->ap", a.x4, tau1104)
        + 2 * einsum("pi,pia->ap", tau544, tau123)
        + 2 * einsum("poi,oia->ap", tau1105, h.l.pov)
        + einsum("pi,pia->ap", tau15, tau87)
        - 2 * einsum("pi,pia->ap", tau458, tau65)
        + einsum("ip,pia->ap", a.x3, tau1106)
        - einsum("ip,pia->ap", a.x3, tau1109) / 2
        - 2 * einsum("poi,oia->ap", tau1110, h.l.pov)
        - 2 * einsum("pq,ip,qia->ap", tau310, a.x4, tau149)
        + einsum("pi,pia->ap", tau37, tau65)
        + einsum("pob,oba->ap", tau1111, h.l.pvv)
        + 2 * einsum("ip,pia->ap", a.x4, tau1112)
        + einsum("poi,oia->ap", tau1113, h.l.pov)
        - 2 * einsum("pq,qa->ap", tau125, tau186)
        + einsum("ip,pia->ap", a.x3, tau1114)
        + einsum("qa,pq->ap", tau103, tau889)
        + einsum("ip,pia->ap", a.x3, tau1116)
        - einsum("pi,pia->ap", tau172, tau101) / 2
        + einsum("ip,pia->ap", a.x4, tau1118)
        + einsum("ip,pia->ap", a.x4, tau1119)
        + einsum("ip,pia->ap", a.x3, tau1121)
        + einsum("qp,ip,qia->ap", tau803, a.x3, tau29)
        + 4 * einsum("pb,ba->ap", tau345, tau44)
        - 4 * einsum("ia,pi->ap", tau7, tau924)
        - einsum("ip,pia->ap", a.x3, tau1122)
        - einsum("pi,pia->ap", tau573, tau149)
        + einsum("ip,pia->ap", a.x3, tau1123)
        - einsum("pob,oba->ap", tau1124, h.l.pvv)
        + 4 * einsum("ip,pia->ap", a.x4, tau1126)
        - einsum("poi,oia->ap", tau1127, h.l.pov) / 2
        + 2 * einsum("bp,pab->ap", a.x1, tau535)
        + 2 * einsum("ip,pia->ap", a.x4, tau1130)
        - einsum("ip,pia->ap", a.x4, tau1131)
        + einsum("pob,oba->ap", tau1132, h.l.pvv)
        + einsum("pob,oba->ap", tau1133, h.l.pvv)
        - einsum("ip,pia->ap", a.x4, tau1134)
        + einsum("poi,oia->ap", tau1135, h.l.pov)
        + einsum("qa,pq->ap", tau103, tau894)
        - 2 * einsum("ip,pia->ap", a.x4, tau1137)
        - 2 * einsum("qp,ip,qia->ap", tau803, a.x3, tau22)
        - 2 * einsum("ip,pia->ap", a.x3, tau1138)
        + einsum("poi,oia->ap", tau1139, h.l.pov)
        + einsum("bp,pba->ap", a.x1, tau390)
        - 2 * einsum("ip,pia->ap", a.x3, tau1140)
        + einsum("ip,pia->ap", a.x4, tau1142)
        - 2 * einsum("ip,pia->ap", a.x4, tau1143)
        + einsum("pob,oba->ap", tau1144, h.l.pvv)
        - einsum("bp,pba->ap", a.x1, tau576) / 2
        + einsum("poi,oia->ap", tau1145, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau1147)
        - einsum("ip,pia->ap", a.x3, tau1149)
        - 2 * einsum("ip,pia->ap", a.x3, tau1150)
        + einsum("pi,pia->ap", tau152, tau65)
        - einsum("ip,pia->ap", a.x4, tau1151)
        - 2 * einsum("ip,pia->ap", a.x4, tau1152)
        + einsum("qa,pq->ap", tau203, tau245)
        + einsum("qa,pq->ap", tau184, tau793)
        + einsum("ip,pia->ap", a.x3, tau1155)
        - einsum("ip,pia->ap", a.x3, tau1158)
        + einsum("qa,pq->ap", tau55, tau837)
        + einsum("ip,pia->ap", a.x3, tau1161)
        + 2 * einsum("ia,pi->ap", tau119, tau828)
        + 2 * einsum("bp,pba->ap", a.x1, tau427)
        + 2 * einsum("ip,pia->ap", a.x4, tau1162)
        - 2 * einsum("ip,pia->ap", a.x4, tau1164)
        - einsum("ip,pia->ap", a.x3, tau1165)
        - einsum("pi,pia->ap", tau15, tau101) / 2
        + 2 * einsum("bp,pba->ap", a.x1, tau518)
        - 2 * einsum("pob,oba->ap", tau1166, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau1167)
        - einsum("ip,pia->ap", a.x4, tau1168) / 2
        - einsum("pq,ip,qia->ap", tau221, a.x3, tau123) / 2
        + einsum("poi,oia->ap", tau1169, h.l.pov)
        + einsum("pq,ip,qia->ap", tau805, a.x4, tau29)
        - 2 * einsum("pob,oba->ap", tau1170, h.l.pvv)
        - einsum("ip,pia->ap", a.x3, tau1172) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau1173)
        - 2 * einsum("qp,ip,qia->ap", tau869, a.x4, tau29)
        + einsum("ip,pia->ap", a.x3, tau1175)
        - 4 * einsum("ip,pia->ap", a.x4, tau1176)
        + 2 * einsum("pi,pia->ap", tau628, tau87)
        - 2 * einsum("ip,pia->ap", a.x3, tau1177)
        - einsum("bp,pba->ap", a.x1, tau217)
        + 4 * einsum("ip,pia->ap", a.x4, tau1179)
        - einsum("ip,pia->ap", a.x4, tau1180)
        - 2 * einsum("ia,pi->ap", h.f.ov, tau828)
        + einsum("poi,oia->ap", tau1181, h.l.pov)
        + 4 * einsum("pq,ip,qia->ap", tau869, a.x4, tau22)
        + 2 * einsum("ip,pia->ap", a.x4, tau1182)
        + einsum("pi,pia->ap", tau295, tau123)
        + einsum("ip,pia->ap", a.x4, tau1184)
        - 2 * einsum("ip,pia->ap", a.x4, tau1185)
        - einsum("qp,ip,qia->ap", tau876, a.x3, tau16) / 2
        + einsum("poi,oia->ap", tau1186, h.l.pov)
        - 2 * einsum("pob,oba->ap", tau1187, h.l.pvv)
        + einsum("ip,pia->ap", a.x4, tau1188)
        - 2 * einsum("ip,pia->ap", a.x4, tau1190)
        + einsum("ip,pia->ap", a.x3, tau1191)
        - einsum("poi,oia->ap", tau1192, h.l.pov)
        + einsum("ip,pia->ap", a.x3, tau1193)
        - 2 * einsum("pq,ip,qia->ap", tau333, a.x4, tau87)
        - einsum("poi,oia->ap", tau1194, h.l.pov) / 2
        + einsum("poi,oia->ap", tau1195, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau1197)
        - 2 * einsum("ip,pia->ap", a.x3, tau1198)
        - einsum("pi,pia->ap", tau495, tau169) / 2
        - einsum("pq,ip,qia->ap", tau221, a.x3, tau145) / 2
        + 4 * einsum("ip,pia->ap", a.x4, tau1199)
        - einsum("poi,oia->ap", tau1200, h.l.pov) / 2
        - 2 * einsum("pb,ba->ap", tau102, tau44)
        - 2 * einsum("ip,pia->ap", a.x3, tau1201)
        - 2 * einsum("ip,pia->ap", a.x4, tau1202)
        + einsum("poi,oia->ap", tau1203, h.l.pov)
        - 4 * einsum("pi,pia->ap", tau442, tau87)
        - 2 * einsum("ip,pia->ap", a.x4, tau1205)
        + einsum("ip,pia->ap", a.x4, tau1206)
        + 4 * einsum("ip,pia->ap", a.x4, tau1208)
        + 2 * einsum("ip,pia->ap", a.x3, tau1209)
        - 2 * einsum("ip,pia->ap", a.x3, tau1210)
        - einsum("ip,pia->ap", a.x3, tau1212) / 2
        + einsum("pq,ip,qia->ap", tau106, a.x3, tau0)
        + 2 * einsum("ip,pia->ap", a.x3, tau1213)
        + 2 * einsum("ip,pia->ap", a.x4, tau1214)
        + einsum("ip,pia->ap", a.x3, tau1216)
        - 2 * einsum("ip,pia->ap", a.x3, tau1218)
        - 2 * einsum("pq,ip,qia->ap", tau803, a.x4, tau201)
        - 2 * einsum("ip,pia->ap", a.x4, tau1220)
        + 4 * einsum("ip,pia->ap", a.x4, tau1222)
        + einsum("ip,pia->ap", a.x4, tau1223)
        + 2 * einsum("poi,oia->ap", tau1224, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x4, tau1225)
        - 4 * einsum("ip,pia->ap", a.x4, tau1228)
        - 2 * einsum("poi,oia->ap", tau1229, h.l.pov)
        - 2 * einsum("qa,pq->ap", tau124, tau245)
        + 2 * einsum("ba,pb->ap", h.f.vv, tau345)
        - einsum("ip,pia->ap", a.x4, tau1230)
        - einsum("ip,pia->ap", a.x3, tau1231) / 2
        + einsum("pq,ip,qia->ap", tau221, a.x3, tau222)
        - einsum("ip,pia->ap", a.x4, tau1232)
        + 4 * einsum("ip,pia->ap", a.x4, tau1235)
        - 2 * einsum("ip,pia->ap", a.x4, tau1236)
        + einsum("pi,pia->ap", tau418, tau149)
        + einsum("ip,pia->ap", a.x4, tau1237)
        + einsum("pq,ip,qia->ap", tau183, a.x3, tau123)
        - 2 * einsum("ip,pia->ap", a.x3, tau1238)
        - 2 * einsum("ip,pia->ap", a.x4, tau1239)
        + einsum("pi,pia->ap", tau172, tau87)
        + einsum("ip,pia->ap", a.x4, tau1241)
        + einsum("ip,pia->ap", a.x4, tau1242)
        + einsum("poi,oia->ap", tau1243, h.l.pov)
        + einsum("pi,pia->ap", tau148, tau169)
        + einsum("ip,pia->ap", a.x3, tau1244)
        - einsum("poi,oia->ap", tau1245, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau1246)
        + einsum("bp,pba->ap", a.x1, tau516)
        + einsum("pq,ip,qia->ap", tau803, a.x4, tau16)
        + einsum("poi,oia->ap", tau1247, h.l.pov)
        + 2 * einsum("poi,oia->ap", tau1248, h.l.pov)
        + einsum("ip,pia->ap", a.x4, tau1249)
        - 2 * einsum("pi,pia->ap", tau155, tau65)
        - einsum("pi,pia->ap", tau484, tau169) / 2
        - einsum("bp,pab->ap", a.x1, tau390) / 2
        + 2 * einsum("pob,oba->ap", tau1250, h.l.pvv)
        - 2 * einsum("ip,pia->ap", a.x4, tau1251)
        - 2 * einsum("pi,pia->ap", tau613, tau87)
        - 2 * einsum("ip,pia->ap", a.x4, tau1253)
        + einsum("ip,pia->ap", a.x4, tau1254)
        + einsum("poi,oia->ap", tau1255, h.l.pov)
        - einsum("pq,ip,qia->ap", tau542, a.x3, tau0) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau1256)
        - 4 * einsum("ip,pia->ap", a.x4, tau1257)
        - 2 * einsum("ip,pia->ap", a.x4, tau1259)
        + einsum("pq,ip,qia->ap", tau221, a.x3, tau149)
        - einsum("pq,ip,qia->ap", tau231, a.x4, tau145) / 2
        - 2 * einsum("qa,pq->ap", tau23, tau793)
        - 2 * einsum("pob,oba->ap", tau1260, h.l.pvv)
        + einsum("pq,ip,qia->ap", tau310, a.x4, tau123)
        - 2 * einsum("ip,pia->ap", a.x3, tau1261)
        + einsum("pq,qa->ap", tau125, tau187)
        + 2 * einsum("ip,pia->ap", a.x4, tau1262)
        - einsum("ip,pia->ap", a.x4, tau1263)
        - einsum("bp,pab->ap", a.x1, tau409)
        + einsum("qa,pq->ap", tau102, tau245)
        - 4 * einsum("ip,pia->ap", a.x4, tau1266)
        + einsum("ip,pia->ap", a.x3, tau1267)
        - einsum("ip,pia->ap", a.x4, tau1268) / 2
        - 2 * einsum("ip,pia->ap", a.x3, tau1271)
        + einsum("poi,oia->ap", tau1272, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x3, tau1273)
        + einsum("poi,oia->ap", tau1274, h.l.pov)
        - 2 * einsum("ip,pia->ap", a.x3, tau1276)
        + einsum("pq,ip,qia->ap", tau542, a.x3, tau264)
        + einsum("ip,pia->ap", a.x3, tau1277)
        + 4 * einsum("ip,pia->ap", a.x4, tau1279)
        - einsum("qp,ip,qia->ap", tau842, a.x3, tau47) / 2
        - einsum("pi,pia->ap", tau43, tau101)
        + einsum("poi,oia->ap", tau1281, h.l.pov)
        - einsum("pq,qa->ap", tau188, tau203) / 2
        + einsum("ip,pia->ap", a.x4, tau1282)
        - einsum("poi,oia->ap", tau1283, h.l.pov) / 2
        + einsum("ip,pia->ap", a.x4, tau1285)
        - 2 * einsum("ip,pia->ap", a.x4, tau1286)
        - 2 * einsum("pq,ip,qia->ap", tau601, a.x4, tau101)
        - 2 * einsum("ip,pia->ap", a.x4, tau1288)
        - einsum("poi,oia->ap", tau1289, h.l.pov) / 2
        - einsum("ip,pia->ap", a.x3, tau1291) / 2
        - einsum("ip,pia->ap", a.x3, tau1292) / 2
        - einsum("qp,ip,qia->ap", tau802, a.x3, tau29) / 2
        + einsum("ip,pia->ap", a.x3, tau1293)
        + einsum("pq,ip,qia->ap", tau802, a.x4, tau201)
        + einsum("poi,oia->ap", tau1294, h.l.pov)
        - einsum("ip,pia->ap", a.x3, tau1296) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau1297)
        - 2 * einsum("ip,pia->ap", a.x4, tau1299)
        - 2 * einsum("pq,ip,qia->ap", tau310, a.x4, tau222)
        + 2 * einsum("ia,pi->ap", tau7, tau903)
        - einsum("qp,ip,qia->ap", tau802, a.x3, tau34) / 2
        + 4 * einsum("pq,ip,qia->ap", tau601, a.x4, tau264)
        + einsum("qp,ip,qia->ap", tau802, a.x3, tau22)
        + einsum("poi,oia->ap", tau1300, h.l.pov)
        - einsum("poi,oia->ap", tau1301, h.l.pov)
        - einsum("ba,pb->ap", h.f.vv, tau102)
        + einsum("ip,pia->ap", a.x3, tau1302)
        - einsum("ba,pb->ap", h.f.vv, tau203)
        - 2 * einsum("ip,pia->ap", a.x4, tau1303)
        - 2 * einsum("pi,pia->ap", tau479, tau65)
        - 4 * einsum("ip,pia->ap", a.x4, tau1305)
        - 2 * einsum("ip,pia->ap", a.x3, tau1307)
        + einsum("ip,pia->ap", a.x4, tau1308)
        - einsum("poi,oia->ap", tau1309, h.l.pov) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau1310)
        + einsum("pq,ip,qia->ap", tau333, a.x4, tau0)
        - 4 * einsum("ip,pia->ap", a.x3, tau1311)
        - einsum("ia,pi->ap", tau119, tau908)
        - einsum("ip,pia->ap", a.x3, tau1314)
        - einsum("pq,ip,qia->ap", tau802, a.x4, tau136) / 2
        - einsum("bp,pba->ap", a.x1, tau535)
        + einsum("pq,ip,qia->ap", tau183, a.x3, tau145)
        - einsum("ip,pia->ap", a.x3, tau1315)
        - einsum("ip,pia->ap", a.x3, tau1317) / 2
        + 2 * einsum("ip,pia->ap", a.x4, tau1320)
        - einsum("pi,pia->ap", tau297, tau149) / 2
        + einsum("ip,pia->ap", a.x4, tau1321)
        + einsum("pi,pia->ap", tau455, tau65)
        - einsum("ip,pia->ap", a.x3, tau1322)
        + 2 * einsum("ip,pia->ap", a.x4, tau1323)
        + einsum("pob,oba->ap", tau1324, h.l.pvv)
        - 2 * einsum("pi,pia->ap", tau397, tau145)
        + einsum("ip,pia->ap", a.x3, tau1325)
        - einsum("poi,oia->ap", tau1326, h.l.pov)
        + 2 * einsum("ip,pia->ap", a.x3, tau1327)
        + einsum("qa,pq->ap", tau116, tau824)
        - 4 * einsum("ip,pia->ap", a.x3, tau1328)
        - einsum("poi,oia->ap", tau1329, h.l.pov) / 2
        + 2 * einsum("ip,pia->ap", a.x3, tau1330)
        + 2 * einsum("ip,pia->ap", a.x4, tau1331)
        + einsum("ip,pia->ap", a.x3, tau1334)
        + 2 * einsum("ip,pia->ap", a.x3, tau1335)
    )

    rx3 = (
        einsum("pq,qi->ip", tau108, tau35)
        - einsum("qi,pq->ip", tau13, tau36) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau141)
        - einsum("ap,pia->ip", a.x2, tau1212) / 2
        + einsum("ap,pia->ip", a.x2, tau1302)
        - einsum("ap,pia->ip", a.x2, tau1014)
        + einsum("qi,pq->ip", tau137, tau171)
        - 2 * einsum("ap,pia->ip", a.x1, tau33)
        - 2 * einsum("ap,pia->ip", a.x2, tau1336)
        + einsum("ap,pia->ip", a.x2, tau1006)
        + einsum("ap,pia->ip", a.x2, tau1161)
        - einsum("poj,oji->ip", tau1337, tau40)
        + einsum("pa,pia->ip", tau1056, tau5)
        - einsum("poj,oji->ip", tau1338, tau79)
        + einsum("ap,pia->ip", a.x2, tau1339)
        - 4 * einsum("ap,pia->ip", a.x1, tau1340)
        + einsum("pa,pia->ip", tau404, tau222)
        - einsum("pq,jp,qji->ip", tau388, a.x4, tau367) / 2
        + einsum("pq,jp,qji->ip", tau450, a.x4, tau132)
        - 4 * einsum("ap,pia->ip", a.x2, tau1311)
        - 2 * einsum("ap,pia->ip", a.x2, tau1341)
        - 2 * einsum("ap,pia->ip", a.x1, tau465)
        - 4 * einsum("ap,pia->ip", a.x1, tau1342)
        - 2 * einsum("ap,pia->ip", a.x1, tau681)
        + einsum("ap,pia->ip", a.x1, tau784)
        + einsum("pqo,qoi->ip", tau1343, tau95)
        - einsum("qi,pq->ip", tau1, tau171) / 2
        + einsum("jp,pji->ip", a.x4, tau1346)
        + einsum("pq,qpi->ip", tau842, tau17)
        + 2 * einsum("pj,ji->ip", tau153, tau42)
        - 2 * einsum("ap,pia->ip", a.x2, tau1061)
        - 2 * einsum("pqo,qoi->ip", tau1347, tau89)
        + 2 * einsum("ap,pia->ip", a.x2, tau1209)
        + einsum("pqo,qoi->ip", tau1348, tau19)
        - 2 * einsum("qp,qpi->ip", tau803, tau196)
        + 2 * einsum("pj,ij->ip", tau150, tau386)
        - einsum("ap,pia->ip", a.x2, tau989)
        - einsum("pq,jp,qji->ip", tau133, a.x4, tau212) / 2
        + einsum("ap,pia->ip", a.x2, tau1116)
        - einsum("pq,jp,qji->ip", tau368, a.x4, tau132) / 2
        - 2 * einsum("jp,pji->ip", a.x4, tau1349)
        + einsum("qi,pq->ip", tau453, tau612)
        - 2 * einsum("poj,oji->ip", tau1350, tau40)
        + einsum("pq,qpi->ip", tau542, tau1351)
        - 4 * einsum("ap,pia->ip", a.x2, tau925)
        - 2 * einsum("ap,pia->ip", a.x2, tau1261)
        - einsum("jp,pij->ip", a.x4, tau1355)
        + 2 * einsum("ap,pia->ip", a.x1, tau206)
        + einsum("pq,qi->ip", tau171, tau318)
        - 2 * einsum("pqo,qoi->ip", tau1356, tau95)
        - einsum("ap,pia->ip", a.x2, tau1357)
        + einsum("poj,oji->ip", tau1358, tau40)
        - 2 * einsum("ap,pia->ip", a.x1, tau61)
        + 2 * einsum("ap,pia->ip", a.x2, tau1087)
        - 2 * einsum("ap,pia->ip", a.x1, tau239)
        - 2 * einsum("ap,pia->ip", a.x1, tau734)
        + einsum("ap,pia->ip", a.x2, tau1175)
        - 2 * einsum("qi,pq->ip", tau137, tau612)
        - einsum("pq,qpi->ip", tau183, tau504) / 2
        + einsum("ap,pia->ip", a.x2, tau796)
        - 2 * einsum("pa,pia->ip", tau246, tau123)
        - 4 * einsum("ap,pia->ip", a.x2, tau1328)
        - einsum("ap,pia->ip", a.x2, tau1359) / 2
        - 2 * einsum("ij,pj->ip", h.f.oo, tau153)
        + einsum("pq,qpi->ip", tau98, tau881)
        + 2 * einsum("poj,oij->ip", tau1360, h.l.poo)
        + einsum("qp,pqi->ip", tau542, tau1361)
        - 2 * einsum("ap,pia->ip", a.x1, tau396)
        - 2 * einsum("pq,qi->ip", tau108, tau146)
        + 2 * einsum("poj,oij->ip", tau1362, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau325)
        + 2 * einsum("ap,pia->ip", a.x2, tau852)
        + einsum("ap,pia->ip", a.x2, tau1015)
        + einsum("pa,pia->ip", tau246, tau222)
        + einsum("pqo,qoi->ip", tau1363, tau89)
        + einsum("pa,pia->ip", tau1107, tau5)
        + einsum("qp,qpi->ip", tau92, tau1364)
        + einsum("poj,oij->ip", tau1365, h.l.poo)
        - 4 * einsum("ap,pia->ip", a.x1, tau776)
        + 4 * einsum("ap,pia->ip", a.x2, tau1366)
        + einsum("qp,qpi->ip", tau803, tau1367)
        - 4 * einsum("ap,pia->ip", a.x1, tau434)
        - einsum("ap,pia->ip", a.x2, tau1368)
        + einsum("ap,pia->ip", a.x2, tau1267)
        + 4 * einsum("ap,pia->ip", a.x1, tau738)
        - 2 * einsum("ap,pia->ip", a.x2, tau919)
        + einsum("ap,pia->ip", a.x2, tau1106)
        - einsum("qp,qpi->ip", tau92, tau1369) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1138)
        - 2 * einsum("qi,pq->ip", tau318, tau612)
        - 2 * einsum("ap,pia->ip", a.x1, tau286)
        - einsum("pa,pia->ip", tau798, tau29) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau1327)
        - einsum("pq,qpi->ip", tau221, tau322) / 2
        + 2 * einsum("poj,oji->ip", tau1370, tau40)
        - 2 * einsum("ap,pia->ip", a.x2, tau1028)
        - 2 * einsum("pa,pia->ip", tau404, tau123)
        + 2 * einsum("ap,pia->ip", a.x1, tau280)
        + 2 * einsum("ap,pia->ip", a.x1, tau168)
        - einsum("poj,oij->ip", tau1371, h.l.poo)
        + 2 * einsum("ap,pia->ip", a.x2, tau916)
        - einsum("qi,pq->ip", tau13, tau151) / 2
        + 4 * einsum("pa,pia->ip", tau45, tau145)
        - 2 * einsum("ap,pia->ip", a.x2, tau1003)
        - 4 * einsum("ap,pia->ip", a.x1, tau54)
        + einsum("pqo,qoi->ip", tau1372, tau89)
        - einsum("qp,qpi->ip", tau876, tau17) / 2
        + einsum("ap,pia->ip", a.x2, tau875)
        + einsum("ap,pia->ip", a.x2, tau1052)
        - 2 * einsum("ap,pia->ip", a.x2, tau866)
        - 4 * einsum("ap,pia->ip", a.x1, tau722)
        + einsum("ap,pia->ip", a.x1, tau684)
        + einsum("qp,qpi->ip", tau98, tau1373)
        - 4 * einsum("ap,pia->ip", a.x1, tau257)
        - einsum("qi,pq->ip", tau170, tau443) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1177)
        - einsum("poj,oji->ip", tau1374, tau40)
        + 4 * einsum("ap,pia->ip", a.x1, tau723)
        - 2 * einsum("qp,pqi->ip", tau601, tau1375)
        - einsum("pq,qpi->ip", tau542, tau1376) / 2
        + 4 * einsum("ap,pia->ip", a.x1, tau604)
        - 2 * einsum("ap,pia->ip", a.x2, tau1210)
        + einsum("qp,pqi->ip", tau542, tau301)
        - 2 * einsum("poj,oij->ip", tau1377, h.l.poo)
        + einsum("qp,qpi->ip", tau92, tau860)
        + einsum("pa,pia->ip", tau501, tau299)
        + einsum("ap,pia->ip", a.x2, tau1051)
        + einsum("ap,pia->ip", a.x1, tau1378)
        - einsum("ap,pia->ip", a.x2, tau1109) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1307)
        + einsum("ap,pia->ip", a.x2, tau1379)
        - 2 * einsum("ap,pia->ip", a.x2, tau1218)
        - einsum("pa,pia->ip", tau99, tau149)
        + 2 * einsum("poj,oji->ip", tau1360, tau79)
        - 2 * einsum("jp,pji->ip", a.x4, tau1382)
        + einsum("pa,pia->ip", tau471, tau123)
        + 2 * einsum("ap,pia->ip", a.x2, tau1037)
        - 2 * einsum("ap,pia->ip", a.x2, tau1047)
        - 2 * einsum("ap,pia->ip", a.x1, tau178)
        - einsum("ap,pia->ip", a.x1, tau627) / 2
        + einsum("pa,pia->ip", tau865, tau29)
        - einsum("pqo,qoi->ip", tau1383, tau89) / 2
        + einsum("ap,pia->ip", a.x2, tau1216)
        - einsum("pq,jp,qij->ip", tau133, a.x4, tau367) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau1335)
        - 2 * einsum("qp,pqi->ip", tau333, tau162)
        - 4 * einsum("pa,pia->ip", tau9, tau123)
        - 2 * einsum("poj,oij->ip", tau1350, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x1, tau421)
        - 2 * einsum("ap,pia->ip", a.x1, tau268)
        - einsum("ap,pia->ip", a.x2, tau1322)
        + einsum("ap,pia->ip", a.x1, tau400)
        - 2 * einsum("qp,qpi->ip", tau208, tau860)
        + 2 * einsum("poj,oji->ip", tau1384, tau79)
        - einsum("pa,pia->ip", tau99, tau222)
        + einsum("pa,pia->ip", tau292, tau222)
        - 2 * einsum("pq,qi->ip", tau108, tau153)
        + einsum("pa,pia->ip", tau926, tau34)
        + einsum("pq,qi->ip", tau147, tau170)
        - 2 * einsum("poj,oji->ip", tau1385, tau40)
        + 2 * einsum("ap,pia->ip", a.x1, tau665)
        + 2 * einsum("poj,oji->ip", tau1386, tau40)
        - einsum("ap,pia->ip", a.x2, tau1165)
        - 4 * einsum("pj,ij->ip", tau146, tau76)
        + einsum("qi,pq->ip", tau146, tau294)
        - 2 * einsum("ap,pia->ip", a.x1, tau456)
        - 2 * einsum("pq,qpi->ip", tau106, tau436)
        + einsum("pq,qpi->ip", tau221, tau504)
        - einsum("pqo,qoi->ip", tau1387, tau19) / 2
        + einsum("ap,pia->ip", a.x2, tau1193)
        - 2 * einsum("pq,qpi->ip", tau98, tau1388)
        - 2 * einsum("pq,qpi->ip", tau371, tau1373)
        - einsum("poj,oij->ip", tau1389, h.l.poo)
        - einsum("ap,pia->ip", a.x2, tau1314)
        + 2 * einsum("ap,pia->ip", a.x1, tau235)
        - 2 * einsum("pj,ij->ip", tau153, tau272)
        - 2 * einsum("pa,pia->ip", tau210, tau145)
        - 2 * einsum("pa,pia->ip", tau886, tau5)
        + einsum("ap,pia->ip", a.x2, tau1013)
        + einsum("qp,pqi->ip", tau333, tau236)
        - einsum("pqo,qoi->ip", tau1390, tau19) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1090)
        + einsum("pa,pia->ip", tau942, tau29)
        - 2 * einsum("ap,pia->ip", a.x1, tau165)
        + einsum("pqo,qoi->ip", tau1391, tau19)
        - 2 * einsum("ap,pia->ip", a.x2, tau1271)
        + einsum("ap,pia->ip", a.x1, tau528)
        + einsum("poj,oji->ip", tau1392, tau40)
        - 2 * einsum("pa,pia->ip", tau292, tau123)
        + einsum("pq,qi->ip", tau151, tau66)
        + 2 * einsum("poj,oij->ip", tau1393, h.l.poo)
        + einsum("ij,pj->ip", tau272, tau35)
        - einsum("pa,pia->ip", tau701, tau299) / 2
        + 2 * einsum("poj,oji->ip", tau1393, tau79)
        + 2 * einsum("ap,pia->ip", a.x2, tau1096)
        - 2 * einsum("pj,ij->ip", tau146, tau272)
        - einsum("pqo,qoi->ip", tau1394, tau19) / 2
        + einsum("pa,pia->ip", tau890, tau5)
        + 2 * einsum("pa,pia->ip", tau1033, tau63)
        + 2 * einsum("pa,pia->ip", tau867, tau22)
        + 2 * einsum("pj,ji->ip", tau146, tau42)
        + einsum("ap,pia->ip", a.x1, tau571)
        - 2 * einsum("ij,pj->ip", h.f.oo, tau146)
        - 2 * einsum("ap,pia->ip", a.x1, tau199)
        + 2 * einsum("ap,pia->ip", a.x1, tau424)
        + einsum("qp,qpi->ip", tau842, tau1395)
        - 2 * einsum("ap,pia->ip", a.x2, tau1238)
        + 2 * einsum("ap,pia->ip", a.x1, tau1396)
        - 2 * einsum("pa,pia->ip", tau1038, tau5)
        + 2 * einsum("ap,pia->ip", a.x2, tau1088)
        + 2 * einsum("ap,pia->ip", a.x1, tau553)
        - 2 * einsum("qi,pq->ip", tau107, tau341)
        + einsum("ap,pia->ip", a.x2, tau1079)
        - einsum("ap,pia->ip", a.x2, tau1017) / 2
        + 2 * einsum("jp,pji->ip", a.x4, tau1355)
        + 2 * einsum("ap,pia->ip", a.x2, tau1397)
        - einsum("ap,pia->ip", a.x2, tau1172) / 2
        + einsum("pqo,qoi->ip", tau1398, tau89)
        + einsum("ap,pia->ip", a.x2, tau1325)
        + 2 * einsum("pa,pia->ip", tau99, tau145)
        - 2 * einsum("ap,pia->ip", a.x2, tau963)
        - 2 * einsum("pa,pia->ip", tau806, tau34)
        + einsum("ap,pia->ip", a.x1, tau562)
        - 2 * einsum("qp,pqi->ip", tau333, tau1399)
        + einsum("qp,qpi->ip", tau876, tau1400)
        + einsum("poj,oij->ip", tau1401, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau686)
        + einsum("ap,pia->ip", a.x2, tau1402)
        + einsum("poj,oji->ip", tau1401, tau40)
        - einsum("pa,pia->ip", tau1033, tau29)
        + einsum("pa,pia->ip", tau895, tau5)
        + 2 * einsum("ap,pia->ip", a.x1, tau522)
        + einsum("ap,pia->ip", a.x1, tau312)
        + einsum("ap,pia->ip", a.x1, tau556)
        - 2 * einsum("ap,pia->ip", a.x1, tau316)
        - 2 * einsum("ap,pia->ip", a.x1, tau507)
        - einsum("ap,pia->ip", a.x1, tau399) / 2
        + einsum("pq,jp,qij->ip", tau133, a.x4, tau212)
        - 2 * einsum("ap,pia->ip", a.x1, tau524)
        + einsum("ap,pia->ip", a.x2, tau1123)
        + einsum("jp,pij->ip", a.x4, tau1349)
        + 4 * einsum("ap,pia->ip", a.x1, tau578)
        + 2 * einsum("ap,pia->ip", a.x2, tau1068)
        + 2 * einsum("ji,pj->ip", tau241, tau35)
        - einsum("pqo,qoi->ip", tau1403, tau19) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau321)
        + einsum("pq,qpi->ip", tau183, tau322)
        - einsum("qp,qpi->ip", tau802, tau1367) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau639)
        - einsum("pa,pia->ip", tau867, tau29)
        - 2 * einsum("ap,pia->ip", a.x1, tau128)
        + 2 * einsum("poj,oij->ip", tau1386, h.l.poo)
        + einsum("pq,qpi->ip", tau183, tau1404)
        - einsum("poj,oji->ip", tau1405, tau79)
        + 2 * einsum("pa,pia->ip", tau99, tau123)
        + 2 * einsum("ap,pia->ip", a.x1, tau1406)
        - 2 * einsum("pqo,qoi->ip", tau1407, tau19)
        + 2 * einsum("ap,pia->ip", a.x1, tau543)
        + einsum("qp,pqi->ip", tau106, tau356)
        - 2 * einsum("ap,pia->ip", a.x2, tau1140)
        - 2 * einsum("pa,pia->ip", tau1153, tau5)
        + 4 * einsum("qp,qpi->ip", tau371, tau1388)
        - 2 * einsum("poj,oij->ip", tau1408, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x1, tau336)
        + 4 * einsum("pa,pia->ip", tau806, tau22)
        - einsum("pqo,qoi->ip", tau1409, tau88) / 2
        + einsum("ap,pia->ip", a.x1, tau614)
        + einsum("qi,pq->ip", tau153, tau294)
        - einsum("ap,pia->ip", a.x2, tau981) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau1410)
        + 2 * einsum("pj,ji->ip", tau153, tau572)
        + einsum("poj,oij->ip", tau1358, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau1411)
        - einsum("pa,pia->ip", tau392, tau169) / 2
        + 4 * einsum("ap,pia->ip", a.x1, tau673)
        - einsum("poj,oij->ip", tau1412, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau714)
        + einsum("ap,pia->ip", a.x1, tau253)
        + einsum("poj,oji->ip", tau1365, tau40)
        + einsum("pa,pia->ip", tau25, tau299)
        + einsum("ap,pia->ip", a.x2, tau946)
        - einsum("ap,pia->ip", a.x2, tau1413) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau353)
        + einsum("pa,pia->ip", tau798, tau63)
        - einsum("pa,pia->ip", tau867, tau34)
        + einsum("qp,qpi->ip", tau208, tau854)
        - 4 * einsum("pa,pia->ip", tau812, tau63)
        - einsum("ap,pia->ip", a.x2, tau1099) / 2
        + einsum("qi,pq->ip", tau1, tau612)
        + 4 * einsum("ap,pia->ip", a.x1, tau646)
        - einsum("ap,pia->ip", a.x2, tau933)
        + einsum("pqo,qoi->ip", tau1414, tau89)
        - 2 * einsum("ap,pia->ip", a.x1, tau469)
        - 2 * einsum("ap,pia->ip", a.x2, tau1150)
        + 2 * einsum("poj,oij->ip", tau1415, h.l.poo)
        + 2 * einsum("jp,pij->ip", a.x4, tau1418)
        + einsum("pa,pia->ip", tau955, tau29)
        + 2 * einsum("ap,pia->ip", a.x1, tau526)
        - 4 * einsum("ap,pia->ip", a.x1, tau1419)
        + einsum("ap,pia->ip", a.x1, tau338)
        - einsum("ap,pia->ip", a.x1, tau728) / 2
        + 2 * einsum("poj,oij->ip", tau1420, h.l.poo)
        - einsum("ap,pia->ip", a.x2, tau1149)
        + einsum("ap,pia->ip", a.x1, tau1421)
        + 4 * einsum("ap,pia->ip", a.x1, tau330)
        - einsum("jp,pji->ip", a.x4, tau1418)
        - einsum("ap,pia->ip", a.x2, tau1422) / 2
        - einsum("ap,pia->ip", a.x2, tau1158)
        - einsum("pqo,qoi->ip", tau1423, tau89) / 2
        + 4 * einsum("pq,qpi->ip", tau371, tau920)
        + einsum("ap,pia->ip", a.x2, tau1191)
        + einsum("ap,pia->ip", a.x1, tau703)
        + einsum("pq,jp,qji->ip", tau133, a.x4, tau367)
        - einsum("ap,pia->ip", a.x2, tau1315)
        + einsum("ap,pia->ip", a.x2, tau821)
        - 2 * einsum("pq,qi->ip", tau457, tau66)
        + 2 * einsum("pa,pia->ip", tau121, tau123)
        - einsum("qp,qpi->ip", tau842, tau110) / 2
        + 4 * einsum("ap,pia->ip", a.x1, tau683)
        - 2 * einsum("pa,pia->ip", tau825, tau5)
        + 2 * einsum("pj,ij->ip", tau146, tau80)
        - einsum("pj,ji->ip", tau35, tau42)
        - 2 * einsum("ap,pia->ip", a.x1, tau1424)
        + einsum("qp,qpi->ip", tau803, tau30)
        - einsum("pq,qpi->ip", tau542, tau175) / 2
        + einsum("pj,ij->ip", tau150, tau272)
        - einsum("poj,oij->ip", tau1425, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau699)
        + einsum("ap,pia->ip", a.x1, tau729)
        - einsum("pqo,qoi->ip", tau1426, tau19) / 2
        - einsum("poj,oji->ip", tau1425, tau40)
        - einsum("poj,oji->ip", tau1371, tau40)
        + einsum("ap,pia->ip", a.x2, tau1277)
        - einsum("poj,oij->ip", tau1427, h.l.poo)
        - einsum("ap,pia->ip", a.x1, tau717)
        - einsum("jp,pji->ip", a.x4, tau1429)
        + 2 * einsum("poj,oij->ip", tau1430, h.l.poo)
        - einsum("pq,qpi->ip", tau183, tau1431) / 2
        + einsum("ij,pj->ip", h.f.oo, tau35)
        - einsum("pq,jp,qij->ip", tau388, a.x4, tau212) / 2
        - 2 * einsum("pqo,qoi->ip", tau1432, tau89)
        + 4 * einsum("ap,pia->ip", a.x1, tau742)
        + 2 * einsum("ap,pia->ip", a.x2, tau1433)
        - 2 * einsum("qi,pq->ip", tau107, tau147)
        - einsum("pq,jp,qij->ip", tau213, a.x4, tau132) / 2
        + einsum("pq,qpi->ip", tau106, tau175)
        - einsum("qp,qpi->ip", tau92, tau854) / 2
        - einsum("ap,pia->ip", a.x1, tau261) / 2
        - einsum("pj,ji->ip", tau150, tau42)
        - 2 * einsum("ap,pia->ip", a.x1, tau1434)
        - 4 * einsum("ap,pia->ip", a.x1, tau624)
        + 4 * einsum("qp,pqi->ip", tau601, tau162)
        - einsum("ap,pia->ip", a.x2, tau902)
        - einsum("ap,pia->ip", a.x2, tau1435) / 2
        - einsum("ap,pia->ip", a.x1, tau378)
        - einsum("ap,pia->ip", a.x1, tau326)
        - 2 * einsum("ap,pia->ip", a.x2, tau1026)
        + einsum("ap,pia->ip", a.x2, tau1293)
        - 2 * einsum("ap,pia->ip", a.x1, tau619)
        - einsum("pqo,qoi->ip", tau1436, tau88) / 2
        + einsum("pa,pia->ip", tau126, tau169)
        - einsum("pq,qpi->ip", tau221, tau1404) / 2
        - einsum("qp,pqi->ip", tau542, tau1437) / 2
        - einsum("poj,oij->ip", tau1337, h.l.poo)
        - 2 * einsum("qp,qpi->ip", tau371, tau881)
        + 2 * einsum("ap,pia->ip", a.x1, tau1438)
        + einsum("ap,pia->ip", a.x1, tau1439)
        + einsum("pq,qpi->ip", tau106, tau1376)
        + einsum("pa,pia->ip", tau210, tau149)
        - einsum("ap,pia->ip", a.x1, tau602)
        - 2 * einsum("pa,pia->ip", tau942, tau63)
        + 2 * einsum("ap,pia->ip", a.x2, tau1213)
        - 2 * einsum("poj,oji->ip", tau1377, tau40)
        - 2 * einsum("ap,pia->ip", a.x1, tau603)
        + einsum("pa,pia->ip", tau347, tau169)
        + einsum("qp,pqi->ip", tau333, tau1375)
        + einsum("pqo,qoi->ip", tau1440, tau88)
        + einsum("ap,pia->ip", a.x2, tau1441)
        - einsum("ap,pia->ip", a.x2, tau913)
        - 4 * einsum("pj,ij->ip", tau153, tau386)
        - 2 * einsum("ap,pia->ip", a.x1, tau786)
        - 2 * einsum("pq,qi->ip", tau154, tau66)
        + 2 * einsum("poj,oji->ip", tau1442, tau40)
        + einsum("ap,pia->ip", a.x1, tau431)
        - 2 * einsum("ap,pia->ip", a.x1, tau211)
        + einsum("pq,jp,qij->ip", tau373, a.x4, tau132)
        - einsum("pj,ji->ip", tau150, tau572)
        + 4 * einsum("ap,pia->ip", a.x1, tau449)
        - einsum("jp,pij->ip", a.x4, tau1444)
        - 2 * einsum("poj,oji->ip", tau1408, tau40)
        + einsum("pqo,qoi->ip", tau1445, tau88)
        + 4 * einsum("ap,pia->ip", a.x1, tau709)
        + 2 * einsum("ap,pia->ip", a.x1, tau642)
        - einsum("pq,jp,qji->ip", tau373, a.x4, tau132) / 2
        + einsum("pq,jp,qji->ip", tau213, a.x4, tau132)
        + einsum("qi,pq->ip", tau13, tau457)
        + einsum("ap,pia->ip", a.x2, tau827)
        + 2 * einsum("pa,pia->ip", tau867, tau63)
        + einsum("pqo,qoi->ip", tau1446, tau19)
        + einsum("jp,pij->ip", a.x4, tau1382)
        + einsum("pq,qpi->ip", tau221, tau1431)
        + 2 * einsum("pj,ij->ip", tau35, tau76)
        + einsum("pqo,qoi->ip", tau1447, tau19)
        - 2 * einsum("qp,pqi->ip", tau601, tau236)
        - 2 * einsum("ap,pia->ip", a.x1, tau514)
        - 2 * einsum("pa,pia->ip", tau865, tau63)
        + einsum("ap,pia->ip", a.x2, tau1102)
        + 2 * einsum("ap,pia->ip", a.x2, tau1100)
        - 2 * einsum("poj,oij->ip", tau1385, h.l.poo)
        + 2 * einsum("poj,oij->ip", tau1448, h.l.poo)
        + einsum("ap,pia->ip", a.x2, tau943)
        + einsum("qp,qpi->ip", tau802, tau1449)
        + einsum("ap,pia->ip", a.x1, tau1450)
        - 2 * einsum("ap,pia->ip", a.x1, tau503)
        + einsum("qi,pq->ip", tau170, tau341)
        - 2 * einsum("ap,pia->ip", a.x1, tau1451)
        + 4 * einsum("ap,pia->ip", a.x1, tau589)
        - 2 * einsum("ap,pia->ip", a.x2, tau1276)
        + einsum("ap,pia->ip", a.x2, tau1009)
        - einsum("ap,pia->ip", a.x2, tau1296) / 2
        + einsum("poj,oij->ip", tau1392, h.l.poo)
        - 4 * einsum("ap,pia->ip", a.x1, tau727)
        - einsum("ap,pia->ip", a.x2, tau1083)
        + einsum("ap,pia->ip", a.x2, tau1092)
        - 2 * einsum("ap,pia->ip", a.x1, tau1452)
        - einsum("pa,pia->ip", tau251, tau299) / 2
        + einsum("pa,pia->ip", tau189, tau123)
        + einsum("pqo,qoi->ip", tau1453, tau89)
        + einsum("qi,pq->ip", tau13, tau154)
        + einsum("ap,pia->ip", a.x2, tau1155)
        + einsum("pqo,qoi->ip", tau1454, tau19)
        - 2 * einsum("pa,pia->ip", tau926, tau22)
        - einsum("qp,qpi->ip", tau802, tau30) / 2
        - einsum("pq,jp,qij->ip", tau450, a.x4, tau132) / 2
        - einsum("qi,pq->ip", tau150, tau294) / 2
        + einsum("ap,pia->ip", a.x1, tau1455)
        - einsum("pa,pia->ip", tau471, tau222) / 2
        - einsum("pqo,qoi->ip", tau1456, tau89) / 2
        + 2 * einsum("pa,pia->ip", tau812, tau29)
        - einsum("ap,pia->ip", a.x2, tau864) / 2
        + 4 * einsum("ap,pia->ip", a.x1, tau595)
        + einsum("ap,pia->ip", a.x1, tau39)
        - 4 * einsum("pj,ji->ip", tau146, tau241)
        + einsum("qi,pq->ip", tau107, tau443)
        + einsum("pq,jp,qij->ip", tau368, a.x4, tau132)
        - 2 * einsum("jp,pij->ip", a.x4, tau1346)
        + einsum("pq,qi->ip", tau108, tau150)
        - einsum("ap,pia->ip", a.x2, tau1457)
        - einsum("qi,pq->ip", tau170, tau483) / 2
        + 2 * einsum("poj,oji->ip", tau1415, tau40)
        + einsum("qp,qpi->ip", tau208, tau1369)
        + 2 * einsum("ap,pia->ip", a.x2, tau938)
        - einsum("pa,pia->ip", tau547, tau169) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau650)
        - einsum("ap,pia->ip", a.x2, tau1291) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau998)
        - 2 * einsum("qp,qpi->ip", tau98, tau920)
        + einsum("pq,jp,qij->ip", tau388, a.x4, tau367)
        - einsum("ap,pia->ip", a.x1, tau671)
        + einsum("ap,pia->ip", a.x1, tau1458)
        - einsum("qp,pqi->ip", tau542, tau356) / 2
        - 2 * einsum("jp,pij->ip", a.x4, tau1459)
        - 2 * einsum("qp,qpi->ip", tau208, tau1364)
        + 4 * einsum("qp,pqi->ip", tau601, tau1399)
        + einsum("qp,qpi->ip", tau802, tau196)
        - einsum("poj,oij->ip", tau1460, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x1, tau27)
        + einsum("jp,pji->ip", a.x4, tau1459)
        - einsum("ap,pia->ip", a.x1, tau270)
        + einsum("pq,qi->ip", tau36, tau66)
        + 2 * einsum("jp,pij->ip", a.x4, tau1429)
        + einsum("ap,pia->ip", a.x2, tau1334)
        - einsum("ap,pia->ip", a.x2, tau960) / 2
        - einsum("poj,oji->ip", tau1389, tau79)
        + 2 * einsum("ap,pia->ip", a.x1, tau480)
        + einsum("ap,pia->ip", a.x2, tau948)
        + einsum("ij,pj->ip", h.f.oo, tau150)
        - einsum("poj,oji->ip", tau1461, tau40)
        - 2 * einsum("qp,qpi->ip", tau803, tau1449)
        - 4 * einsum("pj,ji->ip", tau153, tau241)
        + einsum("ap,pia->ip", a.x2, tau1070)
        - einsum("pa,pia->ip", tau189, tau222) / 2
        + einsum("pqo,qoi->ip", tau1462, tau19)
        + 2 * einsum("pj,ji->ip", tau150, tau241)
        + 2 * einsum("ap,pia->ip", a.x2, tau1463)
        - einsum("ap,pia->ip", a.x2, tau1065) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau711)
        - 2 * einsum("ap,pia->ip", a.x2, tau1464)
        + einsum("ap,pia->ip", a.x2, tau1121)
        + einsum("ap,pia->ip", a.x1, tau413)
        - 2 * einsum("ap,pia->ip", a.x2, tau1198)
        - einsum("pj,ij->ip", tau35, tau80)
        + einsum("ap,pia->ip", a.x2, tau910)
        + einsum("pq,qpi->ip", tau542, tau436)
        - 2 * einsum("qp,pqi->ip", tau106, tau301)
        + einsum("ap,pia->ip", a.x2, tau1040)
        - 2 * einsum("ap,pia->ip", a.x1, tau600)
        + einsum("ap,pia->ip", a.x2, tau971)
        + einsum("pqo,qoi->ip", tau1465, tau88)
        - einsum("pa,pia->ip", tau121, tau222)
        - 2 * einsum("ap,pia->ip", a.x1, tau618)
        - einsum("pq,qi->ip", tau171, tau453) / 2
        - 2 * einsum("pq,qpi->ip", tau106, tau1351)
        + einsum("pqo,qoi->ip", tau1466, tau19)
        - einsum("pqo,qoi->ip", tau1467, tau19) / 2
        - einsum("pa,pia->ip", tau957, tau29) / 2
        - einsum("pq,qi->ip", tau294, tau35) / 2
        + 2 * einsum("poj,oji->ip", tau1362, tau40)
        + 2 * einsum("ap,pia->ip", a.x2, tau1330)
        - 2 * einsum("pa,pia->ip", tau955, tau63)
        - einsum("pq,qpi->ip", tau876, tau1395) / 2
        - einsum("pqo,qoi->ip", tau1468, tau88) / 2
        + einsum("qp,pqi->ip", tau106, tau1437)
        - 2 * einsum("qp,pqi->ip", tau106, tau1361)
        + einsum("pq,qpi->ip", tau876, tau110)
        - 2 * einsum("pa,pia->ip", tau45, tau149)
        + einsum("pq,jp,qji->ip", tau388, a.x4, tau212)
        - 2 * einsum("ap,pia->ip", a.x2, tau1201)
        + 2 * einsum("jp,pji->ip", a.x4, tau1444)
        + einsum("qi,pq->ip", tau107, tau483)
        - 4 * einsum("ap,pia->ip", a.x1, tau207)
        - einsum("poj,oij->ip", tau1405, h.l.poo)
        + einsum("pa,pia->ip", tau957, tau63)
        + 2 * einsum("pa,pia->ip", tau9, tau222)
        + einsum("ap,pia->ip", a.x1, tau778)
        - einsum("ap,pia->ip", a.x2, tau1317) / 2
        - einsum("pq,qpi->ip", tau842, tau1400) / 2
        - einsum("ap,pia->ip", a.x1, tau748)
    )

    rx4 = (
        2 * einsum("ap,pia->ip", a.x1, tau762)
        + einsum("jp,pij->ip", a.x3, tau1459)
        - einsum("pa,pia->ip", tau189, tau101) / 2
        + einsum("pa,pia->ip", tau189, tau264)
        + einsum("ap,pia->ip", a.x2, tau1249)
        - 2 * einsum("pa,pia->ip", tau45, tau0)
        - einsum("pj,ij->ip", tau1, tau80)
        + 2 * einsum("pa,pia->ip", tau867, tau16)
        - 2 * einsum("ap,pia->ip", a.x1, tau1469)
        + 4 * einsum("ap,pia->ip", a.x2, tau872)
        - einsum("ap,pia->ip", a.x1, tau632)
        + einsum("qp,pqi->ip", tau183, tau1437)
        - 2 * einsum("ap,pia->ip", a.x2, tau993)
        + 2 * einsum("pj,ij->ip", tau1, tau76)
        + einsum("jp,pji->ip", a.x3, tau1382)
        + einsum("ap,pia->ip", a.x1, tau669)
        - 2 * einsum("pa,pia->ip", tau246, tau264)
        + 2 * einsum("pa,pia->ip", tau867, tau136)
        - 2 * einsum("ap,pia->ip", a.x2, tau918)
        - 2 * einsum("ap,pia->ip", a.x2, tau1072)
        - 2 * einsum("pa,pia->ip", tau865, tau16)
        - 2 * einsum("ap,pia->ip", a.x1, tau361)
        - einsum("pq,qpi->ip", tau802, tau17) / 2
        + einsum("pq,qpi->ip", tau332, tau860)
        + 2 * einsum("jp,pji->ip", a.x3, tau1418)
        - einsum("poj,oij->ip", tau1470, h.l.poo)
        + 2 * einsum("ap,pia->ip", a.x1, tau462)
        - 2 * einsum("qp,pqi->ip", tau310, tau1399)
        - einsum("pqo,qoi->ip", tau1471, tau95) / 2
        - einsum("poj,oji->ip", tau1472, tau40)
        - 2 * einsum("pa,pia->ip", tau210, tau87)
        - einsum("pq,qpi->ip", tau332, tau1369) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau975)
        + 4 * einsum("qp,qpi->ip", tau869, tau1449)
        + 2 * einsum("pa,pia->ip", tau812, tau201)
        - 2 * einsum("ap,pia->ip", a.x2, tau928)
        + einsum("ap,pia->ip", a.x1, tau769)
        - 2 * einsum("ap,pia->ip", a.x1, tau640)
        + einsum("qi,pq->ip", tau170, tau319)
        + einsum("ap,pia->ip", a.x2, tau1097)
        - 4 * einsum("pj,ij->ip", tau318, tau386)
        + 2 * einsum("ap,pia->ip", a.x2, tau1310)
        + einsum("qp,pqi->ip", tau221, tau301)
        - einsum("ap,pia->ip", a.x2, tau958) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau927)
        + 2 * einsum("poj,oji->ip", tau1473, tau40)
        - 2 * einsum("poj,oji->ip", tau1474, tau40)
        - einsum("ap,pia->ip", a.x2, tau1230)
        + 2 * einsum("ap,pia->ip", a.x2, tau1475)
        - einsum("pqo,qoi->ip", tau1476, tau89) / 2
        + einsum("ap,pia->ip", a.x1, tau1477)
        + einsum("ap,pia->ip", a.x2, tau1478)
        - 2 * einsum("ap,pia->ip", a.x1, tau678)
        - einsum("ap,pia->ip", a.x2, tau1268) / 2
        + einsum("ap,pia->ip", a.x2, tau1479)
        - einsum("poj,oij->ip", tau1480, h.l.poo)
        + einsum("ap,pia->ip", a.x2, tau836)
        - einsum("pq,qi->ip", tau14, tau453) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau405)
        + 2 * einsum("ap,pia->ip", a.x1, tau481)
        - 2 * einsum("pq,qpi->ip", tau803, tau1400)
        - einsum("qp,pqi->ip", tau183, tau301) / 2
        - einsum("poj,oji->ip", tau1481, tau40)
        + einsum("pa,pia->ip", tau798, tau16)
        - 2 * einsum("pa,pia->ip", tau942, tau16)
        + einsum("ap,pia->ip", a.x2, tau897)
        - 4 * einsum("pj,ij->ip", tau137, tau76)
        - 2 * einsum("pq,qpi->ip", tau310, tau1431)
        + 2 * einsum("poj,oij->ip", tau1482, h.l.poo)
        + 2 * einsum("ap,pia->ip", a.x2, tau1182)
        - 2 * einsum("pa,pia->ip", tau501, tau65)
        + einsum("qp,qpi->ip", tau220, tau1369)
        + 4 * einsum("ap,pia->ip", a.x2, tau1235)
        + 4 * einsum("pq,qpi->ip", tau601, tau436)
        + 2 * einsum("pj,ji->ip", tau318, tau42)
        - einsum("pa,pia->ip", tau867, tau201)
        + 2 * einsum("pj,ij->ip", tau137, tau80)
        - 2 * einsum("ap,pia->ip", a.x1, tau568)
        - 2 * einsum("ap,pia->ip", a.x2, tau1483)
        - 2 * einsum("qi,pq->ip", tau153, tau417)
        - einsum("qp,qpi->ip", tau332, tau854) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1137)
        + 2 * einsum("ap,pia->ip", a.x1, tau637)
        - 2 * einsum("ap,pia->ip", a.x1, tau500)
        - 4 * einsum("ap,pia->ip", a.x2, tau1055)
        - 2 * einsum("pa,pia->ip", tau926, tau136)
        - einsum("ap,pia->ip", a.x1, tau688) / 2
        - einsum("ap,pia->ip", a.x1, tau1484)
        + einsum("pq,qpi->ip", tau802, tau110)
        + einsum("poj,oji->ip", tau1485, tau40)
        - 2 * einsum("ij,pj->ip", h.f.oo, tau137)
        + einsum("ap,pia->ip", a.x2, tau1285)
        - 2 * einsum("ap,pia->ip", a.x2, tau1143)
        - einsum("ap,pia->ip", a.x1, tau144)
        + 4 * einsum("pa,pia->ip", tau806, tau136)
        + einsum("pq,qi->ip", tau160, tau66)
        + einsum("jp,pji->ip", a.x3, tau1349)
        + einsum("ap,pia->ip", a.x2, tau1119)
        - einsum("pq,qpi->ip", tau92, tau881) / 2
        + einsum("pq,qpi->ip", tau805, tau30)
        + einsum("pqo,qoi->ip", tau1486, tau19)
        + einsum("pq,qpi->ip", tau310, tau1404)
        + 4 * einsum("ap,pia->ip", a.x2, tau1199)
        + einsum("pq,jp,qji->ip", tau388, a.x3, tau367)
        - einsum("poj,oji->ip", tau1487, tau40)
        - einsum("ap,pia->ip", a.x1, tau181)
        - 2 * einsum("qi,pq->ip", tau107, tau529)
        + einsum("ap,pia->ip", a.x2, tau961)
        + einsum("poj,oij->ip", tau1485, h.l.poo)
        + 2 * einsum("ap,pia->ip", a.x1, tau622)
        + einsum("qp,pqi->ip", tau231, tau1399)
        - einsum("qi,pq->ip", tau170, tau2) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1259)
        + 2 * einsum("poj,oij->ip", tau1473, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x2, tau1190)
        - 2 * einsum("ap,pia->ip", a.x1, tau630)
        + einsum("pa,pia->ip", tau942, tau201)
        - einsum("jp,pji->ip", a.x3, tau1444)
        + 2 * einsum("ap,pia->ip", a.x1, tau1488)
        - 2 * einsum("ij,pj->ip", tau272, tau318)
        - einsum("pqo,qoi->ip", tau1489, tau95) / 2
        + 2 * einsum("poj,oij->ip", tau1490, h.l.poo)
        - einsum("ap,pia->ip", a.x1, tau759) / 2
        + einsum("pa,pia->ip", tau251, tau65)
        - 2 * einsum("ij,pj->ip", h.f.oo, tau318)
        - 2 * einsum("ap,pia->ip", a.x1, tau493)
        - 2 * einsum("jp,pij->ip", a.x3, tau1382)
        + 4 * einsum("ap,pia->ip", a.x2, tau1279)
        + einsum("ap,pia->ip", a.x1, tau382)
        + einsum("ap,pia->ip", a.x1, tau695)
        + 2 * einsum("pa,pia->ip", tau99, tau87)
        + einsum("ap,pia->ip", a.x1, tau638)
        - einsum("poj,oij->ip", tau1481, h.l.poo)
        - einsum("pj,ji->ip", tau453, tau572)
        - 2 * einsum("pa,pia->ip", tau25, tau65)
        + einsum("pa,pia->ip", tau825, tau115)
        + einsum("ap,pia->ip", a.x1, tau227)
        + 2 * einsum("ap,pia->ip", a.x2, tau1018)
        + einsum("qi,pq->ip", tau13, tau138)
        + einsum("qi,pq->ip", tau453, tau67)
        + einsum("ap,pia->ip", a.x2, tau1000)
        - 2 * einsum("poj,oij->ip", tau1491, h.l.poo)
        + 2 * einsum("ap,pia->ip", a.x2, tau1002)
        + einsum("ap,pia->ip", a.x1, tau359)
        + einsum("ap,pia->ip", a.x1, tau700)
        + einsum("ap,pia->ip", a.x1, tau362)
        + einsum("ap,pia->ip", a.x1, tau746)
        - einsum("qi,pq->ip", tau170, tau494) / 2
        + einsum("pq,qpi->ip", tau231, tau504)
        - einsum("ap,pia->ip", a.x1, tau659) / 2
        + einsum("pa,pia->ip", tau886, tau115)
        + einsum("jp,pij->ip", a.x3, tau1346)
        - einsum("qp,pqi->ip", tau183, tau1361) / 2
        - einsum("ap,pia->ip", a.x1, tau582)
        - 2 * einsum("poj,oji->ip", tau1492, tau40)
        - einsum("ap,pia->ip", a.x1, tau1493) / 2
        + 2 * einsum("pa,pia->ip", tau9, tau101)
        - 2 * einsum("ap,pia->ip", a.x1, tau411)
        - einsum("qp,qpi->ip", tau220, tau860) / 2
        - einsum("pqo,qoi->ip", tau1494, tau89) / 2
        + einsum("ap,pia->ip", a.x1, tau344)
        + 4 * einsum("ap,pia->ip", a.x2, tau1179)
        - 2 * einsum("ap,pia->ip", a.x1, tau782)
        + einsum("ap,pia->ip", a.x1, tau585)
        + einsum("qp,pqi->ip", tau231, tau162)
        - 2 * einsum("ap,pia->ip", a.x2, tau1225)
        - einsum("poj,oji->ip", tau1470, tau40)
        - einsum("qi,pq->ip", tau13, tau454) / 2
        - 2 * einsum("pq,qpi->ip", tau601, tau175)
        + einsum("ap,pia->ip", a.x2, tau1142)
        + einsum("ap,pia->ip", a.x2, tau1147)
        + 2 * einsum("ap,pia->ip", a.x2, tau907)
        + einsum("ap,pia->ip", a.x1, tau293)
        - einsum("ap,pia->ip", a.x2, tau1151)
        + einsum("ij,pj->ip", tau272, tau453)
        + einsum("pq,qpi->ip", tau310, tau322)
        + einsum("qi,pq->ip", tau107, tau494)
        - einsum("qi,pq->ip", tau1, tau14) / 2
        + 4 * einsum("pa,pia->ip", tau45, tau87)
        + einsum("qi,pq->ip", tau170, tau529)
        - einsum("ap,pia->ip", a.x1, tau706) / 2
        + einsum("ap,pia->ip", a.x2, tau892)
        - einsum("pa,pia->ip", tau1033, tau201)
        - 2 * einsum("ap,pia->ip", a.x2, tau1303)
        - einsum("ap,pia->ip", a.x1, tau94)
        - 2 * einsum("ap,pia->ip", a.x1, tau760)
        - 2 * einsum("ap,pia->ip", a.x2, tau1164)
        - einsum("pqo,qoi->ip", tau1495, tau89) / 2
        - 2 * einsum("ap,pia->ip", a.x1, tau1496)
        + einsum("ap,pia->ip", a.x2, tau1497)
        - 2 * einsum("ap,pia->ip", a.x2, tau1066)
        + einsum("ap,pia->ip", a.x2, tau1188)
        + einsum("pa,pia->ip", tau865, tau201)
        - einsum("pa,pia->ip", tau1056, tau12) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau775)
        + einsum("ap,pia->ip", a.x1, tau725)
        - einsum("ap,pia->ip", a.x1, tau1498)
        - einsum("pa,pia->ip", tau890, tau115) / 2
        - einsum("ap,pia->ip", a.x1, tau1499)
        - einsum("poj,oji->ip", tau1480, tau79)
        + einsum("pa,pia->ip", tau547, tau65)
        + einsum("pq,qpi->ip", tau231, tau1431)
        + einsum("ap,pia->ip", a.x2, tau1184)
        - 4 * einsum("ap,pia->ip", a.x2, tau1266)
        + einsum("ap,pia->ip", a.x2, tau1242)
        - einsum("pq,qpi->ip", tau802, tau1395) / 2
        + einsum("ap,pia->ip", a.x1, tau113)
        + einsum("pa,pia->ip", tau1153, tau12)
        - 4 * einsum("ap,pia->ip", a.x1, tau78)
        + 4 * einsum("ap,pia->ip", a.x2, tau1001)
        - 2 * einsum("pqo,qoi->ip", tau1500, tau88)
        - 2 * einsum("ap,pia->ip", a.x1, tau781)
        + 2 * einsum("ap,pia->ip", a.x1, tau691)
        + 2 * einsum("ij,pj->ip", tau386, tau453)
        + einsum("ap,pia->ip", a.x1, tau485)
        - einsum("ap,pia->ip", a.x2, tau799) / 2
        - 2 * einsum("poj,oji->ip", tau1491, tau40)
        + einsum("ap,pia->ip", a.x1, tau307)
        - 2 * einsum("ap,pia->ip", a.x2, tau953)
        - 2 * einsum("ap,pia->ip", a.x2, tau1236)
        - einsum("pq,jp,qji->ip", tau450, a.x3, tau132) / 2
        - einsum("pq,jp,qij->ip", tau368, a.x3, tau132) / 2
        - einsum("ap,pia->ip", a.x2, tau1232)
        + 2 * einsum("ap,pia->ip", a.x2, tau1063)
        - einsum("pa,pia->ip", tau867, tau47)
        - 2 * einsum("qi,pq->ip", tau318, tau67)
        + 2 * einsum("ap,pia->ip", a.x1, tau672)
        + einsum("pqo,qoi->ip", tau1501, tau89)
        + einsum("pqo,qoi->ip", tau1502, tau19)
        + 2 * einsum("pa,pia->ip", tau121, tau264)
        + einsum("ap,pia->ip", a.x1, tau385)
        + einsum("qi,pq->ip", tau35, tau417)
        - einsum("pa,pia->ip", tau1107, tau12) / 2
        + einsum("pa,pia->ip", tau292, tau101)
        - einsum("pqo,qoi->ip", tau1503, tau19) / 2
        - einsum("pa,pia->ip", tau798, tau201) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1504)
        + 2 * einsum("ap,pia->ip", a.x1, tau626)
        - 2 * einsum("ap,pia->ip", a.x1, tau1505)
        - 4 * einsum("ap,pia->ip", a.x2, tau1506)
        + 2 * einsum("poj,oji->ip", tau1507, tau40)
        - einsum("poj,oij->ip", tau1508, h.l.poo)
        + einsum("poj,oij->ip", tau1509, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau736)
        + 2 * einsum("ap,pia->ip", a.x2, tau1510)
        - 2 * einsum("ap,pia->ip", a.x2, tau968)
        - 4 * einsum("ap,pia->ip", a.x1, tau512)
        + einsum("qi,pq->ip", tau13, tau478)
        - einsum("pa,pia->ip", tau895, tau115) / 2
        - 2 * einsum("pq,qpi->ip", tau601, tau1376)
        - 2 * einsum("ap,pia->ip", a.x2, tau888)
        - 2 * einsum("poj,oij->ip", tau1492, h.l.poo)
        - 4 * einsum("ap,pia->ip", a.x2, tau1511)
        - 2 * einsum("pa,pia->ip", tau347, tau65)
        - einsum("ap,pia->ip", a.x2, tau835)
        + einsum("qi,pq->ip", tau153, tau296)
        - einsum("ap,pia->ip", a.x1, tau772) / 2
        - einsum("ap,pia->ip", a.x2, tau1180)
        - 4 * einsum("pj,ji->ip", tau137, tau241)
        - 2 * einsum("jp,pji->ip", a.x3, tau1459)
        - einsum("poj,oji->ip", tau1512, tau40)
        + einsum("ap,pia->ip", a.x1, tau1513)
        - einsum("ap,pia->ip", a.x1, tau549) / 2
        + einsum("pa,pia->ip", tau210, tau0)
        - 2 * einsum("ap,pia->ip", a.x2, tau1514)
        - 2 * einsum("ap,pia->ip", a.x1, tau118)
        - einsum("pqo,qoi->ip", tau1515, tau19) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau845)
        - 2 * einsum("ap,pia->ip", a.x2, tau985)
        - 2 * einsum("pa,pia->ip", tau292, tau264)
        - einsum("ap,pia->ip", a.x1, tau682)
        - 2 * einsum("qp,qpi->ip", tau805, tau196)
        + einsum("pq,qpi->ip", tau92, tau1388)
        - einsum("pa,pia->ip", tau121, tau101)
        - einsum("ap,pia->ip", a.x1, tau488) / 2
        + einsum("qi,pq->ip", tau146, tau296)
        - 4 * einsum("ap,pia->ip", a.x1, tau10)
        + einsum("ap,pia->ip", a.x2, tau1241)
        + einsum("ap,pia->ip", a.x2, tau1516)
        - 2 * einsum("pq,qpi->ip", tau333, tau1351)
        + einsum("pa,pia->ip", tau1038, tau12)
        - einsum("poj,oij->ip", tau1472, h.l.poo)
        - einsum("poj,oij->ip", tau1517, h.l.poo)
        - einsum("ap,pia->ip", a.x1, tau86)
        - 2 * einsum("pq,qpi->ip", tau333, tau436)
        + einsum("pq,jp,qij->ip", tau450, a.x3, tau132)
        + einsum("ap,pia->ip", a.x1, tau398)
        + einsum("ap,pia->ip", a.x1, tau403)
        + einsum("ap,pia->ip", a.x2, tau935)
        - einsum("pqo,qoi->ip", tau1518, tau19) / 2
        + einsum("ap,pia->ip", a.x2, tau1519)
        + einsum("ap,pia->ip", a.x2, tau1308)
        + einsum("pq,qpi->ip", tau208, tau1373)
        + 2 * einsum("ap,pia->ip", a.x2, tau1130)
        - 4 * einsum("ap,pia->ip", a.x2, tau1520)
        + einsum("pqo,qoi->ip", tau1521, tau89)
        - einsum("pq,qpi->ip", tau92, tau1373) / 2
        - einsum("ap,pia->ip", a.x1, tau1522) / 2
        + einsum("pa,pia->ip", tau701, tau65)
        - 2 * einsum("pq,qpi->ip", tau208, tau920)
        + einsum("pqo,qoi->ip", tau1523, tau89)
        + einsum("poj,oji->ip", tau1524, tau40)
        - 2 * einsum("poj,oij->ip", tau1474, h.l.poo)
        + einsum("poj,oij->ip", tau1525, h.l.poo)
        + 2 * einsum("pj,ji->ip", tau137, tau42)
        - 2 * einsum("ap,pia->ip", a.x2, tau848)
        - einsum("poj,oij->ip", tau1526, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x1, tau756)
        - 2 * einsum("pj,ij->ip", tau137, tau272)
        - 4 * einsum("ap,pia->ip", a.x2, tau832)
        - einsum("ap,pia->ip", a.x2, tau1134)
        + einsum("ap,pia->ip", a.x1, tau225)
        + einsum("pa,pia->ip", tau955, tau201)
        + 2 * einsum("pa,pia->ip", tau99, tau264)
        + einsum("poj,oij->ip", tau1524, h.l.poo)
        - einsum("qi,pq->ip", tau13, tau160) / 2
        - einsum("qp,pqi->ip", tau221, tau1437) / 2
        + einsum("pqo,qoi->ip", tau1527, tau95)
        + einsum("qp,pqi->ip", tau183, tau356)
        - 2 * einsum("ap,pia->ip", a.x1, tau435)
        - 2 * einsum("ap,pia->ip", a.x2, tau1205)
        + einsum("pj,ij->ip", tau1, tau272)
        + 2 * einsum("pa,pia->ip", tau1033, tau16)
        - 4 * einsum("ap,pia->ip", a.x2, tau1305)
        - einsum("pqo,qoi->ip", tau1528, tau89) / 2
        + einsum("pqo,qoi->ip", tau1529, tau88)
        - 2 * einsum("pq,qpi->ip", tau805, tau1449)
        + einsum("pqo,qoi->ip", tau1530, tau19)
        + einsum("pa,pia->ip", tau404, tau101)
        + 4 * einsum("ap,pia->ip", a.x2, tau1222)
        - 2 * einsum("ap,pia->ip", a.x1, tau550)
        - einsum("pqo,qoi->ip", tau1531, tau89) / 2
        - einsum("qp,pqi->ip", tau231, tau236) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1239)
        + 2 * einsum("poj,oij->ip", tau1532, h.l.poo)
        - 2 * einsum("ap,pia->ip", a.x2, tau1533)
        + 2 * einsum("pj,ji->ip", tau1, tau241)
        + 4 * einsum("ap,pia->ip", a.x1, tau1534)
        - einsum("pqo,qoi->ip", tau1535, tau89) / 2
        - 2 * einsum("jp,pji->ip", a.x3, tau1346)
        + 2 * einsum("ap,pia->ip", a.x2, tau1331)
        - einsum("ap,pia->ip", a.x1, tau1536) / 2
        + einsum("qi,pq->ip", tau107, tau2)
        + einsum("pq,jp,qji->ip", tau373, a.x3, tau132)
        - 4 * einsum("ji,pj->ip", tau241, tau318)
        - einsum("ap,pia->ip", a.x1, tau667) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau984)
        - einsum("pq,jp,qij->ip", tau133, a.x3, tau212) / 2
        - einsum("ap,pia->ip", a.x1, tau460) / 2
        - einsum("pa,pia->ip", tau471, tau101) / 2
        - einsum("jp,pji->ip", a.x3, tau1355)
        - einsum("ap,pia->ip", a.x1, tau1537) / 2
        - 2 * einsum("ap,pia->ip", a.x2, tau1253)
        - 2 * einsum("qi,pq->ip", tau146, tau417)
        + einsum("ap,pia->ip", a.x1, tau766)
        - einsum("ap,pia->ip", a.x1, tau740)
        + 2 * einsum("ap,pia->ip", a.x2, tau1323)
        + 2 * einsum("poj,oij->ip", tau1538, h.l.poo)
        + einsum("ap,pia->ip", a.x1, tau598)
        - einsum("pq,jp,qij->ip", tau373, a.x3, tau132) / 2
        + einsum("ij,pj->ip", h.f.oo, tau1)
        + einsum("qp,pqi->ip", tau310, tau236)
        + einsum("ap,pia->ip", a.x2, tau1539)
        - 2 * einsum("qp,qpi->ip", tau869, tau30)
        + einsum("ap,pia->ip", a.x1, tau406)
        + 4 * einsum("ap,pia->ip", a.x2, tau1076)
        - 2 * einsum("pqo,qoi->ip", tau1540, tau19)
        - einsum("ap,pia->ip", a.x1, tau477)
        - 4 * einsum("pa,pia->ip", tau812, tau16)
        + 2 * einsum("ap,pia->ip", a.x1, tau661)
        + 2 * einsum("ap,pia->ip", a.x2, tau1541)
        - einsum("jp,pij->ip", a.x3, tau1429)
        - 2 * einsum("pq,qpi->ip", tau803, tau110)
        - einsum("pa,pia->ip", tau99, tau0)
        + einsum("pq,jp,qji->ip", tau133, a.x3, tau212)
        - 2 * einsum("pa,pia->ip", tau126, tau65)
        - 2 * einsum("qi,pq->ip", tau107, tau319)
        + einsum("pqo,qoi->ip", tau1542, tau89)
        + 2 * einsum("poj,oij->ip", tau1543, h.l.poo)
        + einsum("qp,qpi->ip", tau805, tau1367)
        + 2 * einsum("ap,pia->ip", a.x1, tau82)
        + einsum("poj,oji->ip", tau1525, tau40)
        - einsum("pj,ji->ip", tau1, tau42)
        - einsum("qp,pqi->ip", tau221, tau356) / 2
        + 4 * einsum("ap,pia->ip", a.x2, tau1019)
        - einsum("pq,jp,qij->ip", tau388, a.x3, tau367) / 2
        - einsum("pq,qpi->ip", tau231, tau1404) / 2
        + 2 * einsum("poj,oji->ip", tau1544, tau40)
        - einsum("ji,pj->ip", tau42, tau453)
        - 2 * einsum("pq,qi->ip", tau138, tau66)
        + einsum("pq,qpi->ip", tau803, tau1395)
        + 2 * einsum("ap,pia->ip", a.x1, tau244)
        + 2 * einsum("poj,oji->ip", tau1543, tau40)
        - einsum("pq,qi->ip", tau296, tau35) / 2
        + einsum("ap,pia->ip", a.x1, tau349)
        - 2 * einsum("pq,qi->ip", tau478, tau66)
        + einsum("pqo,qoi->ip", tau1545, tau19)
        - einsum("ap,pia->ip", a.x1, tau561)
        + einsum("pqo,qoi->ip", tau1546, tau95)
        - einsum("pq,qpi->ip", tau231, tau322) / 2
        + einsum("pq,qpi->ip", tau333, tau1376)
        - 2 * einsum("ap,pia->ip", a.x1, tau439)
        - einsum("ap,pia->ip", a.x1, tau394) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau565)
        + einsum("ap,pia->ip", a.x1, tau541)
        + 2 * einsum("poj,oij->ip", tau1507, h.l.poo)
        - einsum("pq,jp,qji->ip", tau388, a.x3, tau212) / 2
        - einsum("ap,pia->ip", a.x2, tau1263)
        + einsum("ap,pia->ip", a.x1, tau510)
        + einsum("ap,pia->ip", a.x2, tau1058)
        - einsum("ap,pia->ip", a.x1, tau675) / 2
        - einsum("pa,pia->ip", tau957, tau201) / 2
        + einsum("pqo,qoi->ip", tau1547, tau95)
        - einsum("ap,pia->ip", a.x1, tau610)
        + einsum("qi,pq->ip", tau150, tau417)
        - 2 * einsum("ap,pia->ip", a.x2, tau1288)
        + einsum("pq,jp,qji->ip", tau368, a.x3, tau132)
        + 4 * einsum("pq,qpi->ip", tau601, tau1351)
        + einsum("pqo,qoi->ip", tau1548, tau89)
        + 2 * einsum("poj,oji->ip", tau1549, tau40)
        - 2 * einsum("pqo,qoi->ip", tau1550, tau89)
        - 2 * einsum("ap,pia->ip", a.x1, tau733)
        + einsum("pa,pia->ip", tau392, tau65)
        + einsum("ap,pia->ip", a.x2, tau1551)
        - einsum("jp,pij->ip", a.x3, tau1418)
        - einsum("poj,oij->ip", tau1552, h.l.poo)
        - einsum("pq,jp,qji->ip", tau133, a.x3, tau367) / 2
        + einsum("pq,qpi->ip", tau802, tau1400)
        + einsum("ap,pia->ip", a.x1, tau306)
        - einsum("ap,pia->ip", a.x1, tau747)
        - 2 * einsum("ap,pia->ip", a.x2, tau966)
        + einsum("qi,pq->ip", tau1, tau67)
        - 4 * einsum("ap,pia->ip", a.x2, tau1228)
        - 2 * einsum("jp,pij->ip", a.x3, tau1349)
        - einsum("pq,qpi->ip", tau220, tau1364) / 2
        - 4 * einsum("ap,pia->ip", a.x2, tau1257)
        + einsum("pq,jp,qij->ip", tau388, a.x3, tau212)
        - einsum("ap,pia->ip", a.x1, tau446) / 2
        - einsum("pqo,qoi->ip", tau1553, tau95) / 2
        + einsum("ap,pia->ip", a.x1, tau1554)
        - 2 * einsum("poj,oji->ip", tau1555, tau40)
        - einsum("pq,jp,qji->ip", tau213, a.x3, tau132) / 2
        - 2 * einsum("poj,oij->ip", tau1555, h.l.poo)
        - einsum("ap,pia->ip", a.x2, tau1168) / 2
        + einsum("pq,qpi->ip", tau220, tau854)
        - 2 * einsum("qi,pq->ip", tau137, tau67)
        + 2 * einsum("jp,pji->ip", a.x3, tau1429)
        - 2 * einsum("pq,qpi->ip", tau310, tau504)
        - einsum("pa,pia->ip", tau99, tau101)
        + 2 * einsum("poj,oji->ip", tau1538, tau40)
        - 2 * einsum("pa,pia->ip", tau955, tau16)
        - 2 * einsum("pq,qpi->ip", tau869, tau1367)
        - 2 * einsum("ap,pia->ip", a.x2, tau880)
        - 2 * einsum("ap,pia->ip", a.x1, tau283)
        + einsum("ap,pia->ip", a.x1, tau558)
        - einsum("poj,oji->ip", tau1556, tau40)
        + 4 * einsum("ap,pia->ip", a.x2, tau1208)
        + 2 * einsum("ap,pia->ip", a.x2, tau1022)
        - 2 * einsum("ap,pia->ip", a.x1, tau131)
        - 2 * einsum("qp,pqi->ip", tau310, tau162)
        + einsum("pq,qi->ip", tau454, tau66)
        + 2 * einsum("ap,pia->ip", a.x1, tau1557)
        + einsum("qp,qpi->ip", tau332, tau1364)
        + 2 * einsum("pj,ji->ip", tau318, tau572)
        - 2 * einsum("ap,pia->ip", a.x2, tau1220)
        - 4 * einsum("ap,pia->ip", a.x2, tau792)
        + 2 * einsum("jp,pij->ip", a.x3, tau1444)
        - 2 * einsum("pq,qpi->ip", tau208, tau1388)
        - 2 * einsum("ap,pia->ip", a.x2, tau1299)
        + einsum("pa,pia->ip", tau246, tau101)
        + einsum("ap,pia->ip", a.x1, tau379)
        + 2 * einsum("jp,pij->ip", a.x3, tau1355)
        - 2 * einsum("pqo,qoi->ip", tau1558, tau19)
        + 4 * einsum("pq,qpi->ip", tau869, tau196)
        + 2 * einsum("ap,pia->ip", a.x2, tau1112)
        - 2 * einsum("ap,pia->ip", a.x2, tau1286)
        + einsum("ap,pia->ip", a.x1, tau275)
        + 4 * einsum("ap,pia->ip", a.x2, tau951)
        - einsum("qi,pq->ip", tau150, tau296) / 2
        + 2 * einsum("ap,pia->ip", a.x1, tau1559)
        - 2 * einsum("pa,pia->ip", tau404, tau264)
        + 2 * einsum("ap,pia->ip", a.x2, tau1320)
        - 4 * einsum("pa,pia->ip", tau9, tau264)
        + einsum("pq,jp,qij->ip", tau213, a.x3, tau132)
        - 2 * einsum("ap,pia->ip", a.x2, tau1104)
        + einsum("pq,qpi->ip", tau803, tau17)
        + einsum("ap,pia->ip", a.x1, tau277)
        + einsum("pqo,qoi->ip", tau1560, tau19)
        + einsum("pq,qpi->ip", tau333, tau175)
        - 2 * einsum("ap,pia->ip", a.x2, tau1561)
        + 2 * einsum("ji,pj->ip", tau241, tau453)
        + einsum("poj,oji->ip", tau1509, tau40)
        + einsum("pqo,qoi->ip", tau1562, tau89)
        + einsum("ap,pia->ip", a.x1, tau1563)
        + 2 * einsum("poj,oji->ip", tau1564, tau40)
        - 2 * einsum("ap,pia->ip", a.x1, tau467)
        + 2 * einsum("ap,pia->ip", a.x1, tau605)
        - 2 * einsum("pa,pia->ip", tau806, tau47)
        - 2 * einsum("ap,pia->ip", a.x1, tau718)
        + einsum("ap,pia->ip", a.x1, tau592)
        + einsum("qi,pq->ip", tau137, tau14)
        + einsum("ap,pia->ip", a.x2, tau1118)
        - 2 * einsum("ap,pia->ip", a.x2, tau823)
        + einsum("qp,pqi->ip", tau221, tau1361)
        + einsum("ij,pj->ip", h.f.oo, tau453)
        + einsum("pq,qpi->ip", tau208, tau881)
        + einsum("ap,pia->ip", a.x1, tau716)
        - 2 * einsum("ap,pia->ip", a.x2, tau978)
        + 2 * einsum("ap,pia->ip", a.x1, tau654)
        - einsum("poj,oji->ip", tau1508, tau40)
        + einsum("pq,qi->ip", tau14, tau318)
        + einsum("ap,pia->ip", a.x1, tau263)
        + einsum("qp,pqi->ip", tau310, tau1375)
        + einsum("pa,pia->ip", tau471, tau264)
        + einsum("pq,qpi->ip", tau92, tau920)
        + einsum("pa,pia->ip", tau957, tau16)
        + einsum("pqo,qoi->ip", tau1565, tau89)
        + einsum("ap,pia->ip", a.x2, tau1012)
        - einsum("qp,pqi->ip", tau231, tau1375) / 2
        + 2 * einsum("ap,pia->ip", a.x2, tau829)
        - 4 * einsum("ap,pia->ip", a.x2, tau1078)
        + einsum("pa,pia->ip", tau926, tau47)
        + 4 * einsum("ap,pia->ip", a.x2, tau1126)
        + einsum("ap,pia->ip", a.x2, tau1197)
        + 2 * einsum("poj,oji->ip", tau1566, tau79)
        + einsum("pq,jp,qij->ip", tau133, a.x3, tau367)
        + 4 * einsum("ap,pia->ip", a.x2, tau819)
        + 2 * einsum("poj,oij->ip", tau1566, h.l.poo)
    )

    return rx1, rx2, rx3, rx4

