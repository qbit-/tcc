from numpy import einsum
import numpy as np

def gen_energy(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z):

    energy = (
        2 * einsum("ai,ia->", T1, f.ov)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    energy += (
        - einsum("ai,ia->", T1, tau3)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W1.o, W2.v)
    )

    energy += (
        2 * einsum("ai,ia->", T1, tau3)
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("op,om,pm,pn,on->mn", Z, tau0, tau1, tau2, tau3)
    )

    energy += (
        - einsum("mn,mn->", X, tau4)
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("om,om,pm->op", tau0, tau1, tau4)
    )

    energy += (
        2 * einsum("op,op->", Z, tau5)
    )

    return energy


def gen_R1(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z):

    R1 = np.zeros_like(np.transpose(f.ov))

    R1 += (
        einsum("ia->ai", f.ov.conj())
    )
    R1 += (
        - einsum("aj,ji->ai", T1, f.oo)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W4.o, tau1)
    )

    R1 += (
        - einsum("am,mi->ai", W1.v, tau2)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    R1 += (
        2 * einsum("m,am,im->ai", tau2, W1.v, W2.o)
    )
    R1 += (
        einsum("bi,ab->ai", T1, f.vv)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    R1 += (
        - einsum("bi,ab->ai", T1, tau3)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    R1 += (
        2 * einsum("bi,ab->ai", T1, tau3)
    )
    tau0 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    R1 += (
        - einsum("aj,ij->ai", T1, tau0)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,mj->ij", W2.o, tau2)
    )

    R1 += (
        einsum("aj,ij->ai", T1, tau3)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,jm->ij", tau2, W1.o, W2.o)
    )

    R1 += (
        - 2 * einsum("aj,ji->ai", T1, tau3)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W1.v, tau0, tau4)
    )

    R1 += (
        einsum("io,oa->ai", Y2, tau5)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W1.v, tau0, tau4)
    )

    R1 += (
        einsum("io,oa->ai", Y4, tau5)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,po,pm,pn,on->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("am,om->oa", W1.v, tau3)
    )

    R1 += (
        - einsum("io,oa->ai", Y4, tau4) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,op,pm,pn,on->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("am,om->oa", W1.v, tau3)
    )

    R1 += (
        - einsum("io,oa->ai", Y2, tau4) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,op,pm,on,pn->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("im,om->oi", W2.o, tau3)
    )

    R1 += (
        einsum("ao,oi->ai", Y1, tau4) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,po,pm,on,pn->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("im,om->oi", W2.o, tau3)
    )

    R1 += (
        einsum("ao,oi->ai", Y3, tau4) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W2.o, tau0, tau4)
    )

    R1 += (
        - einsum("ao,oi->ai", Y1, tau5)
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W2.o, tau0, tau4)
    )

    R1 += (
        - einsum("ao,oi->ai", Y3, tau5)
    )
    tau0 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau1 = (
        einsum("io,oi->o", Y4, tau0)
    )

    tau2 = (
        einsum("p,op->o", tau1, Z)
    )

    R1 += (
        einsum("o,ao,io->ai", tau2, Y1, Y2)
    )
    tau0 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau1 = (
        einsum("io,oi->o", Y2, tau0)
    )

    tau2 = (
        einsum("p,po->o", tau1, Z)
    )

    R1 += (
        einsum("o,ao,io->ai", tau2, Y3, Y4)
    )
    tau0 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau1 = (
        einsum("io,pi->op", Y2, tau0)
    )

    tau2 = (
        einsum("op,ip,op->oi", Z, Y4, tau1)
    )

    R1 += (
        - einsum("ao,oi->ai", Y1, tau2) / 2
    )
    tau0 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau1 = (
        einsum("ip,oi->op", Y4, tau0)
    )

    tau2 = (
        einsum("op,ap,op->oa", Z, Y3, tau1)
    )

    R1 += (
        - einsum("io,oa->ai", Y2, tau2) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("im,ma->ia", W1.o, tau2)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    R1 += (
        einsum("aj,ij->ai", T1, tau4)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W1.o, W2.v)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    R1 += (
        - 2 * einsum("aj,ij->ai", T1, tau4)
    )

    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,po,pm,pn,on->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("im,om->oi", W1.o, tau3)
    )

    tau5 = (
        einsum("jo,oi->ij", Y4, tau4)
    )

    R1 += (
        einsum("aj,ji->ai", T1, tau5) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,op,pm,pn,on->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("im,om->oi", W1.o, tau3)
    )

    tau5 = (
        einsum("jo,oi->ij", Y2, tau4)
    )

    R1 += (
        einsum("aj,ji->ai", T1, tau5) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,po,pm,on,pn->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("am,om->oa", W2.v, tau3)
    )

    tau5 = (
        einsum("bo,oa->ab", Y3, tau4)
    )

    R1 += (
        einsum("bi,ba->ai", T1, tau5) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("nm,op,on,pn,pm->om", X, Z, tau0, tau1, tau2)
    )

    tau4 = (
        einsum("am,om->oa", W4.v, tau3)
    )

    tau5 = (
        einsum("bo,oa->ab", Y1, tau4)
    )

    R1 += (
        einsum("bi,ba->ai", T1, tau5) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ao,ia->oi", Y3, tau3)
    )

    tau5 = (
        einsum("io,pi->op", Y2, tau4)
    )

    tau6 = (
        einsum("op,ip,op->oi", Z, Y4, tau5)
    )

    R1 += (
        einsum("ao,oi->ai", Y1, tau6) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ao,ia->oi", Y1, tau3)
    )

    tau5 = (
        einsum("ip,oi->op", Y4, tau4)
    )

    tau6 = (
        einsum("op,ap,op->oa", Z, Y3, tau5)
    )

    R1 += (
        einsum("io,oa->ai", Y2, tau6) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    R1 += (
        - einsum("aj,ji->ai", T1, tau6)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    R1 += (
        - einsum("aj,ji->ai", T1, tau6)
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W2.v, tau0, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y1, tau5)
    )

    R1 += (
        - einsum("bi,ba->ai", T1, tau6)
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,om,om->oa", W4.v, tau3, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    R1 += (
        - einsum("bi,ba->ai", T1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ao,ia->oi", Y3, tau3)
    )

    tau5 = (
        einsum("io,pi->op", Y2, tau4)
    )

    tau6 = (
        einsum("op,ip,op->oi", Z, Y4, tau5)
    )

    R1 += (
        - einsum("ao,oi->ai", Y1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ao,ia->oi", Y1, tau3)
    )

    tau5 = (
        einsum("ip,oi->op", Y4, tau4)
    )

    tau6 = (
        einsum("op,ap,op->oa", Z, Y3, tau5)
    )

    R1 += (
        - einsum("io,oa->ai", Y2, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ao,ia->oi", Y3, tau3)
    )

    tau5 = (
        einsum("io,oi->o", Y4, tau4)
    )

    tau6 = (
        einsum("p,op->o", tau5, Z)
    )

    R1 += (
        - einsum("o,ao,io->ai", tau6, Y1, Y2)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ao,ia->oi", Y1, tau3)
    )

    tau5 = (
        einsum("io,oi->o", Y2, tau4)
    )

    tau6 = (
        einsum("p,po->o", tau5, Z)
    )

    R1 += (
        - einsum("o,ao,io->ai", tau6, Y3, Y4)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ao,ia->oi", Y3, tau3)
    )

    tau5 = (
        einsum("io,oi->o", Y4, tau4)
    )

    tau6 = (
        einsum("p,op->o", tau5, Z)
    )

    R1 += (
        2 * einsum("o,ao,io->ai", tau6, Y1, Y2)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ao,ia->oi", Y1, tau3)
    )

    tau5 = (
        einsum("io,oi->o", Y2, tau4)
    )

    tau6 = (
        einsum("p,po->o", tau5, Z)
    )

    R1 += (
        2 * einsum("o,ao,io->ai", tau6, Y3, Y4)
    )

    return R1


def gen_RY1(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z, D1, D2, D3, D4):

    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau4)
    )

    RY1 = (
        einsum("aw,owa->ao", D1, tau5)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau5)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau6)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau5)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau5 = (
        einsum("im,omw,omw->owi", W3.o, tau3, tau4)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau5)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau5)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau6)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau7)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau5)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau4 = (
        einsum("nm,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W3.o, tau1, tau5)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau6)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("nm,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W3.o, tau0, tau5)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau6)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau7)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau6)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau7)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau2)
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau5, tau3, tau4) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau2 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau4, tau5) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("an,nm,on->oma", W1.v, X, tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau1, tau5)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D1, D2, Y2, tau6)
    )

    RY1 += (
        einsum("io,op,pia->ao", Y2.conj(), Z.conj(), tau7) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("an,nm,on->oma", W1.v, X, tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau1, tau5)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D1, D2, Y4, tau6)
    )

    RY1 += (
        einsum("io,op,pia->ao", Y2.conj(), Z.conj(), tau7) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau8)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau9)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau8)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau8)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("am,om,omi->oia", W1.v, tau6, tau5)
    )

    tau8 = (
        einsum("iw,io,pow,pia->owa", D2, Y2.conj(), tau3, tau7)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("am,om,omi->oia", W1.v, tau6, tau5)
    )

    tau8 = (
        einsum("iw,io,pow,pia->owa", D2, Y2.conj(), tau3, tau7)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opwi->opmw", tau5, tau3)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw->owa", W1.v, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("omi,opwi->opmw", tau3, tau1)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw->owa", W1.v, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau7)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau7)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau6)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau6)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau4)
    )

    tau6 = (
        einsum("pq,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau4)
    )

    tau6 = (
        einsum("pq,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau7) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau6)
    )

    tau8 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau6)
    )

    tau8 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau4, tau5) / 2
    )
    tau0 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau1 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau1, tau5) / 2
    )
    tau0 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau1 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau1, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau4, tau5) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau6)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau7) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau6)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau7) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W3.o, tau1, tau6)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("mn,onw,onw->omw", X, tau3, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau7)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau8)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau6)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau7)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau6)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau7)
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw,omw->owa", W1.v, tau7, tau9)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw,omw->owa", W1.v, tau7, tau9)
    )

    RY1 += (
        einsum("aw,owa->ao", D1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau4, tau8, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau4, tau5, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,am,bm->ab", tau3, W1.v, W2.v)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau8, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D2, Y4, tau6)
    )

    tau8 = (
        einsum("jo,op,pwji->owi", Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D2, Y2, tau6)
    )

    tau8 = (
        einsum("jo,op,pwji->owi", Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("op,jp,pwi->owij", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D4, Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("op,jp,pwi->owij", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D4, Y2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau6, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau6, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("am,om,omi->oia", W1.v, tau7, tau6)
    )

    tau9 = (
        einsum("iw,io,pow,pia->owa", D2, Y2.conj(), tau3, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("am,om,omi->oia", W1.v, tau7, tau6)
    )

    tau9 = (
        einsum("iw,io,pow,pia->owa", D2, Y2.conj(), tau3, tau8)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau3)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau3)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY1 += (
        - einsum("aw,owa->ao", D1, tau10) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau4, tau8, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau4, tau5, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau8, tau9) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau8)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau9)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau8)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau4, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau4, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,jm->ij", tau6, W1.o, W2.o)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,jm->ij", tau6, W1.o, W2.o)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau3, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y2, tau3)
    )

    tau5 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau1, tau5)
    )

    tau7 = (
        einsum("im,omwj->owij", W1.o, tau6)
    )

    tau8 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W1.o, tau6)
    )

    tau8 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W1.o, tau0, tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("jm,om,omi->oij", W3.o, tau2, tau1)
    )

    tau4 = (
        einsum("op,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("jw,jo,owji->owi", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("jm,om,omi->oij", W3.o, tau2, tau1)
    )

    tau4 = (
        einsum("po,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("jw,jo,owji->owi", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("im,om,omj->oij", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y2, tau3)
    )

    tau5 = (
        einsum("im,om,omj->oij", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau8) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("mn,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("mn,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau9) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jm,owij->omwi", W3.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("omi,pmwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau9) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jm,owij->omwi", W3.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("omi,pmwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,mj->ij", W2.o, tau6)
    )

    tau8 = (
        einsum("jo,ij->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,mj->ij", W2.o, tau6)
    )

    tau8 = (
        einsum("jo,ij->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W4.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W1.o, tau4)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W4.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W1.o, tau4)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau6 = (
        einsum("pi,pow,pow->owi", tau5, tau3, tau4)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau6 = (
        einsum("pi,pow,pow->owi", tau5, tau0, tau4)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("qo,qpw->opw", Z, tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau6) / 2
    )
    tau0 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau1 = (
        einsum("jo,ij->oi", Y2, tau0)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau3, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau2, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau5 = (
        einsum("jo,ij->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau3, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau3 = (
        einsum("jo,ij->oi", Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau3 = (
        einsum("jo,ij->oi", Y2, tau2)
    )

    tau4 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("qo,qpw->opw", Z, tau5)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau6) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("mn,onw,onw->omw", X, tau3, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8)
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W1.o, tau7, tau9)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W1.o, tau7, tau9)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau3, tau4)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY1 += (
        - einsum("aw,ai,owi->ao", D1, T1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("im,mi->m", W1.o, tau4)
    )

    tau6 = (
        einsum("n,nm->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,am->ia", tau6, W3.o, W4.v)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau10, tau3)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("im,mi->m", W1.o, tau4)
    )

    tau6 = (
        einsum("n,nm->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,am->ia", tau6, W3.o, W4.v)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau10, tau3)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("jm,omi,omk->oijk", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,iljk->owij", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,omj,omk->oijk", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,ilkj->owij", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau7)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau6, tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau6, tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("op,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("jw,jo,owji->owi", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("po,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("jw,jo,owji->owi", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau2, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    tau9 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau2, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    tau9 = (
        einsum("jw,jo,op,pwij->owi", D2, Y2.conj(), Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau3, tau4)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY1 += (
        einsum("aw,ai,owi->ao", D1, T1, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y2, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("ip,po,pm->omi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau10) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau4, tau6)
    )

    tau8 = (
        einsum("op,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau4, tau6)
    )

    tau8 = (
        einsum("op,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("op,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("po,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("am,mi->ia", W2.v, tau6)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau10, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("am,mi->ia", W2.v, tau6)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau10, tau3) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,ip,pq,oqwi->opw", D2, Y2.conj(), Z.conj(), tau8)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("pm,pm,om->op", tau4, tau5, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,om,om->oa", W2.v, tau1, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y1, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,om,pm->opi", W3.o, tau5, tau6)
    )

    tau8 = (
        einsum("io,pqi->opq", Y2, tau7)
    )

    tau9 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("am,om,om->oa", W2.v, tau5, tau9)
    )

    tau11 = (
        einsum("bo,oa->ab", Y1, tau10)
    )

    tau12 = (
        einsum("bo,ba->oa", Y3, tau11)
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau12, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("im,om,pm->opi", W1.o, tau6, tau7)
    )

    tau9 = (
        einsum("io,oq,pqi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,opq->opi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D2, Y2.conj(), tau10, tau2)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau2, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,om,om->oa", W4.v, tau4, tau5)
    )

    tau7 = (
        einsum("ap,oa->op", Y3, tau6)
    )

    tau8 = (
        einsum("ap,po->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("am,om,om->oa", W2.v, tau5, tau9)
    )

    tau11 = (
        einsum("bo,oa->ab", Y1, tau10)
    )

    tau12 = (
        einsum("bo,ba->oa", Y1, tau11)
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau12, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,pm->opi", W3.o, tau2, tau6)
    )

    tau8 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,qop->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau5, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,pm,om->opi", W1.o, tau3, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D2, Y2.conj(), tau10, tau2)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y1, tau11) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau3, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y1, tau12, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("nm,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("am,om,om->oa", W4.v, tau8, tau9)
    )

    tau11 = (
        einsum("ap,oa->op", Y3, tau10)
    )

    tau12 = (
        einsum("ap,po->oa", Y3, tau11)
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau12, tau3, tau4) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("io,opi->op", Y2, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau9 = (
        einsum("im,om,om->oi", W1.o, tau7, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y4, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau12, tau3) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,pm,om->opi", W1.o, tau6, tau7)
    )

    tau9 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D2, Y2.conj(), tau10, tau2)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,om,om->oa", W2.v, tau2, tau6)
    )

    tau8 = (
        einsum("bo,oa->ab", Y1, tau7)
    )

    tau9 = (
        einsum("bo,ba->oa", Y1, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,pm,om->opi", W3.o, tau2, tau6)
    )

    tau8 = (
        einsum("io,pqi->opq", Y2, tau7)
    )

    tau9 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,om,pm->opi", W1.o, tau3, tau7)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D2, Y2.conj(), tau10, tau2)
    )

    RY1 += (
        - einsum("aw,ap,pow->ao", D1, Y3, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("nm,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("am,om,om->oa", W4.v, tau8, tau9)
    )

    tau11 = (
        einsum("ao,pa->op", Y1, tau10)
    )

    tau12 = (
        einsum("ap,op->oa", Y3, tau11)
    )

    RY1 += (
        - einsum("aw,pa,pow,pow->ao", D1, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,pm,om->opi", W3.o, tau5, tau6)
    )

    tau8 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,qop->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,om,om->oa", W4.v, tau5, tau6)
    )

    tau8 = (
        einsum("ao,pa->op", Y1, tau7)
    )

    tau9 = (
        einsum("ap,op->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y2, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY1 += (
        - einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W4.v, tau4)
    )

    tau6 = (
        einsum("ap,oa->op", Y3, tau5)
    )

    tau7 = (
        einsum("ap,po->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("io,op,qpi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,oqp->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("io,op,qpi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,oqp->opi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D2, Y2.conj(), tau10, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W4.v, tau8)
    )

    tau10 = (
        einsum("ap,oa->op", Y3, tau9)
    )

    tau11 = (
        einsum("ap,po->oa", Y1, tau10)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau11, tau3, tau4) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("io,pmi->opm", Y2, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,opwi->opmw", W3.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("opm,pqmw,oqmw->opqw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("iw,iq,oqpw->opwi", D2, Y4, tau9)
    )

    tau11 = (
        einsum("ip,pq,oqwi->opw", Y2.conj(), Z.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau3, tau6)
    )

    tau8 = (
        einsum("mn,onw->omw", X, tau7)
    )

    tau9 = (
        einsum("op,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau9)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("po,poi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("im,opwi->opmw", W3.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("ip,omi->opm", Y2, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("opm,oqmw,pqmw->opqw", tau5, tau2, tau8)
    )

    tau10 = (
        einsum("iw,iq,qopw->opwi", D2, Y2, tau9)
    )

    tau11 = (
        einsum("ip,pq,oqwi->opw", Y2.conj(), Z.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,iljk->oijk", Y4, tau7)
    )

    tau9 = (
        einsum("jw,kw,jp,opwk,oijk->opwi", D2, D4, Y2.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau10) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,opq->opi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D2, Y2.conj(), tau10, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau12, tau8) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,opi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau12) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W2.v, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W4.v, tau8)
    )

    tau10 = (
        einsum("ao,pa->op", Y1, tau9)
    )

    tau11 = (
        einsum("ap,op->oa", Y1, tau10)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y1, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau7, tau6, tau9)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,iljk->oijk", Y4, tau7)
    )

    tau9 = (
        einsum("jw,kw,jp,opwk,oikj->opwi", D2, D4, Y2.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y1, tau10) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("po,poi->oi", Z, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y4, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau12, tau3) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("po,poi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W2.v, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y3, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y3, tau10)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau11, tau3, tau4) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau3, tau6)
    )

    tau8 = (
        einsum("mn,onw->omw", X, tau7)
    )

    tau9 = (
        einsum("op,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau9)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,lijk->oijk", Y2, tau7)
    )

    tau9 = (
        einsum("jw,kw,jp,opwk,oikj->opwi", D2, D4, Y2.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau10) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D2, Y2.conj(), tau10, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,lijk->oijk", Y2, tau7)
    )

    tau9 = (
        einsum("jw,kw,jp,opwk,oijk->opwi", D2, D4, Y2.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau9)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau10) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,opi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau12) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,pa->op", Y1, tau6)
    )

    tau8 = (
        einsum("ap,op->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("im,opwi->opmw", W3.o, tau7)
    )

    tau9 = (
        einsum("opm,pqmw,oqmw->opqw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("iw,iq,qopw->opwi", D2, Y4, tau9)
    )

    tau11 = (
        einsum("ip,pq,oqwi->opw", Y2.conj(), Z.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,opq->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau9, tau6, tau8)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D2, Y2.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W2.v, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y3, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y1, tau10)
    )

    RY1 += (
        einsum("aw,pa,pow,pow->ao", D1, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,oqp->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W2.v, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y3, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY1 += (
        einsum("aw,ap,pow,pow->ao", D1, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,oqp->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D2, Y2.conj(), tau10, tau3)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("im,opwi->opmw", W3.o, tau7)
    )

    tau9 = (
        einsum("opm,pqmw,oqmw->opqw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("iw,iq,qopw->opwi", D2, Y2, tau9)
    )

    tau11 = (
        einsum("ip,pq,oqwi->opw", Y2.conj(), Z.conj(), tau10)
    )

    RY1 += (
        einsum("aw,ap,pow->ao", D1, Y3, tau11) / 4
    )

    return RY1

def gen_RY2(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z, D1, D2, D3, D4):

    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau4)
    )

    RY2 = (
        einsum("iw,owi->io", D2, tau5)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W4.o, tau3, tau5)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau5)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("mi,omw,omw->owi", tau5, tau0, tau4)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau5)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau0, tau5)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W2.o, tau4, tau6)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau3, tau5)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W4.o, tau4, tau6)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau0, tau5)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau6)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau7)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("an,nm,on->oma", W1.v, X, tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau1, tau5)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D1, D2, Y2, tau6)
    )

    tau8 = (
        einsum("ao,pia->opi", Y1.conj(), tau7)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("an,nm,on->oma", W1.v, X, tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau1, tau5)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D1, D2, Y4, tau6)
    )

    tau8 = (
        einsum("ao,pia->opi", Y1.conj(), tau7)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau7)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau7)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau4, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau5, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau4, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau2)
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau5) / 2
    )
    tau0 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau1 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau1, tau5) / 2
    )
    tau0 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau1 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau1, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau2 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau1)
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau4, tau5) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau8)
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    tau8 = (
        einsum("aw,iw,pia,pow->oia", D1, D2, tau3, tau7)
    )

    RY2 += (
        - einsum("ao,oia->io", Y1.conj(), tau8) / 2
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    tau8 = (
        einsum("aw,iw,pia,pow->oia", D1, D2, tau3, tau7)
    )

    RY2 += (
        - einsum("ao,oia->io", Y1.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("omw,omwi->owi", tau0, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau1)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("omi,opwi->opmw", tau4, tau2)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("ip,po,pm->omi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau7, tau5)
    )

    tau9 = (
        einsum("omw,omwi->owi", tau0, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau7)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau7)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau7)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau7)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau4)
    )

    tau6 = (
        einsum("pq,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau4)
    )

    tau6 = (
        einsum("pq,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau7)
    )

    tau9 = (
        einsum("pow,powi->owi", tau0, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau7)
    )

    tau9 = (
        einsum("pow,powi->owi", tau0, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau4, tau6)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau8)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau0, tau6)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau8)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau4, tau6)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau8)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau1, tau3)
    )

    tau5 = (
        einsum("op,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W2.o, tau5, tau7)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau8)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau2 = (
        einsum("oi,pwi->opw", tau1, tau0)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau3, tau4)
    )

    tau6 = (
        einsum("qo,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau2, tau6) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau2 = (
        einsum("oi,pwi->opw", tau1, tau0)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau3, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau2, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("qo,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau6, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau6, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau3 = (
        einsum("jo,ij->oi", Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau3 = (
        einsum("jo,ij->oi", Y2, tau2)
    )

    tau4 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau3)
    )

    tau5 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("qo,qpw->opw", Z, tau5)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau6) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W2.o, tau7, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W2.o, tau7, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau4, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau4, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,jm->ij", tau7, W1.o, W2.o)
    )

    tau9 = (
        einsum("jo,ji->oi", Y2, tau8)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau9, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,jm->ij", tau7, W1.o, W2.o)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau9, tau3, tau4)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y2, tau3)
    )

    tau5 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau2, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau1, tau5)
    )

    tau7 = (
        einsum("im,omwj->owij", W1.o, tau6)
    )

    tau8 = (
        einsum("op,pwij->owij", Z.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owji->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W1.o, tau6)
    )

    tau8 = (
        einsum("op,pwij->owij", Z.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owji->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau6, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau6, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("jm,om,omi->oij", W3.o, tau2, tau1)
    )

    tau4 = (
        einsum("op,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owij->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("jm,om,omi->oij", W3.o, tau2, tau1)
    )

    tau4 = (
        einsum("po,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owij->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("im,om,omj->oij", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y2, tau3)
    )

    tau5 = (
        einsum("im,om,omj->oij", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("pq,oqw->opw", Z.conj(), tau7)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau7 = (
        einsum("mn,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("mn,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("pow,powi->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W3.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jm,owij->omwi", W3.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("omi,pmwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau1, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("pow,powi->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau7 = (
        einsum("jo,mji->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau7 = (
        einsum("jo,mji->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("im,mj->ij", W2.o, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y2, tau8)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau9, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("im,mj->ij", W2.o, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y4, tau8)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau9, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W4.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W1.o, tau4)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W4.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W1.o, tau4)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau0, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau0, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,am,bm->ab", tau6, W1.v, W2.v)
    )

    tau8 = (
        einsum("bo,ab->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,am,bm->ab", tau6, W1.v, W2.v)
    )

    tau8 = (
        einsum("bo,ab->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,am,bm->ab", tau3, W1.v, W2.v)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau8, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D2, Y4, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("owj,pwij->opi", tau8, tau7)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D2, Y2, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("owj,pwij->opi", tau8, tau7)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("op,jp,pwi->owij", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D4, Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau10 = (
        einsum("omw,omwi->owi", tau9, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("op,jp,pwi->owij", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D4, Y2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau10 = (
        einsum("omw,omwi->owi", tau9, tau8)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau8, tau0, tau7)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau8, tau0, tau7)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,iw,pia,pow->oia", D1, D2, tau4, tau8)
    )

    RY2 += (
        - einsum("ao,oia->io", Y1.conj(), tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,iw,pia,pow->oia", D1, D2, tau4, tau8)
    )

    RY2 += (
        - einsum("ao,oia->io", Y1.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau8)
    )

    RY2 += (
        - einsum("op,opi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau4)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("iw,pq,iq,oqw->opwi", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau4)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("in,mi->mn", W3.o, tau4)
    )

    tau6 = (
        einsum("mn,an,mn->ma", X, W4.v, tau5)
    )

    tau7 = (
        einsum("am,mb->ab", W1.v, tau6)
    )

    tau8 = (
        einsum("bo,ab->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("in,mi->mn", W3.o, tau4)
    )

    tau6 = (
        einsum("mn,an,mn->ma", X, W4.v, tau5)
    )

    tau7 = (
        einsum("am,mb->ab", W1.v, tau6)
    )

    tau8 = (
        einsum("bo,ab->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau8, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau0, tau6, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    tau10 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau11 = (
        einsum("mi,omw,omw->owi", tau10, tau7, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau11)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    tau10 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau11 = (
        einsum("mi,omw,omw->owi", tau10, tau7, tau9)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau11)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau10, tau3)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau10, tau3)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W3.o, W4.v)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y2, tau9)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau10, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W3.o, W4.v)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y4, tau9)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau10, tau3, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("jm,omi,omk->oijk", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,iljk->owij", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("op,pwij->owij", Z.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owji->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,omj,omk->oijk", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,ilkj->owij", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("op,pwij->owij", Z.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau10 = (
        einsum("owj,owji->owi", tau9, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau6, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("op,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau6, tau8)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("op,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau11 = (
        einsum("owj,owij->owi", tau10, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("po,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau8 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau11 = (
        einsum("owj,owij->owi", tau10, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    tau10 = (
        einsum("op,pwij->owij", Z.conj(), tau9)
    )

    tau11 = (
        einsum("owj,owji->owi", tau0, tau10)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    tau10 = (
        einsum("op,pwij->owij", Z.conj(), tau9)
    )

    tau11 = (
        einsum("owj,owji->owi", tau0, tau10)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau10, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau10, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau4)
    )

    tau6 = (
        einsum("jm,owij->omwi", W1.o, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("op,ip,pm->omi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("omi,pmwi->opmw", tau8, tau6)
    )

    tau10 = (
        einsum("omi,opmw->opwi", tau3, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("iw,op,ip,pwj->owij", D4, Z.conj(), Y4.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("ip,po,pm->omi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau11 = (
        einsum("pow,powi->owi", tau10, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("nm,onw,onw->omw", X, tau5, tau7)
    )

    tau9 = (
        einsum("op,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("nm,onw,onw->omw", X, tau5, tau7)
    )

    tau9 = (
        einsum("op,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("op,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("po,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("pq,oqw->opw", Z.conj(), tau8)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y2, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    tau10 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau9)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y4, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    tau10 = (
        einsum("iw,qow,qpwi->opi", D2, tau0, tau9)
    )

    RY2 += (
        einsum("op,opi->io", Z.conj(), tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W2.v, tau7)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y2, tau9)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau10, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W2.v, tau7)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y4, tau9)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau10, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("pm,pm,om->op", tau4, tau5, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,om,om->oa", W2.v, tau1, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y1, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,om,pm->opi", W3.o, tau5, tau6)
    )

    tau8 = (
        einsum("io,pqi->opq", Y2, tau7)
    )

    tau9 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau2, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,om,om->oa", W4.v, tau4, tau5)
    )

    tau7 = (
        einsum("ap,oa->op", Y3, tau6)
    )

    tau8 = (
        einsum("ap,po->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", tau11, tau3)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,pm->opi", W3.o, tau2, tau6)
    )

    tau8 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,qop->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau5, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau8, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y4, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y2, tau11)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,pm,om->opi", W1.o, tau4, tau8)
    )

    tau10 = (
        einsum("io,pqi->opq", Y2, tau9)
    )

    tau11 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau3)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau9 = (
        einsum("im,om,pm->opi", W1.o, tau7, tau8)
    )

    tau10 = (
        einsum("io,oq,pqi->opq", Y2, Z, tau9)
    )

    tau11 = (
        einsum("iq,opq->opi", Y4, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau3)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,om,om->oa", W2.v, tau4, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y1, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y3, tau10)
    )

    tau12 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau12, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau5, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y2, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y2, tau11)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W2.v, tau0, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y1, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau9 = (
        einsum("am,om,om->oa", W4.v, tau7, tau8)
    )

    tau10 = (
        einsum("ap,oa->op", Y3, tau9)
    )

    tau11 = (
        einsum("ap,po->oa", Y3, tau10)
    )

    tau12 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau12, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,om,om->oa", W2.v, tau2, tau6)
    )

    tau8 = (
        einsum("bo,oa->ab", Y1, tau7)
    )

    tau9 = (
        einsum("bo,ba->oa", Y1, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,pm,om->opi", W3.o, tau2, tau6)
    )

    tau8 = (
        einsum("io,pqi->opq", Y2, tau7)
    )

    tau9 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("io,opi->op", Y2, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau8, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y4, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y4, tau11)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau12, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau5, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y2, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y4, tau11)
    )

    RY2 += (
        - einsum("iw,pi,pow,pow->io", D2, tau12, tau3, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau9 = (
        einsum("im,pm,om->opi", W1.o, tau7, tau8)
    )

    tau10 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iq,qop->opi", Y2, tau10)
    )

    tau12 = (
        einsum("qoi,qpw->opwi", tau11, tau3)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,om,pm->opi", W1.o, tau4, tau8)
    )

    tau10 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iq,qop->opi", Y2, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau3)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        - einsum("iw,owi->io", D2, tau13) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,om,om->oa", W4.v, tau3, tau4)
    )

    tau6 = (
        einsum("ao,pa->op", Y1, tau5)
    )

    tau7 = (
        einsum("ap,op->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y2, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,pm,om->opi", W3.o, tau5, tau6)
    )

    tau8 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,qop->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("pq,oqw->opw", Z.conj(), tau10)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,om,om->oa", W4.v, tau5, tau6)
    )

    tau8 = (
        einsum("ao,pa->op", Y1, tau7)
    )

    tau9 = (
        einsum("ap,op->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y2, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", tau11, tau3)
    )

    RY2 += (
        - einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W4.v, tau4)
    )

    tau6 = (
        einsum("ap,oa->op", Y3, tau5)
    )

    tau7 = (
        einsum("ap,po->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("io,op,qpi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,oqp->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau5, tau4, tau7)
    )

    tau9 = (
        einsum("mn,onw->omw", X, tau8)
    )

    tau10 = (
        einsum("op,pmw->omw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("om,pmw,omi->opwi", tau1, tau10, tau3)
    )

    tau12 = (
        einsum("pow,powi->owi", tau0, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("po,poi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,opq->opi", Y4, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    tau12 = (
        einsum("qpw,qoi->opwi", tau11, tau7)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau13) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,opi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W2.v, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,omj,omk->oijk", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,iljk->oijk", Y4, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau8 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powi->owij", tau10, tau6, tau8)
    )

    RY2 += (
        einsum("iw,jw,owji->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau10 = (
        einsum("op,ip,pm->omi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("om,omi,pmw->opwi", tau8, tau10, tau7)
    )

    tau12 = (
        einsum("pow,powi->owi", tau0, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("om,pm,omi->opi", tau7, tau8, tau6)
    )

    tau10 = (
        einsum("io,op,qpi->opq", Y2, Z, tau9)
    )

    tau11 = (
        einsum("iq,oqp->opi", Y4, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau4)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau13) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau6, tau5)
    )

    tau8 = (
        einsum("am,om->oa", W4.v, tau7)
    )

    tau9 = (
        einsum("ap,oa->op", Y3, tau8)
    )

    tau10 = (
        einsum("ap,po->oa", Y1, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau11, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau0)
    )

    tau2 = (
        einsum("iw,ip,oqwi->opqw", D4, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,opwi->opmw", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("mn,in,on->omi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("io,pmi->opm", Y2, tau7)
    )

    tau9 = (
        einsum("oqm,oqpw,qpmw->opmw", tau8, tau2, tau5)
    )

    tau10 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau11 = (
        einsum("op,ip,pm->omi", Z, Y4, tau10)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau11, tau9)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("jm,omi,omk->oijk", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,iljk->oijk", Y4, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau8 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powj->owij", tau10, tau6, tau8)
    )

    RY2 += (
        einsum("iw,jw,owij->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("ip,omi->opm", Y4, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("im,opwi->opmw", W1.o, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau8)
    )

    tau10 = (
        einsum("iw,ip,oqwi->opqw", D4, Y4, tau9)
    )

    tau11 = (
        einsum("qom,oqpw,qpmw->opmw", tau4, tau10, tau7)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau1, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W4.v, tau3)
    )

    tau5 = (
        einsum("ao,pa->op", Y1, tau4)
    )

    tau6 = (
        einsum("ap,op->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau11, tau7) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau7, tau11, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("jm,omi,omk->oijk", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,lijk->oijk", Y2, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau8 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powi->owij", tau10, tau6, tau8)
    )

    RY2 += (
        einsum("iw,jw,owji->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("po,poi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau6, tau5)
    )

    tau8 = (
        einsum("am,om->oa", W2.v, tau7)
    )

    tau9 = (
        einsum("bo,oa->ab", Y3, tau8)
    )

    tau10 = (
        einsum("bo,ba->oa", Y3, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau11, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,pa->op", Y1, tau6)
    )

    tau8 = (
        einsum("ap,op->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,opi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("iw,ip,oqwi->opqw", D4, Y2, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("im,opwi->opmw", W1.o, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau9 = (
        einsum("mn,in,on->omi", X, W3.o, tau8)
    )

    tau10 = (
        einsum("ip,omi->opm", Y2, tau9)
    )

    tau11 = (
        einsum("qom,oqpw,qpmw->opmw", tau10, tau4, tau7)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau1, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,opq->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau5, tau4, tau7)
    )

    tau9 = (
        einsum("mn,onw->omw", X, tau8)
    )

    tau10 = (
        einsum("op,pmw->omw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("om,pmw,omi->opwi", tau3, tau10, tau2)
    )

    tau12 = (
        einsum("pow,powi->owi", tau0, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RY2 += (
        einsum("iw,pi,pow,pow->io", D2, tau7, tau11, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau9 = (
        einsum("pm,om,pmi->opi", tau7, tau8, tau6)
    )

    tau10 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iq,qop->opi", Y2, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau4)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau13) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,omj,omk->oijk", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,lijk->oijk", Y2, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau8 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powj->owij", tau10, tau6, tau8)
    )

    RY2 += (
        einsum("iw,jw,owij->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("pq,iq,oqw->opwi", Z.conj(), Y4.conj(), tau2)
    )

    tau4 = (
        einsum("iw,io,pqwi->opqw", D4, Y2, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("im,opwi->opmw", W1.o, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau9 = (
        einsum("mn,in,on->omi", X, W3.o, tau8)
    )

    tau10 = (
        einsum("ip,omi->opm", Y4, tau9)
    )

    tau11 = (
        einsum("qom,qopw,qpmw->opmw", tau10, tau4, tau7)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau1, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("op,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau9 = (
        einsum("ip,po,pm->omi", Y2, Z, tau8)
    )

    tau10 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau11 = (
        einsum("om,pmw,omi->opwi", tau10, tau7, tau9)
    )

    tau12 = (
        einsum("pow,powi->owi", tau0, tau11)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W2.v, tau3)
    )

    tau5 = (
        einsum("bo,oa->ab", Y3, tau4)
    )

    tau6 = (
        einsum("bo,ba->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y2, tau11, tau7) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,oqp->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("pq,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W2.v, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y3, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY2 += (
        einsum("iw,ip,pow,pow->io", D2, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y2, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau10 = (
        einsum("pq,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    tau12 = (
        einsum("qpw,qoi->opwi", tau11, tau7)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY2 += (
        einsum("iw,owi->io", D2, tau13) / 4
    )

    return RY2

def gen_RY3(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z, D1, D2, D3, D4):

    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("am,omw,omw->owa", W3.v, tau0, tau4)
    )

    RY3 = (
        einsum("aw,owa->ao", D3, tau5)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("am,omw,omw->owa", W3.v, tau4, tau5)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("am,omw,omw->owa", W3.v, tau3, tau5)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau6)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau5)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau6)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("im,omw,omw->owi", W3.o, tau0, tau4)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau5)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W1.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("im,omw,omw->owi", W3.o, tau4, tau5)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W3.v, tau1, tau6)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau7)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau6)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau7)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("am,omw,omw->owa", W1.v, tau4, tau6)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau6 = (
        einsum("im,omw,omw->owi", W3.o, tau4, tau5)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W3.o, tau3, tau5)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau6)
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau6)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau7) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau6)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau8)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau8)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau8)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau9)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau8)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("qp,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("qp,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau6)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau6)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau7) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau6)
    )

    tau8 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau6)
    )

    tau8 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opwi->opmw", tau5, tau3)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw->owa", W1.v, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("omi,opwi->opmw", tau3, tau1)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw->owa", W1.v, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("am,om,omi->oia", W1.v, tau6, tau5)
    )

    tau8 = (
        einsum("iw,io,pow,pia->owa", D4, Y4.conj(), tau3, tau7)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("am,om,omi->oia", W1.v, tau6, tau5)
    )

    tau8 = (
        einsum("iw,io,pow,pia->owa", D4, Y4.conj(), tau3, tau7)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau7)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau0, tau7)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau8) / 2
    )
    tau0 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau1 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau1, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau5) / 2
    )
    tau0 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau1 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau1, tau5) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y4, tau6)
    )

    RY3 += (
        einsum("po,io,pia->ao", Z.conj(), Y4.conj(), tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y2, tau6)
    )

    RY3 += (
        einsum("po,io,pia->ao", Z.conj(), Y4.conj(), tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau1, tau3)
    )

    tau5 = (
        einsum("po,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("am,omw,omw->owa", W1.v, tau5, tau7)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau8)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W3.o, tau1, tau6)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau7)
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W3.o, tau0, tau6)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W3.o, tau1, tau6)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau7)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W1.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W1.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau9 = (
        einsum("im,omw,omw->owi", W3.o, tau7, tau8)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau9 = (
        einsum("im,omw,omw->owi", W3.o, tau7, tau8)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,jm->ij", tau2, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,ji->oi", Y4, tau3)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau5, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,jm->ij", tau2, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,ji->oi", Y2, tau3)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau5, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W1.o, W2.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W1.o, W2.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("jm,owij->omwi", W1.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("omi,pmwi->opmw", tau4, tau2)
    )

    tau6 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau7 = (
        einsum("jo,mji->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("jm,owij->omwi", W1.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("omi,pmwi->opmw", tau4, tau2)
    )

    tau6 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau7 = (
        einsum("jo,mji->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau9) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W3.o, tau4, tau3)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W3.o, tau4, tau3)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("im,om,omj->oij", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("op,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("jw,jo,owij->owi", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("im,om,omj->oij", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("po,pij->oij", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pij,pow->owij", tau4, tau7)
    )

    tau9 = (
        einsum("jw,jo,owij->owi", D4, Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("im,omw,omw->owi", W3.o, tau6, tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau8 = (
        einsum("im,omw,omw->owi", W3.o, tau6, tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W3.o, tau6)
    )

    tau8 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W3.o, tau6)
    )

    tau8 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau8) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W2.o, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W2.o, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W4.o, tau1)
    )

    tau3 = (
        einsum("im,mj->ij", W1.o, tau2)
    )

    tau4 = (
        einsum("jo,ji->oi", Y4, tau3)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau5, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W4.o, tau1)
    )

    tau3 = (
        einsum("im,mj->ij", W1.o, tau2)
    )

    tau4 = (
        einsum("jo,ji->oi", Y2, tau3)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau5, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau5, tau8)
    )

    tau10 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau9)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau10)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau8, tau4)
    )

    tau10 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau9)
    )

    RY3 += (
        einsum("aw,owa->ao", D3, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau4, tau5, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau4, tau5, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("jp,po,pwi->owij", Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D2, Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("jp,po,pwi->owij", Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("jw,jo,pwij->opwi", D2, Y2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D4, Y4, tau6)
    )

    tau8 = (
        einsum("po,jo,pwji->owi", Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("iw,ip,powj->owij", D4, Y2, tau6)
    )

    tau8 = (
        einsum("po,jo,pwji->owi", Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau7)
    )

    tau9 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau5)
    )

    tau7 = (
        einsum("qp,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau5)
    )

    tau7 = (
        einsum("qp,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau3)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau3)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau1, tau7)
    )

    tau9 = (
        einsum("iw,io,omwi->omw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("am,omw->owa", W1.v, tau9)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau2, tau5)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau2, tau4)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("am,omw,omw->owa", W1.v, tau1, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("am,om,omi->oia", W1.v, tau7, tau6)
    )

    tau9 = (
        einsum("iw,io,pow,pia->owa", D4, Y4.conj(), tau3, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("am,om,omi->oia", W1.v, tau7, tau6)
    )

    tau9 = (
        einsum("iw,io,pow,pia->owa", D4, Y4.conj(), tau3, tau8)
    )

    RY3 += (
        - einsum("aw,owa->ao", D3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau4, tau5, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau4, tau5, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau3 = (
        einsum("oi,pwi->opw", tau2, tau1)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau5, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau6 = (
        einsum("pi,pow,pow->owi", tau5, tau0, tau4)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau6 = (
        einsum("pi,pow,pow->owi", tau5, tau0, tau4)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y2, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau5 = (
        einsum("jo,ij->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau3, tau6) / 2
    )
    tau0 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau1 = (
        einsum("jo,ij->oi", Y2, tau0)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau3, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau2, tau6) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau3, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W3.o, tau1, tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau5, tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau9)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau10)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau8, tau4)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau9)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY3 += (
        - einsum("aw,ai,owi->ao", D3, T1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau10)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    tau5 = (
        einsum("jo,ij->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau6)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau10)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("im,mi->m", W1.o, tau0)
    )

    tau2 = (
        einsum("n,nm->m", tau1, X)
    )

    tau3 = (
        einsum("m,im,am->ia", tau2, W3.o, W4.v)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    tau5 = (
        einsum("jo,ij->oi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau6)
    )
    tau0 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau4, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau0, tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau4, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau2, tau1, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("op,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("po,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y2, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("jm,owij->omwi", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("ip,po,pm->omi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("omi,pmwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau2, tau8)
    )

    tau10 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("jm,omi,omk->oijk", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,likj->owij", D2, Y2.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,omj,omk->oijk", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ko,oijl->ijkl", Y2, tau5)
    )

    tau7 = (
        einsum("kw,ko,owl,lijk->owij", D2, Y2.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau7)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau8) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau2, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    tau9 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau2, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau1, tau6)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    tau9 = (
        einsum("jw,po,jo,pwij->owi", D4, Z.conj(), Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau2, tau5)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau2, tau4)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W1.o, tau1, tau8)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("op,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("jw,jo,owji->owi", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau10) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("jm,om,omi->oij", W1.o, tau0, tau3)
    )

    tau5 = (
        einsum("po,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("jw,jo,owji->owi", D4, Y4.conj(), tau9)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y1, tau8)
    )

    tau10 = (
        einsum("pi,pow,pow->owi", tau9, tau0, tau4)
    )

    RY3 += (
        einsum("aw,ai,owi->ao", D3, T1, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau10) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau10) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qp,ip,oqwi->opw", D4, Z.conj(), Y4.conj(), tau8)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    tau5 = (
        einsum("jo,ij->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau6) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,ni->mn", W1.o, tau0)
    )

    tau2 = (
        einsum("mn,in,mn->mi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("am,mi->ia", W2.v, tau2)
    )

    tau4 = (
        einsum("ai,ja->ij", T1, tau3)
    )

    tau5 = (
        einsum("jo,ij->oi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("oq,pq->op", Z, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("pm,pm,om->op", tau3, tau4, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("qp,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qo,qp->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,om,om->oi", W1.o, tau4, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y2, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau12, tau3) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau9 = (
        einsum("im,om,om->oi", W1.o, tau7, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y4, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau12, tau3) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y4, tau7)
    )

    tau9 = (
        einsum("oq,pq->op", Z, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau11, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("am,om,om->oa", W2.v, tau5, tau9)
    )

    tau11 = (
        einsum("bo,oa->ab", Y1, tau10)
    )

    tau12 = (
        einsum("bo,ba->oa", Y3, tau11)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("io,oq,pqi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,opq->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,om,om->oa", W2.v, tau1, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y1, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,om,om->oa", W2.v, tau2, tau6)
    )

    tau8 = (
        einsum("bo,oa->ab", Y1, tau7)
    )

    tau9 = (
        einsum("bo,ba->oa", Y1, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("nm,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("am,om,om->oa", W4.v, tau8, tau9)
    )

    tau11 = (
        einsum("ap,oa->op", Y3, tau10)
    )

    tau12 = (
        einsum("ap,po->oa", Y3, tau11)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y4, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("io,pqi->opq", Y2, tau6)
    )

    tau8 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau11) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau2, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,om,om->oa", W4.v, tau4, tau5)
    )

    tau7 = (
        einsum("ap,oa->op", Y3, tau6)
    )

    tau8 = (
        einsum("ap,po->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("am,om,om->oa", W2.v, tau5, tau9)
    )

    tau11 = (
        einsum("bo,oa->ab", Y1, tau10)
    )

    tau12 = (
        einsum("bo,ba->oa", Y1, tau11)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y1, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("pm,om,pmi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("io,opi->op", Y2, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau11, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau5, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau3, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y1, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y2, tau7)
    )

    tau9 = (
        einsum("qp,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qo,qp->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau11, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,om,om->oa", W4.v, tau5, tau6)
    )

    tau8 = (
        einsum("ao,pa->op", Y1, tau7)
    )

    tau9 = (
        einsum("ap,op->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        - einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("nm,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("am,om,om->oa", W4.v, tau8, tau9)
    )

    tau11 = (
        einsum("ao,pa->op", Y1, tau10)
    )

    tau12 = (
        einsum("ap,op->oa", Y3, tau11)
    )

    RY3 += (
        - einsum("aw,pa,pow,pow->ao", D3, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RY3 += (
        - einsum("aw,ap,pow->ao", D3, Y3, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("po,poi->oi", Z, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y4, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau12, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,lijk->oijk", Y2, tau7)
    )

    tau9 = (
        einsum("jw,kw,kp,opwj,oijk->opwi", D2, D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau10) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("po,poi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("op,opi->oi", Z, tau8)
    )

    tau10 = (
        einsum("jo,oi->ij", Y2, tau9)
    )

    tau11 = (
        einsum("jo,ji->oi", Y4, tau10)
    )

    tau12 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau12, tau3) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,opi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,lijk->oijk", Y2, tau7)
    )

    tau9 = (
        einsum("jw,kw,kp,opwj,oikj->opwi", D2, D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau10) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W4.v, tau8)
    )

    tau10 = (
        einsum("ap,oa->op", Y3, tau9)
    )

    tau11 = (
        einsum("ap,po->oa", Y1, tau10)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("io,op,qpi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,oqp->opi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau3)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W4.v, tau4)
    )

    tau6 = (
        einsum("ap,oa->op", Y3, tau5)
    )

    tau7 = (
        einsum("ap,po->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("io,op,qpi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau7, tau6, tau9)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("io,pmi->opm", Y2, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,ip,oqwi->opqw", D2, Y4, tau7)
    )

    tau9 = (
        einsum("qom,qpmw,qopw->opmw", tau5, tau2, tau8)
    )

    tau10 = (
        einsum("im,opmw->opwi", W3.o, tau9)
    )

    tau11 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("op,ip,pm->omi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("ip,omi->opm", Y2, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,ip,oqwi->opqw", D2, Y2, tau7)
    )

    tau9 = (
        einsum("oqm,qpmw,qopw->opmw", tau5, tau2, tau8)
    )

    tau10 = (
        einsum("im,opmw->opwi", W3.o, tau9)
    )

    tau11 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau3)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,opq->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,pa->op", Y1, tau6)
    )

    tau8 = (
        einsum("ap,op->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W2.v, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y3, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y3, tau10)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau9, tau6, tau8)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau7, tau6, tau9)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("po,poi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,iljk->oijk", Y4, tau7)
    )

    tau9 = (
        einsum("jw,kw,kp,opwj,oikj->opwi", D2, D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau10) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,qop->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau10, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,opi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,iljk->oijk", Y4, tau7)
    )

    tau9 = (
        einsum("jw,kw,kp,opwj,oijk->opwi", D2, D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau9)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau10) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W2.v, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("in,nm,on->omi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W4.v, tau8)
    )

    tau10 = (
        einsum("ao,pa->op", Y1, tau9)
    )

    tau11 = (
        einsum("ap,op->oa", Y1, tau10)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y1, tau12, tau8) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,ip,oqwi->opqw", D2, Y4, tau7)
    )

    tau9 = (
        einsum("oqm,qpmw,qopw->opmw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("im,opmw->opwi", W3.o, tau9)
    )

    tau11 = (
        einsum("io,oq,qpwi->opw", Y2, Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,opq->opi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau3)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y1, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,io,pqwi->opqw", D2, Y2, tau7)
    )

    tau9 = (
        einsum("oqm,qpmw,oqpw->opmw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("im,opmw->opwi", W3.o, tau9)
    )

    tau11 = (
        einsum("qo,io,qpwi->opw", Z, Y4, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W2.v, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y3, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau0, tau11) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau7, tau6)
    )

    tau9 = (
        einsum("am,om->oa", W2.v, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y3, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y1, tau10)
    )

    RY3 += (
        einsum("aw,pa,pow,pow->ao", D3, tau11, tau0, tau4) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,oqp->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau3)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RY3 += (
        einsum("aw,ap,pow,pow->ao", D3, Y3, tau10, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau9, tau6, tau8)
    )

    tau11 = (
        einsum("iw,ip,opwi->opw", D4, Y4.conj(), tau10)
    )

    RY3 += (
        einsum("aw,ap,pow->ao", D3, Y3, tau11) / 4
    )

    return RY3

def gen_RY4(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z, D1, D2, D3, D4):

    tau0 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("im,omw,omw->owi", W4.o, tau0, tau4)
    )

    RY4 = (
        einsum("iw,owi->io", D4, tau5)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau6 = (
        einsum("im,omw,omw->owi", W4.o, tau4, tau5)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("nm,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("mi,omw,omw->owi", tau0, tau1, tau5)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau5)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("im,omw,omw->owi", W4.o, tau3, tau5)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau6)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau6)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W3.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pmw->omw", Z.conj(), tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau0, tau5)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W4.o, tau1, tau6)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pmw->omw", Z.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau3, tau5)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pmw->omw", Z.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mi,omw,omw->owi", tau6, tau4, tau5)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,omw,omw->owi", W4.o, tau1, tau6)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau7)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau8)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("qp,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("qp,oqw->opw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau7)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau7)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau7)
    )

    tau9 = (
        einsum("pow,powi->owi", tau0, tau8)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau7)
    )

    tau9 = (
        einsum("pow,powi->owi", tau0, tau8)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opwi->opmw", tau6, tau4)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("omw,omwi->owi", tau0, tau8)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau1)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("omi,opwi->opmw", tau4, tau2)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("ip,po,pm->omi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau7, tau5)
    )

    tau9 = (
        einsum("omw,omwi->owi", tau0, tau8)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    tau8 = (
        einsum("aw,iw,pia,pow->oia", D3, D4, tau3, tau7)
    )

    RY4 += (
        - einsum("ao,oia->io", Y3.conj(), tau8) / 2
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau6 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("oq,qpw->opw", Z, tau6)
    )

    tau8 = (
        einsum("aw,iw,pia,pow->oia", D3, D4, tau3, tau7)
    )

    RY4 += (
        - einsum("ao,oia->io", Y3.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau7)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W2.o, tau0, tau7)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau5, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau7)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau7)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau1 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau1, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau3 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau5) / 2
    )
    tau0 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau1 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau1, tau5) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y4, tau6)
    )

    tau8 = (
        einsum("ap,oia->opi", Y3.conj(), tau7)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y2, tau6)
    )

    tau8 = (
        einsum("ap,oia->opi", Y3.conj(), tau7)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau3, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,omw,omw->owi", W4.o, tau1, tau7)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau8)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau1, tau6)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau8)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W3.o, tau3)
    )

    tau5 = (
        einsum("mn,onw,onw->omw", X, tau2, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau0, tau6)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau8)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau8 = (
        einsum("mi,omw,omw->owi", tau7, tau1, tau6)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau4 = (
        einsum("oi,pwi->opw", tau3, tau2)
    )

    tau5 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau3 = (
        einsum("oi,pwi->opw", tau2, tau1)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau4)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau3, tau6) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau2 = (
        einsum("oi,pwi->opw", tau1, tau0)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau5 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau3, tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau2, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y2, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau3)
    )

    tau5 = (
        einsum("oq,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau6, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau6, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau4, tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau0, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("pm,pow->omw", tau7, tau3)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau0, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,am,bm->ab", tau4, W1.v, W2.v)
    )

    tau6 = (
        einsum("bo,ab->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau5, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("im,mi->m", W3.o, tau0)
    )

    tau2 = (
        einsum("n,mn->m", tau1, X)
    )

    tau3 = (
        einsum("m,am,bm->ab", tau2, W1.v, W2.v)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau5, tau9)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("jp,po,pwi->owij", Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("jw,jo,pwij->opwi", D2, Y4, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("mn,in,on->omi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("jp,po,pwi->owij", Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("jw,jo,pwij->opwi", D2, Y2, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("mn,in,on->omi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau5)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,ip,powj->owij", D4, Y4, tau7)
    )

    tau9 = (
        einsum("pwj,owij->opi", tau0, tau8)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("iw,ip,powj->owij", D4, Y2, tau7)
    )

    tau9 = (
        einsum("pwj,owij->opi", tau0, tau8)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau5)
    )

    tau7 = (
        einsum("qp,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau5)
    )

    tau7 = (
        einsum("qp,oqw->opw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau4)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("iw,iq,qp,oqw->opwi", D2, Y2.conj(), Z.conj(), tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opwi->opmw", tau7, tau4)
    )

    tau9 = (
        einsum("pmi,pomw->omwi", tau2, tau8)
    )

    tau10 = (
        einsum("omw,omwi->owi", tau0, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        - einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("op,ip,pm->omi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau2, tau1, tau4)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau8, tau0, tau7)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("mn,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau8, tau0, tau7)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("qo,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,iw,pia,pow->oia", D3, D4, tau4, tau8)
    )

    RY4 += (
        - einsum("ao,oia->io", Y3.conj(), tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau7 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau5, tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    tau9 = (
        einsum("aw,iw,pia,pow->oia", D3, D4, tau4, tau8)
    )

    RY4 += (
        - einsum("ao,oia->io", Y3.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,an,mn->ma", X, W4.v, tau3)
    )

    tau5 = (
        einsum("am,mb->ab", W1.v, tau4)
    )

    tau6 = (
        einsum("bo,ab->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, tau3)
    )

    tau5 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau5, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("in,mi->mn", W3.o, tau0)
    )

    tau2 = (
        einsum("mn,an,mn->ma", X, W4.v, tau1)
    )

    tau3 = (
        einsum("am,mb->ab", W1.v, tau2)
    )

    tau4 = (
        einsum("bo,ab->oa", Y1, tau3)
    )

    tau5 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau5, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W1.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("im,owi->omw", W1.o, tau1)
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau4, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau3, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W3.o, tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W4.o, tau7, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("pm,pow->omw", tau6, tau2)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W3.o, tau8)
    )

    tau10 = (
        einsum("im,omw,omw->owi", W4.o, tau7, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,jm->ij", tau7, W3.o, W4.o)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau9, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,jm->ij", tau7, W3.o, W4.o)
    )

    tau9 = (
        einsum("jo,ji->oi", Y2, tau8)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau9, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W1.o, W2.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,jm->ij", tau4, W1.o, W2.o)
    )

    tau6 = (
        einsum("jo,ji->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau1)
    )

    tau3 = (
        einsum("jm,owij->omwi", W1.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("omi,pmwi->opmw", tau5, tau3)
    )

    tau7 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau8 = (
        einsum("jo,mji->omi", Y2, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau6)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau1)
    )

    tau3 = (
        einsum("jm,owij->omwi", W1.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("omi,pmwi->opmw", tau5, tau3)
    )

    tau7 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau8 = (
        einsum("jo,mji->omi", Y4, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau6)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau8)
    )

    tau10 = (
        einsum("pow,powi->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W3.o, tau4, tau3)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W3.o, tau4, tau3)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau6)
    )

    tau8 = (
        einsum("qp,oqw->opw", Z.conj(), tau7)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("im,om,omj->oij", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("op,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("im,om,omj->oij", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("po,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pij,pow->owij", tau5, tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W3.o, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W4.o, tau6, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W3.o, tau7)
    )

    tau9 = (
        einsum("im,omw,omw->owi", W4.o, tau6, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau6, tau4)
    )

    tau8 = (
        einsum("im,omwj->owij", W3.o, tau7)
    )

    tau9 = (
        einsum("po,pwij->owij", Z.conj(), tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau6, tau4)
    )

    tau8 = (
        einsum("im,omwj->owij", W3.o, tau7)
    )

    tau9 = (
        einsum("po,pwij->owij", Z.conj(), tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau7 = (
        einsum("jo,mji->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau7 = (
        einsum("jo,mji->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    tau9 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau8)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W2.o, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,mj->ij", W2.o, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W4.o, tau6)
    )

    tau8 = (
        einsum("im,mj->ij", W1.o, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau9, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W4.o, tau6)
    )

    tau8 = (
        einsum("im,mj->ij", W1.o, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y2, tau8)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau9, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau3, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau9 = (
        einsum("mi,omw,omw->owi", tau8, tau1, tau7)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau2, tau4)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W1.o, W2.v)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W1.o, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau8 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau6, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau5, tau8)
    )

    tau10 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau11 = (
        einsum("mi,omw,omw->owi", tau10, tau1, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau11)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W1.o, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("pm,pow->omw", tau8, tau4)
    )

    tau10 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau11 = (
        einsum("mi,omw,omw->owi", tau10, tau1, tau9)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau11)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,im,am->ia", tau3, W1.o, W2.v)
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, tau4)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau0)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau6)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,im,am->ia", tau3, W1.o, W2.v)
    )

    tau5 = (
        einsum("ao,ia->oi", Y1, tau4)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau0)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau6)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("im,mi->m", W1.o, tau2)
    )

    tau4 = (
        einsum("n,nm->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W3.o, W4.v)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W3.o, W4.v)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y4, tau9)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau10, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("im,mi->m", W1.o, tau5)
    )

    tau7 = (
        einsum("n,nm->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W3.o, W4.v)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y2, tau9)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau10, tau0, tau4)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("kw,ko,owl,likj->owij", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("po,pwij->owij", Z.conj(), tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("kw,ko,owl,lijk->owij", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("po,pwij->owij", Z.conj(), tau8)
    )

    tau10 = (
        einsum("owj,owji->owi", tau0, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("nm,onw,onw->omw", X, tau5, tau7)
    )

    tau9 = (
        einsum("po,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau1, tau3, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("nm,onw,onw->omw", X, tau5, tau7)
    )

    tau9 = (
        einsum("po,pmw->omw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("om,omi,pmw->opwi", tau3, tau2, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y4, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("op,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W1.o, tau2, tau5)
    )

    tau7 = (
        einsum("po,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau1, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y2, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau4)
    )

    tau6 = (
        einsum("jm,owij->omwi", W1.o, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("op,ip,pm->omi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("omi,pmwi->opmw", tau8, tau6)
    )

    tau10 = (
        einsum("omi,opmw->opwi", tau3, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau1)
    )

    tau3 = (
        einsum("jo,mij->omi", Y4, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("iw,ip,po,pwj->owij", D2, Y2.conj(), Z.conj(), tau4)
    )

    tau6 = (
        einsum("jm,owij->omwi", W1.o, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("omi,pmwi->opmw", tau8, tau6)
    )

    tau10 = (
        einsum("omi,opmw->opwi", tau3, tau9)
    )

    tau11 = (
        einsum("pow,powi->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y3, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("in,mi->mn", W3.o, tau3)
    )

    tau5 = (
        einsum("in,nm,nm->mi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,ia->oi", Y1, tau6)
    )

    tau8 = (
        einsum("oi,pwi->opw", tau7, tau2)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    tau10 = (
        einsum("po,pwij->owij", Z.conj(), tau9)
    )

    tau11 = (
        einsum("owj,owji->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    tau10 = (
        einsum("po,pwij->owij", Z.conj(), tau9)
    )

    tau11 = (
        einsum("owj,owji->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W1.o, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau2, tau5)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau1, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W1.o, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau2, tau4)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    tau8 = (
        einsum("po,pmw->omw", Z.conj(), tau7)
    )

    tau9 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau10 = (
        einsum("mi,omw,omw->owi", tau9, tau1, tau8)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau2)
    )

    tau4 = (
        einsum("jo,mij->omi", Y4, tau3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W1.o, tau1, tau4)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("pij,pow->owij", tau6, tau9)
    )

    tau11 = (
        einsum("owj,owij->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau2)
    )

    tau4 = (
        einsum("jo,mij->omi", Y2, tau3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W1.o, tau1, tau4)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("pij,pow->owij", tau6, tau9)
    )

    tau11 = (
        einsum("owj,owij->owi", tau0, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau11) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("in,nm,nm->mi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,mi->ia", W4.v, tau3)
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, tau4)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau0)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau6) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("in,nm,nm->mi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,mi->ia", W4.v, tau3)
    )

    tau5 = (
        einsum("ao,ia->oi", Y1, tau4)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau0)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau7, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y4, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    tau10 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau9)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y2, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    tau10 = (
        einsum("iw,qpw,qowi->opi", D4, tau0, tau9)
    )

    RY4 += (
        einsum("po,poi->io", Z.conj(), tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,ni->mn", W1.o, tau2)
    )

    tau4 = (
        einsum("mn,in,mn->mi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W2.v, tau4)
    )

    tau6 = (
        einsum("ai,ja->ij", T1, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W2.v, tau7)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y4, tau9)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau10, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,ni->mn", W1.o, tau5)
    )

    tau7 = (
        einsum("mn,in,mn->mi", X, W3.o, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W2.v, tau7)
    )

    tau9 = (
        einsum("ai,ja->ij", T1, tau8)
    )

    tau10 = (
        einsum("jo,ij->oi", Y2, tau9)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau10, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("oq,pq->op", Z, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("pm,pm,om->op", tau3, tau4, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau8 = (
        einsum("pm,om,om->op", tau5, tau6, tau7)
    )

    tau9 = (
        einsum("qp,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qo,qp->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y4, tau7)
    )

    tau9 = (
        einsum("oq,pq->op", Z, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,om,om->oa", W2.v, tau4, tau8)
    )

    tau10 = (
        einsum("bo,oa->ab", Y1, tau9)
    )

    tau11 = (
        einsum("bo,ba->oa", Y3, tau10)
    )

    tau12 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau12, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("qoi,qpw->opwi", tau10, tau11)
    )

    tau13 = (
        einsum("pow,powi->owi", tau2, tau12)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,om,om->oa", W2.v, tau1, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y1, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("io,oq,pqi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,opq->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau5, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y2, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y4, tau11)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau9 = (
        einsum("am,om,om->oa", W4.v, tau7, tau8)
    )

    tau10 = (
        einsum("ap,oa->op", Y3, tau9)
    )

    tau11 = (
        einsum("ap,po->oa", Y3, tau10)
    )

    tau12 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau12, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y4, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,om,om->oa", W2.v, tau2, tau6)
    )

    tau8 = (
        einsum("bo,oa->ab", Y1, tau7)
    )

    tau9 = (
        einsum("bo,ba->oa", Y1, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("qoi,qpw->opwi", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau13 = (
        einsum("pow,powi->owi", tau12, tau11)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau8, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y4, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y4, tau11)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("io,pqi->opq", Y2, tau6)
    )

    tau8 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W2.v, tau0, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y1, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau5, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y2, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y2, tau11)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau10 = (
        einsum("im,om,om->oi", W1.o, tau8, tau9)
    )

    tau11 = (
        einsum("jo,oi->ij", Y4, tau10)
    )

    tau12 = (
        einsum("jo,ji->oi", Y2, tau11)
    )

    RY4 += (
        - einsum("iw,pi,pow,pow->io", D4, tau12, tau0, tau4) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau2, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,om,om->oa", W4.v, tau4, tau5)
    )

    tau7 = (
        einsum("ap,oa->op", Y3, tau6)
    )

    tau8 = (
        einsum("ap,po->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("oqi,qpw->opwi", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau13 = (
        einsum("pow,powi->owi", tau12, tau11)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("pm,om,pmi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("io,opi->op", Y2, tau7)
    )

    tau9 = (
        einsum("pq,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qp,qo->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("im,om,om->oi", W1.o, tau5, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("qp,oqw->opw", Z.conj(), tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau11) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,om,om->oa", W4.v, tau3, tau4)
    )

    tau6 = (
        einsum("ao,pa->op", Y1, tau5)
    )

    tau7 = (
        einsum("ap,op->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y2, tau12, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("qoi,qpw->opwi", tau10, tau11)
    )

    tau13 = (
        einsum("pow,powi->owi", tau2, tau12)
    )

    RY4 += (
        - einsum("iw,owi->io", D4, tau13) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,om,om->oa", W4.v, tau5, tau6)
    )

    tau8 = (
        einsum("ao,pa->op", Y1, tau7)
    )

    tau9 = (
        einsum("ap,op->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,opi->op", Y2, tau7)
    )

    tau9 = (
        einsum("qp,oq->op", Z, tau8)
    )

    tau10 = (
        einsum("qo,qp->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", tau10, tau2)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        - einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau6, tau5)
    )

    tau8 = (
        einsum("am,om->oa", W4.v, tau7)
    )

    tau9 = (
        einsum("ap,oa->op", Y3, tau8)
    )

    tau10 = (
        einsum("ap,po->oa", Y1, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("qo,qpw->opw", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau8 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau9 = (
        einsum("om,pm,omi->opi", tau7, tau8, tau6)
    )

    tau10 = (
        einsum("io,op,qpi->opq", Y2, Z, tau9)
    )

    tau11 = (
        einsum("iq,oqp->opi", Y4, tau10)
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau11, tau4)
    )

    tau13 = (
        einsum("pow,powi->owi", tau0, tau12)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau13) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("io,op,qpi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W4.v, tau4)
    )

    tau6 = (
        einsum("ap,oa->op", Y3, tau5)
    )

    tau7 = (
        einsum("ap,po->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau7, tau6, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("pow,powi->owi", tau11, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("pm,om,pmi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("oqi,qpw->opwi", tau10, tau3)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau13 = (
        einsum("pow,powi->owi", tau12, tau11)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau13) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("im,opwi->opmw", W3.o, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("ip,omi->opm", Y2, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,ip,oqwi->opqw", D2, Y2, tau7)
    )

    tau9 = (
        einsum("qom,qpmw,oqpw->opmw", tau5, tau2, tau8)
    )

    tau10 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau11 = (
        einsum("op,ip,pm->omi", Z, Y4, tau10)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau11, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("jm,omi,omk->oijk", W3.o, tau5, tau7)
    )

    tau9 = (
        einsum("ko,oijl->ijkl", Y2, tau8)
    )

    tau10 = (
        einsum("lo,lijk->oijk", Y2, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powi->owij", tau1, tau10, tau3)
    )

    RY4 += (
        einsum("jw,iw,owji->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,opq->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y4, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau6, tau5)
    )

    tau8 = (
        einsum("am,om->oa", W2.v, tau7)
    )

    tau9 = (
        einsum("bo,oa->ab", Y3, tau8)
    )

    tau10 = (
        einsum("bo,ba->oa", Y3, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau11, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,opi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W4.v, tau5)
    )

    tau7 = (
        einsum("ao,pa->op", Y1, tau6)
    )

    tau8 = (
        einsum("ap,op->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau9, tau6, tau8)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("pow,powi->owi", tau11, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,omj,omk->oijk", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,lijk->oijk", Y2, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau8 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau9)
    )

    tau11 = (
        einsum("powj,pkij,powk->owij", tau10, tau6, tau8)
    )

    RY4 += (
        einsum("jw,iw,owij->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("io,pmi->opm", Y2, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,opwi->opmw", W3.o, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau7 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau6)
    )

    tau8 = (
        einsum("iw,ip,oqwi->opqw", D2, Y4, tau7)
    )

    tau9 = (
        einsum("oqm,qpmw,oqpw->opmw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau11 = (
        einsum("op,ip,pm->omi", Z, Y4, tau10)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau11, tau9)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("po,poi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("qo,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau4, tau1, tau3)
    )

    tau6 = (
        einsum("nm,onw->omw", X, tau5)
    )

    tau7 = (
        einsum("po,pmw->omw", Z.conj(), tau6)
    )

    tau8 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau9 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau10 = (
        einsum("op,ip,pm->omi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("om,omi,pmw->opwi", tau8, tau10, tau7)
    )

    tau12 = (
        einsum("pow,powi->owi", tau0, tau11)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("jm,omi,omk->oijk", W3.o, tau5, tau7)
    )

    tau9 = (
        einsum("ko,oijl->ijkl", Y2, tau8)
    )

    tau10 = (
        einsum("lo,iljk->oijk", Y4, tau9)
    )

    tau11 = (
        einsum("powk,pkij,powj->owij", tau1, tau10, tau3)
    )

    RY4 += (
        einsum("jw,iw,owij->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,omj,omk->oijk", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,iljk->oijk", Y4, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau8 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau9)
    )

    tau11 = (
        einsum("powi,pkij,powk->owij", tau10, tau6, tau8)
    )

    RY4 += (
        einsum("jw,iw,owji->io", D2, D4, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W4.v, tau3)
    )

    tau5 = (
        einsum("ao,pa->op", Y1, tau4)
    )

    tau6 = (
        einsum("ap,op->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau7) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau10, tau9)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,pi,pow,pow->io", D4, tau7, tau12, tau8) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("ip,omi->opm", Y4, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("im,opwi->opmw", W3.o, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau8)
    )

    tau10 = (
        einsum("iw,ip,oqwi->opqw", D2, Y4, tau9)
    )

    tau11 = (
        einsum("qom,oqpw,qpmw->opmw", tau4, tau10, tau7)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau1, tau11)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("om,pm,omi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("po,poi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y4, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("pm,om,pmi->opi", tau4, tau5, tau3)
    )

    tau7 = (
        einsum("op,opi->oi", Z, tau6)
    )

    tau8 = (
        einsum("jo,oi->ij", Y2, tau7)
    )

    tau9 = (
        einsum("jo,ji->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    tau11 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau10)
    )

    tau12 = (
        einsum("oq,qpw->opw", Z, tau11)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,qop->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau10, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W2.v, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau8)
    )

    tau10 = (
        einsum("iq,opq->opi", Y4, tau9)
    )

    tau11 = (
        einsum("qoi,qpw->opwi", tau10, tau3)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau13 = (
        einsum("pow,powi->owi", tau12, tau11)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau13) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("ip,po,pm->omi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("ip,omi->opm", Y4, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau6 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("im,opwi->opmw", W3.o, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iq,qp,oqw->opwi", Y2.conj(), Z.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,pqwi->opqw", D2, Y2, tau9)
    )

    tau11 = (
        einsum("qom,qopw,qpmw->opmw", tau4, tau10, tau7)
    )

    tau12 = (
        einsum("pmi,pomw->owi", tau1, tau11)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("po,pmw->omw", Z.conj(), tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("om,pmw,omi->opwi", tau9, tau6, tau8)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("pow,powi->owi", tau11, tau10)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W2.v, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y3, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau1, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau0, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qp,oqw->opw", Z.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw->opw", Z, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y4, tau10, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W2.v, tau3)
    )

    tau5 = (
        einsum("bo,oa->ab", Y3, tau4)
    )

    tau6 = (
        einsum("bo,ba->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau8, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RY4 += (
        einsum("iw,ip,pow,pow->io", D4, Y2, tau11, tau7) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("qp,oqw,oqw->opw", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("oq,qpw->opw", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,oqp->opi", Y2, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("oqi,qpw->opwi", tau10, tau11)
    )

    tau13 = (
        einsum("pow,powi->owi", tau3, tau12)
    )

    RY4 += (
        einsum("iw,owi->io", D4, tau13) / 4
    )

    return RY4

def gen_RZ(f, W1, W2, W3, W4, X, T1, Y1, Y2, Y3, Y4, Z, D1, D2, D3, D4):

    tau0 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("mn,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ = (
        einsum("pmw,omw,omw->op", tau2, tau3, tau4)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    tau3 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    RZ += (
        - einsum("omw,omw,pmw->op", tau1, tau2, tau5)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("nm,onw,onw->omw", X, tau3, tau4)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau1, tau2, tau5)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("mn,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau2, tau4, tau5)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau3, tau4, tau5)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("nm,onw,onw->omw", X, tau0, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W3.o, tau5)
    )

    RZ += (
        - einsum("pmw,omw,omw->op", tau2, tau4, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D2, W4.o, Y2.conj())
    )

    RZ += (
        - einsum("pmw,omw,omw->op", tau3, tau5, tau6)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau1, tau2, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau1, tau3, tau6)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("im,owi->omw", W1.o, tau3)
    )

    tau5 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau2, tau6)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W3.v, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau6 = (
        einsum("nm,onw,onw->omw", X, tau4, tau5)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau2, tau6)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau2 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau5 = (
        einsum("qo,qpw,qpw->opw", Z, tau3, tau4)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau2, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("bo,ab->oa", Y3, f.vv)
    )

    tau5 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau4)
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau2, tau3, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau4 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau3)
    )

    tau5 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau4)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau5) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("bo,ab->oa", Y1, f.vv)
    )

    tau2 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau1)
    )

    tau3 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau3, tau4, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau2, tau3)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau8)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("om,pmw,pmw->opw", tau5, tau0, tau1)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau6, tau7, tau8)
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("om,pmw,pmw->opw", tau5, tau0, tau1)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau6, tau7, tau8)
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau7 = (
        einsum("qo,qpw,qpw->opw", Z, tau5, tau6)
    )

    RZ += (
        - einsum("qow,qpw->op", tau4, tau7) / 2
    )
    tau0 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau1 = (
        einsum("jo,mji->omi", Y2, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("am,om,omi->oia", W1.v, tau2, tau1)
    )

    tau4 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau3)
    )

    tau5 = (
        einsum("qo,qpw->opw", Z, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau5, tau6, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RZ += (
        - einsum("iw,ip,qpw,qowi->op", D4, Y4.conj(), tau7, tau6) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RZ += (
        - einsum("iw,ip,qpw,qowi->op", D4, Y4.conj(), tau7, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau7 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ += (
        - einsum("pmw,omw,omw->op", tau5, tau6, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau7 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ += (
        - einsum("pmw,omw,omw->op", tau5, tau6, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    RZ += (
        - einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau6 = (
        einsum("jo,mji->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    RZ += (
        - einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("am,om,omi->oia", W1.v, tau5, tau4)
    )

    tau7 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau6)
    )

    RZ += (
        - einsum("qow,qpw->op", tau2, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("am,om,omi->oia", W1.v, tau4, tau3)
    )

    tau6 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw->opw", Z, tau6)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau2, tau5)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau0, tau1, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W2.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau2, tau4)
    )

    tau7 = (
        einsum("mn,onw->omw", X, tau6)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau0, tau1, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y4, tau6)
    )

    RZ += (
        einsum("ap,ip,oia->op", Y3.conj(), Y4.conj(), tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,an,on->oma", X, W3.v, tau4)
    )

    tau6 = (
        einsum("oma,opmw->opwa", tau5, tau3)
    )

    tau7 = (
        einsum("aw,iw,ip,powa->oia", D3, D4, Y2, tau6)
    )

    RZ += (
        einsum("ap,ip,oia->op", Y3.conj(), Y4.conj(), tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau5 = (
        einsum("qo,qpw,qpw->opw", Z, tau3, tau4)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau2, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau4 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau4)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("jo,ji->oi", Y4, f.oo)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau4)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau2, tau3, tau5) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("jo,ji->oi", Y2, f.oo)
    )

    tau2 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau3, tau4, tau5) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau7, tau6) / 2
    )
    tau0 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau2 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau0, tau2)
    )

    tau4 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau5 = (
        einsum("jo,mji->omi", Y4, tau4)
    )

    tau6 = (
        einsum("omi,opmw->opwi", tau5, tau3)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau7, tau6) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau3)
    )

    tau5 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau6 = (
        einsum("im,owi->omw", W1.o, tau5)
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau4, tau6)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau2, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau3, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("nm,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W3.o, tau6)
    )

    RZ += (
        - einsum("pmw,omw,omw->op", tau3, tau5, tau7)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau7 = (
        einsum("nm,onw,onw->omw", X, tau5, tau6)
    )

    RZ += (
        - einsum("pmw,pmw,omw->op", tau1, tau3, tau7)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau3, tau4)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau6, tau1, tau2)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,am,bm->ab", tau3, W1.v, W2.v)
    )

    tau5 = (
        einsum("bo,ab->oa", Y3, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,mi->m", W3.o, tau1)
    )

    tau3 = (
        einsum("n,mn->m", tau2, X)
    )

    tau4 = (
        einsum("m,am,bm->ab", tau3, W1.v, W2.v)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau9 = (
        einsum("oq,qpw,qpw->opw", Z, tau7, tau8)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau6, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,am,bm->ab", tau5, W1.v, W2.v)
    )

    tau7 = (
        einsum("bo,ab->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau7)
    )

    tau9 = (
        einsum("oq,qpw,qpw->opw", Z, tau2, tau8)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,am,bm->ab", tau6, W1.v, W2.v)
    )

    tau8 = (
        einsum("bo,ab->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau2, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("jp,powi->owij", Y4, tau7)
    )

    RZ += (
        - einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("jp,powi->owij", Y2, tau7)
    )

    RZ += (
        - einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("jp,powi->owij", Y4, tau7)
    )

    RZ += (
        - einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("mn,in,on->omi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau4)
    )

    tau8 = (
        einsum("jp,powi->owij", Y2, tau7)
    )

    RZ += (
        - einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau4, tau3, tau6)
    )

    tau8 = (
        einsum("mn,onw->omw", X, tau7)
    )

    RZ += (
        - einsum("omw,omw,pmw->op", tau1, tau2, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau0)
    )

    tau2 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau6, tau3, tau5)
    )

    tau8 = (
        einsum("mn,onw->omw", X, tau7)
    )

    RZ += (
        - einsum("omw,omw,pmw->op", tau1, tau2, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("oq,qpw->opw", Z, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau6, tau7, tau8) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau0)
    )

    tau2 = (
        einsum("jo,mij->omi", Y2, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,om,omi->oia", W1.v, tau3, tau2)
    )

    tau5 = (
        einsum("aw,iw,ap,ip,oia->opw", D1, D2, Y1.conj(), Y2.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau8 = (
        einsum("oq,qpw,qpw->opw", Z, tau6, tau7)
    )

    RZ += (
        - einsum("qow,qpw->op", tau5, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    RZ += (
        - einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("ip,po,pm->omi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("iw,ip,pmw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    RZ += (
        - einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RZ += (
        - einsum("iw,ip,qpw,qowi->op", D4, Y4.conj(), tau8, tau7) / 2
    )
    tau0 = (
        einsum("aw,am,ao->omw", D1, W1.v, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("iw,ip,pmw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("omi,opmw->opwi", tau6, tau3)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    RZ += (
        - einsum("iw,ip,qpw,qowi->op", D4, Y4.conj(), tau8, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    RZ += (
        - einsum("omw,pmw,pmw->op", tau5, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("aw,am,ao->omw", D3, W1.v, Y3.conj())
    )

    RZ += (
        - einsum("omw,pmw,pmw->op", tau5, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau2)
    )

    tau4 = (
        einsum("jo,mij->omi", Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("am,om,omi->oia", W1.v, tau5, tau4)
    )

    tau7 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau6)
    )

    tau8 = (
        einsum("oq,qpw->opw", Z, tau7)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau3)
    )

    tau5 = (
        einsum("jo,mij->omi", Y2, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("am,om,omi->oia", W1.v, tau6, tau5)
    )

    tau8 = (
        einsum("aw,iw,ap,ip,oia->opw", D3, D4, Y3.conj(), Y4.conj(), tau7)
    )

    RZ += (
        - einsum("qow,qpw->op", tau2, tau8) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y3, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau7, tau8, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau9 = (
        einsum("oq,qpw,qpw->opw", Z, tau7, tau8)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau6, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y3, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau7, tau8, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,an,mn->ma", X, W4.v, tau2)
    )

    tau4 = (
        einsum("am,mb->ab", W1.v, tau3)
    )

    tau5 = (
        einsum("bo,ab->oa", Y1, tau4)
    )

    tau6 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("oq,qpw,qpw->opw", Z, tau7, tau8)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau6, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("op,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau3, tau4)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("mn,on,on->om", X, tau5, tau6)
    )

    tau8 = (
        einsum("po,pm->om", Z, tau7)
    )

    tau9 = (
        einsum("om,pmw,pmw->opw", tau8, tau3, tau4)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("op,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau3, tau5, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W3.o, tau4)
    )

    tau6 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau7 = (
        einsum("om,pmw,pmw->opw", tau3, tau5, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau7, tau8, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("im,mi->m", W1.o, tau4)
    )

    tau6 = (
        einsum("n,nm->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,jm->ij", tau6, W3.o, W4.o)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau2, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("im,mi->m", W1.o, tau3)
    )

    tau5 = (
        einsum("n,nm->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,jm->ij", tau5, W3.o, W4.o)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau8)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,mi->m", W3.o, tau3)
    )

    tau5 = (
        einsum("n,mn->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,jm->ij", tau5, W1.o, W2.o)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau8)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,jm->ij", tau6, W1.o, W2.o)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau2, tau3, tau9)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau5 = (
        einsum("jo,mji->omi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau3, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau2, tau6)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    RZ += (
        einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau1, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("pmi,pomw->omwi", tau6, tau4)
    )

    tau8 = (
        einsum("im,omwj->owij", W1.o, tau7)
    )

    RZ += (
        einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau5, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau7 = (
        einsum("im,owi->omw", W1.o, tau6)
    )

    tau8 = (
        einsum("iw,im,io->omw", D2, W2.o, Y2.conj())
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau5, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau4 = (
        einsum("jo,mji->omi", Y4, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W3.o, tau5, tau4)
    )

    tau7 = (
        einsum("op,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau2, tau7)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau4 = (
        einsum("jo,mji->omi", Y2, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("jm,om,omi->oij", W3.o, tau5, tau4)
    )

    tau7 = (
        einsum("po,pij->oij", Z, tau6)
    )

    tau8 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau2, tau7)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y4, tau2)
    )

    tau4 = (
        einsum("im,om,omj->oij", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("op,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau0, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau6, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("mn,in,jn->mij", X, W3.o, W4.o)
    )

    tau3 = (
        einsum("jo,mji->omi", Y2, tau2)
    )

    tau4 = (
        einsum("im,om,omj->oij", W1.o, tau1, tau3)
    )

    tau5 = (
        einsum("po,pij->oij", Z, tau4)
    )

    tau6 = (
        einsum("iw,ip,pwj,oji->opw", D4, Y4.conj(), tau0, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau6, tau7, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau4, tau7)
    )

    RZ += (
        einsum("pmw,omw->op", tau3, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("im,owi->omw", W3.o, tau0)
    )

    tau2 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau3 = (
        einsum("mn,onw,onw->omw", X, tau1, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau7, tau4, tau6)
    )

    RZ += (
        einsum("pmw,omw->op", tau3, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau2 = (
        einsum("jo,mji->omi", Y2, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W3.o, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau8, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau2 = (
        einsum("jo,mji->omi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau0, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("ip,po,pm->omi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("pmi,pomw->omwi", tau5, tau3)
    )

    tau7 = (
        einsum("im,omwj->owij", W3.o, tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau8, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,im,io->omw", D4, W4.o, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau2)
    )

    tau4 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau5)
    )

    tau7 = (
        einsum("jo,mij->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau4)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau7 = (
        einsum("jo,mji->omi", Y2, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("in,jn,nm->mij", W1.o, W2.o, X)
    )

    tau7 = (
        einsum("jo,mji->omi", Y4, tau6)
    )

    tau8 = (
        einsum("omi,opmw->opwi", tau7, tau5)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,ni->mn", W1.o, tau3)
    )

    tau5 = (
        einsum("mn,in,mn->mi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,mj->ij", W2.o, tau5)
    )

    tau7 = (
        einsum("jo,ij->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau8)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("im,mj->ij", W2.o, tau6)
    )

    tau8 = (
        einsum("jo,ij->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau2, tau3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,in,mn->mi", X, W4.o, tau2)
    )

    tau4 = (
        einsum("im,mj->ij", W1.o, tau3)
    )

    tau5 = (
        einsum("jo,ji->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau9 = (
        einsum("qo,qpw,qpw->opw", Z, tau7, tau8)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau6, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("in,mi->mn", W3.o, tau1)
    )

    tau3 = (
        einsum("mn,in,mn->mi", X, W4.o, tau2)
    )

    tau4 = (
        einsum("im,mj->ij", W1.o, tau3)
    )

    tau5 = (
        einsum("jo,ji->oi", Y2, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau5)
    )

    tau7 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau7, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau6 = (
        einsum("oi,pwi->opw", tau5, tau4)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau2, tau3, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau5 = (
        einsum("oi,pwi->opw", tau4, tau3)
    )

    tau6 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("ao,ia->oi", Y3, f.ov)
    )

    tau3 = (
        einsum("oi,pwi->opw", tau2, tau1)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("qo,qpw,qpw->opw", Z, tau4, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau3, tau6) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("ao,ia->oi", Y1, f.ov)
    )

    tau3 = (
        einsum("oi,pwi->opw", tau2, tau1)
    )

    tau4 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau4, tau5, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau4 = (
        einsum("jo,ij->oi", Y2, tau3)
    )

    tau5 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau4)
    )

    tau6 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau5 = (
        einsum("jo,ij->oi", Y4, tau4)
    )

    tau6 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau5)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau2, tau3, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y4, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau6 = (
        einsum("qo,qpw,qpw->opw", Z, tau4, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau3, tau6) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,ja->ij", T1, f.ov)
    )

    tau2 = (
        einsum("jo,ij->oi", Y2, tau1)
    )

    tau3 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau2)
    )

    tau4 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau3)
    )

    tau5 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau6 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau4, tau5, tau6) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W3.o, tau2)
    )

    tau4 = (
        einsum("mn,onw,onw->omw", X, tau1, tau3)
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau5)
    )

    tau7 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau8 = (
        einsum("im,owi->omw", W1.o, tau7)
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau4, tau6, tau8)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("op,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("om,pmw,pmw->opw", tau9, tau3, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau2)
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("im,owi->omw", W1.o, tau4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("po,pm->om", Z, tau8)
    )

    tau10 = (
        einsum("om,pmw,pmw->opw", tau9, tau3, tau5)
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau0, tau1, tau10)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W1.o, W2.v)
    )

    tau6 = (
        einsum("ao,ia->oi", Y3, tau5)
    )

    tau7 = (
        einsum("oi,pwi->opw", tau6, tau1)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau8, tau9)
    )

    RZ += (
        - einsum("qow,qpw,qow->op", tau0, tau10, tau7)
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("im,mi->m", W3.o, tau2)
    )

    tau4 = (
        einsum("n,mn->m", tau3, X)
    )

    tau5 = (
        einsum("m,im,am->ia", tau4, W1.o, W2.v)
    )

    tau6 = (
        einsum("ao,ia->oi", Y1, tau5)
    )

    tau7 = (
        einsum("oi,pwi->opw", tau6, tau1)
    )

    tau8 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qow,qpw->op", tau10, tau8, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau1, tau3)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        - einsum("qow,qpw,qow->op", tau10, tau8, tau9)
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("om,pmw,pmw->opw", tau7, tau1, tau3)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        - einsum("qow,qpw,qow->op", tau10, tau8, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau6 = (
        einsum("im,mi->m", W3.o, tau5)
    )

    tau7 = (
        einsum("n,mn->m", tau6, X)
    )

    tau8 = (
        einsum("m,im,am->ia", tau7, W1.o, W2.v)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RZ += (
        - einsum("qpw,qow,qpw->op", tau10, tau2, tau3)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,mi->m", W3.o, tau4)
    )

    tau6 = (
        einsum("n,mn->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,am->ia", tau6, W1.o, W2.v)
    )

    tau8 = (
        einsum("ao,ia->oi", Y1, tau7)
    )

    tau9 = (
        einsum("oi,pwi->opw", tau8, tau3)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau9)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau10)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("im,mi->m", W1.o, tau1)
    )

    tau3 = (
        einsum("n,nm->m", tau2, X)
    )

    tau4 = (
        einsum("m,im,am->ia", tau3, W3.o, W4.v)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qow,qpw->op", tau10, tau8, tau9)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("im,mi->m", W1.o, tau1)
    )

    tau3 = (
        einsum("n,nm->m", tau2, X)
    )

    tau4 = (
        einsum("m,im,am->ia", tau3, W3.o, W4.v)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau8, tau9)
    )

    RZ += (
        - einsum("qow,qpw,qow->op", tau0, tau10, tau7)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("im,mi->m", W1.o, tau4)
    )

    tau6 = (
        einsum("n,nm->m", tau5, X)
    )

    tau7 = (
        einsum("m,im,am->ia", tau6, W3.o, W4.v)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau9)
    )

    RZ += (
        - einsum("qpw,qow,qpw->op", tau10, tau2, tau3)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau4 = (
        einsum("im,mi->m", W1.o, tau3)
    )

    tau5 = (
        einsum("n,nm->m", tau4, X)
    )

    tau6 = (
        einsum("m,im,am->ia", tau5, W3.o, W4.v)
    )

    tau7 = (
        einsum("ai,ja->ij", T1, tau6)
    )

    tau8 = (
        einsum("jo,ij->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau9)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau10)
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("kw,ko,owl,likj->owij", D2, Y2.conj(), tau1, tau7)
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("kw,ko,owl,lijk->owij", D2, Y2.conj(), tau1, tau7)
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau8) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau5, tau7, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("mn,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("iw,io,mi->omw", D2, Y2.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau9 = (
        einsum("im,owi->omw", W1.o, tau8)
    )

    RZ += (
        einsum("pmw,omw,omw->op", tau5, tau7, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("jm,om,omi->oij", W1.o, tau3, tau6)
    )

    tau8 = (
        einsum("op,pij->oij", Z, tau7)
    )

    tau9 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau2, tau8)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("jm,om,omi->oij", W1.o, tau3, tau6)
    )

    tau8 = (
        einsum("po,pij->oij", Z, tau7)
    )

    tau9 = (
        einsum("iw,ip,pwj,oij->opw", D2, Y2.conj(), tau2, tau8)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    RZ += (
        einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D4, Y4.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    RZ += (
        einsum("iw,io,owj,pwji->op", D2, Y2.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau4 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau5 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau6 = (
        einsum("in,mi->mn", W3.o, tau5)
    )

    tau7 = (
        einsum("in,nm,nm->mi", W1.o, X, tau6)
    )

    tau8 = (
        einsum("am,mi->ia", W4.v, tau7)
    )

    tau9 = (
        einsum("ao,ia->oi", Y3, tau8)
    )

    tau10 = (
        einsum("oi,pwi->opw", tau9, tau4)
    )

    RZ += (
        einsum("qow,qpw,qow->op", tau10, tau2, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau3 = (
        einsum("aw,ai,ao->owi", D1, T1, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau5 = (
        einsum("in,mi->mn", W3.o, tau4)
    )

    tau6 = (
        einsum("in,nm,nm->mi", W1.o, X, tau5)
    )

    tau7 = (
        einsum("am,mi->ia", W4.v, tau6)
    )

    tau8 = (
        einsum("ao,ia->oi", Y1, tau7)
    )

    tau9 = (
        einsum("oi,pwi->opw", tau8, tau3)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau9)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau10) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("op,ip,pm->omi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau4)
    )

    tau6 = (
        einsum("jo,mij->omi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,opw,omi->opmw", D2, Y2.conj(), tau3, tau6)
    )

    tau8 = (
        einsum("pmi,pomw->omwi", tau2, tau7)
    )

    tau9 = (
        einsum("im,omwj->owij", W1.o, tau8)
    )

    RZ += (
        einsum("iw,ip,pwj,owji->op", D4, Y4.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("op,ip,pm->omi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau5, tau4, tau7)
    )

    tau9 = (
        einsum("mn,onw->omw", X, tau8)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau3, tau9) / 2
    )
    tau0 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau1 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau0)
    )

    tau2 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau3 = (
        einsum("im,owi->omw", W1.o, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau5 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau6 = (
        einsum("ip,po,pm->omi", Y2, Z, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau7, tau4, tau6)
    )

    tau9 = (
        einsum("mn,onw->omw", X, tau8)
    )

    RZ += (
        einsum("pmw,pmw,omw->op", tau1, tau3, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau2)
    )

    tau4 = (
        einsum("jo,mij->omi", Y4, tau3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W1.o, tau1, tau4)
    )

    tau6 = (
        einsum("op,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau7, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau3 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau2)
    )

    tau4 = (
        einsum("jo,mij->omi", Y2, tau3)
    )

    tau5 = (
        einsum("jm,om,omi->oij", W1.o, tau1, tau4)
    )

    tau6 = (
        einsum("po,pij->oij", Z, tau5)
    )

    tau7 = (
        einsum("iw,ip,pwj,oij->opw", D4, Y4.conj(), tau0, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau7, tau8, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("in,nm,nm->mi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W4.v, tau4)
    )

    tau6 = (
        einsum("ao,ia->oi", Y3, tau5)
    )

    tau7 = (
        einsum("oi,pwi->opw", tau6, tau1)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau8, tau9)
    )

    RZ += (
        einsum("qpw,qow,qpw->op", tau0, tau10, tau7) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau1 = (
        einsum("aw,ai,ao->owi", D3, T1, Y3.conj())
    )

    tau2 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau3 = (
        einsum("in,mi->mn", W3.o, tau2)
    )

    tau4 = (
        einsum("in,nm,nm->mi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("am,mi->ia", W4.v, tau4)
    )

    tau6 = (
        einsum("ao,ia->oi", Y1, tau5)
    )

    tau7 = (
        einsum("oi,pwi->opw", tau6, tau1)
    )

    tau8 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qow,qpw,qow->op", tau10, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau4 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W3.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau7 = (
        einsum("jn,nm,ni->mij", W1.o, X, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y2, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W2.v)
    )

    tau2 = (
        einsum("iw,io,mi->omw", D4, Y4.conj(), tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau4 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("im,pmw,opwi->opmw", W1.o, tau2, tau4)
    )

    tau6 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau7 = (
        einsum("mn,jn,ni->mij", X, W3.o, tau6)
    )

    tau8 = (
        einsum("jo,mij->omi", Y4, tau7)
    )

    tau9 = (
        einsum("omi,opmw->opwi", tau8, tau5)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau3 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau4 = (
        einsum("im,ni->mn", W1.o, tau3)
    )

    tau5 = (
        einsum("mn,in,mn->mi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("am,mi->ia", W2.v, tau5)
    )

    tau7 = (
        einsum("ai,ja->ij", T1, tau6)
    )

    tau8 = (
        einsum("jo,ij->oi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau2, tau9)
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau0, tau1, tau10) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau2 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau4 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau5 = (
        einsum("im,ni->mn", W1.o, tau4)
    )

    tau6 = (
        einsum("mn,in,mn->mi", X, W3.o, tau5)
    )

    tau7 = (
        einsum("am,mi->ia", W2.v, tau6)
    )

    tau8 = (
        einsum("ai,ja->ij", T1, tau7)
    )

    tau9 = (
        einsum("jo,ij->oi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau9)
    )

    RZ += (
        einsum("qow,qpw,qow->op", tau10, tau2, tau3) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,ni->mn", W1.o, tau1)
    )

    tau3 = (
        einsum("mn,in,mn->mi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,mi->ia", W2.v, tau3)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y4, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau8, tau9)
    )

    RZ += (
        einsum("qpw,qow,qpw->op", tau0, tau10, tau7) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("ai,am->mi", T1, W4.v)
    )

    tau2 = (
        einsum("im,ni->mn", W1.o, tau1)
    )

    tau3 = (
        einsum("mn,in,mn->mi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,mi->ia", W2.v, tau3)
    )

    tau5 = (
        einsum("ai,ja->ij", T1, tau4)
    )

    tau6 = (
        einsum("jo,ij->oi", Y2, tau5)
    )

    tau7 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau6)
    )

    tau8 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    RZ += (
        einsum("qow,qpw,qow->op", tau10, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("mn,on,on->om", X, tau6, tau7)
    )

    tau9 = (
        einsum("pm,pm,om->op", tau4, tau5, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau12)
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("pm,om,om->op", tau2, tau3, tau4)
    )

    tau6 = (
        einsum("pq,oq->op", Z, tau5)
    )

    tau7 = (
        einsum("qp,qo->op", Z, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", tau7, tau8, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau10, tau11, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau8 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau9 = (
        einsum("pm,om,om->op", tau6, tau7, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau12)
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("oq,pq->op", Z, tau9)
    )

    tau11 = (
        einsum("oq,pq->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,om,om->oa", W2.v, tau2, tau6)
    )

    tau8 = (
        einsum("bo,oa->ab", Y1, tau7)
    )

    tau9 = (
        einsum("bo,ba->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("op,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,om,om->oi", W1.o, tau3, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y2, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau11, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("nm,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau7 = (
        einsum("am,om,om->oa", W4.v, tau5, tau6)
    )

    tau8 = (
        einsum("ap,oa->op", Y3, tau7)
    )

    tau9 = (
        einsum("ap,po->oa", Y3, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y4, tau8)
    )

    tau10 = (
        einsum("pq,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qp,qo->op", Z, tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,om,pm->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oqi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau2 = (
        einsum("mn,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau3, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau12, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("io,pqi->opq", Y2, tau6)
    )

    tau8 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 2
    )
    tau0 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,om,om->oa", W2.v, tau0, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y1, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau12 = (
        einsum("qo,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau12, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("io,oq,pqi->opq", Y2, Z, tau6)
    )

    tau8 = (
        einsum("iq,opq->opi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("am,om,om->oa", W2.v, tau1, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y1, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qpw,qow,qow->op", tau12, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("op,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau6 = (
        einsum("im,pm,om->opi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("oq,iq,opi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("nm,on,on->om", X, tau0, tau1)
    )

    tau3 = (
        einsum("po,pm->om", Z, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("am,om,om->oa", W4.v, tau3, tau4)
    )

    tau6 = (
        einsum("ap,oa->op", Y3, tau5)
    )

    tau7 = (
        einsum("ap,po->oa", Y3, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau8, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("nm,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("po,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau3, tau7)
    )

    tau9 = (
        einsum("io,pqi->opq", Y2, tau8)
    )

    tau10 = (
        einsum("oq,iq,opq->opi", Z, Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("io,opi->op", Y2, tau4)
    )

    tau6 = (
        einsum("pq,oq->op", Z, tau5)
    )

    tau7 = (
        einsum("qp,qo->op", Z, tau6)
    )

    tau8 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau9 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", tau7, tau8, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("im,om,om->oi", W1.o, tau4, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau5 = (
        einsum("mn,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau8 = (
        einsum("im,om,om->oi", W1.o, tau6, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y4, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau11, tau12, tau2) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("mn,on,on->om", X, tau4, tau5)
    )

    tau7 = (
        einsum("op,pm->om", Z, tau6)
    )

    tau8 = (
        einsum("am,om,om->oa", W2.v, tau3, tau7)
    )

    tau9 = (
        einsum("bo,oa->ab", Y1, tau8)
    )

    tau10 = (
        einsum("bo,ba->oa", Y1, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau10)
    )

    tau12 = (
        einsum("qo,qpw,qpw->opw", Z, tau11, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("mn,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("op,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,om,om->oi", W1.o, tau0, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau12, tau8, tau9) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("mn,on,on->om", X, tau2, tau3)
    )

    tau5 = (
        einsum("po,pm->om", Z, tau4)
    )

    tau6 = (
        einsum("im,om,pm->opi", W1.o, tau1, tau5)
    )

    tau7 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau6)
    )

    tau8 = (
        einsum("iq,qop->opi", Y2, tau7)
    )

    tau9 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 2
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("nm,on,on->om", X, tau1, tau2)
    )

    tau4 = (
        einsum("po,pm->om", Z, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,om,om->oa", W4.v, tau4, tau5)
    )

    tau7 = (
        einsum("ao,pa->op", Y1, tau6)
    )

    tau8 = (
        einsum("ap,op->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau8)
    )

    tau10 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        - einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau8 = (
        einsum("am,om,om->oa", W4.v, tau6, tau7)
    )

    tau9 = (
        einsum("ao,pa->op", Y1, tau8)
    )

    tau10 = (
        einsum("ap,op->oa", Y3, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau10)
    )

    tau12 = (
        einsum("qo,qpw,qpw->opw", Z, tau11, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("nm,on,on->om", X, tau3, tau4)
    )

    tau6 = (
        einsum("po,pm->om", Z, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("im,pm,om->opi", W3.o, tau6, tau7)
    )

    tau9 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iq,qop->opi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,qoi,qpw->opw", D4, Y4.conj(), tau10, tau2)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("om,pm,omi->opi", tau6, tau7, tau5)
    )

    tau9 = (
        einsum("ip,opi->op", Y2, tau8)
    )

    tau10 = (
        einsum("qp,oq->op", Z, tau9)
    )

    tau11 = (
        einsum("qo,qp->op", Z, tau10)
    )

    tau12 = (
        einsum("qo,qpw,qpw->opw", tau11, tau2, tau3)
    )

    RZ += (
        - einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 2
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau5 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W4.v, tau5)
    )

    tau7 = (
        einsum("ap,oa->op", Y3, tau6)
    )

    tau8 = (
        einsum("ap,po->oa", Y1, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("mn,in,on->omi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("io,op,qpi->opq", Y2, Z, tau7)
    )

    tau9 = (
        einsum("iq,oqp->opi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau2, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau7, tau6, tau9)
    )

    RZ += (
        einsum("pmw,omw->op", tau10, tau5) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("io,pmi->opm", Y2, tau2)
    )

    tau4 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau5 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau4)
    )

    tau6 = (
        einsum("im,opwi->opmw", W3.o, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("op,ip,pm->omi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("opm,pqmw,oqmw->opqw", tau3, tau6, tau9)
    )

    tau11 = (
        einsum("iw,iq,oqpw->opwi", D2, Y4, tau10)
    )

    tau12 = (
        einsum("qow,qpwi->opi", tau0, tau11)
    )

    RZ += (
        einsum("io,opi->op", Y2.conj(), tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("mn,in,on->omi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("pm,om,pmi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,qop->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,qoi->opw", D4, Y4.conj(), tau2, tau9)
    )

    tau11 = (
        einsum("oq,qpw->opw", Z, tau10)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("po,poi->oi", Z, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y4, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau11, tau2)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau4 = (
        einsum("om,pm,omi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("po,poi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y4, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau12, tau8, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,opq->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("qo,qpw->opw", Z, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("pm,om,pmi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("op,opi->oi", Z, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y2, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y4, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau11, tau2)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau5 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau4, tau3)
    )

    tau6 = (
        einsum("am,om->oa", W2.v, tau5)
    )

    tau7 = (
        einsum("bo,oa->ab", Y3, tau6)
    )

    tau8 = (
        einsum("bo,ba->oa", Y3, tau7)
    )

    tau9 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau8)
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W4.v, tau3)
    )

    tau5 = (
        einsum("ap,oa->op", Y3, tau4)
    )

    tau6 = (
        einsum("ap,po->oa", Y1, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau11 = (
        einsum("qo,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau11, tau7, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("io,op,qpi->opq", Y2, Z, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y4, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau11 = (
        einsum("qo,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qpw,qow->op", tau11, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau1 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau3 = (
        einsum("in,nm,on->omi", W1.o, X, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("jm,omi,omk->oijk", W3.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,lijk->oijk", Y2, tau7)
    )

    tau9 = (
        einsum("jw,jp,opwk,okij->opwi", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau10, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("im,opwi->opmw", W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau5 = (
        einsum("in,nm,on->omi", W1.o, X, tau4)
    )

    tau6 = (
        einsum("ip,omi->opm", Y2, tau5)
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau8 = (
        einsum("op,ip,pm->omi", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau8)
    )

    tau10 = (
        einsum("opm,oqmw,pqmw->opqw", tau6, tau3, tau9)
    )

    tau11 = (
        einsum("iw,iq,qopw->opwi", D2, Y2, tau10)
    )

    tau12 = (
        einsum("qow,qpwi->opi", tau0, tau11)
    )

    RZ += (
        einsum("io,opi->op", Y2.conj(), tau12) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W4.v, tau4)
    )

    tau6 = (
        einsum("ao,pa->op", Y1, tau5)
    )

    tau7 = (
        einsum("ap,op->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("qo,qpw,qpw->opw", Z, tau0, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("pm,om,pmi->opi", tau2, tau3, tau1)
    )

    tau5 = (
        einsum("op,opi->oi", Z, tau4)
    )

    tau6 = (
        einsum("jo,oi->ij", Y2, tau5)
    )

    tau7 = (
        einsum("jo,ji->oi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau12 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau11)
    )

    RZ += (
        einsum("qpw,qow,qow->op", tau12, tau8, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau2 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("mn,in,on->omi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau6 = (
        einsum("op,ip,pm->omi", Z, Y4, tau5)
    )

    tau7 = (
        einsum("im,omj,omk->oijk", W1.o, tau4, tau6)
    )

    tau8 = (
        einsum("ko,oijl->ijkl", Y2, tau7)
    )

    tau9 = (
        einsum("lo,lijk->oijk", Y2, tau8)
    )

    tau10 = (
        einsum("jw,jp,opwk,okji->opwi", D4, Y4.conj(), tau2, tau9)
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau0, tau10) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y1, Y1.conj())
    )

    tau1 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau2 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau1, tau0, tau3)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau9, tau6, tau8)
    )

    RZ += (
        einsum("pmw,omw->op", tau10, tau5) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("po,poi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y4, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("po,poi->oi", Z, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y4, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau11, tau12, tau2) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,opi->oi", Z, tau5)
    )

    tau7 = (
        einsum("jo,oi->ij", Y2, tau6)
    )

    tau8 = (
        einsum("jo,ji->oi", Y4, tau7)
    )

    tau9 = (
        einsum("iw,ip,oi->opw", D2, Y2.conj(), tau8)
    )

    tau10 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau12 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qow,qpw,qpw->op", tau10, tau11, tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("pm,om,pmi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("op,ip,oqi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,qop->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,qoi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("oq,qpw->opw", Z, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 4
    )
    tau0 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau1 = (
        einsum("mn,in,on->omi", X, W3.o, tau0)
    )

    tau2 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau3 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau2, tau1)
    )

    tau4 = (
        einsum("am,om->oa", W2.v, tau3)
    )

    tau5 = (
        einsum("bo,oa->ab", Y3, tau4)
    )

    tau6 = (
        einsum("bo,ba->oa", Y3, tau5)
    )

    tau7 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau6)
    )

    tau8 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau9 = (
        einsum("oq,qpw,qpw->opw", Z, tau7, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau11 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau10, tau11, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("mn,in,on->omi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("ip,pq,oqi->opq", Y2, Z, tau7)
    )

    tau9 = (
        einsum("iq,opq->opi", Y4, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau2, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau8 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau9 = (
        einsum("op,ip,pm->omi", Z, Y4, tau8)
    )

    tau10 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau7, tau6, tau9)
    )

    RZ += (
        einsum("pmw,omw->op", tau10, tau5) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y3)
    )

    tau3 = (
        einsum("op,ip,pm->omi", Z, Y4, tau2)
    )

    tau4 = (
        einsum("jm,omi,omk->oijk", W3.o, tau1, tau3)
    )

    tau5 = (
        einsum("ko,oijl->ijkl", Y2, tau4)
    )

    tau6 = (
        einsum("lo,iljk->oijk", Y4, tau5)
    )

    tau7 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau8 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("jw,jp,okji,opwk->opwi", D4, Y4.conj(), tau6, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau10, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau1 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau0)
    )

    tau2 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau3 = (
        einsum("mn,in,on->omi", X, W3.o, tau2)
    )

    tau4 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau5 = (
        einsum("op,ip,pm->omi", Z, Y4, tau4)
    )

    tau6 = (
        einsum("im,omj,omk->oijk", W1.o, tau3, tau5)
    )

    tau7 = (
        einsum("ko,oijl->ijkl", Y2, tau6)
    )

    tau8 = (
        einsum("lo,iljk->oijk", Y4, tau7)
    )

    tau9 = (
        einsum("jw,jp,opwk,okij->opwi", D4, Y4.conj(), tau1, tau8)
    )

    tau10 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    RZ += (
        einsum("iw,io,qow,qpwi->op", D2, Y2.conj(), tau10, tau9) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("io,op,pm,pmi->om", Y2, Z, tau5, tau4)
    )

    tau7 = (
        einsum("am,om->oa", W4.v, tau6)
    )

    tau8 = (
        einsum("ao,pa->op", Y1, tau7)
    )

    tau9 = (
        einsum("ap,op->oa", Y1, tau8)
    )

    tau10 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau9)
    )

    tau11 = (
        einsum("qo,qpw,qpw->opw", Z, tau10, tau2)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau4 = (
        einsum("in,nm,on->omi", W1.o, X, tau3)
    )

    tau5 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau6 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau7 = (
        einsum("pm,om,pmi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("op,opi->oi", Z, tau7)
    )

    tau9 = (
        einsum("jo,oi->ij", Y2, tau8)
    )

    tau10 = (
        einsum("jo,ji->oi", Y2, tau9)
    )

    tau11 = (
        einsum("iw,ip,oi->opw", D4, Y4.conj(), tau10)
    )

    tau12 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    RZ += (
        einsum("qpw,qpw,qow->op", tau11, tau12, tau2) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y3)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y1, Y3.conj())
    )

    tau7 = (
        einsum("iq,qo,qpw->opwi", Y2, Z, tau6)
    )

    tau8 = (
        einsum("im,opwi->opmw", W3.o, tau7)
    )

    tau9 = (
        einsum("opm,pqmw,oqmw->opqw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("iw,iq,qopw->opwi", D2, Y4, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau12 = (
        einsum("qow,qpwi->opi", tau11, tau10)
    )

    RZ += (
        einsum("io,opi->op", Y2.conj(), tau12) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("ip,po,pm->omi", Y2, Z, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y4)
    )

    tau4 = (
        einsum("iw,io,pm,pow,pmi->omw", D2, Y2.conj(), tau3, tau0, tau2)
    )

    tau5 = (
        einsum("nm,onw->omw", X, tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau8 = (
        einsum("ip,po,pm->omi", Y2, Z, tau7)
    )

    tau9 = (
        einsum("im,io->om", W3.o, Y4)
    )

    tau10 = (
        einsum("iw,io,pm,pow,pmi->omw", D4, Y4.conj(), tau9, tau6, tau8)
    )

    RZ += (
        einsum("pmw,omw->op", tau10, tau5) / 4
    )
    tau0 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau1 = (
        einsum("in,nm,on->omi", W1.o, X, tau0)
    )

    tau2 = (
        einsum("ip,omi->opm", Y4, tau1)
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("ip,po,pm->omi", Y2, Z, tau3)
    )

    tau5 = (
        einsum("iw,ip,omi->opmw", D4, Y4.conj(), tau4)
    )

    tau6 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau7 = (
        einsum("oq,iq,qpw->opwi", Z, Y4, tau6)
    )

    tau8 = (
        einsum("im,opwi->opmw", W3.o, tau7)
    )

    tau9 = (
        einsum("opm,pqmw,oqmw->opqw", tau2, tau5, tau8)
    )

    tau10 = (
        einsum("iw,iq,qopw->opwi", D2, Y2, tau9)
    )

    tau11 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau12 = (
        einsum("qow,qpwi->opi", tau11, tau10)
    )

    RZ += (
        einsum("io,opi->op", Y2.conj(), tau12) / 4
    )
    tau0 = (
        einsum("iw,io,ip->opw", D2, Y2, Y2.conj())
    )

    tau1 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau2 = (
        einsum("mn,in,on->omi", X, W3.o, tau1)
    )

    tau3 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau4 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau3, tau2)
    )

    tau5 = (
        einsum("am,om->oa", W2.v, tau4)
    )

    tau6 = (
        einsum("bo,oa->ab", Y3, tau5)
    )

    tau7 = (
        einsum("bo,ba->oa", Y1, tau6)
    )

    tau8 = (
        einsum("aw,ap,oa->opw", D1, Y1.conj(), tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qow,qpw,qow->op", tau0, tau11, tau8) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("oq,qpw,qpw->opw", Z, tau0, tau1)
    )

    tau3 = (
        einsum("iw,io,ip->opw", D4, Y2, Y4.conj())
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("mn,in,on->omi", X, W3.o, tau4)
    )

    tau6 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau7 = (
        einsum("po,io,pm,pmi->om", Z, Y4, tau6, tau5)
    )

    tau8 = (
        einsum("am,om->oa", W2.v, tau7)
    )

    tau9 = (
        einsum("bo,oa->ab", Y3, tau8)
    )

    tau10 = (
        einsum("bo,ba->oa", Y1, tau9)
    )

    tau11 = (
        einsum("aw,ap,oa->opw", D3, Y3.conj(), tau10)
    )

    RZ += (
        einsum("qpw,qow,qpw->op", tau11, tau2, tau3) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("iw,io,ip->opw", D2, Y4, Y2.conj())
    )

    tau2 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau3 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau4 = (
        einsum("mn,in,on->omi", X, W3.o, tau3)
    )

    tau5 = (
        einsum("im,io->om", W1.o, Y2)
    )

    tau6 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau7 = (
        einsum("om,pm,omi->opi", tau5, tau6, tau4)
    )

    tau8 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau7)
    )

    tau9 = (
        einsum("iq,oqp->opi", Y2, tau8)
    )

    tau10 = (
        einsum("iw,ip,qpw,oqi->opw", D4, Y4.conj(), tau2, tau9)
    )

    tau11 = (
        einsum("qo,qpw->opw", Z, tau10)
    )

    RZ += (
        einsum("qow,qow,qpw->op", tau0, tau1, tau11) / 4
    )
    tau0 = (
        einsum("aw,ao,ap->opw", D1, Y3, Y1.conj())
    )

    tau1 = (
        einsum("am,ao->om", W2.v, Y1)
    )

    tau2 = (
        einsum("in,nm,on->omi", W1.o, X, tau1)
    )

    tau3 = (
        einsum("im,io->om", W3.o, Y2)
    )

    tau4 = (
        einsum("am,ao->om", W4.v, Y1)
    )

    tau5 = (
        einsum("om,pm,omi->opi", tau3, tau4, tau2)
    )

    tau6 = (
        einsum("pq,iq,opi->opq", Z, Y4, tau5)
    )

    tau7 = (
        einsum("iq,oqp->opi", Y2, tau6)
    )

    tau8 = (
        einsum("iw,ip,qpw,oqi->opw", D2, Y2.conj(), tau0, tau7)
    )

    tau9 = (
        einsum("aw,ao,ap->opw", D3, Y3, Y3.conj())
    )

    tau10 = (
        einsum("iw,io,ip->opw", D4, Y4, Y4.conj())
    )

    tau11 = (
        einsum("oq,qpw,qpw->opw", Z, tau10, tau9)
    )

    RZ += (
        einsum("qpw,qow->op", tau11, tau8) / 4
    )

    return RZ


def gen_A1(Y1, Y2, Y3, Y4, Z):

    tau0 = (
        einsum("io,ip->op", Y2, Y2.conj())
    )

    tau1 = (
        einsum("ao,ap->op", Y3, Y3.conj())
    )

    tau2 = (
        einsum("io,ip->op", Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oq,oq->op", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qp->op", Z, tau3)
    )

    A1 = (
        einsum("po,po->op", tau0, tau4)
    )

    return A1

def gen_A2(Y1, Y2, Y3, Y4, Z):
    tau0 = (
        einsum("ao,ap->op", Y1, Y1.conj())
    )

    tau1 = (
        einsum("ao,ap->op", Y3, Y3.conj())
    )

    tau2 = (
        einsum("io,ip->op", Y4, Y4.conj())
    )

    tau3 = (
        einsum("pq,oq,oq->op", Z.conj(), tau1, tau2)
    )

    tau4 = (
        einsum("oq,qp->op", Z, tau3)
    )

    A2 = (
        einsum("po,po->op", tau0, tau4)
    )

    return A2

def gen_A3(Y1, Y2, Y3, Y4, Z):

    tau0 = (
        einsum("ao,ap->op", Y1, Y1.conj())
    )

    tau1 = (
        einsum("io,ip->op", Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oq,oq->op", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qp->op", Z, tau2)
    )

    tau4 = (
        einsum("io,ip->op", Y4, Y4.conj())
    )

    A3 = (
        einsum("po,po->op", tau3, tau4)
    )

    return A3

def gen_A4(Y1, Y2, Y3, Y4, Z):

    tau0 = (
        einsum("ao,ap->op", Y1, Y1.conj())
    )

    tau1 = (
        einsum("io,ip->op", Y2, Y2.conj())
    )

    tau2 = (
        einsum("qp,oq,oq->op", Z.conj(), tau0, tau1)
    )

    tau3 = (
        einsum("qo,qp->op", Z, tau2)
    )

    tau4 = (
        einsum("ao,ap->op", Y3, Y3.conj())
    )

    A4 = (
        einsum("po,po->op", tau3, tau4)
    )

    return A4

def gen_AZr(Y1, Y2, Y3, Y4, Z):

    tau0 = (
        einsum("ao,ap->op", Y1, Y1.conj())
    )

    tau1 = (
        einsum("io,ip->op", Y2, Y2.conj())
    )

    AZr = (
        einsum("po,po->op", tau0, tau1)
    )

    return AZr

def gen_AZl(Y1, Y2, Y3, Y4, Z):

    tau0 = (
        einsum("ao,ap->op", Y3, Y3.conj())
    )

    tau1 = (
        einsum("io,ip->op", Y4, Y4.conj())
    )

    AZl = (
        einsum("po,po->op", tau0, tau1)
    )

    return AZl
