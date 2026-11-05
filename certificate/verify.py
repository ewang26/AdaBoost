"""Reproduce the finite certificate for the accompanying solution.
Run with Python 3 and SymPy, with assertions enabled.
All acceptance checks use exact rational arithmetic.
"""
from pathlib import Path
from fractions import Fraction as Q
import json, sys, time, hashlib
assert __debug__, 'Run without Python optimization flags'
sys.dont_write_bytecode=True
sys.set_int_max_str_digits(0)
started=time.time()
print('Deriving rational maps and exact algebraic data',flush=True)
import algebra as c
def mat(M):return [[c.qiv(v) for v in row] for row in M]
# Input (x,t,z) -> old phase-zero (x,y,z); output old phase-one
# (x,y,z) -> reflected normalized (x,t,z).
Ci=mat([[1,0,0],[0,1,-Q(1,2)],[0,0,1]])
Do=mat([[1,0,0],[0,-1,-Q(1,2)],[0,0,1]])
LB=c.matmul_q(Do,c.matmul_q(c.L0B_iv,Ci))
LA=c.matmul_q(mat([[1,0],[0,-1]]),c.L0A_iv)
# Algebraically, these first-column entries vanish because the old ones do.
# Their interval enclosures need only contain the already-proved exact zeros.
assert LA[1][0].contains_zero()
assert all(LB[i][0].contains_zero() for i in [1,2])
theta_hat=c.thetaB0_iv+c.chiB_iv*Q(1,2)

# Check the outward enclosures printed in the solution. In particular, the
# separated A-marginal limits in the scalar noncycling proof require d_* > 1/4.
for interval,lo,hi in [
    (c.d_iv,'0.25799538295','0.25799538296'),
    (c.cA_iv,'0.30739560746557','0.30739560746558'),
    (c.cB_iv,'0.45153264872461','0.45153264872462'),
    (c.thetaA_iv,'0.580405272518281104','0.580405291461442916'),
    (theta_hat,'0.705067523141833','0.705067554266809'),
    (c.chiB_iv,'1.01349886578531906','1.01349888527878103'),
    (c.thetaB0_iv,'0.198318090249174404','0.198318111627417225'),
]:
    assert Q(lo)<=interval.lo<=interval.hi<=Q(hi), (lo,hi,c.fmt_iv(interval,24))
assert c.d_iv.lo>Q(1,4)

# Row permutation S swaps examples 3,4; its column permutation swaps hypotheses
# 3,4. Thus edges, branch maxima and all signed competitors permute exactly.
perm=[0,1,3,2]
for M in [c.LA,c.LB]:
    rperm=perm+([4] if M.rows==5 else [])
    assert all(M[rperm[r],j]==M[r,perm[j]] for r in range(M.rows) for j in range(4))
assert [perm[j] for j in c.H0]==c.H1

def compact(data):return {k:float(v) for k,v in data.items()}
def pair(iv):return [str(iv.lo),str(iv.hi)]
def short_enclosure(v):
    scale=10**18
    lo=v.numerator*scale//v.denominator
    hi=-((-v.numerator*scale)//v.denominator)
    return [str(Q(lo,scale)),str(Q(hi,scale))]
radius=Q(1,100000)
rho=Q(1,1000)
results=[]
ka=c.thetaA_iv.hi+rho
kd=c.thetaB0_iv.hi+Q(3,2)*rho
ke=c.chiB_iv.hi+rho
assert c.thetaB0_iv.lo>Q(3,2)*rho and c.chiB_iv.lo>rho
abox=[c.P_iv+c.qiv(0,radius),c.d_iv+c.qiv(0,ka*radius)]
bbox=[c.P_iv+c.qiv(0,radius),c.d_iv+c.qiv(0,kd*radius),c.qiv(0,ke*radius)]
print('Computing phase-zero Hessians at radius',radius,flush=True)
AH=[c.hessian_sup_matrix(f,(c.p,c.d),abox) for f in [c.F0A_p,c.F0A_d]]
BH=[c.hessian_sup_matrix(f,(c.p,c.d,c.e),bbox) for f in [c.F0B_p,c.F0B_d,c.F0B_e]]
ak=[c.A_component_K(h,ka) for h in AH]
bk=[c.B_component_K(h,kd,ke) for h in BH]
ax,ay=ak[0],ak[1]+c.thetaA_iv.hi*ak[0]
bx=bk[0];bt=bk[1]+Q(1,2)*bk[2]+theta_hat.hi*bx;bz=bk[2]+c.chiB_iv.hi*bx
asector=c.A_sector_constants(LA,ax,ay,radius=radius,rho=rho)
bsector=c.B_sector_constants(LB,bx,bt,bz,radius=radius,rho=rho)
wordA=c.certify_word_on_box(c.build_state_A_from_box(abox),c.LA_list,c.H0)
wordB=c.certify_word_on_box(c.build_state_B_from_box(bbox),c.LB_list,c.H0)
states=[c.build_state_A_from_box(abox),c.build_state_B_from_box(bbox)]
matrices=[c.LA_list,c.LB_list]
frozen=[c.edge_list_iv(states[g],matrices[g]) for g in [0,1]]
neggaps=[]
for g in [0,1]:
    state=states[g]
    for j in c.H0:
        edges=c.edge_list_iv(state,matrices[g])
        neggaps.append(edges[j].lo-max(-min(v.lo for v in edges),-min(v.lo for v in frozen[1-g])))
        state,_=c.update_iv(state,matrices[g],j)
branch=min(wordA[0],wordB[0])
lock=min(wordA[1],wordB[1])-max(wordA[2],wordB[2])
ngap=min(neggaps)
passes=all(0<s['x_lo']<s['x_hi']<1 and s['tau_hi']<rho and s['alpha']<1 for s in [asector,bsector]) and min(branch,lock,ngap)>0
record={'radius':str(radius),'rho':str(rho),'passed':passes,
        'A':{k:short_enclosure(v) for k,v in asector.items()},'B':{k:short_enclosure(v) for k,v in bsector.items()},
        'branch_gap':short_enclosure(branch),'burst_lock_gap':short_enclosure(lock),'negation_gap':short_enclosure(ngap)}
results.append(record)
print('Passed:',passes,'A:',compact(asector),'B:',compact(bsector),'gaps:',*[float(v) for v in [branch,lock,ngap]],flush=True)

assert len(results)==1 and results[0]['passed']
for stats in [asector,bsector]:
    assert stats['x_lo']>Q(3,10) and stats['x_hi']<Q(1,2)
    assert stats['alpha']<Q(7,10) and stats['beta']<10
    assert stats['gamma']<Q(1,3) and stats['delta']<6
assert branch>Q(1,40) and lock>Q(3,40) and ngap>Q(1,4)
assert Q(7,10)*Q(1,1000)+10*Q(1,100000)<Q(1,1000)
print('Normalized regions and common rational bounds passed',flush=True)

# Direct half-burst multiplier polynomials and norm certificates.
s=c.sp;z=s.Symbol('z')
qa=4*z**14-268*z**13-5676*z**12-76559*z**11+61076*z**10+5403070*z**9+3714640*z**8-44707134*z**7+159688998*z**6-80404208*z**5+1806746*z**4+2355093*z**3+127106*z**2+1846*z+2
qb=z**7-9*z**6+23*z**5-3*z**4-45*z**3+13*z**2+5*z-1
pa=c.P_lambda_manuscript.subs(c.x,z);pb=c.P_kappa_manuscript.subs(c.x,z)
assert s.expand(pa.subs(z,z*z)-qa*qa.subs(z,-z))==0
assert s.expand(pb.subs(z,z*z)+qb*qb.subs(z,-z))==0
norms={}
for name,poly,parent,interval,actual in [
    ('A',qa,pa,('0.30739560746557','0.30739560746558'),c.cA_iv),
    ('B',qb,pb,('0.45153264872461','0.45153264872462'),c.cB_iv)]:
    pp=s.Poly(poly,z);lo,hi=map(s.Rational,interval)
    assert pp.is_irreducible and pp.count_roots(lo,hi)==1
    assert s.Poly(poly.subs(z,-z),z).count_roots(lo,hi)==0
    assert Q(lo)<actual.lo<=actual.hi<Q(hi)
    assert s.Poly(parent,z).count_roots(lo*lo,hi*hi)==1
    norms[name]=str((-1)**pp.degree()*pp.TC()/pp.LC())
assert norms=={'A':'1/2','B':'1'}

# Full twenty-weight finite prefix; original and negated columns are all checked.
A=c.LA_list;B=c.LB_list
M=[a+b for a in A for b in B];M=[r+[-v for v in r]for r in M]
a_counts=[7,18,12,13];b_counts=[8,17,11,13,1]
counts=[a*b for a in a_counts for b in b_counts]
assert sum(counts)==2500 and min(counts)>0
W=[c.qiv(Q(v,2500))for v in counts]
scale=2**1024

def outward(v):
    lo=(v.lo*scale).__floor__();hi=(v.hi*scale).__ceil__()
    return c.qiv(Q(lo,scale),Q(hi,scale))
def total(vs):
    ans=c.qiv(0)
    for v in vs:ans=ans+v
    return ans
seq=[];prefix_gaps=[];last=[None,None]
for step in range(140):
    edges=c.edge_list_iv(W,M)
    chosen=[j for j in range(16)if all(edges[j].lo>edges[k].hi for k in range(16)if k!=j)]
    assert len(chosen)==1,(step,chosen)
    j=chosen[0];assert j<8
    prefix_gaps.append(edges[j].lo-max(edges[k].hi for k in range(16)if k!=j))
    W,_=c.update_iv(W,M,j)
    W=[outward(v)for v in W]
    assert min(v.lo for v in W)>0
    seq.append(j);last[j//4]=j%4
assert min(prefix_gaps)>Q(1,10**8) and last==[1,1]
phases=[0,0];bursts=[]
for k in range(28):
    word=seq[k*5:(k+1)*5];g=word[0]//4
    assert all(j//4==g for j in word)
    assert [j%4 for j in word]==[c.H0,c.H1][phases[g]]
    phases[g]^=1;bursts.append('AB'[g])
assert phases==[0,0]
assert ''.join(bursts)=='BBBABABBABABBABABBABABBABABB'
marginals=[[total(W[5*i:5*i+5])for i in range(4)],
           [total(W[5*i+j]for i in range(4))for j in range(5)]]
lo,hi=c.f_poly.refine_root(c.p_lo,c.p_hi,eps=s.Rational(1,10**50))
assert c.p_lo<=lo<hi<=c.p_hi and c.f_poly.count_roots(lo,hi)==1
assert c.f_poly.eval(lo)!=0 and c.f_poly.eval(hi)!=0
P=c.qiv(lo,hi);D=c.eval_expr_iv(c.Dexpr,{c.p:P})
endpoint=[]
for g,w in enumerate(marginals):
    x=c.qiv(1)-w[1]-P
    assert 0<x.lo<=x.hi<Q(1,40000000)
    theta=c.thetaA_iv if g==0 else c.thetaB0_iv
    y=(w[3]-D)/x-theta
    rr=[y]
    if g==1:
        zz=w[4]/x-c.chiB_iv
        rr=[y+zz/2,zz]
    assert all(v.abs_upper()<Q(1,1000)for v in rr)
    # Verify the exact outward bounds displayed in the solution's entry table.
    xb=[(Q('2.13810e-8'),Q('2.13811e-8')),(Q('1.31657e-8'),Q('1.31659e-8'))][g]
    assert xb[0]<x.lo<=x.hi<xb[1]
    rb=[[(Q('-1.54e-7'),Q('-1.34e-7'))],[(Q('4.84e-5'),Q('4.85e-5')),(Q('3.94e-4'),Q('3.95e-4'))]][g]
    assert all(lo<v.lo<=v.hi<hi for v,(lo,hi) in zip(rr,rb))
    endpoint.append({'phase':0,'x':pair(x),'normalized_residuals':[pair(v)for v in rr]})
# Exact primal and dual margin witnesses for both column sets.
colweights=[Q(v,10)for v in [2,1,1,1,2,1,1,1]]+[Q(0)]*8
assert min(sum(v*q for v,q in zip(row,colweights))for row in M)==Q(1,5)
u=[Q(1,5),Q(2,5),Q(1,5),Q(1,5)]
rowweights=[a*b for a in u for b in u+[Q(0)]]
assert max(sum(w*row[j]for w,row in zip(rowweights,M))for j in range(16))==Q(1,5)
output={'passed':True,'sympy_version':s.__version__,'elapsed_seconds':time.time()-started,
        'uniform_rows':2500,'margin':'1/5','strict_prefix_steps':140,
        'prefix_gap_lower':str(min(prefix_gaps)),'burst_word':''.join(bursts),
        'selected_columns_zeroindexed':seq,'endpoint':endpoint,'norms':norms,
        'normalized_local_certificate':results,
        'files_sha256':{name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()for name in ['verify.py','algebra.py']}}
Path(__file__).with_name('verification.json').write_text(json.dumps(output,indent=2)+'\n')
print('PASS: algebra, normalized regions, 140 strict updates, 2,500 rows, margin 1/5',flush=True)
