"""Continuing two-skill learning pilot; an abstract auxiliary sensor, not fMRI.

Observations precede feedback and transitions. Assessment blocks do not change
state. Research logs are complete; only the personalization algorithm's retained history is compressed.
The neural interpretation requires an independently validated observation model.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import minimize
try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        return lambda f: f

STATES = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.int64)
REQ = np.array([[1,0],[0,1],[1,1]], dtype=np.int64)
ACTIVITIES = ('A','B','AB')

@dataclass(frozen=True)
class Config:
    n: int = 120
    blocks: int = 6
    capacity: int = 18
    epsilon: float = .10
    initial: str = 'skewed'
    gains: tuple = (.08,.045,.055,.04)  # a:A, a:AB, b:B, b:AB
    losses: tuple = (.008,.008)
    initial_mastery: tuple = (.20,.20)
    guess: float = .25
    slip: float = .10
    heterogeneity: float = .8  # persistent +/- shift of gain logits, equal mixture
    aux_loading: float = 1.0
    aux_sd: float = .8
    aux_drift: float = 0.0
    aux_subject_sd: float = 0.0
    retention_loss: tuple = (.15,.15)

@dataclass
class Observed:
    action: np.ndarray
    response: np.ndarray
    transition: np.ndarray  # 0 identity, 1 practice, 2 overnight
    auxiliary: np.ndarray
    assigned_probability: np.ndarray
    phase: np.ndarray

def retained_indices(history, rule, capacity=18):
    if capacity < 3 or capacity % 3:
        raise ValueError('Capacity must be a positive multiple of three, at least 3.')
    if rule == 'recency':
        return list(range(max(0,len(history)-capacity),len(history)))
    if rule != 'balanced':
        raise ValueError(rule)
    keep = set()
    for e in range(3):
        keep.update([j for j,(a,_) in enumerate(history) if a==e][-capacity//3:])
    need = capacity-len(keep)
    if need > 0:
        keep.update([j for j in range(len(history)) if j not in keep][-need:])
    return sorted(keep)

def policy(history, rule, capacity=18, epsilon=.1):
    if rule == 'random':
        return np.ones(3)/3
    if not 0 <= epsilon <= 1:
        raise ValueError('epsilon outside [0,1]')
    retained = [history[j] for j in retained_indices(history,rule,capacity)]
    weight = np.ones(3)
    for e in range(3):
        ys = [y for a,y in retained if a==e]
        weight[e] = 1 - (1+sum(ys))/(2+len(ys))
    return epsilon/3 + (1-epsilon)*weight/weight.sum()

def transition_matrix(gains,losses,e,shift=0):
    effective = expit(logit(np.asarray(gains))+shift)
    gain = np.zeros(2)
    if e==0: gain[0]=effective[0]
    if e==1: gain[1]=effective[2]
    if e==2: gain[:]=effective[[1,3]]
    out = np.ones((4,4))
    for j,s in enumerate(STATES):
        p = np.where(s==1,1-np.asarray(losses),gain)
        for k,sp in enumerate(STATES):
            out[j,k] = np.prod(np.where(sp==1,p,1-p))
    return out

def initial_dist(p):
    return np.prod(np.where(STATES==1,np.asarray(p),1-np.asarray(p)),axis=1)

def target(gains, heterogeneity):
    """Standardized equal-type gain contrasts; independent of policy occupancy."""
    rates = np.mean(expit(logit(np.asarray(gains))[None,:]+np.array([-1,1])[:,None]*heterogeneity),axis=0)
    return rates[[1,3]]-rates[[0,2]]

def simulate(config=Config(), rule='recency', seed=20261025):
    rng=np.random.default_rng(seed); c=config
    if rule not in ('recency','balanced','random'):
        raise ValueError(rule)
    if c.initial not in ('balanced','skewed'):
        raise ValueError(c.initial)
    # 18 initial + probe9 + each block(12 practice + probe9), immediate24,
    # overnight1, delayed24. Probe Z occurs once per 9-item block.
    tmax=18+9+c.blocks*21+24+1+24
    a=np.zeros((c.n,tmax),dtype=np.int64); y=np.full_like(a,-1)
    trans=np.zeros_like(a); z=np.full(a.shape,np.nan)
    prob=np.ones(a.shape); phase=np.zeros_like(a)
    before=np.zeros_like(a); after=np.zeros_like(a); types=np.zeros(c.n)
    for i in range(c.n):
        u=rng.choice([-1,1]);types[i]=u
        matrices=[transition_matrix(c.gains,c.losses,e,u*c.heterogeneity) for e in range(3)]
        s=rng.choice(4,p=initial_dist(c.initial_mastery));hist=[];t=0
        offset=rng.normal(0,c.aux_subject_sd)
        def event(e,kind,ph,p=1.0,sensor=False):
            nonlocal s,t
            a[i,t]=e;trans[i,t]=kind;phase[i,t]=ph;prob[i,t]=p;before[i,t]=s
            if kind==2:
                current=STATES[s].copy()
                current *= rng.random(2)>=np.asarray(c.retention_loss)
                s=int(current[0]+2*current[1])
            else:
                eta=int(np.all(STATES[s]>=REQ[e]))
                y[i,t]=int(rng.random()<c.guess+(1-c.slip-c.guess)*eta)
                if sensor:
                    z[i,t]=c.aux_loading*STATES[s].sum()+c.aux_drift*t/(tmax-1)+offset+rng.normal(0,c.aux_sd)
                if kind==1:
                    hist.append((e,int(y[i,t])))
                    s=rng.choice(4,p=matrices[e][s])
            after[i,t]=s;t+=1
        initial=np.repeat(np.arange(3),[6,6,6] if c.initial=='balanced' else [12,3,3]);rng.shuffle(initial)
        for e in initial:event(int(e),1,0)
        for k,e in enumerate(np.tile(np.arange(3),3)):event(int(e),0,1,sensor=k==0)
        for block in range(c.blocks):
            positions=rng.choice(12,3,replace=False);order=rng.permutation(3)
            scheduled=dict(zip(positions,order))
            for j in range(12):
                if j in scheduled:
                    event(int(scheduled[j]),1,2)
                else:
                    pi=policy(hist,rule,c.capacity,c.epsilon);e=int(rng.choice(3,p=pi))
                    event(e,1,3,float(pi[e]))
            for k,e in enumerate(np.tile(np.arange(3),3)):event(int(e),0,1,sensor=k==0)
        for e in np.tile(np.arange(3),8):event(int(e),0,4)
        event(0,2,5)
        for e in np.tile(np.arange(3),8):event(int(e),0,6)
        assert t==tmax
    obs=Observed(a,y,trans,z,prob,phase)
    truth={'before':before,'after':after,'type':types,'target':target(c.gains,c.heterogeneity),'config':asdict(c)}
    return obs,truth

def pack(theta, auxiliary, fixed_h, retention_loss):
    gains=expit(theta[:4]);losses=expit(theta[4:6]);p0=expit(theta[6:8])
    g=.5*expit(theta[8]);slip=.5*expit(theta[9])
    mats=np.zeros((2,3,4,4))
    for u,sgn in enumerate([-1,1]):
        for e in range(3):mats[u,e]=transition_matrix(gains,losses,e,sgn*fixed_h)
    p=initial_dist(p0)
    response=np.zeros((3,4))
    for e in range(3):
        response[e]=g+(1-g-slip)*np.all(STATES>=REQ[e],axis=1)
    retention=np.ones((4,4))
    for j,s in enumerate(STATES):
        pr=s*(1-np.array(retention_loss))
        for k,sp in enumerate(STATES):retention[j,k]=np.prod(np.where(sp,pr,1-pr))
    if auxiliary:
        mu=theta[10]+np.exp(theta[11])*STATES.sum(axis=1);sd=np.exp(theta[12])
    else:mu=np.zeros(4);sd=1.
    return mats,p,response,retention,mu,sd

@njit(cache=True,nogil=True)
def _forward(a,y,tr,z,mats,initial,response,retention,mu,sd,use_z):
    out=np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        f=np.empty((2,4))
        for u in range(2):
            for s in range(4):f[u,s]=.5*initial[s]
        ll=0.
        for t in range(a.shape[1]):
            e=a[i,t]
            logs=np.zeros(4)
            for s in range(4):
                if y[i,t]>=0:
                    pp=response[e,s]
                    logs[s]=math.log(pp if y[i,t]==1 else 1-pp)
                if use_z and not math.isnan(z[i,t]):
                    logs[s]+=-.5*((z[i,t]-mu[s])/sd)**2-math.log(sd)-.5*math.log(2*math.pi)
            mx=np.max(logs);total=0.
            for u in range(2):
                for s in range(4):
                    f[u,s]*=math.exp(logs[s]-mx);total+=f[u,s]
            if total<=0:return np.full(a.shape[0],-1e100)
            ll+=math.log(total)+mx
            f/=total
            if tr[i,t]>0:
                nxt=np.zeros((2,4))
                for u in range(2):
                    for s in range(4):
                        for sp in range(4):
                            pp=mats[u,e,s,sp] if tr[i,t]==1 else retention[s,sp]
                            nxt[u,sp]+=f[u,s]*pp
                f=nxt
        out[i]=ll
    return out

def loglik(theta,obs,auxiliary=False,fixed_h=0.,retention_loss=(.15,.15)):
    args=pack(theta,auxiliary,fixed_h,retention_loss)
    return _forward(obs.action,obs.response,obs.transition,obs.auxiliary,*args,auxiliary)

def start_theta(auxiliary=False):
    base=np.r_[logit([.06]*4),logit([.01,.01]),logit([.2,.2]),logit([.5,.2])]
    return np.r_[base,0.,np.log(.8),np.log(1.)] if auxiliary else base

def fit(obs,auxiliary=False,fixed_h=0.,retention_loss=(.15,.15),seed=1,starts=3,maxiter=350,intervals=True):
    """Known heterogeneity support and retention; other listed parameters learned.

    fixed_h=0 gives homogeneous dynamics. fixed_h>0 is an optimistic specified-
    mixture benchmark: it does not estimate population learning-rate heterogeneity.
    """
    if starts<1:raise ValueError('starts must be positive')
    rng=np.random.default_rng(seed)
    def objective(x):return -float(loglik(x,obs,auxiliary,fixed_h,retention_loss).sum())
    bounds=[(-7,1)]*4+[(-9,-.5)]*2+[(-5,3)]*2+[(-5,3)]*2
    if auxiliary:bounds += [(-4,4),(-4,2),(-3,3)]
    x0=start_theta(auxiliary); candidates=[]
    for j in range(starts):
        x=x0 if j==0 else x0+rng.normal(0,.55,len(x0))
        rr=minimize(objective,x,method='L-BFGS-B',bounds=bounds,options={'maxiter':maxiter,'ftol':1e-10,'gtol':1e-5})
        candidates.append(rr)
    finite=[r for r in candidates if np.isfinite(r.fun)]
    if not finite:raise RuntimeError('All starts produced nonfinite likelihood')
    best=min(finite,key=lambda r:r.fun);x=best.x
    est=target(expit(x[:4]),fixed_h)
    boundary=any(min(v-lo,hi-v)<.02 for v,(lo,hi) in zip(x,bounds))
    answer={'estimate':est.tolist(),'theta':x.tolist(),'nll':float(best.fun),'converged':bool(best.success),'boundary':bool(boundary),
            'starts':starts,'successful_starts':sum(bool(r.success) for r in candidates),'start_nll':[float(r.fun) for r in candidates],
            'same_best_starts':sum(abs(r.fun-best.fun)<1e-3 for r in candidates),'interval_valid':False,'auxiliary':auxiliary,'fixed_h':fixed_h,
            'ci_model':None,'ci_sandwich':None,'hessian_condition':None,'message':str(best.message)}
    if not intervals:return answer
    d=len(x);h=3e-4*np.maximum(1,np.abs(x));H=np.zeros((d,d));f0=objective(x)
    for j in range(d):
        ej=np.zeros(d);ej[j]=h[j]
        H[j,j]=(objective(x+ej)-2*f0+objective(x-ej))/h[j]**2
        for k in range(j):
            ek=np.zeros(d);ek[k]=h[k]
            H[j,k]=H[k,j]=(objective(x+ej+ek)-objective(x+ej-ek)-objective(x-ej+ek)+objective(x-ej-ek))/(4*h[j]*h[k])
    eig=np.linalg.eigvalsh(H)
    condition=float(eig[-1]/eig[0]) if eig[0]>0 else float('inf')
    answer['hessian_condition']=condition
    # Refuse regular Wald intervals at a failed optimum, boundary or singularity.
    if not best.success or boundary or eig[0]<=1e-6 or condition>1e9:return answer
    cov=np.linalg.inv(H);scores=np.zeros((obs.action.shape[0],d));J=np.zeros((2,d))
    for j in range(d):
        ej=np.zeros(d);ej[j]=h[j]
        scores[:,j]=(loglik(x+ej,obs,auxiliary,fixed_h,retention_loss)-loglik(x-ej,obs,auxiliary,fixed_h,retention_loss))/(2*h[j])
        J[:,j]=(target(expit((x+ej)[:4]),fixed_h)-target(expit((x-ej)[:4]),fixed_h))/(2*h[j])
    scores-=scores.mean(axis=0)
    sandwich=cov@(scores.T@scores)@cov
    for name,C in [('model',cov),('sandwich',sandwich)]:
        v=np.diag(J@C@J.T)
        if np.any(v<0) or not np.all(np.isfinite(v)):return answer
        se=np.sqrt(v);answer['ci_'+name]=np.column_stack([est-1.96*se,est+1.96*se]).tolist()
    answer['interval_valid']=True
    return answer

def score_heldout(theta,obs,cutoff,auxiliary=False,fixed_h=0.,retention_loss=(.15,.15)):
    """Sequential response predictions after cutoff, using only past Y and Z.

    The auxiliary value on the current trial is incorporated only AFTER the
    current response is predicted. Test outcomes update the filtering state;
    they never refit parameters. This evaluates one-step-ahead prediction.
    """
    mats,p,resp,ret,mu,sd=pack(theta,auxiliary,fixed_h,retention_loss)
    predicted=[];actual=[];f=np.tile(p/2,(obs.action.shape[0],2,1))
    for t in range(obs.action.shape[1]):
        e=obs.action[:,t];yp=obs.response[:,t]
        pp=np.sum(f*resp[e,None,:],axis=(1,2))
        valid=(yp>=0)
        if t>=cutoff:
            predicted.extend(pp[valid]);actual.extend(yp[valid])
        loge=np.zeros((len(yp),4))
        loge[valid]=np.where(yp[valid,None]==1,np.log(resp[e[valid]]),np.log1p(-resp[e[valid]]))
        zv=obs.auxiliary[:,t];have=~np.isnan(zv)
        if auxiliary:loge[have]+=-.5*((zv[have,None]-mu)/sd)**2-np.log(sd)-.5*np.log(2*np.pi)
        loge-=loge.max(axis=1,keepdims=True)
        f*=np.exp(loge[:,None,:]);f/=f.sum(axis=(1,2),keepdims=True)
        for kind in (1,2):
            idx=np.flatnonzero(obs.transition[:,t]==kind)
            for i in idx:
                for u in range(2):f[i,u]=f[i,u]@(mats[u,e[i]] if kind==1 else ret)
    p=np.clip(np.asarray(predicted),1e-9,1-1e-9);yy=np.asarray(actual)
    return {'n_predictions':len(p),'brier':float(np.mean((p-yy)**2)),'log_loss':float(-np.mean(yy*np.log(p)+(1-yy)*np.log1p(-p))),
            'mean_prediction':float(p.mean()),'accuracy':float(yy.mean()),'signed_calibration_bias':float(p.mean()-yy.mean())}
