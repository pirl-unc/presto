"""Independent dense-generator checks for the variable-time admission path."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.linalg import expm
from admission import directions, readout_weights
from rational import expectations
from reference import counts, analytical_logs, features
from independent_cut import ordered_bond_features

COEF=json.loads((Path(__file__).resolve().parents[1]/'reproduce/coefficients.json').read_text())


def dense(logs,eta,time):
    n=len(logs)+1
    spans=[(a,b) for a in range(n) for b in range(a+1,n+1)]
    index={s:i for i,s in enumerate(spans)}
    mat=np.zeros((len(spans),len(spans)))
    for j,(a,b) in enumerate(spans):
        for k in range(a+1,b):
            rate=np.exp(logs[k-1])*((b-a)/20)**eta
            mat[j,j]-=rate
            mat[index[a,k],j]+=rate
            mat[index[k,b],j]+=rate
    v=np.zeros(len(spans));v[index[0,n]]=1
    return spans,expm(time*mat)@v


def test_variable_times_against_dense_generator():
    logs=np.log([.03,.7,.12,.4,.02])
    for t in [.25,2.,24.]:
        for eta in [-1.,0.,1.]:
            spans,ref=dense(logs,eta,t)
            a,b=np.array(spans).T;b=b-1
            actual=expectations(torch.tensor(logs),torch.tensor(eta,dtype=torch.float64),t,substeps=2).numpy()
            independent=counts(logs,eta,t,8,COEF)
            np.testing.assert_allclose(actual[a,b],ref,rtol=2e-10,atol=2e-13)
            np.testing.assert_allclose(independent[a,b],ref,rtol=2e-10,atol=2e-13)
            if eta==0:
                np.testing.assert_allclose(analytical_logs(logs,spans,t),np.log(ref),rtol=1e-10,atol=1e-10)


def test_directions_and_signed_full_support_gradient():
    seq='ACDEFG'
    x=features(seq,.7)
    np.testing.assert_array_equal(x,ordered_bond_features(seq,context=.7).numpy())
    p=np.r_[np.sin(np.arange(40))*.1,.37];p[0]=-2
    spans,_=dense(x@p[:40],p[-1],1.7)
    a,b=np.array(spans).T;b=b-1
    w=readout_weights(len(spans))
    assert np.all(w!=0) and abs(np.abs(w).sum()-1)<1e-14
    d=directions();np.testing.assert_allclose(np.linalg.norm(d,axis=1),1)
    q=torch.tensor(p,requires_grad=True)
    y=expectations(torch.tensor(x)@q[:40],q[-1],1.7,substeps=2)
    grad=torch.autograd.grad((torch.tensor(w)*y[a,b].log()).sum(),q)[0].numpy()
    for direction in d:
        vals=[]
        for s in [-1,1]:
            v=p+s*1e-5*direction
            c=counts(x@v[:40],v[-1],1.7,8,COEF)
            vals.append(w@np.log(c[a,b]))
        np.testing.assert_allclose(grad@direction,(vals[1]-vals[0])/2e-5,atol=1e-6,rtol=1e-5)
