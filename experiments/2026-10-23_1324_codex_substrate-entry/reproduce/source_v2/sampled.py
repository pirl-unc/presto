"""Continuous-entry path scores with irrelevant bond scores integrated out."""

import math

import torch

from common import entry_predict
from presto.models.independent_cut import partition_contains
from presto.models.substrate_entry import log_entry_fragment_expectation, sample_entry_partitions


def event_gradients(log_rates, log_entry, design, time_hours, starts, ends, coefficients, count, generator):
    draw=sample_entry_partitions(log_rates,log_entry,time_hours,count,generator=generator)
    present=partition_contains(draw.cuts,starts,ends).double()
    weights=present*coefficients[None,:]/count
    n=len(log_rates);h=draw.ages[:,None]*log_rates.exp()[None,:]
    safe=torch.where(draw.entered[:,None],h,1.)
    cut_score=safe*torch.exp(-safe)/(-torch.expm1(-safe))
    full_score=torch.where(draw.cuts,cut_score,-h)
    entry_score=torch.where(draw.entered,1-log_entry.exp()*draw.entry_times,-log_entry.exp()*time_hours)
    full_rates=(weights.sum(1)[:,None]*full_score).sum(0)@design
    entry_gradient=(weights.sum(1)*entry_score).sum()
    # Difference arrays accumulate internal-span occupancy in O(K*(spans+bonds)).
    internal=torch.zeros((count,n+1),dtype=log_rates.dtype)
    internal.scatter_add_(1,starts[None,:].expand(count,-1),weights)
    internal.scatter_add_(1,(ends-1)[None,:].expand(count,-1),-weights)
    internal=internal.cumsum(1)[:,:n]
    boundary=torch.zeros((count,n),dtype=log_rates.dtype)
    for index,valid in ((starts-1,starts>0),(ends-1,ends<=n)):
        safe_index=index.clamp(0,n-1)
        boundary.scatter_add_(1,safe_index[None,:].expand(count,-1),weights*valid[None,:])
    integrated=((boundary*cut_score-internal*h).sum(0))@design
    full=torch.cat((full_rates,entry_gradient.reshape(1)))
    reduced=torch.cat((integrated,entry_gradient.reshape(1)))
    return full,reduced,dict(partitions=count,entry_draws=count,bond_draws=draw.cuts.numel(),presence_terms=present.numel())


def training_gradient(theta, groups, design, *, order, count, generator, penalty=.01):
    full=torch.zeros_like(theta);reduced=torch.zeros_like(theta);work={}
    for g,x in zip(groups,design):
        if g['split']!='train':continue
        logs=x@theta[:-1];pred=entry_predict(logs,theta[-1],g,order)
        residual=2*g['weight']*(pred-g['target'])/math.log(10)
        combined=torch.zeros(len(g['point_times']),dtype=torch.float64)
        combined.scatter_add_(0,g['target_index'],residual)
        combined.scatter_add_(0,g['reference_index'],-residual)
        masses=log_entry_fragment_expectation(logs,theta[-1],g['point_starts'],g['point_ends'],g['point_times'],order=order,validate=False).exp()
        assert (masses>0).all(),'Quadrature mass underflow; no pseudocount substitution'
        for t in g['point_times'].unique().tolist():
            mask=g['point_times']==t
            a,b,counts=event_gradients(logs,theta[-1],x,t,g['point_starts'][mask],g['point_ends'][mask],combined[mask]/masses[mask],count,generator)
            full+=a;reduced+=b
            for k,v in counts.items():work[k]=work.get(k,0)+v
    reg=penalty*theta.clone();reg[0]=0;reg[-1]=0
    return full+reg,reduced+reg,work
