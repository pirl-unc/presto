# Class I Best Hits — 6 Conditions x 3 Seeds = 18 Runs

## F1: d128 DAG heads_only pretrain lr=3e-4 warmup [ctrl=L2]

- seed=42: `class1-f1-s42-20260328a` app=`None`
- seed=43: `class1-f1-s43-20260328a` app=`None`
- seed=44: `class1-f1-s44-20260328a` app=`None`

## F2: d128 A07 heads_only pretrain lr=3e-4 warmup

- seed=42: `class1-f2-s42-20260328a` app=`None`
- seed=43: `class1-f2-s43-20260328a` app=`None`
- seed=44: `class1-f2-s44-20260328a` app=`None`

## F3: d128 DAG full pretrain lr=3e-4 warmup

- seed=42: `class1-f3-s42-20260328a` app=`None`
- seed=43: `class1-f3-s43-20260328a` app=`None`
- seed=44: `class1-f3-s44-20260328a` app=`None`

## F4: d128 DAG heads_only no-pretrain lr=3e-4 warmup

- seed=42: `class1-f4-s42-20260328a` app=`None`
- seed=43: `class1-f4-s43-20260328a` app=`None`
- seed=44: `class1-f4-s44-20260328a` app=`None`

## F5: d128 DAG full no-pretrain lr=3e-4 warmup

- seed=42: `class1-f5-s42-20260328a` app=`None`
- seed=43: `class1-f5-s43-20260328a` app=`None`
- seed=44: `class1-f5-s44-20260328a` app=`None`

## F6: d128 A03 heads_only pretrain lr=3e-4 warmup

- seed=42: `class1-f6-s42-20260328a` app=`None`
- seed=43: `class1-f6-s43-20260328a` app=`None`
- seed=44: `class1-f6-s44-20260328a` app=`None`

## Questions

| # | Question | Comparison |
|---|----------|------------|
| Q1 | DAG vs A07 at full scale? | F1 vs F2 |
| Q2 | full vs heads_only with more data? | F1 vs F3 |
| Q3 | Pretraining effect at 105 alleles? | F1 vs F4 |
| Q4 | Cold-start + full loss stress test | F5 |
| Q5 | 3-way architecture (DAG vs A07 vs A03) | F1 vs F2 vs F6 |
