# default for pre-training
```
python main_pretrain.py
```

# ablation study
```
python main.py --phase=pretrain encoder_decoder=ablation/aeflow/b12_k7
```

# Assimilation

for the assimilation process, we always use the parameters $\sigma_x^b=0.1$, $\sigma^o=0.1$, $n_{\rm obs}=1024$, ensemble size 32 and inflation 1.05, and varying $\sigma_z^b$ and $\sigma^m$.
```
python assimilate_main.py sigma_z_b=0.1 sigma_m=0.1
```