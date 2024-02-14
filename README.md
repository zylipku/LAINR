# LAINR
Official repository for the article:

Latent assimilation with implicit neural representations for unknown dynamics

https://arxiv.org/abs/2309.09574

Full codes have been updated for reference. Please do not hesitate to reach me if you have any problems.

# training
1. change directory to `src/`:
```
cd src/
```
2. Run
```
python main.py --phase=<phase_name> \ 
encoder_decoder=<encoder_decoder_name> \
latent_dynamics=<latent_dynamics_name> \ # if exists
uncertainty_est=<uncertainty_est_name> \ # if exists
<other_parameters> # if needed
```
to train the model.
See src/run.sh for examples.

# assimilation

See src/assimilate_main.py for examples.