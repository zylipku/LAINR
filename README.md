# LAINR
Official repository for the LAINR framework

** Full codes will be published shortly. Currently only available upon request.
# usage
1. Run `python main.py --ds=<dataset_name> --ed=<encoder_name> --ld=none` to train the encoder-decoder model. (DDP enabled for multi-GPU training)
2. Run `python main.py --ds=<dataset_name> --ed=<encoder_name> --ld=<latent_dynamics_name>` for fine-tuning together with the latent dynamics model. (DDP enabled for multi-GPU training)
3. Run `bash ablation_sinr.sh` for batched assimilation with empirical assimilation parameters.

## optional:

3. Run `python main_uq.py --ds=<dataset_name> --ed=<encoder_name> --ld=<latent_dynamics_name> --uq=cholesky` to get the uncertainty estimator.
4. Run `python assimilate_uq.py --ds=<dataset_name> --uq=cholesky` to assimilate with uncertainty estimator.