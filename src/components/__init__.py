import logging

from .encoder_decoder import *
from .latent_dynamics import *
from .uncertainty import *

ed_name2class = {
    'pca': PCAED,
    'cae': CAEED,
    'aeflow': AEflowED,
    'fouriernetcartes': FourierNetCartesED,
    'fouriernetlatlon': FourierNetLatlonED,
    'sinr': SINRED,
    'sinr_noskip': SINRNoSkipED,
    'sinrv11': SINRv11ED,
    'sinrv11noskip': SINRv11NoSkipED,
    'sinrv12': SINRv12ED,
    'sinrv13': SINRv13ED,
}

ld_name2class = {
    'rezero': ReZeroDyn,
    'neuralode': NeuralODE,
}

uq_name2class = {
    'scalar': Scalar,
    'diagonal': Diagonal,
    'cholesky': Cholesky,
    'none': None,
}


def get_encoder_decoder(logger: logging.Logger, name: str, **kwargs) -> EncoderDecoder:

    lower_name = name.lower()
    ed_class: EncoderDecoder = ed_name2class[lower_name]
    # logger.info(f'Using [{ed_class.name}] as encoder-decoder')

    return ed_class(logger, **kwargs)


def get_latent_dynamics(logger: logging.Logger, name: str, **kwargs) -> LatentDynamics:

    lower_name = name.lower()
    ld_class: LatentDynamics = ld_name2class[lower_name]

    if ld_class is None:
        # logger.info(f'Using [None] as latent dynamics')
        return None
    else:
        # logger.info(f'Using [{ld_class.name}] as latent dynamics')
        return ld_class(logger, **kwargs)


def get_uncertainty_est(logger: logging.Logger, name: str, **kwargs) -> UncertaintyEst:

    lower_name = name.lower()
    uq_class: UncertaintyEst = uq_name2class[lower_name]

    if uq_class is None:
        # logger.info(f'Using [None] as latent dynamics')
        return None
    else:
        # logger.info(f'Using [{ld_class.name}] as latent dynamics')
        return uq_class(logger, **kwargs)
