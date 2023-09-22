import logging

from .encoder_decoder import EncoderDecoder
from .encoder_decoder import CAEED, CAE400ED, AEflowED, SINRED

from .latent_dynamics import LatentDynamics
from .latent_dynamics import LinReg, ReZeroDyn, NeuralODE

from .uncertainty import Uncertainty
from .uncertainty import Vacuous, Diagonal, Cholesky

ed_name2class = {
    'cae': CAEED,
    'cae400': CAE400ED,
    'aeflow': AEflowED,
    'sinr': SINRED,
}

ld_name2class = {
    'linreg': LinReg,
    'rezero': ReZeroDyn,
    'neuralode': NeuralODE,
}

uq_name2class = {
    'none': Vacuous,
    'vacuous': Vacuous,
    'diagonal': Diagonal,
    'cholesky': Cholesky,
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


def get_uncertainty(logger: logging.Logger, name: str, **kwargs) -> Uncertainty:

    lower_name = name.lower()
    uq_class: Uncertainty = uq_name2class[lower_name]

    if uq_class is None:
        # logger.info(f'Using [None] as latent dynamics')
        return None
    else:
        # logger.info(f'Using [{ld_class.name}] as latent dynamics')
        return uq_class(logger, **kwargs)
