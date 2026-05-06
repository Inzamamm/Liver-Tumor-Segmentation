"""
Optional template for running ablation experiments.
Change the flags below to disable selected components and retrain.
"""

ABLATION_SETTINGS = {
    'full_model': {
        'use_context': True,
        'use_multiscale_skips': True,
        'use_decoder_refinement': True,
        'dice_weight': 0.7,
        'bce_weight': 0.3,
    },
    'without_context_module': {
        'use_context': False,
        'use_multiscale_skips': True,
        'use_decoder_refinement': True,
        'dice_weight': 0.7,
        'bce_weight': 0.3,
    },
    'without_multiscale_encoding': {
        'use_context': True,
        'use_multiscale_skips': False,
        'use_decoder_refinement': True,
        'dice_weight': 0.7,
        'bce_weight': 0.3,
    },
    'without_decoder_refinement': {
        'use_context': True,
        'use_multiscale_skips': True,
        'use_decoder_refinement': False,
        'dice_weight': 0.7,
        'bce_weight': 0.3,
    },
    'without_loss_balancing': {
        'use_context': True,
        'use_multiscale_skips': True,
        'use_decoder_refinement': True,
        'dice_weight': 0.5,
        'bce_weight': 0.5,
    },
}

print('This file defines ablation settings. Integrate the flags into model.py if you want to run each variant automatically.')
