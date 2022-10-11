
"""
# --------------------------------------------
# define training model
# --------------------------------------------
"""


def define_Model(opt):
    model = opt['model']      # one input: L

    if model == 'admm':
        from models.model_pnp_admm import PnP_ADMM as M

    elif model == 'approximate_admm':
        from models.model_pnp_approximate_admm import PnP_approx_ADMM as M
    
    elif model == 'linearized_admm':
        from models.model_pnp_ladmm import PnP_linearized_ADMM as M

    elif model == 'fista':
        from models.model_pnp_fista import PnP_FISTA as M
    
    elif model == 'ista':
        from models.model_pnp_ista import PnP_ISTA as M

    elif model == 'richardson_lucy':
        from models.model_rl import RichardsonLucy as M

    else:
        raise NotImplementedError('Model [{:s}] is not defined.'.format(model))

    m = M(opt)

    print('Training model [{:s}] is created.'.format(m.__class__.__name__))
    return m
