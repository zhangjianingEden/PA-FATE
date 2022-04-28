from util import *

ENV_CONF = {
    'lb_emb_type': 'pcc',
    'act_dim': 2,
    'max_fea_ratio': 0.50,
    'rwd': {
        'num_sample': 1000,
        'metric': 'auc',
        'alpha': 0.5,
        'beta': 0.5,
        'gamma': 1,
        'cost': 0.1,
        'classifier_output_size': 2,
    }
}
