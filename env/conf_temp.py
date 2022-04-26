from util import *

ENV_CONF = {
    'lb_emb_type': 'pcc',  # TODO ['pcc', ...]
    'act_dim': 2,
    'max_fea_ratio': 0.50,  # TODO [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00]

    # Reward
    'rwd': {
        'S_CONFIG': {
            's1': {
                'A': 'if s>=k, p=pk/s',
                'B': 'if s>=k, p=p',
            },
            's2': {
                'A': 'p=metric',
                'B': 'p=-Rd',
                'C': 'p=metric-gamma*Rd',
            },
            's3': {
                'A': 'p\' comp p',
                'B': 'p\' comp p_all_fea',
            },
            's4': {
                # 'A': '0',
                # 'B': 'cost',
                'C': 'p\'',
                'D': 'p\'-alpha*p',
                'E': 'p\'-alpha*p_all_fea'
            },
            's5': {
                'A': '0',
                # 'B': 'cost',
                'C': 'p\'',
                'D': 'p\'-alpha*p',
                'E': 'p\'-alpha*p_all_fea'
            },
            's6': {
                'A': '0',
                # 'B': 'cost',
                'C': 'p\'',
                'D': 'p\'-alpha*p',
                'E': 'p\'-alpha*p_all_fea'
            },
            's7': {
                'A': '0',
                'B': 'cost',
                'C': 'p\'',
                'D': 'p\'-alpha*p',
                'E': 'p\'-alpha*p_all_fea'
            },
        },
        'num_sample': 1000,
        'no_fea': 'fea_fill',  # TODO ['fea_fill', 'random', '0.5']
        ############################    use clsf     ############################
        'rwd_use_clsf': True,
        's1': 'B',
        's2': 'A',
        's3': 'A',
        's4': 'E',
        's5': 'E',
        's6': 'E',
        's7': 'E',
        'metric': 'auc',
        'alpha': 0.5,  # TODO [0.1, 0.3, 0.5, 0.7, 1.0]
        'beta': 0.5,  # TODO [0.1, 0.3, 0.5, 0.7, 1.0]
        'gamma': 1,  # TODO [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]
        'fea_num_threshold': 15,
        'cost': 0.1,  # TODO [0.1, 0.3, 0.5, 0.7, 1.0, 2.0, 5.0, 10.0]

        ############## pretrain ##############
        'retrain': False,
        'retrained_clsf': 'SVM',  # TODO ['SVM', 'DNN']
        'fea_fill': 'mean',  # TODO ['mean', 'median', 'zero']
        'pretrained_clsf': 'DNN',  # TODO ['SVM', 'DNN']

        # if DNN
        'classifier_output_size': 2,

        ############## retrain  ##############
        # 'retrain': True,

        ############################ do not use clsf ############################
        # 'rwd_use_clsf': False,

    }

}
