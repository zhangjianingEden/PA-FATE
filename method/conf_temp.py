from util import *

METHOD_CONF = {
    'env_num_per_lb': 1,
    'gpu_id': 0,
    'seed': 1,
    'lr': 2.5e-4,
    'eps': 1e-5,
    'gamma': 0.9,
    'tar_net_update_iter_gap': 8,
    'epsilon_start': 1.0,
    'epsilon_final': 0.001,
    'epsilon_decay': 500,
    'mini_batch_size': 64,
    'sample_times': 2,
    'capacity': 1000,
    'train_iter': 6000,
    'further_train_iter': 0,
    'test_sample_num': 1,
    'ITS': {
        'fq': 10,
        'uc': 0.5,
        'dis': 0.5,
    },
    'ITE': {
        'k': 10,
        'UCB_config': {
            'c': 1.0
        },
    },
    'cal_log_frequency': 50,
    'iter_avg': 1,
}
