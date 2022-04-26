from util import *

METHOD_CONF = {
    'method_name': __file__.split('/')[-2],
    'env_num_per_lb': 1,
    'gpu_id': 0,
    'seed': 1,
    # ---------algo------------
    'lr': 2.5e-4,  # TODO [2.5e-3, 2.5e-4, 2.5e-5]
    'eps': 1e-5,  # TODO [1e-4, 1e-5, 1e-6]
    'gamma': 0.9,  # TODO [0.5, 0.8, 0.9, 0.99, 0.999]
    'tar_net_update_iter_gap': 8,  # TODO [4, 8, 16, 32]
    'epsilon_start': 1.0,  # TODO [0.8, 0.9, 1.0]
    'epsilon_final': 0.001,  # TODO [0.0001, 0.001, 0.01]
    'epsilon_decay': 500,  # TODO [100, 500, 1000]
    # ---------buffer------------
    'mini_batch_size': 64,  # TODO [16, 64, 256]
    'sample_times': 2,  # TODO [2, 5, 10, 20]
    'capacity': 1000,  # TODO [500, 1000, 2000]
    # ----------train test-----------
    'train_iter': 2000,
    'test_iter': 0,
    'test_sample_num': 1,
    # ----------lr decay-----------
    'is_lr_decay': False,
    'decay_rate': 0.9995,
    'decay_start_iter_id': 3000,
    # --------obs-------------
    'obs': {
        'S_CONFIG': {
            's1': {
                'A': 'FNN',
                'B': 'CNN',
            },
            's2': {
                'A': '* no',
                'B': 'lb_pre * fea_binary',
                'C': 'lb_pre * fea_cur_pos',
                'D': 'lb_pre * fea_binary, lb_pre * fea_cur_pos',
            },
            's3': {
                'A': 'lb_pre yes',
                'B': 'lb_pre no',
            },
            's4': {
                'A': 'fea_cur_pos yes',
                'B': 'fea_cur_pos no',
            },
            's5': {
                'A': 'obs->2',
                'B': 'obs->41 41+fea_cur_pos->2',
                'C': 'obs->41',
            },
        },
        's1': 'A',
        's2': 'A',
        's3': 'A',
        's4': 'A',
        's5': 'B',
    },
    # --------network-------------
    'hidden_size': 32,
    # ------------log--------------
    'cal_log_frequency': 50,
    'log_root_path': '/' + os.path.join(*__file__.split('/')[:-3]) + '_log',
    'iter_avg': 1,
    # ------------BIG--------------
    'BIG': {
        'use': True,  # TODO [False, True]
        'fq': 10,  # TODO [1, 5, 10, 20, 50, 100, 200]
        'si': {
            'uc': 0.5,  # TODO [0.00, 0.25, 0.50, 0.75, 1.00]
            'dis': 0.5,  # TODO [0.00, 0.25, 0.50, 0.75, 1.00]
        },
        'up': {
            'uc': 0.5,  # TODO [0.00, 0.25, 0.50, 0.75, 1.00]
            'dis': 0.5,  # TODO [0.00, 0.25, 0.50, 0.75, 1.00]
        },
    },
    # ------------SMALL--------------
    'SMALL': {
        'use': True,  # TODO [False, True]
        'k': 10,  # TODO [1, 5, 10, 20, 50, 100, 200]
        'choose_type': 'choose_c',  # TODO ['choose_a', 'choose_b', 'choose_c']
        'build_type': 'build_a',
        'value_usage_type': 'value_usage_a',
        'value_content_type': 'value_content_a',  # TODO ['value_content_a', 'value_content_b'(put aside)]
        'uct_factor': 1.00,  # TODO [0.25, 0.50, 0.75, 1.00]
        'UCB_config': {
            'c': 1.0  # TODO [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
        },
    }

}
# better_start_state
# 模仿UCT的过程，根据UCB公式给予初始state，期待可以给强化学习带来学习速度或者效果上的提升。
# 方案：
# 为各label分别初始化一颗树Tree
# 用第i(i = 0,1,...)个k个回合的经验伸展树

# Tree_config:
# choose state ways:
# choose_a.根据UCB公式选择节点，作为第(i + 1)个k个回合的初始state
# choose_b.或根据UCB公式选择Tree第(i % fea_num + 1)层的节点，作为第(i + 1)个k个回合的初始state
# choose_c.或根据UCB公式依次选择Tree第1层~第fea_num层的节点，作为第(i + 1)个k个回合的初始state
# build tree ways：
# build_a：将k个回合的经验中的所有state对应的节点均加入树
# value usage ways:
# value_usage_a：仅使用final state的value
# value_usage_b：使用所有state的value (弃用)
# value content ways:
# value_content_a = pretrain_auc
# value_content_b = pretrain_auc - redundancy
# value_content_c = retrain_auc
# value_content_d = retrain_auc - redundancy
# UCB_config
# ...

# DRL可调: reward是否只给final_state，如果是，则是否改用retrain_auc作为reward

# contribution:
# 框架
# BIG manage 和 SMALL manage 兼顾
# experience 的高质量生成 和 高质量利用 兼顾
# 实验
