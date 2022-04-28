from .network import *
from log.sublog import *
from env.env import *


def valid_subp_test(lb_id,
                    log_path,
                    valid_shared_ifdone,
                    dataset_conf,
                    env_conf,
                    method_conf,
                    log_conf,
                    ):
    with torch.no_grad():
        global_dict_init()
        set_global_dict_value('dataset_conf', dataset_conf)
        set_global_dict_value('env_conf', env_conf)
        set_global_dict_value('method_conf', method_conf)
        set_global_dict_value('log_conf', log_conf)

        random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        np.random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        torch.manual_seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        sub_iter_counter = 0
        local_network = Network()
        local_network.eval()
        env = Env()
        sublog = SubLog('valid', lb_id, log_path, env.dataset, type2='test')
        lb = get_global_dict_value('dataset_conf')['te_lb_list'][lb_id]
        epsilon_by_iter = lambda iter_idx: get_global_dict_value('method_conf')['epsilon_final']
        while sub_iter_counter < get_global_dict_value('method_conf')['further_train_iter'] + 1:
            while True:
                if not valid_shared_ifdone.value:
                    # sync shared model to local
                    local_network.load_state_dict(
                        torch.load(os.path.join(log_path, lb, 'tmp_valid_model.pth'), map_location=torch.device('cpu')))
                    for _ in range(get_global_dict_value('method_conf')['test_sample_num']):
                        ################################## interact with env ####################################
                        st = env.reset(lb)
                        obs = env.gen_obs(st)
                        sublog.episode_info['st'] = []
                        sublog.episode_info['st'].append(copy.deepcopy(st))
                        while True:
                            # get action
                            epsilon = epsilon_by_iter(sub_iter_counter)
                            action = local_network.get_action(torch.tensor(obs, dtype=torch.float32), epsilon)
                            # take action
                            st, done = env.step(action)
                            obs = env.gen_obs(st)
                            sublog.episode_info['st'].append(copy.deepcopy(st))
                            if done == 1:
                                break
                        ################################## sublog work ####################################
                        sublog.gen_metrics_result(sub_iter_counter)
                    sublog.record_metrics_result()
                    valid_shared_ifdone.value = True
                    sub_iter_counter += get_global_dict_value('method_conf')['cal_log_frequency']
                    break
