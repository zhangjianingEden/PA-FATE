from util import *
from .network import *
from log.sublog import *
from env.env import *


def subp_test(process_id,
              log_path,
              shared_rollout,
              shared_ifdone,
              shared_te_lb_id,
              dataset_conf,
              env_conf,
              method_conf,
              log_conf,
              ):
    with torch.no_grad():
        start_datetime = datetime.now()
        global_dict_init()
        set_global_dict_value('dataset_conf', dataset_conf)
        set_global_dict_value('env_conf', env_conf)
        set_global_dict_value('method_conf', method_conf)
        set_global_dict_value('log_conf', log_conf)

        random.seed(get_global_dict_value('method_conf')['seed'] + process_id)
        np.random.seed(get_global_dict_value('method_conf')['seed'] + process_id)
        torch.manual_seed(get_global_dict_value('method_conf')['seed'] + process_id)
        sub_iter_counter = 0
        print("---------------------------->", process_id, "subp_test")
        local_network = Network()
        local_network.eval()
        env = Env()
        sublog = SubLog('simulator', process_id, log_path=log_path)
        epsilon_by_iter = lambda iter_idx: get_global_dict_value('method_conf')['epsilon_final'] + (
                get_global_dict_value('method_conf')['epsilon_start'] - get_global_dict_value('method_conf')[
            'epsilon_final']) * math.exp(-1. * iter_idx / get_global_dict_value('method_conf')['epsilon_decay'])
        while sub_iter_counter < get_global_dict_value('method_conf')['test_iter']:
            while True:
                if not shared_ifdone.value:
                    lb = get_global_dict_value('dataset_conf')['te_lb_list'][shared_te_lb_id.value]
                    # sync shared model to local
                    local_network.load_state_dict(
                        torch.load(os.path.join(log_path, lb, 'tmp_model.pth'), map_location=torch.device('cpu')))
                    ################################## interact with env ####################################
                    st = env.reset(lb)
                    env.reward_dict[lb].epi_rwd_reset()
                    obs = env.gen_obs(st)
                    # ----------------------------------------
                    sublog.episode_info['st'] = []
                    sublog.episode_info['obs'] = []
                    sublog.episode_info['action'] = []
                    sublog.episode_info['reward'] = []
                    sublog.episode_info['done'] = []

                    sublog.episode_info['st'].append(copy.deepcopy(st))
                    sublog.episode_info['obs'].append(obs)
                    while True:
                        # get action
                        epsilon = epsilon_by_iter(sub_iter_counter)
                        action = local_network.get_action(torch.tensor(obs, dtype=torch.float32), epsilon)
                        # take action
                        st, done = env.step(action)
                        obs = env.gen_obs(st)
                        sublog.episode_info['action'].append(action)
                        sublog.episode_info['st'].append(copy.deepcopy(st))
                        sublog.episode_info['obs'].append(obs)
                        sublog.episode_info['done'].append(done)
                        if done == 1:
                            break
                    # t1 = datetime.now()
                    env.gen_reward(sublog.episode_info)
                    # print('episode_gen_reward_time:{}'.format(datetime.now() - t1))
                    ################################## feed in sharestorage ####################################
                    shared_rollout.reset()
                    num = len(sublog.episode_info['action'])

                    for idx in range(num):
                        shared_rollout.push(torch.tensor(sublog.episode_info['obs'][idx], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['action'][idx], dtype=torch.int64),
                                            torch.tensor(sublog.episode_info['reward'][idx], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['obs'][idx + 1], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['done'][idx], dtype=torch.float32)
                                            )
                    ################################## sublog work ####################################
                    sublog.save_buffer(shared_rollout.buffer)
                    sublog.save_epi_info()

                    ################################## interact with env from init_fea ####################################
                    if sub_iter_counter < 2:
                        st = env.reset(lb)
                    else:
                        sublog.load_UCT_result(lb)
                        length = len(sublog.UCT_result['SMALL']['init_fea'])
                        idx = np.random.randint(low=0, high=length)
                        init_fea = sublog.UCT_result['SMALL']['init_fea'][idx]
                        st = env.reset(lb, init_fea)
                    env.reward_dict[lb].epi_rwd_reset()
                    obs = env.gen_obs(st)
                    # ----------------------------------------
                    sublog.episode_info['st'] = []
                    sublog.episode_info['obs'] = []
                    sublog.episode_info['action'] = []
                    sublog.episode_info['reward'] = []
                    sublog.episode_info['done'] = []

                    sublog.episode_info['st'].append(copy.deepcopy(st))
                    sublog.episode_info['obs'].append(obs)
                    while True:
                        # get action
                        epsilon = epsilon_by_iter(sub_iter_counter)
                        action = local_network.get_action(torch.tensor(obs, dtype=torch.float32), epsilon)
                        # take action
                        st, done = env.step(action)
                        obs = env.gen_obs(st)
                        sublog.episode_info['action'].append(action)
                        sublog.episode_info['st'].append(copy.deepcopy(st))
                        sublog.episode_info['obs'].append(obs)
                        sublog.episode_info['done'].append(done)
                        if done == 1:
                            break
                    # t1 = datetime.now()
                    env.gen_reward(sublog.episode_info)
                    # print('episode_gen_reward_time:{}'.format(datetime.now() - t1))
                    #### feed in sharestorage ####
                    shared_rollout.reset()
                    num = len(sublog.episode_info['action'])

                    for idx in range(num):
                        shared_rollout.push(torch.tensor(sublog.episode_info['obs'][idx], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['action'][idx], dtype=torch.int64),
                                            torch.tensor(sublog.episode_info['reward'][idx], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['obs'][idx + 1], dtype=torch.float32),
                                            torch.tensor(sublog.episode_info['done'][idx], dtype=torch.float32)
                                            )
                    ##### sublog work ####
                    sublog.save_buffer(shared_rollout.buffer, type='init_fea')
                    sublog.save_epi_info(type='init_fea')

                    shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
        # make check if test is health easier
        delta_time = (datetime.now() - start_datetime).seconds
        if delta_time < 30:
            time.sleep(30 - delta_time)
