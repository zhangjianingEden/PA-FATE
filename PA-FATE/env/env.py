from .dataset import *
from .reward import *


class Env:
    state_info = {}
    step_counter = 0

    def __init__(self):
        super(Env, self).__init__()
        self.dataset = Dataset(get_global_dict_value('dataset_conf')['dataset_name'],
                               get_global_dict_value('dataset_conf')['dataset_path'],
                               get_global_dict_value('dataset_conf')['fea_list'],
                               get_global_dict_value('dataset_conf')['lb_list'])

        rwd_config = get_global_dict_value('env_conf')['rwd']
        self.reward_dict = {}
        for lb in get_global_dict_value('dataset_conf')['tr_lb_list'] + get_global_dict_value('dataset_conf')[
            'te_lb_list']:
            self.reward_dict[lb] = Reward(rwd_config, dataset=self.dataset, lb=lb)
        self.lb2lb_emb_path = os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_pre.npy')
        self.lb2lb_emb = np.load(self.lb2lb_emb_path, allow_pickle=True)[()][
            get_global_dict_value('env_conf')['lb_emb_type']]

    def reset(self, lb, init_fea=None):
        self.lb = lb
        self.lb_emb = self.lb2lb_emb[self.lb][get_global_dict_value('dataset_conf')['remap_fea_list']]
        if init_fea == None:
            self.fea_binary = np.zeros(get_global_dict_value('dataset_conf')['fea_num'], dtype=np.float32)
            self.step_counter = 0
        else:
            self.fea_binary = init_fea['fea_binary']
            self.step_counter = init_fea['step_counter']
        self.fea_cur_pos = np.zeros(get_global_dict_value('dataset_conf')['fea_num'], dtype=np.float32)
        self.fea_cur_pos[self.step_counter] = 1
        self._gen_state_info()
        return self.state_info

    def _gen_state_info(self):
        self.state_info = {}
        self.state_info['lb_emb'] = self.lb_emb
        self.state_info['fea_binary'] = self.fea_binary
        self.state_info['fea_cur_pos'] = self.fea_cur_pos

    def gen_obs(self, state_info):
        obs = np.hstack((self.state_info['lb_emb'], state_info['fea_binary'], state_info['fea_cur_pos']))
        return obs

    def step(self, action):
        self.fea_binary[self.step_counter] = action
        self.fea_cur_pos[self.step_counter] = 0
        self.step_counter += 1
        if self.step_counter < self.fea_cur_pos.shape[0]:
            self.fea_cur_pos[self.step_counter] = 1
        # gen_next_state
        self._gen_state_info()
        done = 0
        # whether done
        if self.step_counter == get_global_dict_value('dataset_conf')['fea_num'] or np.sum(self.fea_binary) >= \
                get_global_dict_value('dataset_conf')['fea_num'] * get_global_dict_value('env_conf')['max_fea_ratio']:
            done = 1
        return self.state_info, done

    def rwd1(self, fea_binary, next_fea_binary, done):
        return self.reward_dict[self.lb].gen_rwd_instance(fea_binary, next_fea_binary, done)

    def gen_reward(self, episode_info):
        num = len(episode_info['action'])
        for reward_id in range(num):
            st = episode_info['st'][reward_id]
            next_st = episode_info['st'][reward_id + 1]
            fea_binary = st['fea_binary']
            next_fea_binary = next_st['fea_binary']
            done = 0
            if reward_id == num - 1:
                done = 1
            episode_info['reward'].append(
                self.rwd1(fea_binary[get_global_dict_value('dataset_conf')['reremap_fea_list']],
                          next_fea_binary[get_global_dict_value('dataset_conf')['reremap_fea_list']], done))
        episode_info['meta_reward_list'] = {}
        episode_info['meta_reward_list']['pretrain_metric_list'] = self.reward_dict[self.lb].epi_p
        episode_info['meta_reward_list']['redundancy_list'] = self.reward_dict[self.lb].epi_rd
