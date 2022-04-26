from util import *


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.env_conf = get_global_dict_value('env_conf')
        self.method_conf = get_global_dict_value('method_conf')
        self.log_conf = get_global_dict_value('log_conf')
        self.hidden_size = self.method_conf['hidden_size']
        self.used_part_num = 1
        if self.method_conf['obs']['S_CONFIG']['s3'][self.method_conf['obs']['s3']] == 'lb_pre yes':
            self.used_part_num += 1
        if self.method_conf['obs']['S_CONFIG']['s4'][self.method_conf['obs']['s4']] == 'fea_cur_pos yes':
            self.used_part_num += 1
        if self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->2' \
                or self.method_conf['obs']['S_CONFIG']['s5'][
            self.method_conf['obs']['s5']] == 'obs->41 41+fea_cur_pos->2':
            self.action_space = 2
        elif self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->41':
            self.action_space = self.dataset_conf['fea_num']
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        cnn_init_ = lambda m: init(m,
                                   nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain('relu'))
        #######################改动部分#############################
        if self.method_conf['obs']['S_CONFIG']['s1'][self.method_conf['obs']['s1']] == 'FNN':
            self.feature = nn.Sequential(
                init_(nn.Linear(self.dataset_conf['fea_num'] * self.used_part_num, self.dataset_conf['fea_num'])),
                nn.ReLU()
            )
            # self.feature = nn.Sequential(
            #     init_(nn.Linear(self.dataset_conf['fea_num'] * self.used_part_num, self.hidden_size)),
            #     nn.ReLU()
            # )
        elif self.method_conf['obs']['S_CONFIG']['s1'][self.method_conf['obs']['s1']] == 'CNN':
            if self.used_part_num == 2:
                self.feature = nn.Sequential(
                    cnn_init_(nn.Conv2d(1, 4, (2, 3), stride=1, padding=(0, 1))),
                    nn.ReLU(),
                    cnn_init_(nn.Conv2d(4, 1, 1, stride=1, padding=0)),
                    nn.ReLU(),
                )
            elif self.used_part_num == 3:
                self.feature = nn.Sequential(
                    cnn_init_(nn.Conv2d(1, 4, 3, stride=1, padding=(0, 1))),
                    nn.ReLU(),
                    cnn_init_(nn.Conv2d(4, 1, 1, stride=1, padding=0)),
                    nn.ReLU(),
                )
        if self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->2':
            self.advantage = nn.Sequential(
                init_(nn.Linear(self.dataset_conf['fea_num'], self.dataset_conf['fea_num'])),
                nn.ReLU(),
                init_(nn.Linear(self.dataset_conf['fea_num'], self.action_space)),
            )
            self.value = nn.Sequential(
                init_(nn.Linear(self.dataset_conf['fea_num'], self.dataset_conf['fea_num'])),
                nn.ReLU(),
                init_(nn.Linear(self.dataset_conf['fea_num'], 1)),

            )
            # self.advantage = nn.Sequential(
            #     init_(nn.Linear(self.hidden_size, self.hidden_size)),
            #     nn.ReLU(),
            #     init_(nn.Linear(self.hidden_size, self.action_space)),
            # )
            # self.value = nn.Sequential(
            #     init_(nn.Linear(self.hidden_size, self.hidden_size)),
            #     nn.ReLU(),
            #     init_(nn.Linear(self.hidden_size, 1)),
            #
            # )
        elif self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->41 41+fea_cur_pos->2':
            self.advantage = nn.Sequential(
                init_(nn.Linear(self.dataset_conf['fea_num'] * 2, self.dataset_conf['fea_num'])),
                nn.ReLU(),
                init_(nn.Linear(self.dataset_conf['fea_num'], self.action_space)),
            )
            self.value = nn.Sequential(
                init_(nn.Linear(self.dataset_conf['fea_num'] * 2, self.dataset_conf['fea_num'])),
                nn.ReLU(),
                init_(nn.Linear(self.dataset_conf['fea_num'], 1)),

            )

    def forward(self, obs):
        lb_pre = obs[:, :self.dataset_conf['fea_num']]
        fea_binary = obs[:, self.dataset_conf['fea_num']:self.dataset_conf['fea_num'] * 2]
        fea_cur_pos = obs[:, -self.dataset_conf['fea_num']:]
        lb_pre_fea_binary = lb_pre * fea_binary
        lb_pre_fea_cur_pos = lb_pre * fea_cur_pos
        used_part = []
        if self.method_conf['obs']['S_CONFIG']['s3'][self.method_conf['obs']['s3']] == 'lb_pre yes':
            used_part.append(lb_pre)
        if self.method_conf['obs']['S_CONFIG']['s2'][self.method_conf['obs']['s2']] == '* no' \
                or self.method_conf['obs']['S_CONFIG']['s2'][self.method_conf['obs']['s2']] == 'lb_pre * fea_cur_pos':
            used_part.append(fea_binary)
        elif self.method_conf['obs']['S_CONFIG']['s2'][self.method_conf['obs']['s2']] == 'lb_pre * fea_binary' \
                or self.method_conf['obs']['S_CONFIG']['s2'][
            self.method_conf['obs']['s2']] == 'lb_pre * fea_binary, lb_pre * fea_cur_pos':
            used_part.append(lb_pre_fea_binary)
        if self.method_conf['obs']['S_CONFIG']['s4'][self.method_conf['obs']['s4']] == 'fea_cur_pos yes':
            if self.method_conf['obs']['S_CONFIG']['s2'][self.method_conf['obs']['s2']] == '* no' \
                    or self.method_conf['obs']['S_CONFIG']['s2'][
                self.method_conf['obs']['s2']] == 'lb_pre * fea_binary':
                used_part.append(fea_cur_pos)
            elif self.method_conf['obs']['S_CONFIG']['s2'][self.method_conf['obs']['s2']] == 'lb_pre * fea_cur_pos' \
                    or self.method_conf['obs']['S_CONFIG']['s2'][
                self.method_conf['obs']['s2']] == 'lb_pre * fea_binary, lb_pre * fea_cur_pos':
                used_part.append(lb_pre_fea_cur_pos)
        if self.method_conf['obs']['S_CONFIG']['s1'][self.method_conf['obs']['s1']] == 'FNN':
            obs_1 = torch.cat(used_part, dim=1)
            feature = self.feature(obs_1)
        elif self.method_conf['obs']['S_CONFIG']['s1'][self.method_conf['obs']['s1']] == 'CNN':
            obs_1 = torch.stack(used_part, dim=1).unsqueeze(1)
            feature = self.feature(obs_1)
            feature = feature.squeeze(-2).squeeze(-2)

        if self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->2':
            advantage = self.advantage(feature)
            value = self.value(feature)
        elif self.method_conf['obs']['S_CONFIG']['s5'][self.method_conf['obs']['s5']] == 'obs->41 41+fea_cur_pos->2':
            feature_fea_cur_pos = torch.cat((feature, fea_cur_pos), dim=1)
            advantage = self.advantage(feature_fea_cur_pos)
            value = self.value(feature_fea_cur_pos)
        # 这里不减去advantage均值的话会导致训练不稳定，因为value的作用可能有的时候被忽略掉了，有的时候又突然非常大。
        return value + advantage - advantage.mean()

    def get_action(self, obs, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(obs.unsqueeze(0))
            action = q_value.max(1)[1].data[0]
            action = int(action)
        else:
            action = random.randrange(self.action_space)  # 返回指定递增基数集合中的一个随机数，基数默认值为1。
        return action
