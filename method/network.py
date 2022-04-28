from util import *


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.dataset_conf = get_global_dict_value('dataset_conf')
        self.used_part_num = 3
        self.action_space = 2
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.feature = nn.Sequential(
            init_(nn.Linear(self.dataset_conf['fea_num'] * self.used_part_num, self.dataset_conf['fea_num'])),
            nn.ReLU()
        )
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
        used_part = []
        used_part.append(lb_pre)
        used_part.append(fea_binary)
        used_part.append(fea_cur_pos)
        obs_1 = torch.cat(used_part, dim=1)
        feature = self.feature(obs_1)
        feature_fea_cur_pos = torch.cat((feature, fea_cur_pos), dim=1)
        advantage = self.advantage(feature_fea_cur_pos)
        value = self.value(feature_fea_cur_pos)
        return value + advantage - advantage.mean()

    def get_action(self, obs, epsilon):
        if random.random() > epsilon:
            q_value = self.forward(obs.unsqueeze(0))
            action = q_value.max(1)[1].data[0]
            action = int(action)
        else:
            action = random.randrange(self.action_space)
        return action
