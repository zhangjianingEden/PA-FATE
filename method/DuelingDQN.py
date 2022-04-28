from util import *


class DuelingDQN:
    def __init__(self, network, target_network):
        self.network = network
        self.target_network = target_network
        self.lr = get_global_dict_value('method_conf')['lr']
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr,
                                    eps=get_global_dict_value('method_conf')['eps'], weight_decay=1e-6)

    def update(self, rollout, lb, type='train'):
        loss_total = 0
        sample_num = 0
        for _ in range(get_global_dict_value('method_conf')['sample_times']):
            obses, actions, rewards, next_obses, dones = rollout.sample(lb)
            sample_num += rewards.size(0)
            if type == 'train':
                obses = obses.to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
                actions = actions.to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
                rewards = rewards.to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
                next_obses = next_obses.to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
                dones = dones.to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))

            q_values = self.network(obses)
            next_q_values = self.target_network(next_obses)

            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_values.max(1)[0]

            expected_q_value = rewards + get_global_dict_value('method_conf')['gamma'] * next_q_value * (1 - dones)

            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            loss_total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_total / sample_num

    def update_tar_net(self):
        self.target_network.load_state_dict(self.network.state_dict())
