from util import *


def adjust_learning_rate(optimizer, lr, iter_id):
    lr = get_global_dict_value('method_conf')['lr'] * \
         get_global_dict_value('method_conf')['decay_rate'] ** \
         max(0, iter_id - get_global_dict_value('method_conf')['decay_start_iter_id'])
    # lr = CONF['lr'] * math.exp(-CONF['decay_rate'] * max(0, iter_id - CONF['decay_start_iter_id']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# -------------------
class DuelingDQN:
    def __init__(self, network, target_network):
        self.network = network
        self.target_network = target_network
        self.lr = get_global_dict_value('method_conf')['lr']
        self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr,
                                    eps=get_global_dict_value('method_conf')['eps'], weight_decay=1e-6)

    def update(self, rollout, lb, iter_id, type='train', type2=None):
        loss_total = 0
        sample_num = 0
        if get_global_dict_value('method_conf')['is_lr_decay']:
            self.lr = adjust_learning_rate(optimizer=self.optimizer, lr=self.lr, iter_id=iter_id)
        for _ in range(get_global_dict_value('method_conf')['sample_times']):
            obses, actions, rewards, next_obses, dones = rollout.sample(lb, type=type2)
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
            # gather可以看作是对q_values的查询，即元素都是q_values中的元素，查询索引都存在action中。输出大小与action.unsqueeze(1)一致。
            # dim=1,它存放的都是第1维度的索引；dim=0，它存放的都是第0维度的索引；
            # 这里增加维度主要是为了方便gather操作，之后再删除该维度
            next_q_value = next_q_values.max(1)[0]

            expected_q_value = rewards + get_global_dict_value('method_conf')['gamma'] * next_q_value * (1 - dones)

            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            if get_global_dict_value('method_conf')['SMALL']['use']:
                if type2 == None:
                    loss *= 1 - get_global_dict_value('method_conf')['SMALL']['uct_factor']
                elif type2 == 'init_fea':
                    loss *= get_global_dict_value('method_conf')['SMALL']['uct_factor']
            else:
                loss *= 0.5
            loss_total += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return loss_total / sample_num

    def update_tar_net(self):
        self.target_network.load_state_dict(self.network.state_dict())
