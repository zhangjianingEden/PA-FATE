from util import *


class RolloutStorage:
    def __init__(self, capacity, type='train'):
        self.buffer_dict = {}
        if type == 'train':
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.buffer_dict[lb] = deque(maxlen=capacity)
        elif type == 'test':
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.buffer_dict[lb] = deque(maxlen=capacity)

    def load_shared_buffer(self, lb, shared_buffer):
        self.buffer_dict[lb].extend(shared_buffer)

    def is_enough(self, lb):
        return len(self.buffer_dict[lb]) >= get_global_dict_value('method_conf')['mini_batch_size']

    def sample(self, lb):
        obses, actions, rewards, next_obses, dones = zip(
            *random.sample(self.buffer_dict[lb], get_global_dict_value('method_conf')['mini_batch_size']))
        return torch.cat(obses, 0), torch.tensor(actions), torch.tensor(rewards), torch.cat(next_obses,
                                                                                            0), torch.tensor(dones)
