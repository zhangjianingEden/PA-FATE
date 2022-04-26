from util import *


class RolloutStorage:
    def __init__(self, capacity, type='train'):
        self.buffer_dict = {}
        self.buffer_dict_init_fea = {}
        if type == 'train':
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.buffer_dict[lb] = deque(maxlen=capacity)
                self.buffer_dict_init_fea[lb] = deque(maxlen=capacity)
        elif type == 'test':
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.buffer_dict[lb] = deque(maxlen=capacity)
                self.buffer_dict_init_fea[lb] = deque(maxlen=capacity)

    def load_shared_buffer(self, lb, shared_buffer, type=None):
        if type == None:
            self.buffer_dict[lb].extend(shared_buffer)
        elif type == 'init_fea':
            self.buffer_dict_init_fea[lb].extend(shared_buffer)

    def is_enough(self, lb, type=None):
        if type == None:
            return len(self.buffer_dict[lb]) >= get_global_dict_value('method_conf')['mini_batch_size']
        elif type == 'init_fea':
            return len(self.buffer_dict_init_fea[lb]) >= get_global_dict_value('method_conf')['mini_batch_size']

    def sample(self, lb, type=None):
        # 将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        if type == None:
            obses, actions, rewards, next_obses, dones = zip(
                *random.sample(self.buffer_dict[lb], get_global_dict_value('method_conf')['mini_batch_size']))
        elif type == 'init_fea':
            obses, actions, rewards, next_obses, dones = zip(
                *random.sample(self.buffer_dict_init_fea[lb], get_global_dict_value('method_conf')['mini_batch_size']))
        # 最后使用concatenate对数组进行拼接，相当于少了一个维度
        return torch.cat(obses, 0), torch.tensor(actions), torch.tensor(rewards), torch.cat(next_obses,
                                                                                            0), torch.tensor(dones)
