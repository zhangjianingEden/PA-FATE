from util import *


class MyNode:
    def __init__(self, parent_node, layer_id, fea_binary):
        self.parent_node = parent_node
        self.child_node_dict = {}

        self.layer_id = layer_id
        self.fea_binary = fea_binary

        self.visit_times = 0
        self.total_value = 0

    def gen_child(self, final_fea_binary):
        if self.layer_id < final_fea_binary.shape[0]:
            action = int(final_fea_binary[self.layer_id])
            if str(action) not in self.child_node_dict:
                new_fea_binary = copy.deepcopy(self.fea_binary)
                new_fea_binary[self.layer_id] = action
                child_node = MyNode(parent_node=self, layer_id=self.layer_id + 1,
                                    fea_binary=new_fea_binary)
                self.child_node_dict[str(action)] = child_node
            return self.child_node_dict[str(action)].gen_child(final_fea_binary)
        else:
            return self

    def update_value(self, final_fea_value):
        self.visit_times += 1
        self.total_value += final_fea_value
        if self.parent_node is not None:
            self.parent_node.update_value(final_fea_value)

    def count_UCB(self):
        UCB = self.total_value / self.visit_times + get_global_dict_value('method_conf')['ITE']['UCB_config']['c'] * (
                np.log(self.parent_node.visit_times + 1) / (self.visit_times + 1)) ** 0.5
        return UCB

    def choose_child(self, target_layer_id):
        if self.layer_id == target_layer_id:
            return self
        child_node_keys = self.child_node_dict.keys()
        if len(child_node_keys) == 0:
            return self
        elif len(child_node_keys) < get_global_dict_value('env_conf')['act_dim']:
            return self.child_node_dict[list(child_node_keys)[0]].choose_child(target_layer_id)
        else:
            UCB_array = np.array([self.child_node_dict[str(action)].count_UCB() for action in
                                  range(get_global_dict_value('env_conf')['act_dim'])])
            winner_action = np.argmax(UCB_array)
            return self.child_node_dict[str(winner_action)].choose_child(target_layer_id)


class ExperienceTree:
    def __init__(self):
        self.node_dict = {}
        self.root_node = MyNode(parent_node=None, layer_id=0,
                                fea_binary=np.zeros(get_global_dict_value('dataset_conf')['fea_num'], dtype=np.float32))

    def build(self, lb_epi_info_deque):
        for epi_info in lb_epi_info_deque:
            final_fea_binary = epi_info['final_state']
            final_fea_value = epi_info['meta_reward_list']['pretrain_metric_list'][-1]
            leaf_node = self.root_node.gen_child(final_fea_binary)
            # update value
            leaf_node.update_value(final_fea_value)

    def choose(self):
        init_fea_list = []
        init_fea = {}
        for target_layer_id in range(1, get_global_dict_value('dataset_conf')['fea_num']):
            target_node = self.root_node.choose_child(target_layer_id)
            init_fea['fea_binary'] = target_node.fea_binary
            init_fea['step_counter'] = target_node.layer_id
            init_fea_list.append(copy.deepcopy(init_fea))
        return init_fea_list
