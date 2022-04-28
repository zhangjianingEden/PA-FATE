from util import *
from .classifiers import *


class Reward:
    def __init__(self, rwd_config, dataset, lb):
        self.conf = rwd_config
        self.dataset = dataset
        self.epi_p = []
        self.epi_metric = []
        self.epi_rd = []
        self.lb_id2statistic_info = {}
        self.lb = lb
        self.lb_metric_path = os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_metric.npy')
        self.lb_metric = np.load(self.lb_metric_path, allow_pickle=True)[()][self.lb]
        self.p_all_fea = self.lb_metric['all_fea'][self.conf['metric']]
        self.clsf_pretr = DNNClassifier(input_size=get_global_dict_value('dataset_conf')['fea_num'],
                                        output_size=self.conf['classifier_output_size'])
        self.clsf_pretr.eval()
        self.clsf_pretr_dict_path = os.path.join(get_global_dict_value('dataset_conf')['dataset_path'],
                                                 'task_model_dict.npy')
        self.clsf_pretr_state_dict = np.load(self.clsf_pretr_dict_path, allow_pickle=True)[()][self.lb]
        self.clsf_pretr.load_state_dict(self.clsf_pretr_state_dict)
        self.lb_positive_value_ratio = self.gen_lb_positive_value_ratio()
        self.lb_positive_value_idx = np.where(self.dataset.lb2y_dict[self.lb]['train_set']['y'] == 1)[0]
        self.lb_negative_value_idx = np.where(self.dataset.lb2y_dict[self.lb]['train_set']['y'] == 0)[0]

    def gen_sample_idx_list(self):
        lb_positive_value_idx_sample = self.lb_positive_value_idx[np.random.choice(self.lb_positive_value_idx.shape[0],
                                                                                   int(self.lb_positive_value_ratio *
                                                                                       self.conf['num_sample']),
                                                                                   replace=False)]
        lb_negative_value_idx_sample = self.lb_negative_value_idx[np.random.choice(self.lb_negative_value_idx.shape[0],
                                                                                   self.conf['num_sample'] - int(
                                                                                       self.lb_positive_value_ratio *
                                                                                       self.conf['num_sample']),
                                                                                   replace=False)]
        sample_idx_list = []
        sample_idx_list.extend(list(lb_positive_value_idx_sample))
        sample_idx_list.extend(list(lb_negative_value_idx_sample))
        return sample_idx_list

    def gen_lb_positive_value_ratio(self):
        train_y = self.dataset.lb2y_dict[self.lb]['train_set']['y']
        total_num = train_y.shape[0]
        positive_value_num = np.sum(train_y)
        lb_positive_value_ratio = positive_value_num / total_num
        return lb_positive_value_ratio

    def epi_rwd_reset(self):
        self.epi_p.clear()
        self.epi_metric.clear()
        self.epi_rd.clear()

    def fea_binary_to_fea_list(self, fea_binary):
        fea = []
        fea.clear()
        for i in range(len(fea_binary)):
            if fea_binary[i] == 1:
                fea.append(i)
        return copy.deepcopy(fea)

    def gen_rwd_instance(self, cur_fea_binary, next_fea_binary, done):
        add_fea_flag = False
        if np.sum(next_fea_binary) > np.sum(cur_fea_binary):
            add_fea_flag = True
        if add_fea_flag is False and len(self.epi_p) > 0:
            next_p = self.epi_p[-1]
        else:
            next_p = self.calculate_p(next_fea_binary)
        self.epi_p.append(next_p)
        if done == 1:
            reward = next_p - self.conf['alpha'] * self.p_all_fea
        else:
            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
        return reward

    def fea_redundancy(self, cur_fea_binary, next_fea_binary):
        rd = 0
        cur_fea_list = self.fea_binary_to_fea_list(cur_fea_binary)
        next_fea_list = self.fea_binary_to_fea_list(next_fea_binary)
        fea = list(set(next_fea_list).difference(set(cur_fea_list)))
        for i in fea:
            for j in cur_fea_list:
                a, b = pearsonr(self.dataset.ds_class2x_dict['train_set'][:, i],
                                self.dataset.ds_class2x_dict['train_set'][:, j])
                rd = rd + abs(a)
            rd = float(rd / len(cur_fea_list))
        return rd

    def calculate_p(self, fea_binary):
        p = 0
        fea = []
        for i in range(len(fea_binary)):
            if fea_binary[i] == 1:
                fea.append(i)
        non_list = list(
            set(range(len(self.dataset.fea_list))).difference(set(fea)))

        train_x = self.dataset.ds_class2x_dict['train_set']
        train_y = self.dataset.lb2y_dict[self.lb]['train_set']['y']

        if self.conf['num_sample'] < train_x.shape[0]:
            sample_idx_list = self.gen_sample_idx_list()
            train_x_sam = copy.deepcopy(train_x[sample_idx_list, :])
            train_y_sam = copy.deepcopy(train_y[sample_idx_list])
        else:
            train_x_sam = copy.deepcopy(train_x)
            train_y_sam = copy.deepcopy(train_y)

        if 'mean' not in self.lb_id2statistic_info:
            self.lb_id2statistic_info['mean'] = {}
        for i in non_list:
            if str(i) not in self.lb_id2statistic_info['mean']:
                self.lb_id2statistic_info['mean'][str(i)] = np.mean(train_x[:, i])
            train_x_sam[:, i] = self.lb_id2statistic_info['mean'][str(i)]

        train_x_sam = torch.tensor(train_x_sam, dtype=torch.float32)
        output_sam = self.clsf_pretr(train_x_sam)
        _, pred_tensor = torch.max(output_sam, 1)
        if self.conf['metric'] == 'auc':
            auc = roc_auc_score(train_y_sam, pred_tensor.numpy())
            p = auc
        elif self.conf['metric'] == 'f1_score':
            f1_s = f1_score(train_y_sam, pred_tensor.numpy())
            p = f1_s
        return p
