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
        if self.conf['rwd_use_clsf']:
            if self.conf['pretrained_clsf'] == 'DNN':
                self.clsf_pretr = DNNClassifier(input_size=get_global_dict_value('dataset_conf')['fea_num'],
                                                output_size=self.conf['classifier_output_size'])
                self.clsf_pretr.eval()
                self.clsf_pretr_dict_path = os.path.join(get_global_dict_value('dataset_conf')['dataset_path'],
                                                         'task_model_dict.npy')
                self.clsf_pretr_state_dict = np.load(self.clsf_pretr_dict_path, allow_pickle=True)[()][self.lb]
                self.clsf_pretr.load_state_dict(self.clsf_pretr_state_dict)
            else:
                self.clsf_pretr = joblib.load(os.path.join(get_global_dict_value('dataset_conf')['dataset_path'],
                                                           self.lb + '_' + self.conf['pretrained_clsf'] + '.pkl'))
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
        if not self.conf['rwd_use_clsf']:
            # ICML2010 KNN
            return
        else:
            add_fea_flag = False
            if np.sum(next_fea_binary) > np.sum(cur_fea_binary):
                add_fea_flag = True
            if len(self.epi_p) > 0:
                cur_p = self.epi_p[-1]
            else:
                cur_p = self.calculate_p(cur_fea_binary)
            if add_fea_flag is False and len(self.epi_p) > 0:
                next_p = self.epi_p[-1]
            else:
                next_p = self.calculate_p(next_fea_binary)
            self.epi_p.append(next_p)
            if done == 1:
                if self.conf['S_CONFIG']['s4'][self.conf['s4']] == 'p\'':
                    reward = next_p
                elif self.conf['S_CONFIG']['s4'][self.conf['s4']] == 'p\'-alpha*p':
                    reward = next_p - self.conf['alpha'] * cur_p
                elif self.conf['S_CONFIG']['s4'][self.conf['s4']] == 'p\'-alpha*p_all_fea':
                    reward = next_p - self.conf['alpha'] * self.p_all_fea
            else:
                if self.conf['S_CONFIG']['s3'][self.conf['s3']] == 'p\' comp p':
                    if add_fea_flag is True and next_p >= cur_p:
                        if self.conf['S_CONFIG']['s5'][self.conf['s5']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                    elif add_fea_flag is False:
                        if self.conf['S_CONFIG']['s6'][self.conf['s6']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                    elif add_fea_flag is True and next_p < cur_p:
                        if self.conf['S_CONFIG']['s7'][self.conf['s7']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'cost':
                            reward = - self.conf['beta'] * self.conf['cost']
                elif self.conf['S_CONFIG']['s3'][self.conf['s3']] == 'p\' comp p_all_fea':
                    if abs(next_p - self.p_all_fea) > 1e-5 and next_p > self.p_all_fea:
                        if self.conf['S_CONFIG']['s5'][self.conf['s5']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s5'][self.conf['s5']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                    elif abs(next_p - self.p_all_fea) < 1e-5:
                        if self.conf['S_CONFIG']['s6'][self.conf['s6']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s6'][self.conf['s6']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                    elif abs(next_p - self.p_all_fea) > 1e-5 and next_p < self.p_all_fea:
                        if self.conf['S_CONFIG']['s7'][self.conf['s7']] == '0':
                            reward = 0
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'':
                            reward = self.conf['beta'] * next_p
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'-alpha*p':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * cur_p)
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'p\'-alpha*p_all_fea':
                            reward = self.conf['beta'] * (next_p - self.conf['alpha'] * self.p_all_fea)
                        elif self.conf['S_CONFIG']['s7'][self.conf['s7']] == 'cost':
                            reward = - self.conf['beta'] * self.conf['cost']
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

    def fea_set_redundancy(self, fea_binary):
        pass

    def calculate_p(self, fea_binary):
        p = 0
        fea = []
        for i in range(len(fea_binary)):
            if fea_binary[i] == 1:
                fea.append(i)
        # 计算过程中不需要分类器
        if not self.conf['rwd_use_clsf']:
            return
        # 计算过程中需要分类器
        elif self.conf['rwd_use_clsf']:
            # 确定分类器
            if not self.conf['retrain']:
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

                if self.conf['fea_fill'] == 'zero':
                    train_x_sam[:, non_list] = 0
                elif self.conf['fea_fill'] == 'mean':
                    if 'mean' not in self.lb_id2statistic_info:
                        self.lb_id2statistic_info['mean'] = {}
                    for i in non_list:
                        if str(i) not in self.lb_id2statistic_info['mean']:
                            self.lb_id2statistic_info['mean'][str(i)] = np.mean(train_x[:, i])
                        train_x_sam[:, i] = self.lb_id2statistic_info['mean'][str(i)]
                elif self.conf['fea_fill'] == 'median':
                    if 'median' not in self.lb_id2statistic_info:
                        self.lb_id2statistic_info['median'] = {}
                    for i in non_list:
                        if str(i) not in self.lb_id2statistic_info['median']:
                            self.lb_id2statistic_info['median'][str(i)] = np.median(train_x[:, i])
                        train_x_sam[:, i] = self.lb_id2statistic_info['median'][str(i)]

                if self.conf['pretrained_clsf'] == 'DNN':
                    train_x_sam = torch.tensor(train_x_sam, dtype=torch.float32)
                    # train_x = torch.tensor(train_x, dtype=torch.float32)

                    output_sam = self.clsf_pretr(train_x_sam)
                    # output_train = self.clsf_pretr(train_x)

                    _, pred_tensor = torch.max(output_sam, 1)
                    # _, output_train_pred = torch.max(output_train, 1)

                    if self.conf['metric'] == 'auc':
                        auc = roc_auc_score(train_y_sam, pred_tensor.numpy())
                        # auc_train = roc_auc_score(train_y, output_train_pred.numpy())
                        p = auc
                    elif self.conf['metric'] == 'acc':
                        acc = accuracy_score(train_y_sam, pred_tensor.numpy())
                        p = acc
                    elif self.conf['metric'] == 'precs':
                        precs = precision_score(train_y_sam, pred_tensor.numpy())
                        p = precs
                    elif self.conf['metric'] == 'recall':
                        recall = recall_score(train_y_sam, pred_tensor.numpy())
                        p = recall
                    elif self.conf['metric'] == 'f1_score':
                        f1_s = f1_score(train_y_sam, pred_tensor.numpy())
                        p = f1_s
                elif self.conf['pretrained_clsf'] == 'SVM':
                    y_predict = self.clsf_pretr.predict(train_x_sam)
                    if self.conf['metric'] == 'acc':
                        acc = accuracy_score(copy.deepcopy(train_y_sam),
                                             y_predict)
                        p = acc
                    elif self.conf['metric'] == 'auc':
                        auc = roc_auc_score(copy.deepcopy(train_y_sam),
                                            y_predict)
                        p = auc

                if np.sum(fea_binary == 1) == 0:
                    if self.conf['no_fea'] == 'random':
                        output_sam = np.random.randint(0, 2, train_y_sam.shape[0])
                        if self.conf['metric'] == 'auc':
                            p = roc_auc_score(train_y_sam, output_sam)
                        elif self.conf['metric'] == 'acc':
                            p = accuracy_score(train_y_sam, output_sam)
                        elif self.conf['metric'] == 'precs':
                            p = precision_score(train_y_sam, output_sam)
                        elif self.conf['metric'] == 'recall':
                            p = recall_score(train_y_sam, output_sam)
                        elif self.conf['metric'] == 'f1_score':
                            p = f1_score(train_y_sam, output_sam)
                        else:
                            print('Metric Error!')
                            return
                    elif self.conf['no_fea'] == '0.5':
                        p = 0.5
            elif self.conf['retrain']:
                train_x = self.dataset.ds_class2x_dict['train_set']
                train_y = self.dataset.lb2y_dict[self.lb]['train_set']['y']

                valid_x = self.dataset.ds_class2x_dict['valid_set']
                valid_y = self.dataset.lb2y_dict[self.lb]['valid_set']['y']

                sub_train_x = copy.deepcopy(train_x[:, fea])
                sub_valid_x = copy.deepcopy(valid_x[:, fea])

                # 如果提前训练好的分类器是SVM
                if self.conf['retrained_clsf'] == 'SVM':
                    param = get_global_dict_value('dataset_conf')['PARAM_PRE_TRAIN']['SVM'][self.lb]
                    clsf = SVClassifier(param['C'], param['kernel'],
                                        param['gamma'])
                    if len(fea) == 0:
                        output_valid = np.random.randint(0, 2, valid_y.shape[0])
                    else:
                        clsf.fit(sub_train_x, train_y)
                        output_valid = clsf.predict(sub_valid_x)
                    if self.conf['metric'] == 'acc':
                        acc = accuracy_score(copy.deepcopy(valid_y),
                                             output_valid)
                        p = acc
                    elif self.conf['metric'] == 'auc':
                        auc = roc_auc_score(copy.deepcopy(valid_y),
                                            output_valid)
                        p = auc
        if self.conf['S_CONFIG']['s2'][self.conf['s2']] == 'p=metric':
            pass
        elif self.conf['S_CONFIG']['s2'][self.conf['s2']] == 'p=-Rd':
            Rd = self.fea_set_redundancy(fea_binary)
            p = -Rd
        elif self.conf['S_CONFIG']['s2'][self.conf['s2']] == 'p=metric-gamma*Rd':
            Rd = self.fea_set_redundancy(fea_binary)
            p = p - self.conf['gamma'] * Rd

        if self.conf['S_CONFIG']['s1'][self.conf['s1']] == 'if s>=k, p=pk/s':
            s = np.sum(fea_binary)
            if int(s) >= self.conf['fea_num_threshold']:
                p = p * self.conf['fea_num_threshold'] / s
        elif self.conf['S_CONFIG']['s1'][self.conf['s1']] == 'if s>=k, p=p':
            pass
        return p
