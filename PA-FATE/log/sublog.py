from util import *
from env.classifiers import *


class SubLog:
    def __init__(self, type, process_id=None, log_path=None, dataset=None, type2='train'):
        self.episode_info = {}
        self.main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
        self.root_log_path = log_path
        if type == 'simulator':
            self.UCT_result = {}
            self.log_path = os.path.join(log_path, 'process_' + str(process_id))
        elif type == 'valid':
            if type2 == 'train':
                self.valid_lb = get_global_dict_value('dataset_conf')['tr_lb_list'][process_id]
            elif type2 == 'test':
                self.valid_lb = get_global_dict_value('dataset_conf')['te_lb_list'][process_id]
            self.log_path = os.path.join(log_path, self.valid_lb)
            self.dataset = dataset
            self.metric_list = {}
            for metric_type in get_global_dict_value('dataset_conf')['metric_type_list']:
                self.metric_list[metric_type + '_list'] = {}
                if type2 == 'train':
                    self.metric_list[metric_type + '_list'][metric_type] = np.zeros(
                        get_global_dict_value('method_conf')['train_iter'] + 1, dtype=np.float32)
                elif type2 == 'test':
                    self.metric_list[metric_type + '_list'][metric_type] = np.zeros(
                        get_global_dict_value('method_conf')['test_iter'] + 1, dtype=np.float32)
                self.metric_list[metric_type + '_list']['max_' + metric_type + '_state'] = \
                    np.zeros(10, dtype=np.float32)
                self.metric_list[metric_type + '_list']['max_' + metric_type] = 0
        elif type == 'UCT':
            if type2 == 'train':
                self.UCT_lb = get_global_dict_value('dataset_conf')['tr_lb_list'][process_id]
            elif type2 == 'test':
                self.UCT_lb = get_global_dict_value('dataset_conf')['te_lb_list'][process_id]
            self.log_path = os.path.join(log_path, self.UCT_lb)

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

    def gen_metric_value(self, fea_binary):
        cur_fea = []
        for i in range(len(fea_binary)):
            if fea_binary[i] == 1:
                cur_fea.append(i)

        train_x = self.dataset.ds_class2x_dict['train_set']
        train_y = self.dataset.lb2y_dict[self.valid_lb]['train_set']['y']
        train_x_cur = copy.deepcopy(train_x[:, cur_fea])

        param = get_global_dict_value('dataset_conf')['PARAM_PRE_TRAIN']['SVM'][
            self.valid_lb]
        clsf = SVClassifier(param['C'], param['kernel'],
                            param['gamma'])

        valid_x = self.dataset.ds_class2x_dict['valid_set']
        valid_y = self.dataset.lb2y_dict[self.valid_lb]['valid_set']['y']

        metric_cur_dict = {}

        if len(cur_fea) == 0:
            pred_y_cur = np.random.randint(0, 2, valid_y.shape[0])
        else:
            clsf.fit(train_x_cur, train_y)
            valid_x_cur = copy.deepcopy(valid_x[:, cur_fea])
            pred_y_cur = clsf.predict(valid_x_cur)
        for metric_type in get_global_dict_value('dataset_conf')['metric_type_list']:
            if metric_type == 'auc':
                metric_cur_dict[metric_type] = roc_auc_score(copy.deepcopy(valid_y), pred_y_cur)
            if metric_type == 'acc':
                metric_cur_dict[metric_type] = accuracy_score(copy.deepcopy(valid_y), pred_y_cur)
            if metric_type == 'precs':
                metric_cur_dict[metric_type] = precision_score(copy.deepcopy(valid_y), pred_y_cur)
            if metric_type == 'recall':
                metric_cur_dict[metric_type] = recall_score(copy.deepcopy(valid_y), pred_y_cur)
            if metric_type == 'f1_score':
                metric_cur_dict[metric_type] = f1_score(copy.deepcopy(valid_y), pred_y_cur)
        return metric_cur_dict

    def gen_metrics_result(self, iter_id):
        final_st = self.episode_info['st'][-1]
        final_fea_binary = final_st['fea_binary']
        # extreme_fea_binary = np.zeros(ENV_CONF['fea_num'], dtype=np.float32)
        # extreme_fea_binary[:41] = 1
        # metric_cur_dict = self.gen_metric_value(extreme_fea_binary[reremap_fea_list])
        metric_cur_dict = self.gen_metric_value(
            final_fea_binary[get_global_dict_value('dataset_conf')['reremap_fea_list']])

        self.metric_list['cur_state'] = final_fea_binary
        for metric_type in get_global_dict_value('dataset_conf')['metric_type_list']:
            self.metric_list[metric_type + '_list'][metric_type][iter_id] = metric_cur_dict[metric_type]
            if metric_cur_dict[metric_type] > self.metric_list[metric_type + '_list']['max_' + metric_type]:
                self.metric_list[metric_type + '_list']['max_' + metric_type] = metric_cur_dict[metric_type]
                self.metric_list[metric_type + '_list']['max_' + metric_type + '_state'] = copy.deepcopy(
                    final_fea_binary)

    def record_metrics_result(self):
        np.save(self.log_path + '/metric_list.npy', self.metric_list)

    def save_buffer(self, shared_buffer, type=None):
        tmp_dict = {}
        tmp_dict['buffer'] = shared_buffer
        if type == None:
            np.save(self.log_path + '/shared_buffer.npy', tmp_dict)
        elif type == 'init_fea':
            np.save(self.log_path + '/shared_buffer_init_fea.npy', tmp_dict)

    def save_epi_info(self, type=None):
        tmp_dict = {}
        tmp_dict['reward_list'] = self.episode_info['reward']
        tmp_dict['final_state'] = self.episode_info['st'][-1]['fea_binary']
        tmp_dict['meta_reward_list'] = self.episode_info['meta_reward_list']
        if type == None:
            np.save(self.log_path + '/epi_info.npy', tmp_dict)
        elif type == 'init_fea':
            np.save(self.log_path + '/epi_info_init_fea.npy', tmp_dict)

    def record_UCT_result(self, UCT_result):
        np.save(self.log_path + '/UCT_result.npy', UCT_result)

    def load_UCT_result(self, lb):
        try:
            tmp_dict = np.load(os.path.join(self.root_log_path, 'lb_UCT_result.npy'), allow_pickle=True)[()][lb]
        except:
            pass
        else:
            self.UCT_result = tmp_dict
