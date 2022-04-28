from util import *


class MainLog:
    loss_list = []
    lbs_info = {}

    def __init__(self, mode='train'):
        self.mode = mode
        self.epi_info_dict = {}
        self.epi_info_dict_init_fea = {}
        self.root_path = get_global_dict_value('log_conf')['log_root_path']
        if mode == 'train':
            self.tr_lb_num = len(get_global_dict_value('dataset_conf')['tr_lb_list'])

            self.dataset_name = get_global_dict_value('dataset_conf')['dataset_name']
            self.log_path = os.path.join(self.root_path, self.dataset_name,
                                         'max_fea_ratio_' + str(get_global_dict_value('env_conf')['max_fea_ratio']))
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.epi_info_dict[lb] = deque()
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
        elif mode == 'test':
            self.tr_lb_list = get_global_dict_value('dataset_conf')['tr_lb_list']
            self.te_lb_list = get_global_dict_value('dataset_conf')['te_lb_list']
            self.tr_lb_num = len(self.tr_lb_list)
            self.te_lb_num = len(self.te_lb_list)
            self.lb_list = self.tr_lb_list + self.te_lb_list
            self.lb_num = self.tr_lb_num + self.te_lb_num
            self.dataset_name = get_global_dict_value('dataset_conf')['dataset_name']
            self.log_root_path = os.path.join(self.root_path, self.dataset_name, 'max_fea_ratio_' + str(
                get_global_dict_value('env_conf')['max_fea_ratio']))
            self.log_path = os.path.join(self.log_root_path, 'test')
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.epi_info_dict[lb] = deque()
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            for lb in self.te_lb_list:
                if not os.path.exists(os.path.join(self.log_path, lb)):
                    os.makedirs(os.path.join(self.log_path, lb))

    def load_lb_E_Tree_result(self):
        self.lb_E_Tree_result = {}
        if self.mode == 'train':
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.lb_E_Tree_result[lb] = \
                np.load(os.path.join(self.log_path, lb, 'E_Tree_result.npy'), allow_pickle=True)[()]
        elif self.mode == 'test':
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.lb_E_Tree_result[lb] = \
                np.load(os.path.join(self.log_path, lb, 'E_Tree_result.npy'), allow_pickle=True)[()]

    def save_lb_E_Tree_result(self):
        np.save(self.log_path + '/lb_E_Tree_result.npy', self.lb_E_Tree_result)

    def reset_epi_info_dict(self):
        for lb in self.epi_info_dict:
            self.epi_info_dict[lb].clear()

    def save_epi_info_dict(self):
        np.save(self.log_path + '/epi_info_dict.npy', self.epi_info_dict)

    def load_shared_buffer(self, env_id):
        shared_buffer = \
            np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'shared_buffer.npy'),
                    allow_pickle=True)[()]['buffer']
        return shared_buffer

    def load_epi_info(self, env_id, lb):
        epi_info = \
            np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'epi_info.npy'), allow_pickle=True)[()]
        self.epi_info_dict[lb].append(epi_info)

    def load_envs_info(self):
        if self.mode == 'train':
            for tr_lb_id in range(self.tr_lb_num):
                tr_lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                self.lbs_info[tr_lb] = \
                    np.load(os.path.join(self.log_path, tr_lb, 'metric_list.npy'), allow_pickle=True)[()]
        elif self.mode == 'test':
            for lb_id in range(self.tr_lb_num + self.te_lb_num):
                lb = self.lb_list[lb_id]
                if lb in self.te_lb_list:
                    self.lbs_info[lb] = np.load(os.path.join(self.log_path, lb, 'metric_list.npy'), allow_pickle=True)[
                        ()]

    def record_metrics_result(self):
        np.save(self.log_path + '/lbs_info.npy', self.lbs_info)

    def record_loss(self, loss):
        self.loss_list.append(loss)
        np.save(self.log_path + '/loss.npy', self.loss_list)

    def save_cur_model(self, model):
        self._model_path = self.log_path + '/tmp_model.pth'
        torch.save(model.state_dict(), self._model_path)

    def save_cur_valid_model(self, model):
        self._model_path = self.log_path + '/tmp_valid_model.pth'
        torch.save(model.state_dict(), self._model_path)

    def save_cur_model_test(self, lb, model):
        self._model_path = os.path.join(self.log_path, lb, 'tmp_model.pth')
        torch.save(model.state_dict(), self._model_path)

    def save_cur_valid_model_test(self, lb, model):
        self._model_path = os.path.join(self.log_path, lb, 'tmp_valid_model.pth')
        torch.save(model.state_dict(), self._model_path)

    def save_model(self, model):
        self._model_path = self.log_path + '/model.pth'
        torch.save(model.state_dict(), self._model_path)
        print('model has been saved to ', self._model_path)
