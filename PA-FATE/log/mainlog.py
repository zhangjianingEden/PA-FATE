from util import *


class MainLog:
    loss_list = []
    lbs_info = {}

    def __init__(self, mode='train'):
        self.mode = mode
        slave_ip = get_host_ip()
        # print(slave_ip)
        for machine in machine_list:
            if machine.ip == slave_ip:
                self.slave_machine = machine
        self.epi_info_dict = {}
        self.epi_info_dict_init_fea = {}
        if mode == 'train':
            self.tr_lb_num = len(get_global_dict_value('dataset_conf')['tr_lb_list'])
            self.root_path = os.path.join(self.slave_machine.code_root_path, project_name + '_log')
            self.dataset_name = get_global_dict_value('dataset_conf')['dataset_name']
            self.method_name = get_global_dict_value('method_conf')['method_name']
            # self._time = str(time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime()))
            self.log_path = os.path.join(self.root_path, self.dataset_name, self.method_name,
                                         get_global_dict_value('log_conf')[
                                             'param_name'] + '___ip__' + self.slave_machine.ip.replace('.', '-'))
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.epi_info_dict[lb] = deque()
                self.epi_info_dict_init_fea[lb] = deque()
            if os.path.exists(self.log_path):
                shutil.rmtree(self.log_path)
            os.makedirs(self.log_path)
        elif mode == 'test':
            self.tr_lb_list = get_global_dict_value('dataset_conf')['tr_lb_list']
            self.te_lb_list = get_global_dict_value('dataset_conf')['te_lb_list']
            self.tr_lb_num = len(self.tr_lb_list)
            self.te_lb_num = len(self.te_lb_list)
            self.lb_list = self.tr_lb_list + self.te_lb_list
            self.lb_num = self.tr_lb_num + self.te_lb_num
            self.root_path = os.path.join(self.slave_machine.code_root_path, project_name + '_log')
            self.dataset_name = get_global_dict_value('dataset_conf')['dataset_name']
            self.method_name = get_global_dict_value('method_conf')['method_name']
            # self._time = str(time.strftime("%Y-%m-%d/%H-%M-%S", time.localtime()))
            self.log_root_path = os.path.join(self.root_path, self.dataset_name, self.method_name,
                                              get_global_dict_value('log_conf')[
                                                  'param_name'] + '___ip__' + self.slave_machine.ip.replace('.', '-'))
            self.log_path = os.path.join(self.log_root_path, 'test')
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.epi_info_dict[lb] = deque()
                self.epi_info_dict_init_fea[lb] = deque()
            if not os.path.exists(self.log_path):
                os.makedirs(self.log_path)
            for lb in self.lb_list:
                if not os.path.exists(os.path.join(self.log_path, lb)):
                    os.makedirs(os.path.join(self.log_path, lb))

    def load_lb_UCT_result(self, ):
        self.lb_UCT_result = {}
        if self.mode == 'train':
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                self.lb_UCT_result[lb] = np.load(os.path.join(self.log_path, lb, 'UCT_result.npy'), allow_pickle=True)[
                    ()]
        elif self.mode == 'test':
            for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                self.lb_UCT_result[lb] = np.load(os.path.join(self.log_path, lb, 'UCT_result.npy'), allow_pickle=True)[
                    ()]

    def save_lb_UCT_result(self):
        np.save(self.log_path + '/lb_UCT_result.npy', self.lb_UCT_result)

    def reset_epi_info_dict(self, type=None):
        if type == None:
            for lb in  self.epi_info_dict:
                self.epi_info_dict[lb].clear()
        elif type == 'init_fea':
            for lb in self.epi_info_dict_init_fea:
                self.epi_info_dict_init_fea[lb].clear()

    def save_epi_info_dict(self, type=None):
        if type == None:
            np.save(self.log_path + '/epi_info_dict.npy', self.epi_info_dict)
        elif type == 'init_fea':
            np.save(self.log_path + '/epi_info_dict_init_fea.npy', self.epi_info_dict_init_fea)

    def record_conf(self):
        self._conf_path = self.log_path + '/confs.txt'
        with open(self._conf_path, 'w') as f:
            lines = []
            lines.append('DATASET_CONF\n')
            for k in get_global_dict_value('dataset_conf'):
                lines.append(str(k) + '\t' + str(get_global_dict_value('dataset_conf')[k]) + '\n')
            lines.append('ENV_CONF\n')
            for k in get_global_dict_value('env_conf'):
                lines.append(str(k) + '\t' + str(get_global_dict_value('env_conf')[k]) + '\n')
            lines.append('METHOD_CONF\n')
            for k in get_global_dict_value('method_conf'):
                lines.append(str(k) + '\t' + str(get_global_dict_value('method_conf')[k]) + '\n')
            lines.append('LOG_CONF\n')
            for k in get_global_dict_value('log_conf'):
                lines.append(str(k) + '\t' + str(get_global_dict_value('log_conf')[k]) + '\n')
            f.writelines(lines)

    # def get_start_time(self):
    #     return self._time

    def record_report(self, report_str):
        self._report_path = self.log_path + '/report.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def load_shared_buffer(self, env_id, type=None):
        if type == None:
            shared_buffer = \
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'shared_buffer.npy'),
                        allow_pickle=True)[()]['buffer']
        elif type == 'init_fea':
            shared_buffer = \
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'shared_buffer_init_fea.npy'),
                        allow_pickle=True)[()]['buffer']
        return shared_buffer

    def load_epi_info(self, env_id, lb, type=None):
        if type == None:
            epi_info = \
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'epi_info.npy'), allow_pickle=True)[()]
            self.epi_info_dict[lb].append(epi_info)
        elif type == 'init_fea':
            epi_info_init_fea = \
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'epi_info_init_fea.npy'),
                        allow_pickle=True)[()]
            self.epi_info_dict_init_fea[lb].append(epi_info_init_fea)

    def load_envs_info(self):
        if self.mode == 'train':
            for tr_lb_id in range(self.tr_lb_num):
                tr_lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                self.lbs_info[tr_lb] = \
                    np.load(os.path.join(self.log_path, tr_lb, 'metric_list.npy'), allow_pickle=True)[()]
        elif self.mode == 'test':
            for lb_id in range(self.tr_lb_num + self.te_lb_num):
                lb = self.lb_list[lb_id]
                if lb not in self.te_lb_list:
                    self.lbs_info[lb] = \
                    np.load(os.path.join(self.log_root_path, lb, 'metric_list.npy'), allow_pickle=True)[()]
                else:
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

    def save_model_top(self, top_best_model_list):
        length = len(top_best_model_list)
        for i in range(length):
            self._model_path = self.log_path + '/No' + str(length - i) + '_model.pth'
            torch.save(top_best_model_list[i][2], self._model_path)

    def update_check_and_transfer_to_log_master(self, mode):
        connect_flag, machine, ssh, sftp = machine_ip2ssh(get_global_dict_value('log_conf')['log_master_ip'])
        if connect_flag:
            try:
                if mode == 'trained':
                    check_path = os.path.join(self.log_path, 'check.npy')
                    check = {}
                    check[mode] = True
                    np.save(check_path, check)
                    local_dir_path = self.log_path
                    remote_dir_path = \
                        os.path.join(machine.code_root_path, 'drlfs' + get_global_dict_value('log_conf')[
                            'code_version'] + '_' +
                                     get_global_dict_value('log_conf')['log_master_ip'].replace('.', '-') +
                                     '_log_master', *self.log_path.split('/')[-3:])
                    sftp_put_dir(machine.ip, local_dir_path, remote_dir_path)
                elif mode == 'tested':
                    check_path = os.path.join(self.log_root_path, 'check.npy')
                    check = np.load(check_path, allow_pickle=True)[()]
                    check[mode] = True
                    np.save(check_path, check)
                    local_dir_path = self.log_path
                    remote_dir_path = \
                        os.path.join(machine.code_root_path, 'drlfs' + get_global_dict_value('log_conf')[
                            'code_version'] + '_' +
                                     get_global_dict_value('log_conf')['log_master_ip'].replace('.', '-') +
                                     '_log_master', *self.log_path.split('/')[-4:])
                    sftp_put_dir(machine.ip, local_dir_path, remote_dir_path)

                    local_file_path = check_path
                    remote_file_path = \
                        os.path.join(machine.code_root_path, 'drlfs' + get_global_dict_value('log_conf')[
                            'code_version'] + '_' +
                                     get_global_dict_value('log_conf')['log_master_ip'].replace('.', '-') +
                                     '_log_master', *self.log_path.split('/')[-4:-1], 'check.npy')
                    sftp_put_file(machine.ip, local_file_path, remote_file_path)
            except:
                print(str(datetime.now()).split('.')[0], 'update_check_and_transfer_to_log_master ERROR1!', mode)
            finally:
                ssh.close()
        else:
            print(str(datetime.now()).split('.')[0], 'update_check_and_transfer_to_log_master ERROR2!', mode)
