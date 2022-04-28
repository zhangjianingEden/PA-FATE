from util import *
from log.sublog import *
from method.E_Tree import *


def E_Tree_subp(lb_id,
                log_path,
                E_Tree_shared_ifdone,
                dataset_conf,
                env_conf,
                method_conf,
                log_conf,
                ):
    with torch.no_grad():
        global_dict_init()
        set_global_dict_value('dataset_conf', dataset_conf)
        set_global_dict_value('env_conf', env_conf)
        set_global_dict_value('method_conf', method_conf)
        set_global_dict_value('log_conf', log_conf)

        random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        np.random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        torch.manual_seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        sub_iter_counter = 0
        sublog = SubLog('E_Tree', lb_id, log_path)
        lb = get_global_dict_value('dataset_conf')['tr_lb_list'][lb_id]
        E_Tree_result = {}
        E_Tree_result['ITS'] = {}
        E_Tree_result['ITE'] = {}
        E_Tree_result['ITS']['uc'] = 'Nan'
        E_Tree_result['ITS']['dis'] = 'Nan'
        E_Tree_result['ITE']['init_fea'] = []
        E_Tree_result['ITE']['init_fea'].append({})
        E_Tree_result['ITE']['init_fea'][0]['fea_binary'] = np.zeros(get_global_dict_value('dataset_conf')['fea_num'],
                                                                     dtype=np.float32)
        E_Tree_result['ITE']['init_fea'][0]['step_counter'] = 0
        lb_epi_info_deque_buffer = deque(maxlen=get_global_dict_value('method_conf')['ITS']['fq'] * 5)
        lb_metric = \
            np.load(os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_metric.npy'),
                    allow_pickle=True)[
                ()][lb]
        my_tree = ExperienceTree()
        build_times = 0
        while sub_iter_counter < get_global_dict_value('method_conf')['train_iter']:
            while True:
                if not E_Tree_shared_ifdone.value:
                    ################################## load lb_epi_info ####################################
                    lb_epi_info_deque = np.load(log_path + '/epi_info_dict.npy', allow_pickle=True)[()][lb]
                    epi_num = len(lb_epi_info_deque)
                    if epi_num != 0:
                        lb_epi_info_deque_buffer.extend(lb_epi_info_deque)
                        # build tree
                        my_tree.build(lb_epi_info_deque)
                        build_times += 1
                    ################################## generate ITS result #################################
                    if (sub_iter_counter + 1) % get_global_dict_value('method_conf')['ITS']['fq'] == 0 and \
                            sub_iter_counter > 0:
                        # uc
                        final_state_list = [epi_info['final_state'] for epi_info in lb_epi_info_deque_buffer]
                        final_state_array = np.array(final_state_list)
                        final_state_num = final_state_array.shape[0]
                        fea_num = final_state_array.shape[1]
                        E_Tree_result['ITS']['uc'] = 1 - (np.sum(
                            abs(final_state_num - 2 * np.sum(final_state_array, axis=0))) / final_state_num / fea_num)
                        # dis
                        final_meta_reward_list = [epi_info['meta_reward_list']['pretrain_metric_list'][-1] for epi_info
                                                  in lb_epi_info_deque_buffer]
                        E_Tree_result['ITS']['dis'] = (lb_metric['all_fea'][
                                                           get_global_dict_value('env_conf')['rwd'][
                                                               'metric']] - np.mean(
                            final_meta_reward_list)) / lb_metric['all_fea'][
                                                          get_global_dict_value('env_conf')['rwd']['metric']]
                    ################################## generate ITE result #################################
                    if build_times % get_global_dict_value('method_conf')['ITE']['k'] == 0 and build_times > 0:
                        E_Tree_result['ITE']['init_fea'] = my_tree.choose()
                    ################################## sublog work ####################################
                    sublog.record_E_Tree_result(E_Tree_result)
                    E_Tree_shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
