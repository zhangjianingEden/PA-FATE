from util import *
from log.sublog import *
from method.E_Tree import *


def E_Tree_subp_test(lb_id,
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
        sublog = SubLog('E_Tree', lb_id, log_path, type2='test')
        lb = get_global_dict_value('dataset_conf')['te_lb_list'][lb_id]
        E_Tree_result = {}
        E_Tree_result['ITE'] = {}
        E_Tree_result['ITE']['init_fea'] = []
        E_Tree_result['ITE']['init_fea'].append({})
        E_Tree_result['ITE']['init_fea'][0]['fea_binary'] = np.zeros(get_global_dict_value('dataset_conf')['fea_num'],
                                                                     dtype=np.float32)
        E_Tree_result['ITE']['init_fea'][0]['step_counter'] = 0
        lb_epi_info_deque_buffer = deque(maxlen=get_global_dict_value('method_conf')['ITS']['fq'] * 5)

        my_tree = ExperienceTree()
        build_times = 0
        while sub_iter_counter < get_global_dict_value('method_conf')['further_train_iter']:
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
                    ################################## generate ITE result #################################
                    if build_times % get_global_dict_value('method_conf')['ITE']['k'] == 0 and build_times > 0:
                        E_Tree_result['ITE']['init_fea'] = my_tree.choose()
                    ################################## sublog work ####################################
                    sublog.record_E_Tree_result(E_Tree_result)
                    E_Tree_shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
