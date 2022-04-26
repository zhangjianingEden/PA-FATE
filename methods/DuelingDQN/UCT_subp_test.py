from util import *
from log.sublog import *
from methods.UCT import *


def UCT_subp_test(lb_id,
                  log_path,
                  UCT_shared_ifdone,
                  dataset_conf,
                  env_conf,
                  method_conf,
                  log_conf,
                  ):
    with torch.no_grad():
        start_datetime = datetime.now()
        global_dict_init()
        set_global_dict_value('dataset_conf', dataset_conf)
        set_global_dict_value('env_conf', env_conf)
        set_global_dict_value('method_conf', method_conf)
        set_global_dict_value('log_conf', log_conf)

        random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        np.random.seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        torch.manual_seed(get_global_dict_value('method_conf')['seed'] + lb_id)
        sub_iter_counter = 0
        print("---------------------------->", lb_id, "UCT_subp")
        sublog = SubLog('UCT', lb_id, log_path, type2='test')
        lb = get_global_dict_value('dataset_conf')['te_lb_list'][lb_id]
        UCT_result = {}
        UCT_result['SMALL'] = {}
        UCT_result['SMALL']['init_fea'] = []
        UCT_result['SMALL']['init_fea'].append({})
        UCT_result['SMALL']['init_fea'][0]['fea_binary'] = np.zeros(get_global_dict_value('dataset_conf')['fea_num'],
                                                                    dtype=np.float32)
        UCT_result['SMALL']['init_fea'][0]['step_counter'] = 0
        lb_epi_info_deque_buffer = deque(maxlen=get_global_dict_value('method_conf')['BIG']['fq'] * 5)

        my_tree = MyTree()
        build_times = 0
        while sub_iter_counter < get_global_dict_value('method_conf')['test_iter']:
            while True:
                if not UCT_shared_ifdone.value:
                    ################################## load lb_epi_info ####################################
                    lb_epi_info_deque = np.load(log_path + '/epi_info_dict.npy', allow_pickle=True)[()][lb]
                    lb_epi_info_deque_init_fea = \
                        np.load(log_path + '/epi_info_dict_init_fea.npy', allow_pickle=True)[()][lb]
                    epi_num = len(lb_epi_info_deque) + len(lb_epi_info_deque_init_fea)
                    if epi_num != 0:
                        lb_epi_info_deque_buffer.extend(lb_epi_info_deque)
                        lb_epi_info_deque_buffer.extend(lb_epi_info_deque_init_fea)
                        # build tree
                        my_tree.build(lb_epi_info_deque)
                        my_tree.build(lb_epi_info_deque_init_fea)
                        build_times += 1
                    ################################## generate SMALL result #################################
                    if get_global_dict_value('method_conf')['SMALL']['use'] and \
                            build_times % get_global_dict_value('method_conf')['SMALL']['k'] == 0 and build_times > 0:
                        UCT_result['SMALL']['init_fea'] = my_tree.choose(build_times)
                    ################################## sublog work ####################################
                    sublog.record_UCT_result(UCT_result)
                    UCT_shared_ifdone.value = True
                    sub_iter_counter += 1
                    break
        # make check if test is health easier
        delta_time = (datetime.now() - start_datetime).seconds
        if delta_time < 30:
            time.sleep(30 - delta_time)
