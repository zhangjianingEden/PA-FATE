from util import *
from log.mainlog import *
from .DuelingDQN import *
from .storage import *
from .sharestorage import *
from .subp import *
from .valid_subp import *
from .UCT_subp import *


def main():
    further_train_time = 0
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    lb_metric = \
        np.load(os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_metric.npy'),
                allow_pickle=True)[()]
    main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
    all_fea_lb_mean_auc = np.mean(
        [lb_metric[lb]['all_fea'][main_metric_type] for lb in get_global_dict_value('dataset_conf')['tr_lb_list']])
    mainlog = MainLog()
    mainlog.record_conf()
    mainlog.save_epi_info_dict()
    mainlog.save_epi_info_dict(type='init_fea')
    network = Network().to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
    network.eval()
    target_network = Network().to('cuda:' + str(get_global_dict_value('method_conf')['gpu_id']))
    target_network.eval()
    agent = DuelingDQN(network=network, target_network=target_network)
    mainlog.save_cur_model(network)
    mainlog.save_cur_valid_model(network)
    rollout = RolloutStorage(get_global_dict_value('method_conf')['capacity'])
    shared_rollout_list = []
    tr_lb_num = len(get_global_dict_value('dataset_conf')['tr_lb_list'])
    env_num = tr_lb_num * get_global_dict_value('method_conf')['env_num_per_lb']
    for env_id in range(env_num):
        shared_rollout_list.append(ShareRolloutStorage())
    cur_datetime = datetime.now()
    print('process time:', cur_datetime - start_datetime)
    mainlog.record_report(str(start_datetime).split('.')[0] + ' ' + str(cur_datetime).split('.')[0])
    shared_ifdone_list = [mp.Value('b', False) for _ in range(env_num)]

    shared_tr_lb_id_list = [mp.Value('i', env_id % tr_lb_num) for env_id in range(env_num)]
    update_tr_lb_id_list = [env_id % tr_lb_num for env_id in range(env_num)]

    simulator_processes = []
    for env_id in range(env_num):
        p = mp.Process(target=subp,
                       args=(env_id,
                             mainlog.log_path,
                             shared_rollout_list[env_id],
                             shared_ifdone_list[env_id],
                             shared_tr_lb_id_list[env_id],
                             get_global_dict_value('dataset_conf'),
                             get_global_dict_value('env_conf'),
                             get_global_dict_value('method_conf'),
                             get_global_dict_value('log_conf'),
                             )
                       )
        simulator_processes.append(p)
        p.start()

    valid_processes = []
    valid_shared_ifdone_list = [mp.Value('b', False) for _ in range(tr_lb_num)]
    for lb_id in range(tr_lb_num):
        p = mp.Process(target=valid_subp,
                       args=(lb_id,
                             mainlog.log_path,
                             valid_shared_ifdone_list[lb_id],
                             get_global_dict_value('dataset_conf'),
                             get_global_dict_value('env_conf'),
                             get_global_dict_value('method_conf'),
                             get_global_dict_value('log_conf'),
                             )
                       )
        valid_processes.append(p)
        p.start()

    UCT_processes = []
    UCT_shared_ifdone_list = [mp.Value('b', False) for _ in range(tr_lb_num)]
    for lb_id in range(tr_lb_num):
        p = mp.Process(target=UCT_subp,
                       args=(lb_id,
                             mainlog.log_path,
                             UCT_shared_ifdone_list[lb_id],
                             get_global_dict_value('dataset_conf'),
                             get_global_dict_value('env_conf'),
                             get_global_dict_value('method_conf'),
                             get_global_dict_value('log_conf'),
                             )
                       )
        UCT_processes.append(p)
        p.start()

    top_best_model_list = []
    lb_mean_main_metric_list = []
    max_iter_avg_lb_mean_main_metric = 0
    max_iter_avg_lb_mean_main_metric_iter = 0

    tmp_valid_model = Network()
    iter_id = 0
    while iter_id < get_global_dict_value('method_conf')['train_iter']:
        iter_start_datetime = datetime.now()
        # print(str(datetime.now()).split('.')[0], iter_id)
        # load UCT result
        while True:
            UCT_global_ifdone = 0
            for UCT_shared_ifdone in UCT_shared_ifdone_list:
                if UCT_shared_ifdone.value:
                    UCT_global_ifdone += 1
                else:
                    break
            if UCT_global_ifdone == tr_lb_num:
                mainlog.load_lb_UCT_result()
                mainlog.save_lb_UCT_result()
                break
        # update network
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break
            if global_ifdone == env_num:
                # load sub_buffer
                for env_id in range(env_num):
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][shared_tr_lb_id_list[env_id].value]
                    new_shared_buffer = mainlog.load_shared_buffer(env_id)
                    rollout.load_shared_buffer(lb, new_shared_buffer)
                    new_shared_buffer_init_fea = mainlog.load_shared_buffer(env_id, type='init_fea')
                    rollout.load_shared_buffer(lb, new_shared_buffer_init_fea, type='init_fea')
                    mainlog.load_epi_info(env_id, lb)
                    mainlog.load_epi_info(env_id, lb, type='init_fea')
                # update params
                network.train()
                target_network.train()
                loss_per_sample = 0
                update_time = 0
                further_train_start_time = time.time()
                for env_id in range(env_num):
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][update_tr_lb_id_list[env_id]]
                    if rollout.is_enough(lb):
                        loss_per_sample += agent.update(rollout, lb, iter_id)
                        update_time += 1
                    if rollout.is_enough(lb, type='init_fea'):
                        loss_per_sample += agent.update(rollout, lb, iter_id, type2='init_fea')
                        update_time += 1
                further_train_end_time = time.time()
                further_train_time += further_train_end_time - further_train_start_time
                if update_time > 0:
                    loss_per_sample /= update_time
                mainlog.record_loss(loss_per_sample)
                if (iter_id + 1) % get_global_dict_value('method_conf')['tar_net_update_iter_gap'] == 0:
                    agent.update_tar_net()

                network.eval()
                target_network.eval()
                mainlog.save_cur_model(network)
                break
        mainlog.save_epi_info_dict()
        if iter_id < 1950:
            mainlog.reset_epi_info_dict()
        mainlog.save_epi_info_dict(type='init_fea')
        if iter_id < 1950:
            mainlog.reset_epi_info_dict(type='init_fea')

        for UCT_shared_ifdone in UCT_shared_ifdone_list:
            UCT_shared_ifdone.value = False

        # if need BIG Management
        if get_global_dict_value('method_conf')['BIG']['use'] and \
                (iter_id + 1) % get_global_dict_value('method_conf')['BIG']['fq'] == 0 and iter_id > 0:
            p_si_array = 0
            p_si_array_flag = False

            p_uc_array = 0
            p_uc_array_flag = False
            p_dis_array = 0
            p_dis_array_flag = False

            if get_global_dict_value('method_conf')['BIG']['si']['uc'] != 0:
                tr_lb_uc_list = []
                for tr_lb_id in range(tr_lb_num):
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                    tr_lb_uc_list.append(mainlog.lb_UCT_result[lb]['BIG']['uc'])
                p_uc_array = np.array(gen_p_list(tr_lb_uc_list))
                p_uc_array_flag = True
                p_si_array += p_uc_array * get_global_dict_value('method_conf')['BIG']['si']['uc']
                p_si_array_flag = True
            if get_global_dict_value('method_conf')['BIG']['si']['dis'] != 0:
                tr_lb_dis_list = []
                for tr_lb_id in range(tr_lb_num):
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                    tr_lb_dis_list.append(mainlog.lb_UCT_result[lb]['BIG']['dis'])
                p_dis_array = np.array(gen_p_list(tr_lb_dis_list))
                p_dis_array_flag = True
                p_si_array += p_dis_array * get_global_dict_value('method_conf')['BIG']['si']['dis']
                p_si_array_flag = True
            if p_si_array_flag:
                p_list = gen_p_list(list(p_si_array))
                new_shared_tr_lb_id_array = np.random.choice(tr_lb_num, env_num, p=p_list)
                for env_id in range(env_num):
                    shared_tr_lb_id_list[env_id].value = new_shared_tr_lb_id_array[env_id]

            p_update_array = 0
            p_update_array_flag = False
            if get_global_dict_value('method_conf')['BIG']['up']['uc'] != 0:
                if not p_uc_array_flag:
                    tr_lb_uc_list = []
                    for tr_lb_id in range(tr_lb_num):
                        lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                        tr_lb_uc_list.append(mainlog.lb_UCT_result[lb]['BIG']['uc'])
                    p_uc_array = np.array(gen_p_list(tr_lb_uc_list))
                p_update_array += p_uc_array * get_global_dict_value('method_conf')['BIG']['up']['uc']
                p_update_array_flag = True
            if get_global_dict_value('method_conf')['BIG']['up']['dis'] != 0:
                if not p_dis_array_flag:
                    tr_lb_dis_list = []
                    for tr_lb_id in range(tr_lb_num):
                        lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                        tr_lb_dis_list.append(mainlog.lb_UCT_result[lb]['BIG']['dis'])
                    p_dis_array = np.array(gen_p_list(tr_lb_dis_list))
                p_update_array += p_dis_array * get_global_dict_value('method_conf')['BIG']['up']['dis']
                p_update_array_flag = True
            if p_update_array_flag:
                p_list = gen_p_list(list(p_update_array))
                update_tr_lb_id_list = list(np.random.choice(tr_lb_num, env_num, p=p_list))
        for shared_ifdone in shared_ifdone_list:
            shared_ifdone.value = False
        # if need valid and log
        if (iter_id + 1) % get_global_dict_value('method_conf')['cal_log_frequency'] == 0:
            print('time_cost_per_100_iter:', further_train_time * 2)
            further_train_time = 0
            # valid
            while True:
                valid_global_ifdone = 0
                for valid_shared_ifdone in valid_shared_ifdone_list:
                    if valid_shared_ifdone.value:
                        valid_global_ifdone += 1
                    else:
                        break
                if valid_global_ifdone == tr_lb_num:
                    ################################## save good model ####################################
                    tmp_valid_model.load_state_dict(
                        torch.load(mainlog.log_path + '/tmp_valid_model.pth', map_location=torch.device('cpu')))
                    mainlog.load_envs_info()
                    mainlog.save_cur_valid_model(network)
                    for valid_shared_ifdone in valid_shared_ifdone_list:
                        valid_shared_ifdone.value = False
                    lb_mean_main_metric = np.mean([mainlog.lbs_info[tr_lb][main_metric_type + '_list'][
                                                       main_metric_type][
                                                       iter_id + 1 - get_global_dict_value('method_conf')[
                                                           'cal_log_frequency']]
                                                   for tr_lb in
                                                   get_global_dict_value('dataset_conf')['tr_lb_list']])
                    lb_mean_main_metric_list.append(lb_mean_main_metric)
                    iter_avg_lb_mean_main_metric = np.mean(
                        lb_mean_main_metric_list[-get_global_dict_value('method_conf')['iter_avg']:])
                    if iter_avg_lb_mean_main_metric > max_iter_avg_lb_mean_main_metric:
                        max_iter_avg_lb_mean_main_metric = iter_avg_lb_mean_main_metric
                        max_iter_avg_lb_mean_main_metric_iter = iter_id + 1 - get_global_dict_value('method_conf')[
                            'cal_log_frequency']
                        mainlog.save_model(tmp_valid_model)

                    if_insert = False
                    insert_pos = -1
                    if len(top_best_model_list) == 0:
                        if_insert = True
                        insert_pos = 0
                    else:
                        for i, model in enumerate(top_best_model_list):
                            if model[1] < lb_mean_main_metric:
                                if_insert = True
                                insert_pos = i + 1
                            if model[1] > lb_mean_main_metric:
                                break
                    if if_insert:
                        model = [iter_id + 1 - get_global_dict_value('method_conf')[
                            'cal_log_frequency'], lb_mean_main_metric, tmp_valid_model.state_dict()]
                        top_best_model_list.insert(insert_pos, model)
                        if len(top_best_model_list) > 5:
                            del top_best_model_list[0]
                        mainlog.save_model_top(top_best_model_list)
                    break
            # log
            mainlog.record_metrics_result()
            max_lb_mean_main_metric = np.mean([mainlog.lbs_info[lb][main_metric_type + '_list'][
                                                   'max_' + main_metric_type] for lb in
                                               get_global_dict_value('dataset_conf')['tr_lb_list']])
            report_str = 'iter: ' + str(iter_id + 1 - get_global_dict_value('method_conf')['cal_log_frequency']) \
                         + ' all_fea_lb_mean_' + main_metric_type + ': ' + str(
                np.round(all_fea_lb_mean_auc, 5)) \
                         + ' max_lb_mean_' + main_metric_type + ': ' + str(
                np.round(max_lb_mean_main_metric, 5)) \
                         + ' max_iter_avg_lb_mean_' + main_metric_type + ': ' + str(
                np.round(max_iter_avg_lb_mean_main_metric, 5)) \
                         + ' max_iter_avg_lb_mean_' + main_metric_type + '_iter: ' + str(
                max_iter_avg_lb_mean_main_metric_iter) \
                         + '\n iter_avg_lb_mean_' + main_metric_type + ': ' + str(
                np.round(iter_avg_lb_mean_main_metric, 5)) \
                         + ' lb_mean_' + main_metric_type + ': ' + str(np.round(lb_mean_main_metric, 5))

            # ------------------------------------------------------------
            for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                    main_metric_type][iter_id + 1 - get_global_dict_value('method_conf')['cal_log_frequency']]
                max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                    'max_' + main_metric_type]
                max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
                    'max_' + main_metric_type + '_state']
                cur_state = mainlog.lbs_info[lb]['cur_state']
                lb_report_str = 'lb: ' + lb \
                                + ' max_' + main_metric_type + ': ' + str(np.round(max_main_metric, 5)) \
                                + ' ' + main_metric_type + ': ' + str(np.round(main_metric, 5))
                mainlog.record_report(lb_report_str)
                print(lb_report_str)
                lb_report_str2 = '\t all_fea_' + main_metric_type + ': ' + str(
                    np.round(lb_metric[lb]['all_fea'][main_metric_type], 5)) \
                                 + ' random_' + main_metric_type + ': ' + str(
                    np.round(lb_metric[lb]['random'][main_metric_type], 5)) \
                                 + ' pretrainedDNN_' + main_metric_type + ': ' + str(
                    np.round(lb_metric[lb]['pretrainedDNN'][main_metric_type], 5)) \
                    #              + ' D3QN_max_' + main_metric_type + ': ' + str(
                # np.round(lb_metric[lb]['D3QN_max'][main_metric_type], 5)) \
                #              + ' D3QN_stable_' + main_metric_type + ': ' + str(
                # np.round(lb_metric[lb]['D3QN_stable'][main_metric_type], 5))
                print(lb_report_str2)

                # print(max_main_metric_state)
                print(str(int(np.sum(cur_state))) + '/' + str(get_global_dict_value('dataset_conf')['fea_num']),
                      cur_state)
            print('pid', os.getpid(), 'seed', get_global_dict_value('method_conf')['seed'],
                  'cuda:' + str(get_global_dict_value('method_conf')['gpu_id']),
                  '\'' + get_global_dict_value('dataset_conf')['dataset_name'] + '/' +
                  get_global_dict_value('method_conf')['method_name'] + '/' +
                  get_global_dict_value('log_conf')['param_name'] + ' ' + get_host_ip().replace('.', '-') + '\',')
            # -------------------------------------------
            mainlog.record_report(str(datetime.now()).split('.')[0] + ' ' + report_str)
            print(report_str)
            cur_datetime = datetime.now()
            print('process time:', cur_datetime - start_datetime, 'iter duration:',
                  cur_datetime - iter_start_datetime,
                  '\n')
        iter_id += 1
    # last valid and log
    # valid
    while True:
        valid_global_ifdone = 0
        for valid_shared_ifdone in valid_shared_ifdone_list:
            if valid_shared_ifdone.value:
                valid_global_ifdone += 1
            else:
                break
        if valid_global_ifdone == tr_lb_num:
            ################################## save good model ####################################
            tmp_valid_model.load_state_dict(
                torch.load(mainlog.log_path + '/tmp_valid_model.pth', map_location=torch.device('cpu')))
            mainlog.load_envs_info()
            lb_mean_main_metric = np.mean([mainlog.lbs_info[tr_lb][main_metric_type + '_list'][
                                               main_metric_type][
                                               iter_id]
                                           for tr_lb in
                                           get_global_dict_value('dataset_conf')['tr_lb_list']])
            lb_mean_main_metric_list.append(lb_mean_main_metric)
            iter_avg_lb_mean_main_metric = np.mean(
                lb_mean_main_metric_list[-get_global_dict_value('method_conf')['iter_avg']:])
            if iter_avg_lb_mean_main_metric > max_iter_avg_lb_mean_main_metric:
                max_iter_avg_lb_mean_main_metric = iter_avg_lb_mean_main_metric
                max_iter_avg_lb_mean_main_metric_iter = iter_id
                mainlog.save_model(tmp_valid_model)

            if_insert = False
            insert_pos = -1
            if len(top_best_model_list) == 0:
                if_insert = True
                insert_pos = 0
            else:
                for i, model in enumerate(top_best_model_list):
                    if model[1] < lb_mean_main_metric:
                        if_insert = True
                        insert_pos = i + 1
                    if model[1] > lb_mean_main_metric:
                        break
            if if_insert:
                model = [iter_id, lb_mean_main_metric, tmp_valid_model.state_dict()]
                top_best_model_list.insert(insert_pos, model)
                if len(top_best_model_list) > 5:
                    del top_best_model_list[0]
                mainlog.save_model_top(top_best_model_list)
            break
    # log
    mainlog.record_metrics_result()
    max_lb_mean_main_metric = np.mean([mainlog.lbs_info[lb][main_metric_type + '_list'][
                                           'max_' + main_metric_type] for lb in
                                       get_global_dict_value('dataset_conf')['tr_lb_list']])
    report_str = 'iter: ' + str(iter_id) \
                 + ' all_fea_lb_mean_' + main_metric_type + ': ' + str(
        np.round(all_fea_lb_mean_auc, 5)) \
                 + ' max_lb_mean_' + main_metric_type + ': ' + str(
        np.round(max_lb_mean_main_metric, 5)) \
                 + ' max_iter_avg_lb_mean_' + main_metric_type + ': ' + str(
        np.round(max_iter_avg_lb_mean_main_metric, 5)) \
                 + ' max_iter_avg_lb_mean_' + main_metric_type + '_iter: ' + str(
        max_iter_avg_lb_mean_main_metric_iter) \
                 + '\n iter_avg_lb_mean_' + main_metric_type + ': ' + str(
        np.round(iter_avg_lb_mean_main_metric, 5)) \
                 + ' lb_mean_' + main_metric_type + ': ' + str(np.round(lb_mean_main_metric, 5))

    # ------------------------------------------------------------

    for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
        main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
            main_metric_type][iter_id]
        max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
            'max_' + main_metric_type]
        max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
            'max_' + main_metric_type + '_state']
        cur_state = mainlog.lbs_info[lb]['cur_state']
        lb_report_str = 'lb: ' + lb \
                        + ' max_' + main_metric_type + ': ' + str(np.round(max_main_metric, 5)) \
                        + ' ' + main_metric_type + ': ' + str(np.round(main_metric, 5))
        mainlog.record_report(lb_report_str)
        print(lb_report_str)
        lb_report_str2 = '\t all_fea_' + main_metric_type + ': ' + str(
            np.round(lb_metric[lb]['all_fea'][main_metric_type], 5)) \
                         + ' random_' + main_metric_type + ': ' + str(
            np.round(lb_metric[lb]['random'][main_metric_type], 5)) \
                         + ' pretrainedDNN_' + main_metric_type + ': ' + str(
            np.round(lb_metric[lb]['pretrainedDNN'][main_metric_type], 5)) \
            #              + ' D3QN_max_' + main_metric_type + ': ' + str(
        # np.round(lb_metric[lb]['D3QN_max'][main_metric_type], 5)) \
        #              + ' D3QN_stable_' + main_metric_type + ': ' + str(
        # np.round(lb_metric[lb]['D3QN_stable'][main_metric_type], 5))
        print(lb_report_str2)

        # print(max_main_metric_state)
        print(str(int(np.sum(cur_state))) + '/' + str(get_global_dict_value('dataset_conf')['fea_num']), cur_state)
    print('pid', os.getpid(), 'seed', get_global_dict_value('method_conf')['seed'],
          'cuda:' + str(get_global_dict_value('method_conf')['gpu_id']),
          '\'' + get_global_dict_value('dataset_conf')['dataset_name'] + '/' +
          get_global_dict_value('method_conf')['method_name'] + '/' +
          get_global_dict_value('log_conf')['param_name'] + ' ' + get_host_ip().replace('.', '-') + '\',')
    # -------------------------------------------
    mainlog.record_report(str(datetime.now()).split('.')[0] + ' ' + report_str)
    print(report_str)

    mainlog.update_check_and_transfer_to_log_master('trained')
    for p in simulator_processes:
        p.join()
    for p in valid_processes:
        p.join()
    for p in UCT_processes:
        p.join()
