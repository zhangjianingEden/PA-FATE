from util import *
from log.mainlog import *
from .DuelingDQN import *
from .storage import *
from .sharestorage import *
from .subp_test import *
from .valid_subp_test import *


def main():
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    lb_metric = np.load(os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_metric.npy'),
                        allow_pickle=True)[()]
    mainlog = MainLog(mode='test')
    tr_lb_num = len(get_global_dict_value('dataset_conf')['tr_lb_list'])
    te_lb_num = len(get_global_dict_value('dataset_conf')['te_lb_list'])
    lb_list = get_global_dict_value('dataset_conf')['tr_lb_list'] + get_global_dict_value('dataset_conf')['te_lb_list']
    lb_num = tr_lb_num + te_lb_num
    network_list = [Network() for _ in range(lb_num)]
    target_network_list = [Network() for _ in range(lb_num)]
    for lb_id in range(lb_num):
        network_list[lb_id].eval()
        target_network_list[lb_id].eval()
        network_list[lb_id].load_state_dict(
            torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
        target_network_list[lb_id].load_state_dict(
            torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
        mainlog.save_cur_model_test(lb_list[lb_id], network_list[lb_id])
        mainlog.save_cur_valid_model_test(lb_list[lb_id], network_list[lb_id])
    agent_list = [DuelingDQN(network=network_list[lb_id], target_network=target_network_list[lb_id]) for lb_id in
                  range(lb_num)]

    rollout = RolloutStorage(get_global_dict_value('method_conf')['capacity'], type='test')
    shared_rollout_list = []

    env_num = lb_num * get_global_dict_value('method_conf')['env_num_per_lb']
    for env_id in range(env_num):
        shared_rollout_list.append(ShareRolloutStorage())
    cur_datetime = datetime.now()
    print('process time:', cur_datetime - start_datetime)
    mainlog.record_report(str(start_datetime).split('.')[0] + ' ' + str(cur_datetime).split('.')[0])

    shared_ifdone_list = [mp.Value('b', False) for _ in range(env_num)]

    shared_lb_id_list = [mp.Value('i', env_id % lb_num) for env_id in range(env_num)]

    simulator_processes = []
    for env_id in range(env_num):
        p = mp.Process(target=subp_test,
                       args=(env_id,
                             mainlog.log_path,
                             shared_rollout_list[env_id],
                             shared_ifdone_list[env_id],
                             shared_lb_id_list[env_id],
                             get_global_dict_value('dataset_conf'),
                             get_global_dict_value('env_conf'),
                             get_global_dict_value('method_conf'),
                             get_global_dict_value('log_conf'),
                             )
                       )
        simulator_processes.append(p)
        p.start()

    valid_processes = []
    valid_shared_ifdone_list = [mp.Value('b', False) for _ in range(lb_num)]
    for lb_id in range(lb_num):
        p = mp.Process(target=valid_subp_test,
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

    main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
    tr_lb_mean_max_main_metric = 0
    te_lb_mean_max_main_metric = 0
    max_tr_lb_mean_max_main_metric = 0
    max_te_lb_mean_max_main_metric = 0
    max_tr_lb_mean_max_main_metric_iter = -1
    max_te_lb_mean_max_main_metric_iter = -1

    for iter_id in range(get_global_dict_value('method_conf')['test_iter'] + 1):
        iter_start_datetime = datetime.now()
        ################################## gen samples ####################################
        while True:
            global_ifdone = 0
            for shared_ifdone in shared_ifdone_list:
                if shared_ifdone.value:
                    global_ifdone += 1
                else:
                    break

            if global_ifdone == env_num:
                # valid
                if iter_id % get_global_dict_value('method_conf')['cal_log_frequency'] == 0 and iter_id > 0:
                    # valid
                    while True:
                        valid_global_ifdone = 0
                        for valid_shared_ifdone in valid_shared_ifdone_list:
                            if valid_shared_ifdone.value:
                                valid_global_ifdone += 1
                            else:
                                break
                        if valid_global_ifdone == lb_num:
                            mainlog.load_envs_info()
                            tr_lb_mean_max_main_metric = np.mean([mainlog.lbs_info[tr_lb][main_metric_type + '_list'][
                                                                      'max_' + main_metric_type]
                                                                  for tr_lb in
                                                                  get_global_dict_value('dataset_conf')['tr_lb_list']])
                            te_lb_mean_max_main_metric = np.mean([mainlog.lbs_info[te_lb][main_metric_type + '_list'][
                                                                      'max_' + main_metric_type]
                                                                  for te_lb in
                                                                  get_global_dict_value('dataset_conf')['te_lb_list']])
                            if tr_lb_mean_max_main_metric > max_tr_lb_mean_max_main_metric:
                                max_tr_lb_mean_max_main_metric = tr_lb_mean_max_main_metric
                                max_tr_lb_mean_max_main_metric_iter = iter_id - get_global_dict_value('method_conf')[
                                    'cal_log_frequency']
                            if te_lb_mean_max_main_metric > max_te_lb_mean_max_main_metric:
                                max_te_lb_mean_max_main_metric = te_lb_mean_max_main_metric
                                max_te_lb_mean_max_main_metric_iter = iter_id - get_global_dict_value('method_conf')[
                                    'cal_log_frequency']

                            for lb_id in range(lb_num):
                                mainlog.save_cur_valid_model_test(lb_list[lb_id], network_list[lb_id])
                            for valid_shared_ifdone in valid_shared_ifdone_list:
                                valid_shared_ifdone.value = False
                            break

                # update
                for env_id in range(env_num):
                    new_shared_buffer = mainlog.load_shared_buffer(env_id)
                    lb = lb_list[shared_lb_id_list[env_id].value]
                    rollout.load_shared_buffer(lb, new_shared_buffer)

                ################################## update params ####################################
                for lb_id in range(lb_num):
                    network_list[lb_id].train()
                    target_network_list[lb_id].train()
                loss_per_sample = 0
                update_time = 0
                for env_id in range(env_num):
                    lb = lb_list[shared_lb_id_list[env_id].value]
                    if rollout.is_enough(lb):
                        loss_per_sample += agent_list[shared_lb_id_list[env_id].value].update(rollout, lb, iter_id,
                                                                                              type='test')
                        update_time += 1
                if update_time > 0:
                    loss_per_sample /= update_time
                if iter_id % get_global_dict_value('method_conf')['tar_net_update_iter_gap'] == 0:
                    for agent in agent_list:
                        agent.update_tar_net()
                for lb_id in range(lb_num):
                    network_list[lb_id].eval()
                    target_network_list[lb_id].eval()

                ################################## mainlog work ####################################
                mainlog.record_loss(loss_per_sample)
                for lb_id in range(lb_num):
                    mainlog.save_cur_model_test(lb_list[lb_id], network_list[lb_id])

                if iter_id % get_global_dict_value('method_conf')['cal_log_frequency'] == 0 and iter_id > 0:
                    mainlog.record_metrics_result()
                    report_str = 'iter: ' + str(iter_id - get_global_dict_value('method_conf')['cal_log_frequency']) \
                                 + ' max_tr_lb_mean_max_' + main_metric_type + ': ' + str(
                        np.round(max_tr_lb_mean_max_main_metric, 5)) \
                                 + ' max_tr_lb_mean_max_' + main_metric_type + '_iter: ' + str(
                        max_tr_lb_mean_max_main_metric_iter) \
                                 + ' max_te_lb_mean_max_' + main_metric_type + ': ' + str(
                        np.round(max_te_lb_mean_max_main_metric, 5)) \
                                 + ' max_te_lb_mean_max_' + main_metric_type + '_iter: ' + str(
                        max_te_lb_mean_max_main_metric_iter)
                    # ------------------------------------------------------------

                    print('########### TR LB ###########')
                    for lb in get_global_dict_value('dataset_conf')['tr_lb_list']:
                        main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            main_metric_type][iter_id - get_global_dict_value('method_conf')['cal_log_frequency']]
                        max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            'max_' + main_metric_type]
                        max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            'max_' + main_metric_type + '_state']
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
                            # + ' D3QN_max_' + main_metric_type + ': ' + str(
                        # np.round(lb_metric[lb]['D3QN_max'][main_metric_type], 5)) \
                        #              + ' D3QN_stable_' + main_metric_type + ': ' + str(
                        # np.round(lb_metric[lb]['D3QN_stable'][main_metric_type], 5))
                        print(lb_report_str2)
                        print(np.sum(max_main_metric_state), max_main_metric_state)
                    print('########### TE LB ###########')
                    for lb in get_global_dict_value('dataset_conf')['te_lb_list']:
                        main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            main_metric_type][iter_id - get_global_dict_value('method_conf')['cal_log_frequency']]
                        max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            'max_' + main_metric_type]
                        max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
                            'max_' + main_metric_type + '_state']
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
                            # + ' D3QN_max_' + main_metric_type + ': ' + str(
                        # np.round(lb_metric[lb]['D3QN_max'][main_metric_type], 5)) \
                        #              + ' D3QN_stable_' + main_metric_type + ': ' + str(
                        # np.round(lb_metric[lb]['D3QN_stable'][main_metric_type], 5))
                        print(lb_report_str2)
                        print(np.sum(max_main_metric_state), max_main_metric_state)

                    print('pid', os.getpid(), 'seed', get_global_dict_value('method_conf')['seed'],
                          'cuda:' + str(get_global_dict_value('method_conf')['gpu_id']),
                          '\'' + mainlog.log_path + '\',')
                    # -------------------------------------------
                    mainlog.record_report(str(datetime.now()).split('.')[0] + ' ' + report_str)
                    print(report_str)
                    cur_datetime = datetime.now()
                    print('process time:', cur_datetime - start_datetime, 'iter duration:',
                          cur_datetime - iter_start_datetime,
                          '\n')
                for shared_ifdone in shared_ifdone_list:
                    shared_ifdone.value = False
                break
    mainlog.update_check_and_transfer_to_log_master('tested')
    for p in simulator_processes:
        p.join()
    for p in valid_processes:
        p.join()
