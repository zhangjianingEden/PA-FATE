from util import *
from log.mainlog import *
from .DuelingDQN import *
from .storage import *
from .sharestorage import *
from .subp_test import *
from .valid_subp_test import *
from .UCT_subp_test import *


def main():
    mp.set_start_method("spawn", force=True)
    start_datetime = datetime.now()
    test_iter = get_global_dict_value('method_conf')['test_iter']
    lb_metric = np.load(os.path.join(get_global_dict_value('dataset_conf')['dataset_path'], 'lb_metric.npy'),
                        allow_pickle=True)[()]
    main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
    mainlog = MainLog(mode='test')
    mainlog.record_conf()
    env_num = mainlog.te_lb_num * get_global_dict_value('method_conf')['env_num_per_lb']

    tmp_network = Network()
    tmp_network.load_state_dict(
        torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
    for te_lb_id in range(mainlog.te_lb_num):
        mainlog.save_cur_valid_model_test(mainlog.te_lb_list[te_lb_id], tmp_network)
    cur_datetime = datetime.now()
    print('process time:', cur_datetime - start_datetime)
    mainlog.record_report(str(start_datetime).split('.')[0] + ' ' + str(cur_datetime).split('.')[0])

    valid_processes = []
    valid_shared_ifdone_list = [mp.Value('b', False) for _ in range(mainlog.te_lb_num)]
    for lb_id in range(mainlog.te_lb_num):
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

    iter_id = 0
    if test_iter > 0:
        mainlog.save_epi_info_dict()
        mainlog.save_epi_info_dict(type='init_fea')
        network_list = [Network() for _ in range(mainlog.te_lb_num)]
        target_network_list = [Network() for _ in range(mainlog.te_lb_num)]
        for te_lb_id in range(mainlog.te_lb_num):
            network_list[te_lb_id].load_state_dict(
                torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
            target_network_list[te_lb_id].load_state_dict(
                torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
            network_list[te_lb_id].eval()
            target_network_list[te_lb_id].eval()
            mainlog.save_cur_model_test(mainlog.te_lb_list[te_lb_id], network_list[te_lb_id])
        agent_list = [DuelingDQN(network=network_list[te_lb_id], target_network=target_network_list[te_lb_id]) for
                      te_lb_id in range(mainlog.te_lb_num)]

        rollout = RolloutStorage(get_global_dict_value('method_conf')['capacity'], type='test')
        shared_rollout_list = []
        for env_id in range(env_num):
            shared_rollout_list.append(ShareRolloutStorage())
        shared_ifdone_list = [mp.Value('b', False) for _ in range(env_num)]
        shared_te_lb_id_list = [mp.Value('i', env_id % mainlog.te_lb_num) for env_id in range(env_num)]
        simulator_processes = []
        for env_id in range(env_num):
            p = mp.Process(target=subp_test,
                           args=(env_id,
                                 mainlog.log_path,
                                 shared_rollout_list[env_id],
                                 shared_ifdone_list[env_id],
                                 shared_te_lb_id_list[env_id],
                                 get_global_dict_value('dataset_conf'),
                                 get_global_dict_value('env_conf'),
                                 get_global_dict_value('method_conf'),
                                 get_global_dict_value('log_conf'),
                                 )
                           )
            simulator_processes.append(p)
            p.start()

        UCT_processes = []
        UCT_shared_ifdone_list = [mp.Value('b', False) for _ in range(mainlog.te_lb_num)]
        for lb_id in range(mainlog.te_lb_num):
            p = mp.Process(target=UCT_subp_test,
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

        while iter_id < test_iter:
            iter_start_datetime = datetime.now()
            # load UCT result
            while True:
                UCT_global_ifdone = 0
                for UCT_shared_ifdone in UCT_shared_ifdone_list:
                    if UCT_shared_ifdone.value:
                        UCT_global_ifdone += 1
                    else:
                        break
                if UCT_global_ifdone == mainlog.te_lb_num:
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
                        lb = mainlog.te_lb_list[shared_te_lb_id_list[env_id].value]
                        new_shared_buffer = mainlog.load_shared_buffer(env_id)
                        rollout.load_shared_buffer(lb, new_shared_buffer)
                        new_shared_buffer_init_fea = mainlog.load_shared_buffer(env_id, type='init_fea')
                        rollout.load_shared_buffer(lb, new_shared_buffer_init_fea, type='init_fea')
                        mainlog.load_epi_info(env_id, lb)
                        mainlog.load_epi_info(env_id, lb, type='init_fea')
                    # update params
                    for lb_id in range(mainlog.te_lb_num):
                        network_list[lb_id].train()
                        target_network_list[lb_id].train()
                    loss_per_sample = 0
                    update_time = 0
                    for env_id in range(env_num):
                        lb = mainlog.te_lb_list[shared_te_lb_id_list[env_id].value]
                        if rollout.is_enough(lb):
                            loss_per_sample += agent_list[shared_te_lb_id_list[env_id].value].update(rollout, lb,
                                                                                                     iter_id,
                                                                                                     type='test')
                            update_time += 1
                        if rollout.is_enough(lb, type='init_fea'):
                            loss_per_sample += agent_list[shared_te_lb_id_list[env_id].value].update(rollout, lb,
                                                                                                     iter_id,
                                                                                                     type='test',
                                                                                                     type2='init_fea')
                            update_time += 1
                    if update_time > 0:
                        loss_per_sample /= update_time
                    mainlog.record_loss(loss_per_sample)
                    if (iter_id + 1) % get_global_dict_value('method_conf')['tar_net_update_iter_gap'] == 0:
                        for agent in agent_list:
                            agent.update_tar_net()
                    for lb_id in range(mainlog.te_lb_num):
                        network_list[lb_id].eval()
                        target_network_list[lb_id].eval()
                    for lb_id in range(mainlog.te_lb_num):
                        mainlog.save_cur_model_test(mainlog.te_lb_list[lb_id], network_list[lb_id])
                    break
            mainlog.save_epi_info_dict()
            mainlog.reset_epi_info_dict()
            mainlog.save_epi_info_dict(type='init_fea')
            mainlog.reset_epi_info_dict(type='init_fea')

            for UCT_shared_ifdone in UCT_shared_ifdone_list:
                UCT_shared_ifdone.value = False
            for shared_ifdone in shared_ifdone_list:
                shared_ifdone.value = False
            # if need valid and log
            if (iter_id + 1) % get_global_dict_value('method_conf')['cal_log_frequency'] == 0:
                # valid
                while True:
                    valid_global_ifdone = 0
                    for valid_shared_ifdone in valid_shared_ifdone_list:
                        if valid_shared_ifdone.value:
                            valid_global_ifdone += 1
                        else:
                            break
                    if valid_global_ifdone == mainlog.te_lb_num:
                        mainlog.load_envs_info()
                        for lb_id in range(mainlog.te_lb_num):
                            mainlog.save_cur_valid_model_test(mainlog.te_lb_list[lb_id], network_list[lb_id])
                        for valid_shared_ifdone in valid_shared_ifdone_list:
                            valid_shared_ifdone.value = False
                        break
                # log
                mainlog.record_metrics_result()
                max_tr_lb_mean_main_metric = np.mean([mainlog.lbs_info[tr_lb][main_metric_type + '_list'][
                                                          'max_' + main_metric_type]
                                                      for tr_lb in mainlog.tr_lb_list])
                max_te_lb_mean_main_metric = np.mean([mainlog.lbs_info[te_lb][main_metric_type + '_list'][
                                                          'max_' + main_metric_type]
                                                      for te_lb in mainlog.te_lb_list])
                report_str = 'iter: ' + str(iter_id + 1 - get_global_dict_value('method_conf')['cal_log_frequency']) \
                             + ' max_tr_lb_mean_' + main_metric_type + ': ' + str(
                    np.round(max_tr_lb_mean_main_metric, 5)) \
                             + ' max_te_lb_mean_' + main_metric_type + ': ' + str(
                    np.round(max_te_lb_mean_main_metric, 5))
                # ------------------------------------------------------------
                print('########### TR LB ###########')
                for lb in mainlog.tr_lb_list:
                    max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                        'max_' + main_metric_type]
                    max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
                        'max_' + main_metric_type + '_state']
                    lb_report_str = 'lb: ' + lb \
                                    + ' max_' + main_metric_type + ': ' + str(np.round(max_main_metric, 5))
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
                    print(str(int(np.sum(max_main_metric_state))) + '/' + str(
                        get_global_dict_value('dataset_conf')['fea_num']),
                          max_main_metric_state)
                print('########### TE LB ###########')
                for lb in mainlog.te_lb_list:
                    main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
                        main_metric_type][iter_id + 1 - get_global_dict_value('method_conf')['cal_log_frequency']]
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
                    print(str(int(np.sum(max_main_metric_state))) + '/' + str(
                        get_global_dict_value('dataset_conf')['fea_num']),
                          max_main_metric_state)
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
            iter_id += 1
        for p in simulator_processes:
            p.join()
        for p in UCT_processes:
            p.join()
    # last valid and log
    # valid
    while True:
        valid_global_ifdone = 0
        for valid_shared_ifdone in valid_shared_ifdone_list:
            if valid_shared_ifdone.value:
                valid_global_ifdone += 1
            else:
                break
        if valid_global_ifdone == mainlog.te_lb_num:
            mainlog.load_envs_info()
            break
    # log
    mainlog.record_metrics_result()
    max_tr_lb_mean_main_metric = np.mean([mainlog.lbs_info[tr_lb][main_metric_type + '_list'][
                                              'max_' + main_metric_type]
                                          for tr_lb in mainlog.tr_lb_list])
    max_te_lb_mean_main_metric = np.mean([mainlog.lbs_info[te_lb][main_metric_type + '_list'][
                                              'max_' + main_metric_type]
                                          for te_lb in mainlog.te_lb_list])
    report_str = 'iter: ' + str(iter_id) \
                 + ' max_tr_lb_mean_' + main_metric_type + ': ' + str(
        np.round(max_tr_lb_mean_main_metric, 5)) \
                 + ' max_te_lb_mean_' + main_metric_type + ': ' + str(
        np.round(max_te_lb_mean_main_metric, 5))
    # ------------------------------------------------------------
    print('########### TR LB ###########')
    for lb in mainlog.tr_lb_list:
        max_main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
            'max_' + main_metric_type]
        max_main_metric_state = mainlog.lbs_info[lb][main_metric_type + '_list'][
            'max_' + main_metric_type + '_state']
        lb_report_str = 'lb: ' + lb \
                        + ' max_' + main_metric_type + ': ' + str(np.round(max_main_metric, 5))
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
        print(str(int(np.sum(max_main_metric_state))) + '/' + str(get_global_dict_value('dataset_conf')['fea_num']),
              max_main_metric_state)
    print('########### TE LB ###########')
    for lb in mainlog.te_lb_list:
        main_metric = mainlog.lbs_info[lb][main_metric_type + '_list'][
            main_metric_type][iter_id]
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
        print(str(int(np.sum(max_main_metric_state))) + '/' + str(get_global_dict_value('dataset_conf')['fea_num']),
              max_main_metric_state)
    print('pid', os.getpid(), 'seed', get_global_dict_value('method_conf')['seed'],
          'cuda:' + str(get_global_dict_value('method_conf')['gpu_id']),
          '\'' + mainlog.log_path + '\',')
    # -------------------------------------------
    mainlog.record_report(str(datetime.now()).split('.')[0] + ' ' + report_str)
    print(report_str)

    mainlog.update_check_and_transfer_to_log_master('tested')
    for p in valid_processes:
        p.join()
