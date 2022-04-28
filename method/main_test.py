from util import *
from log.mainlog import *
from .DuelingDQN import *
from .storage import *
from .sharestorage import *
from .subp_test import *
from .valid_subp_test import *
from .E_Tree_subp_test import *


def main():
    mp.set_start_method("spawn", force=True)
    further_train_iter = get_global_dict_value('method_conf')['further_train_iter']
    main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
    mainlog = MainLog(mode='test')
    env_num = mainlog.te_lb_num * get_global_dict_value('method_conf')['env_num_per_lb']

    tmp_network = Network()
    tmp_network.load_state_dict(
        torch.load(mainlog.log_root_path + '/model.pth', map_location=torch.device('cpu')))
    for te_lb_id in range(mainlog.te_lb_num):
        mainlog.save_cur_valid_model_test(mainlog.te_lb_list[te_lb_id], tmp_network)
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
    if further_train_iter > 0:
        mainlog.save_epi_info_dict()
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

        E_Tree_processes = []
        E_Tree_shared_ifdone_list = [mp.Value('b', False) for _ in range(mainlog.te_lb_num)]
        for lb_id in range(mainlog.te_lb_num):
            p = mp.Process(target=E_Tree_subp_test,
                           args=(lb_id,
                                 mainlog.log_path,
                                 E_Tree_shared_ifdone_list[lb_id],
                                 get_global_dict_value('dataset_conf'),
                                 get_global_dict_value('env_conf'),
                                 get_global_dict_value('method_conf'),
                                 get_global_dict_value('log_conf'),
                                 )
                           )
            E_Tree_processes.append(p)
            p.start()

        while iter_id < further_train_iter:
            while True:
                E_Tree_global_ifdone = 0
                for E_Tree_shared_ifdone in E_Tree_shared_ifdone_list:
                    if E_Tree_shared_ifdone.value:
                        E_Tree_global_ifdone += 1
                    else:
                        break
                if E_Tree_global_ifdone == mainlog.te_lb_num:
                    mainlog.load_lb_E_Tree_result()
                    mainlog.save_lb_E_Tree_result()
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
                        mainlog.load_epi_info(env_id, lb)
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
                                                                                                     type='test')
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

            for E_Tree_shared_ifdone in E_Tree_shared_ifdone_list:
                E_Tree_shared_ifdone.value = False
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
            iter_id += 1
        for p in simulator_processes:
            p.join()
        for p in E_Tree_processes:
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
    max_te_lb_mean_side_metric = np.mean([mainlog.lbs_info[te_lb]['f1_score' + '_list'][
                                              'max_' + 'f1_score']
                                          for te_lb in mainlog.te_lb_list])
    report_str = 'Unseen Tasks Avg ' + 'F1-score' + ': ' + str(np.round(max_te_lb_mean_side_metric, 5))
    print(report_str)
    max_te_lb_mean_main_metric = np.mean([mainlog.lbs_info[te_lb][main_metric_type + '_list'][
                                              'max_' + main_metric_type]
                                          for te_lb in mainlog.te_lb_list])
    report_str = 'Unseen Tasks Avg ' + main_metric_type + ': ' + str(np.round(max_te_lb_mean_main_metric, 5))
    print(report_str)
    for p in valid_processes:
        p.join()
