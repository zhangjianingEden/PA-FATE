from util import *
from log.mainlog import *
from .DuelingDQN import *
from .storage import *
from .sharestorage import *
from .subp import *
from .valid_subp import *
from .E_Tree_subp import *


def main():
    mp.set_start_method("spawn", force=True)
    main_metric_type = get_global_dict_value('env_conf')['rwd']['metric']
    mainlog = MainLog()
    mainlog.save_epi_info_dict()
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
    shared_ifdone_list = [mp.Value('b', False) for _ in range(env_num)]
    shared_tr_lb_id_list = [mp.Value('i', env_id % tr_lb_num) for env_id in range(env_num)]
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

    E_Tree_processes = []
    E_Tree_shared_ifdone_list = [mp.Value('b', False) for _ in range(tr_lb_num)]
    for lb_id in range(tr_lb_num):
        p = mp.Process(target=E_Tree_subp,
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

    lb_mean_main_metric_list = []
    max_iter_avg_lb_mean_main_metric = 0

    tmp_valid_model = Network()
    iter_id = 0
    while iter_id < get_global_dict_value('method_conf')['train_iter']:
        # load E_Tree result
        while True:
            E_Tree_global_ifdone = 0
            for E_Tree_shared_ifdone in E_Tree_shared_ifdone_list:
                if E_Tree_shared_ifdone.value:
                    E_Tree_global_ifdone += 1
                else:
                    break
            if E_Tree_global_ifdone == tr_lb_num:
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
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][shared_tr_lb_id_list[env_id].value]
                    new_shared_buffer = mainlog.load_shared_buffer(env_id)
                    rollout.load_shared_buffer(lb, new_shared_buffer)
                    mainlog.load_epi_info(env_id, lb)
                # update params
                network.train()
                target_network.train()
                loss_per_sample = 0
                update_time = 0
                for env_id in range(env_num):
                    lb = get_global_dict_value('dataset_conf')['tr_lb_list'][shared_tr_lb_id_list[env_id].value]
                    if rollout.is_enough(lb):
                        loss_per_sample += agent.update(rollout, lb)
                        update_time += 1
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
        if iter_id < get_global_dict_value('method_conf')['train_iter'] - 50:
            mainlog.reset_epi_info_dict()

        for E_Tree_shared_ifdone in E_Tree_shared_ifdone_list:
            E_Tree_shared_ifdone.value = False

        if (iter_id + 1) % get_global_dict_value('method_conf')['ITS']['fq'] == 0 and iter_id > 0:
            p_si_array = 0
            tr_lb_uc_list = []
            for tr_lb_id in range(tr_lb_num):
                lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                tr_lb_uc_list.append(mainlog.lb_E_Tree_result[lb]['ITS']['uc'])
            p_uc_array = np.array(gen_p_list(tr_lb_uc_list))
            p_si_array += p_uc_array * get_global_dict_value('method_conf')['ITS']['uc']
            tr_lb_dis_list = []
            for tr_lb_id in range(tr_lb_num):
                lb = get_global_dict_value('dataset_conf')['tr_lb_list'][tr_lb_id]
                tr_lb_dis_list.append(mainlog.lb_E_Tree_result[lb]['ITS']['dis'])
            p_dis_array = np.array(gen_p_list(tr_lb_dis_list))
            p_si_array += p_dis_array * get_global_dict_value('method_conf')['ITS']['dis']
            p_list = gen_p_list(list(p_si_array))
            new_shared_tr_lb_id_array = np.random.choice(tr_lb_num, env_num, p=p_list)
            for env_id in range(env_num):
                shared_tr_lb_id_list[env_id].value = new_shared_tr_lb_id_array[env_id]

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
                        mainlog.save_model(tmp_valid_model)
                    break
            # log
            mainlog.record_metrics_result()
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
                mainlog.save_model(tmp_valid_model)
            break
    # log
    mainlog.record_metrics_result()
    for p in simulator_processes:
        p.join()
    for p in valid_processes:
        p.join()
    for p in E_Tree_processes:
        p.join()
