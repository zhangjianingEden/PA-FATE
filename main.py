from util import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', required=True, type=str, help='the name of dataset')
    parser.add_argument('--max_fea_ratio', required=True, type=float, help='the maximum ratio of selectable features')
    parser.add_argument('--mode', required=True, type=str, help='train or test')
    parser.add_argument('--further_train_iter', default=0, type=int,
                        help='the No. of iteration for further train on unseen tasks')
    args = parser.parse_args()

    dataset_conf = importlib.import_module('datasets.' + args.dataset_name + '.conf_temp').DATASET_CONF
    env_conf = importlib.import_module('env.conf_temp').ENV_CONF
    method_conf = importlib.import_module('method.conf_temp').METHOD_CONF
    log_conf = importlib.import_module('log.conf_temp').LOG_CONF

    dataset_conf['dataset_path'] = '/' + os.path.join(*__file__.split('/')[:-1], 'datasets', args.dataset_name)

    global_dict_init()
    set_global_dict_value('dataset_conf', gen_conf(args, dataset_conf))
    set_global_dict_value('env_conf', gen_conf(args, env_conf))
    set_global_dict_value('method_conf', gen_conf(args, method_conf))
    set_global_dict_value('log_conf', gen_conf(args, log_conf))

    fix_random_seed(get_global_dict_value('method_conf')['seed'])
    if args.mode == 'train':
        main = importlib.import_module('method.main')
        main.main()
    elif args.mode == 'test':
        main_test = importlib.import_module('method.main_test')
        main_test.main()
