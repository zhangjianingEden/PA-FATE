from util import *


class Dataset:
    def __init__(self, dataset_name, dataset_path, fea_list, lb_list):
        super(Dataset, self).__init__()
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.fea_list = fea_list
        self.lb_list = lb_list
        # self.dataset_class_list = ['train_set', 'valid_set', 'test_set']
        self.dataset_class_list = ['train_set', 'valid_set']
        # self.lb_class_list = ['tr_lb', 'te_lb']
        self.ds_class2x_dict = {}
        self.lb2y_dict = {}
        self.gen_ds_class2x_dict()
        self.gen_lb2y_dict()

    def x_abstract(self, dataset_class):
        dataset_path = os.path.join(self.dataset_path, 'data_split', dataset_class + '.csv')
        dataset = pd.read_csv(dataset_path)
        x = dataset[self.fea_list].values
        return x

    def y_abstract(self, lb, dataset_class):
        dataset_path = os.path.join(self.dataset_path, 'data_split', dataset_class + '.csv')
        dataset = pd.read_csv(dataset_path)
        y = dataset[lb].values
        return y

    def gen_ds_class2x_dict(self):
        for dataset_class in self.dataset_class_list:
            if dataset_class not in self.ds_class2x_dict:
                self.ds_class2x_dict[dataset_class] = self.x_abstract(dataset_class)

    def gen_lb2y_dict(self):
        for lb in self.lb_list:
            if lb not in self.lb2y_dict:
                self.lb2y_dict[lb] = {}
            for dataset_class in self.dataset_class_list:
                if dataset_class not in self.lb2y_dict[lb]:
                    self.lb2y_dict[lb][dataset_class] = {}
                self.lb2y_dict[lb][dataset_class]['y'] = self.y_abstract(lb, dataset_class)
