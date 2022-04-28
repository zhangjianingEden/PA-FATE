from util import *


def gen_reremap_fea_list(remap_fea_list):
    reremap_fea_list = []
    for i in range(len(remap_fea_list)):
        for j in range(len(remap_fea_list)):
            if remap_fea_list[j] == i:
                reremap_fea_list.append(j)
                break
    return reremap_fea_list


fea_list = [
    '3', '20', '21', '26', '27', '31', '33', '49', '57', '58', '91', '107', '114', '118', '132', '133',
    '136', '139', '142', '143', '151', '152', '166', '170', '176', '180', '182', '184', '190', '192',
    '194', '196', '202', '204', '211', '212', '216', '217', '221', '222', '229', '231', '233', '239',
    '246', '296', '298', '301', '302', '309', '313', '314', '320', '324', '327', '329', '330', '332',
    '336', '345', '348', '353', '356', '358', '360', '362', '370', '374', '383', '385', '403', '407',
    '421', '253', '426', '430', '431', '434', '436', '455', '483', '494', '522', '776', '840', '1375',
    '1494', '2458', '2519', '2817', '2821', '2829', '2842', '676', '1130', '1732', '2138', '2877', '2889',
    '2891', '1065', '2950', '1236', '3179', '3254', '3453', '2790', '3038', '526', '529', '637', '698',
    '1717', '2200', '2253', '2469', '3156', '3485', '1366', '2461', '1634', '1795', '3305', '3954', '1335',
    '1902', '1963', '2927', '3010', '3851', '267', '718', '1209', '1415', '1700', '2092', '2355', '2357',
    '2618', '2649', '3292', '4354', '1537', '4741', '1296', '288', '1024', '3104', '836', '2105', '2699',
    '1516', '1803', '2462', '5046', '3124', '680', '3098', '3719'
]

lb_list = [
    '34096', '34099', '34100', '34101', '34103', '34104', '34105', '34106', '34107', '34108',
    '34109', '34110', '34113', '34116', '34119', '34121', '34123', '34127'
]

tr_lb_list = [
    '34096', '34101', '34105', '34106', '34107', '34108', '34109'
]
te_lb_list = [
    '34099', '34100', '34103', '34104', '34110', '34113', '34116', '34119', '34123', '34127', '34121'
]
remap_fea_list = [18, 89, 57, 110, 146, 61, 58, 102, 90, 76, 156, 72, 52, 116, 87, 117, 126, 64, 41, 78, 75, 34, 36, 8,
                  24, 28, 84, 80, 42, 2, 101, 56, 16, 93, 140, 10, 69, 107, 47, 22, 30, 108, 95, 103, 40, 145, 158, 157,
                  155, 154, 153, 152, 151, 150, 149, 148, 147, 144, 143, 142, 141, 139, 138, 137, 136, 135, 134, 133,
                  132, 131, 130, 129, 128, 127, 125, 124, 123, 122, 121, 120, 119, 118, 115, 114, 113, 112, 111, 109,
                  106, 105, 104, 100, 99, 98, 97, 96, 94, 92, 91, 88, 86, 85, 83, 82, 81, 79, 77, 74, 73, 71, 70, 68,
                  67, 66, 65, 63, 62, 60, 59, 55, 54, 53, 51, 50, 49, 48, 46, 45, 44, 43, 39, 38, 37, 35, 33, 32, 31,
                  29, 27, 26, 25, 23, 21, 20, 19, 17, 15, 14, 13, 12, 11, 9, 7, 6, 5, 4, 3, 1, 0]

reremap_fea_list = gen_reremap_fea_list(remap_fea_list)

DATASET_CONF = {
    'dataset_name': __file__.split('/')[-2],
    'dataset_path': '',
    'metric_type_list': ['auc', 'f1_score'],
    'fea_list': fea_list,
    'fea_num': len(fea_list),
    'lb_list': lb_list,
    'tr_lb_list': tr_lb_list,
    'te_lb_list': te_lb_list,
    'use_remap': True,
    'remap_fea_list': remap_fea_list,
    'reremap_fea_list': reremap_fea_list,
    'PARAM_PRE_TRAIN': {
        'SVM': {
            '34096': {
                'C': 3,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34099': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34100': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34101': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34103': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34104': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34105': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34106': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34107': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34108': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34109': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34110': {
                'C': 13,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
            '34113': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34116': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34119': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34121': {
                'C': 11,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34123': {
                'C': 9,
                'kernel': 'rbf',
                'gamma': 'auto'
            },
            '34127': {
                'C': 1,
                'kernel': 'rbf',
                'gamma': 'scale'
            },
        }

    }
}