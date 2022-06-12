import os 
from sklearn import metrics
import common as com
import yaml
import glob
import numpy as np
import csv

with open("param.yaml") as stream:
    param = yaml.safe_load(stream)

mode = True

def read_csv(save_file_path):
    with open(save_file_path, "r", newline="") as f:
        reader = csv.reader(f, delimiter=',')
        return (list(reader))


def test_file_list_generator(target_dir,
                             id_name,
                             dir_name="test",
                             prefix_normal="normal",
                             prefix_anomaly="anomaly",
                             ext="wav"):

    com.logger.info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n========================================")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        com.logger.info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")
        print("\n=========================================")

    return files, labels


def load_score_csv(machine_class, machine_id):
    anomaly_score_csv = "{0}/anomaly_score_{1}_{2}.csv".format(param["result_root"], machine_class, machine_id)
    anomaly_score_list = read_csv(anomaly_score_csv)
    normal_list = []
    abnormal_list = []
    for row in anomaly_score_list:
        if 'anomaly' in row[0]:
            abnormal_list.append(row)
        else:
            normal_list.append(row)
    anomaly_score_list = sorted(normal_list, key=lambda x: x[0]) + sorted(abnormal_list, key=lambda x: x[0])

    #anomaly_score_list = sorted(anomaly_score_list, key=lambda x: x[0])
    test_files = []
    y_pred = []
    for row in anomaly_score_list:
        test_files.append(row[0])
        y_pred.append(float(row[1]))

    return test_files, y_pred
    

def evaluate(machine_class, machine_id):
    test_files, y_true = test_file_list_generator(os.path.join(param['data_root'], machine_class), machine_id, dir_name='test')
    test_files2, y_pred = load_score_csv(machine_class, machine_id)

    assert len(test_files) == len(test_files2)
    for i in range(len(test_files)):
        #print (test_files[i], test_files2[i])
        assert os.path.basename(test_files[i]) == test_files2[i]
        
    auc = metrics.roc_auc_score(y_true, y_pred)

    p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
    com.logger.info("AUC : {}".format(auc))
    com.logger.info("pAUC : {}".format(p_auc))

    
if __name__ == '__main__':

    machine_class = 'fan'
    machine_id_list = ['id_00', 'id_02', 'id_04', 'id_06']
    
    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)

    machine_class = 'pump'
    machine_id_list = ['id_00', 'id_02', 'id_04', 'id_06']
    
    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)

    machine_class = 'slider'
    machine_id_list = ['id_00', 'id_02', 'id_04', 'id_06']
    
    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)

    machine_class = 'valve'
    machine_id_list = ['id_00', 'id_02', 'id_04', 'id_06']
    
    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)
        
    machine_class = 'ToyCar'
    machine_id_list = ['id_01', 'id_02', 'id_03', 'id_04']

    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)

    machine_class = 'ToyConveyor'
    machine_id_list = ['id_01', 'id_02', 'id_03']

    for machine_id in machine_id_list:
        evaluate(machine_class, machine_id)

    
"""
auc = metrics.roc_auc_score(y_true, y_pred)
p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=param["max_fpr"])
csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
performance.append([auc, p_auc])
com.logger.info("AUC : {}".format(auc))
com.logger.info("pAUC : {}".format(p_auc))
"""
