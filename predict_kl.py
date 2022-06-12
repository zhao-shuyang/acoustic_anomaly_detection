import sys
import csv
import yaml
import h5py
import numpy as np

n_mfccs = 40

with open("param.yaml") as stream:
    param = yaml.safe_load(stream)


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def gau_kl(pm, pv, qm, qv):

    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
        
    #Diagonal
    if len(pv .shape) == 1:

        # Determinants of diagonal covariances pv, qv
        dpv = pv.prod() + sys.float_info.min
        dqv = qv.prod(axis) + sys.float_info.min
    
        # Inverse of diagonal covariance qv
        iqv = 1./qv
        # Difference between means pm, qm
        diff = qm - pm

        return  (0.5 *
            (np.log(dqv / dpv)            # log |\Sigma_q| / |\Sigma_p|
             + np.sum(iqv * pv)          # + tr(\Sigma_q^{-1} * \Sigma_p)
             + np.sum(diff * iqv * diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm)))                     # - N

    #Full
    elif len(pv .shape) == 2:
        
        # Determinants of diagonal covariances pv, qv
        dpv = np.linalg.det(pv)
        dqv = np.linalg.det(qv)
        # Inverse of diagonal covariance qv
        ipv = np.linalg.inv(pv)
        iqv = np.linalg.inv(qv)
        #print dpv, dqv, np.log(dqv / dpv) 

        # Difference between means pm, qm
        diff = qm - pm
        eps = 1e-60
        a = (dqv + eps)/(dpv + eps) + eps
        b = np.log(np.nan_to_num(a))
        return  0.5  *(
             np.log(dqv/dpv)                   # log |\Sigma_q| / |\Sigma_p|
             + np.trace(np.dot(iqv, pv))        # + tr(\Sigma_q^{-1} * \Sigma_p)
             + np.dot(np.dot(diff, iqv), diff) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
             - len(pm))                     # - N
    


        
def generate_predictions(machine_class, machine_id):
    h5_train = h5py.File(param['feature_root'] + '/' + '_'.join([machine_class, machine_id, 'train', 'features'])+'.hdf5', 'r')
    h5_test = h5py.File(param['feature_root'] + '/' + '_'.join([machine_class, machine_id, 'test', 'features'])+'.hdf5', 'r')

    #train_means = h5_train['openl3_means'][:]

    N = 0
    n = 0
    
    for k in h5_train.keys():
        if machine_id in k:
            embs = h5_train[k]['mfcc'][:]
            embs = np.hstack((embs[:,:n_mfccs],embs[:, 40:40+n_mfccs], embs[:, 80:80+n_mfccs]))
            
            l,d = embs.shape
            N += l
            n += 1

            
    all_embs = np.zeros((N, d))
    record_emb_means = np.zeros((n, d))
    record_emb_vars = np.zeros((n, d))
    
    
    i = 0
    for k in h5_train.keys():
        if machine_id in k:
            embs = h5_train[k]['mfcc'][:]
            #embs = np.hstack((embs[16:-16,:n_mfccs],embs[16:-16, 30:30+n_mfccs], embs[16:-16, 60:60+n_mfccs]))
            record_emb_means[i] = np.hstack((np.mean(embs[:,:n_mfccs],axis=0), \
                                   np.mean(embs[4:-4, 40:40+n_mfccs],axis=0), \
                                   np.mean(embs[8:-8, 80:80+n_mfccs],axis=0)))
            record_emb_vars[i] = np.hstack((np.var(embs[:,:n_mfccs],axis=0), \
                                   np.var(embs[4:-4, 40:40+n_mfccs],axis=0), \
                                   np.var(embs[8:-8, 80:80+n_mfccs],axis=0)))

            i += 1

            
            
    anomaly_score_csv = "{0}/anomaly_score_{1}_{2}.csv".format(param["result_root"], machine_class, machine_id)
    anomaly_score_list = []
    
    
    for k in h5_test.keys():
        if machine_id in k:
            k_embs = h5_test[k]['mfcc'][:]
            #k_embs = np.hstack((k_embs[16:-16,:n_mfccs],k_embs[16:-16, 30:30+n_mfccs], k_embs[16:-16, 60:60+n_mfccs]))
            pm = np.hstack((np.mean(k_embs[:,:n_mfccs],axis=0), \
                                   np.mean(k_embs[4:-4, 40:40+n_mfccs, ],axis=0), \
                                   np.mean(k_embs[8:-8, 80:80+n_mfccs],axis=0)))
            pv = np.hstack((np.var(k_embs[:,:n_mfccs],axis=0), \
                                   np.var(k_embs[4:-4, 40:40+n_mfccs],axis=0), \
                                   np.var(k_embs[8:-8, 80:80+n_mfccs],axis=0)))
            
            #print (pm)
            #pm = np.mean(k_embs, axis=0)
            #pv = np.var(k_embs, axis=0)

            pm[n_mfccs:] = 0 #Mean of deltas should be theoretically 0
            
            dist_vec = np.zeros(n)
            for i in range(n):
                qm = record_emb_means[i]
                qm[n_mfccs:] = 0 #Mean of deltas should be theoretically 0
                qv = record_emb_vars[i]
                pm = pm.astype(np.longdouble)
                pv = pv.astype(np.longdouble)
                qm = qm.astype(np.longdouble)
                qv = qv.astype(np.longdouble)
                dist_vec[i] = gau_kl(pm,pv,qm,qv) + gau_kl(qm,qv,pm,pv)
                
            
            dist_vec = dist_vec[dist_vec.argsort()]
            
            a = np.min(dist_vec)
            #a = np.mean(dist_vec[:3])
            
            print (k, a)
            anomaly_score_list.append([k+'.wav', a])

            
    save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
            
    


if __name__ == '__main__':
    generate_predictions('ToyCar', 'id_01')
    generate_predictions('ToyCar', 'id_02')
    generate_predictions('ToyCar', 'id_03')
    generate_predictions('ToyCar', 'id_04')
    generate_predictions('ToyConveyor', 'id_01')
    generate_predictions('ToyConveyor', 'id_02')
    generate_predictions('ToyConveyor', 'id_03')
    generate_predictions('pump', 'id_00')
    generate_predictions('pump', 'id_02')
    generate_predictions('pump', 'id_04')
    generate_predictions('pump', 'id_06')
    generate_predictions('fan', 'id_00')
    generate_predictions('fan', 'id_02')
    generate_predictions('fan', 'id_04')
    generate_predictions('fan', 'id_06')
    generate_predictions('valve', 'id_00')
    generate_predictions('valve', 'id_02')
    generate_predictions('valve', 'id_04')
    generate_predictions('valve', 'id_06')
    generate_predictions('slider', 'id_00')
    generate_predictions('slider', 'id_02')
    generate_predictions('slider', 'id_04')
    generate_predictions('slider', 'id_06')
    """
    generate_predictions('ToyCar', 'id_05')
    generate_predictions('ToyCar', 'id_06')
    generate_predictions('ToyCar', 'id_07')
    generate_predictions('ToyConveyor', 'id_04')
    generate_predictions('ToyConveyor', 'id_05')
    generate_predictions('ToyConveyor', 'id_06')
    generate_predictions('pump', 'id_01')
    generate_predictions('pump', 'id_03')
    generate_predictions('pump', 'id_05')
    generate_predictions('fan', 'id_01')
    generate_predictions('fan', 'id_03')
    generate_predictions('fan', 'id_05')
    generate_predictions('valve', 'id_01')
    generate_predictions('valve', 'id_03')
    generate_predictions('valve', 'id_05')
    generate_predictions('slider', 'id_01')
    generate_predictions('slider', 'id_03')
    generate_predictions('slider', 'id_05')
    """
