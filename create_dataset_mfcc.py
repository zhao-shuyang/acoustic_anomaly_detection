import h5py
import glob
import os
import itertools, re
import numpy as np
import yaml
import librosa
with open("param.yaml") as stream:
    param = yaml.safe_load(stream)


def compute_mfcc(filename):
    wav = librosa.load(filename, sr=16000)[0]
    melspec = librosa.feature.melspectrogram(
        wav,
        n_fft=2400,
        hop_length=600,
        window='hamming',
        n_mels=128        
    )
    
    log_mel = librosa.power_to_db(melspec)
    mfccs = librosa.feature.mfcc(S=log_mel,n_mfcc=40)
    delta = librosa.feature.delta(mfccs, width=9)
    delta2 = librosa.feature.delta(mfccs, order=2, width=9)

    mfcc_delta = np.vstack((mfccs,  delta, delta2))

    return mfcc_delta.T


def compute_mfcc2(filename):
    wav = librosa.load(filename, sr=16000)[0]
    melspec = librosa.feature.melspectrogram(
        wav,
        n_fft=3200,
        hop_length=800,
        window='hamming',
        n_mels=128)
    
    log_mel = librosa.power_to_db(melspec)
    mfccs = librosa.feature.mfcc(S=log_mel,n_mfcc=40)
    delta = librosa.feature.delta(mfccs, width=9)
    delta2 = librosa.feature.delta(mfccs, order=2, width=9)

    mfcc_delta = np.vstack((mfccs,  delta, delta2))

    return mfcc_delta.T

    

def get_machine_id_list_for_test(target_dir,
                                 dir_name="test",
                                 ext="wav"):
    """
    target_dir : str
        base directory path of "dev_data" or "eval_data"
    test_dir_name : str (default="test")

    directory containing test data
    ext : str (default="wav)
        file extension of audio files

    return :
        machine_id_list : list [ str ]
            list of machine IDs extracted from the names of test files
    """
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def create_dataset(machine_class, split):
    machine_id_list = get_machine_id_list_for_test(os.path.join(param['data_root'], machine_class), dir_name=split, ext="wav")
    print (machine_id_list)
    for machine_id in machine_id_list:
        h5 = h5py.File(param['feature_root'] + '/' + '_'.join([machine_class, machine_id, split, 'mfcc2'])+'.hdf5', 'w')
    
        for filename in glob.glob(os.path.join(param['data_root'], machine_class, split, '*.wav')):
            if  machine_id in filename:
                print(filename)
                f = compute_mfcc2(filename)
                print (f.shape)
                h5.create_dataset('/{0}/mfccs'.format(os.path.basename(filename).split('.')[0]), data=f)
     
        h5.close()



if __name__ == '__main__':
    create_dataset('ToyCar', 'train')
    create_dataset('ToyCar', 'test')
    create_dataset('fan', 'train')
    create_dataset('fan', 'test')
    create_dataset('slider', 'train')
    create_dataset('slider', 'test')
    create_dataset('pump', 'train')
    create_dataset('pump', 'test')

    create_dataset('ToyConveyor', 'train')
    create_dataset('ToyConveyor', 'test')
    create_dataset('valve', 'train')
    create_dataset('valve', 'test')
