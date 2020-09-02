import json
from DRflow.metrics import ABW, CAL, DSC, HM, NH, SC, sepme
from DRflow.metrics import AUClogRNX, CCA, CC, LCMC, NeRV, NLM, Stress, Trustworthiness, Continuity #,scagnostics
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
import ray
import json

@ray.remote
def apply_supervised(filename, output_dir):
    path = output_dir
    print(path)
    name = 'supervised_' + filename.split('/')[-1].split('.csv')[0]
    # if os.path.exists(path + name + '.json'):
    #     print('Skipping: {}'.format(name))
    #     return

    print('Starting... {}'.format(name))

    dr = pd.read_csv(filename)
    visu = np.array(dr.iloc[:, [0, 1]])
    labels = dr.labels


    res_sepme = sepme.compute(visu, labels, beta=0.2)
    print('SEPME complete for: {}'.format(name))
    abw = ABW.compute(visu, labels)
    print('AWB complete for: {}'.format(name))
    cal = CAL.compute(visu, labels)
    print('CAL complete for: {}'.format(name))
    dsc = DSC.compute(visu, labels)
    print('DSC complete for: {}'.format(name))
    hm = HM.compute(visu, labels)
    print('HM complete for: {}'.format(name))
    nh = NH.compute(dr.iloc[:, [0, 1, 2]])
    print('NH complete for: {}'.format(name))
    sc = SC.compute(visu, labels)
    print('SC complete for: {}'.format(name))

    res = {}
    res[name] = {'abw': abw, 'cal': cal, 'dsc': dsc, 'hm': hm, 'nh': nh, 'sc': sc}

    res[name] = {**res[name], **res_sepme}

    # Serialize data into file:
    json.dump(res, open(path + name + '.json', 'w'))

    return res

@ray.remote
def apply_highlow(drname, flatname, output_dir):

    name = 'highlow_' + drname.split('/')[-1].split('.csv')[0]
    path = output_dir
    # if os.path.exists(path + name + '.json'):
    #     print('Skipping: {}'.format(name))
    #     return
    print('Starting... {}'.format(name))

    high = pd.read_csv(flatname)

    dr = pd.read_csv(drname).reset_index(drop=True)
    visu = np.array(dr.iloc[:, [0, 1]])
    labels = dr.labels

    print(high.shape, dr.shape)


    high = high.loc[high['filename'].isin(dr['filename']),:].reset_index(drop=True)
    print(high.shape, dr.shape)

    # return high, dr

    auclog = AUClogRNX.compute(high.iloc[:, 0:-2], visu)
    print('AUClogRNX complete for: {}'.format(name))

    cca = CCA.compute(high.iloc[:, 0:-2], visu)
    print('CCA complete for: {}'.format(name))

    cc = CC.compute(high.iloc[:, 0:-2], visu)
    print('CC complete for: {}'.format(name))

    lcmc = LCMC.compute(high.iloc[:, 0:-2], visu)
    print('LCMC complete for: {}'.format(name))

    # high.to_csv('/Users/morarica/Developer/DRflow/high_coil_tnc.csv', index=False)
    # dr.to_csv('/Users/morarica/Developer/DRflow/visu_coil_tnc.csv', index=False)

    try:
        nerv = NeRV.compute(high.iloc[:, 0:-2], visu)
    except:
        nerv=-1

    print('NeRV complete for: {}'.format(name))

    nlm = NLM.compute(high.iloc[:, 0:-2], visu)
    print('NLM complete for: {}'.format(name))

    trust = Trustworthiness.compute(high.iloc[:, 0:-2], visu)
    print('Trust complete for: {}'.format(name))

    cont = Continuity.compute(high.iloc[:, 0:-2], visu)
    print('Cont complete for: {}'.format(name))

    # tnc = TandC.compute(high.iloc[:, 0:-2], visu)
    # print('TandC complete for: {}'.format(name))

    stress = Stress.compute(high.iloc[:, 0:-2], visu)
    print('Stress complete for: {}'.format(name))

    res = {}
    res[name] = {'auclog': auclog, 'cca': cca, 'cc': cc, 'lcmc': lcmc,
                'nerv':nerv, 'nlm':nlm, 'stress':stress, 'trustworthiness': trust, 'continuity': cont}


    json.dump(res, open(path + name + '.json', 'w'))

    return res

@ray.remote
def apply_scagnostics(drname):
    name = 'scagnostics_' + drname.split('/')[-1].split('.csv')[0]
    path = drname.split('dr')[0] + 'metrics/'
    if os.path.exists(path + name + '.json'):
        print('Skipping: {}'.format(name))
        return

    dr = pd.read_csv(drname)
    x = np.array(dr.iloc[:, [0]])
    y = np.array(dr.iloc[:, [1]])

    res = {name: scagnostics.compute(x, y)}

    json.dump(res, open(path + name + '.json', 'w'))
    print('Scagnostics complete for: {}'.format(name))
    return res


def evaluate_files(types=['supervised', 'scagnostics', 'highlow'],
                   high_dir='../../data/{}/mini_batch/flatfiles/',
                   dr_dir='../../data/{}/mini_batch/dr/',
                   output_dir='../../data/{}/mini_batch/metrics/'):
    for flat in os.listdir(high_dir):
        if not flat.endswith('.csv'):
            continue
        if 'classes' not in flat:
            c = 'all'
            size = flat.split('flat')[-1].split('.csv')[0]
        else:
            tmp = flat.split('.csv')[0]
            tmp = tmp.split('_')
            # print(tmp)

            size = tmp[-2].split('flat')[-1]
            # print(size)
            c = tmp[-1].split('classes')[0]
            # print(c)

        for dr in os.listdir(dr_dir):
            if not dr.endswith('.csv'):
                continue

            if ('c{}'.format(c) in dr):
                print(dr)
                try:
                    if 'supervised' in types:
                        apply_supervised.remote(dr_dir + dr, output_dir)
                    if 'scagnostics' in types:
                        apply_scagnostics(dr_dir + dr, output_dir)
                    if 'highlow' in types:
                        print(dr_dir + dr, high_dir + flat, output_dir)
                        print(apply_highlow.remote(dr_dir + dr, high_dir + flat, output_dir))
                except Exception as e:
                    print(e)
                    continue


def sample_and_save(input_file, output_folder, size_limit=800, idx=None, labels = 'labels'):
    fn = input_file.split('/')[-1].split('.csv')[0] + '_1.csv'
    print(input_file)
    df = pd.read_csv(input_file)

    if idx is None:
        if df.shape[0] > size_limit:
            try:
                _, df = train_test_split(df, test_size = size_limit, stratify = df[labels])
            except:
                _, df = train_test_split(df, test_size = size_limit)

    else:
        df = df.loc[idx].copy()

    df.to_csv(output_folder + fn, index = False)
    idx = list(df.index)

    print('Saved mini batch: {}'.format(fn))
    return idx