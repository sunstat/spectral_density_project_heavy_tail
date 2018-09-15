import os
from spectral_density import *
from generate_weights import *
import pickle
import random
import time

import matplotlib.pyplot as plt
from datetime import datetime
import sys
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
import multiprocessing.pool
from multiprocessing import Manager
from itertools import chain
from itertools import cycle


RES_DIR = 'result'
simu_log_file = 'simu_res.log'


p_values = [12, 24, 48, 96]


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess



def fetch_span(num_obs, gen_model):
    if gen_model == 'ma':
        if num_obs == 800:
            return int(np.sqrt(num_obs))
        elif num_obs == 600:
            return int(np.sqrt(num_obs))
        elif num_obs == 400:
            return int(np.sqrt(num_obs))
        elif num_obs == 200:
            return int(np.sqrt(num_obs))
        elif num_obs == 100:
            return int(np.sqrt(num_obs))
    elif gen_model == 'var':
        if num_obs == 800:
            return int(np.sqrt(num_obs))//3*2
        elif num_obs == 600:
            return int(np.sqrt(num_obs))//3*2
        elif num_obs == 400:
            return int(np.sqrt(num_obs))//3*2
        elif num_obs == 200:
            return int(np.sqrt(num_obs))//3*2
        elif num_obs == 100:
            return int(np.sqrt(num_obs))//3*2


def extract_tuple1(errs_dict):
    ls = sorted(errs_dict.items())
    freq_ind, values = zip(*ls)
    return freq_ind, values




def append_help(ls1, ls2, ls3, v1, v2, v3):
    ls1.append(v1)
    ls2.append(v2)
    ls3.append(v3)



def mean_values(dict_values, true_spectral):
    return np.mean(list(dict_values.values()))/np.mean(list(true_spectral.values()))


def append_relative_err(result):

    errs_dict_sm = result['raw_error']['sm']
    errs_dict_sh = result['raw_error']['sh']
    errs_dict_th = result['raw_error']['th']
    errs_dict_so = result['raw_error']['so']
    errs_dict_al = result['raw_error']['al']
    true_spectral = result['raw_error']['true']

    errs_dict_sm = list(map(lambda x: mean_values(x, true_spectral), errs_dict_sm))
    errs_dict_sh = list(map(lambda x: mean_values(x, true_spectral), errs_dict_sh))
    errs_dict_th = list(map(lambda x: mean_values(x, true_spectral), errs_dict_th))
    errs_dict_al = list(map(lambda x: mean_values(x, true_spectral), errs_dict_al))
    errs_dict_so = list(map(lambda x: mean_values(x, true_spectral), errs_dict_so))

    err_al = np.mean(errs_dict_al)
    err_th = np.mean(errs_dict_th)
    err_so = np.mean(errs_dict_so)
    err_sh = np.mean(errs_dict_sh)
    err_sm = np.mean(errs_dict_sm)

    std_sm = np.std(errs_dict_sm)
    std_sh = np.std(errs_dict_sh)
    std_th = np.std(errs_dict_th)
    std_al = np.std(errs_dict_al)
    std_so = np.std(errs_dict_so)


    result['relative_error'] = {}
    for method in ['al', 'th', 'so', 'sh', 'sm']:
        result['relative_error'][method] = (eval('err_'+method), eval('std_'+method))

    return result




def simu_setting_2_str(p, generating_mode):
    return str(generating_mode)+'_'+str(p)




def simu_help(mode, num_obs, p, generating_mode, individual_level=True, num_iterations=50):
    assert generating_mode in ['ma', 'var']
    print("now doing simulation with setting p = {}, mode = {}".format(p, mode))
    print("================")
    weights = fetch_weights(p, mode, generating_mode)
    stdev = 1
    span = fetch_span(num_obs, generating_mode)

    model_info = {}
    model_info['model'] = generating_mode
    model_info['weights'] = weights
    model_info['span'] = span
    model_info['stdev'] = stdev


    errs_dict_al = []
    errs_dict_th = []
    errs_dict_so = []
    errs_dict_sh = []
    errs_dict_sm = []

    precision_al = []
    precision_th= []
    precision_so = []
    recall_al = []
    recall_so = []
    recall_th = []
    F1_al = []
    F1_so = []
    F1_th = []

    true_spectral_norm_square = {}


    for i in range(num_iterations):
        if generating_mode == 'ma':
            ts = generate_ma(weights, num_obs=num_obs, stdev=stdev)
        elif generating_mode == 'var':
            ts = generate_mvar(weights, num_obs=num_obs, stdev=stdev)
        spec_est = SpecEst(ts, model_info, individual_level=individual_level)
        #test_left_right_norm(spec_est)
        err_al_dict = spec_est.evaluate('al')
        err_th_dict = spec_est.evaluate('th')
        err_so_dict = spec_est.evaluate('so')
        err_sh_dict = spec_est.evaluate('sh')
        err_sm_dict = spec_est.evaluate('sm')

        errs_dict_al.append(err_al_dict)
        errs_dict_th.append(err_th_dict)
        errs_dict_so.append(err_so_dict)
        errs_dict_sh.append(err_sh_dict)
        errs_dict_sm.append(err_sm_dict)
        for mode_threshold in ['th', 'so', 'al']:
            precision, recall, F1 = spec_est.query_recover_three_measures(mode_threshold)
            if mode_threshold == 'th':
                append_help(precision_th, recall_th, F1_th, precision, recall, F1)
            elif mode_threshold == 'so':
                append_help(precision_so, recall_so, F1_so, precision, recall, F1)
            elif mode_threshold == 'al':
                append_help(precision_al, recall_al, F1_al, precision, recall, F1)

        if i == 0:
            true_spectral = spec_est.return_all_true_spectral()
            for key in true_spectral:
                true_spectral_norm_square[key] = HS_norm(true_spectral[key])**2
        print("finishing iteration {}".format(i))

    result = {}

    result['raw_error'] = {'al': errs_dict_al, 'th': errs_dict_th, 'so':errs_dict_so,
                       'sh': errs_dict_sh, 'sm': errs_dict_sm, 'true': true_spectral_norm_square}
    result['precision'] = {'so': (np.mean(precision_so), np.std(precision_so)), 'al': (np.mean(precision_al), np.std(precision_al))
        , 'th': (np.mean(precision_al), np.std(precision_al))}
    result['recall'] = {'so': (np.mean(recall_so), np.std(recall_so)), 'al': (np.mean(recall_al), np.std(recall_al))
        , 'th': (np.mean(recall_th), np.std(recall_th))}
    result['F1'] = {'so': (np.mean(F1_so), np.std(F1_so)), 'al': (np.mean(F1_al), np.std(F1_al)), 'th': (np.mean(F1_th), np.std(F1_th))}

    append_relative_err(result)


    return result, simu_setting_2_str(p, mode)



def evaluate_iteration(num_obs, model_info, individual_level = True):
    np.random.seed()
    generating_mode = model_info['model']
    weights = model_info['weights']
    span = model_info['span']
    stdev = model_info['stdev']


    if generating_mode == 'ma':
        ts = generate_ma(weights, num_obs=num_obs, stdev=stdev)
    elif generating_mode == 'var':
        ts = generate_mvar(weights, num_obs=num_obs, stdev=stdev)
    spec_est = SpecEst(ts, model_info, individual_level=individual_level)
    true_spectral_norm_square = {}
    true_spectral = spec_est.return_all_true_spectral()
    for key in true_spectral:
        true_spectral_norm_square[key] = HS_norm(true_spectral[key]) ** 2

    res = []

    err_al_dict = spec_est.evaluate('al')
    err_th_dict = spec_est.evaluate('th')
    err_so_dict = spec_est.evaluate('so')
    err_sh_dict = spec_est.evaluate('sh')
    err_sm_dict = spec_est.evaluate('sm')

    res += [err_al_dict]+[err_th_dict]+[err_so_dict]+[err_sh_dict]+[err_sm_dict]

    for mode_threshold in ['th', 'so', 'al']:
        precision, recall, F1 = spec_est.query_recover_three_measures(mode_threshold)
        res += [precision]+[recall]+[F1]

    res+=[true_spectral_norm_square]

    return res



def parallel_simu_help(mode, num_obs, p, generating_mode, individual_level=True, num_iterations=2, noise_type='T'):
    assert generating_mode in ['ma', 'var']
    print("now doing simulation with setting p = {}, mode = {}".format(p, mode))
    print("================")
    weights = fetch_weights(p, mode, generating_mode)
    stdev = 1
    span = fetch_span(num_obs, generating_mode)

    model_info = {}
    model_info['model'] = generating_mode
    model_info['weights'] = weights
    model_info['span'] = span
    model_info['stdev'] = stdev


    if generating_mode == 'ma':
        ts = generate_ma(weights, num_obs=num_obs, stdev=stdev, noise_type=noise_type)
    elif generating_mode == 'var':
        ts = generate_mvar(weights, num_obs=num_obs, stdev=stdev, noise_type=noise_type)

    arguments = list(zip(cycle([num_obs]),  [model_info for _ in range(num_iterations)]))
    #print(arguments)


    iteration_pool = Pool(10)

    res = iteration_pool.starmap(evaluate_iteration, arguments)

    errs_dict_al,  errs_dict_th, errs_dict_so, errs_dict_sh, errs_dict_sm , precision_th, recall_th, F1_th, \
        precision_so, recall_so, F1_so, precision_al, recall_al, F1_al, true_spectral_norm_square = list(zip(*res))

    true_spectral_norm_square = true_spectral_norm_square[0]
    result = {}


    result['raw_error'] = {'al': errs_dict_al, 'th': errs_dict_th, 'so':errs_dict_so,
                       'sh': errs_dict_sh, 'sm': errs_dict_sm, 'true': true_spectral_norm_square}
    result['precision'] = {'so': (np.mean(precision_so), np.std(precision_so)), 'al': (np.mean(precision_al), np.std(precision_al))
        , 'th': (np.mean(precision_al), np.std(precision_al))}
    result['recall'] = {'so': (np.mean(recall_so), np.std(recall_so)), 'al': (np.mean(recall_al), np.std(recall_al))
        , 'th': (np.mean(recall_th), np.std(recall_th))}
    result['F1'] = {'so': (np.mean(F1_so), np.std(F1_so)), 'al': (np.mean(F1_al), np.std(F1_al)), 'th': (np.mean(F1_th), np.std(F1_th))}

    append_relative_err(result)


    return result, simu_setting_2_str(p, mode)



def series_simu(num_obs, generating_mode, individual_level=True, noise_type='T'):
    print(type(generating_mode))
    res_file_name = generating_mode+'_'+'result_'+str(num_obs)
    result = {}
    for p in p_values:
        for mode in ['ho']:
            sub_res, key_name = parallel_simu_help(mode, num_obs = num_obs, p=p,
                    generating_mode = generating_mode, individual_level=individual_level)
            print(key_name)
            result[key_name] = sub_res
    with open(os.path.join(RES_DIR, res_file_name), 'wb') as f:
        pickle.dump(result, f)
    return result


def parallel_simu(num_obs, generating_mode, individual_level=True):

    res_file_name = generating_mode+'_'+'result_'+str(num_obs)
    num_obs = [num_obs for _ in range(6)]
    '''
    p_values1 = [p_values for _ in range(2)]
    p_values1 = list(chain(*p_values1))
    modes = [['ho', 'he'] for _ in range(3)]
    modes = list(chain(*modes))
    generating_modes = [generating_mode for _ in range(6)]
    '''
    arguments = list(zip(cycle(['ho', 'he']), num_obs, cycle(p_values), cycle([generating_mode]), cycle([individual_level])))
    print(arguments)
    p = MyPool(10)
    res = p.starmap(parallel_simu_help, arguments)
    result = {}
    for item in res:
        result[item[1]] = item[0]

    with open(os.path.join(RES_DIR, res_file_name), 'wb') as f:
        pickle.dump(result, f)
    return result



def load_result(result_file_name = 'result'):
    print(os.path.join(RES_DIR, result_file_name))
    with open(os.path.join(RES_DIR, result_file_name), 'rb') as handle:
        res = pickle.load(handle)
    return res



def extract_tuple(errs_dict):
    num_obs = len(errs_dict)
    ls = sorted(errs_dict.items())
    freq_ind, values = zip(*ls)
    freq = [index_to_freq(x, num_obs) for x in freq_ind]
    return freq, values



def main(series=False):
    if series:
        series_simu(100, generating_mode='ma', individual_level=True)
        series_simu(100, generating_mode='var', individual_level=True)
        series_simu(200, generating_mode='ma', individual_level=True)
        series_simu(200, generating_mode='var', individual_level=True)
        #series_simu(400, generating_mode = 'ma', individual_level=True)
        #series_simu(400, generating_mode='var', individual_level=True)
        #series_simu(600, generating_mode='ma', individual_level=True)
        #series_simu(600, generating_mode='var', individual_level=True)
    else:
        #p_simu = MyPool()
        #p_simu.starmap(parallel_simu, [[200,'ma'], [200,'var']])
        parallel_simu(200, 'ma')
        #parallel_simu(200, 'ma', True)




def test_evaluate_iteration():
    pass


def test_parallel_simu_help():
    mode = 'ho'; num_obs=200; p=12; generating_mode='ma'; individual_level = False; num_iterations = 3
    parallel_simu_help(mode, num_obs, p, generating_mode)



if __name__ == "__main__":
    #test_parallel_simu_help()

    start_time = time.time()
    main(series=True)
    print("--- %s seconds ---" % (time.time() - start_time))
    '''

    #simu(200, generating_mode='ma', individual_level=True)
    #simu(200, generating_mode='var', individual_level=True)
    #simu(400, generating_mode = 'ma', individual_level=True)
    #simu(400, generating_mode='var', individual_level=True)
    #simu(600, generating_mode='ma', individual_level=True)
    #simu(600, generating_mode='var', individual_level=True)

    #simu_ma_help(mode = 'ho', num_obs = 600, p=48, graphics=True)
    '''