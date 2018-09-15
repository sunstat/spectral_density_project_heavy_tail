'''
result structure:
ma_result_400, ma_result_800, var_result_400, var_result_800
ma,var is saved in gen_mode
load the result as nested dictionary:
first layer key: ho_12, he_12, ho_48, he_48, ho_96, he_96
second layer key: error, precision, recall, F1,
third layer key: error|'sm', 'sh', 'th', 'so', 'al', 'true'; precision|recall|F1: 'sm', 'sh', 'th', 'so', 'al'
'''



from simulation import *
import numpy as np
from generate_weights import *
from spectral_density import *
import matplotlib.pyplot as plt
import os

dashList = [(5,2),(2,5),(4,10),(3,3,2,2),(5,2,20,2)]
p_values = [12,24,48,96]


def plot_err_curve(ax, errs_sm_dict, errs_sh_dict, errs_th_dict,
                   errs_soft_dict, errs_al_dict, p, true_spectral_norm = None, graphics = False):

    num_obs = len(errs_sm_dict)
    keys = list(errs_sm_dict.keys())
    x_keys = sorted(list(map(lambda x: index_to_freq(x, num_obs), keys)))

    if true_spectral_norm is None:

        _, values = extract_tuple(errs_sm_dict)
        sm, = ax.plot(x_keys, values, 'b', label = 'smoothed', linestyle = '-.')
        _, values = extract_tuple(errs_sh_dict)
        sh, = ax.plot(x_keys, values, 'g', label='shrinkage', linestyle= ':')
        _, values = extract_tuple(errs_th_dict)
        hard, = ax.plot(x_keys, values, 'c', label = 'hard', linestyle='-')

        if errs_soft_dict:
            _, values = extract_tuple(errs_soft_dict)
            so, = ax.plot(x_keys, values, 'y', label='soft', linestyle='--', dashes=(10,2,20,2))
        if errs_al_dict:
            _, values = extract_tuple(errs_al_dict)
            al, = ax.plot(x_keys, values, 'm', label='adaptive.lasso', linestyle='--', dashes=[30, 5, 10, 5])
    else:

        _, values = extract_tuple(errs_sm_dict)
        sm, = ax.plot(x_keys, values, 'b', label='smoothed', linestyle='-.')
        _, values = extract_tuple(errs_sh_dict)
        sh, = ax.plot(x_keys, values, 'g', label='shrinkage', linestyle=':')
        _, values = extract_tuple(errs_th_dict)
        hard, = ax.plot(x_keys, values, 'c', label='hard', linestyle='-')

        _, values = extract_tuple(true_spectral_norm)
        true, = ax.plot(x_keys, values, 'r', label='true', linestyle='--')

        if errs_soft_dict:
            _, values = extract_tuple(errs_soft_dict)
            so, = ax.plot(x_keys, values, 'y', label='soft', linestyle='--', dashes=(10, 2, 20, 2))
        if errs_al_dict:
            _, values = extract_tuple(errs_al_dict)
            al, = ax.plot(x_keys, values, 'm', label='adaptive.lasso', linestyle='--', dashes=[30, 5, 10, 5])

    ax.set_xlabel('frequencies')
    ax.set_ylabel('relative error norm square')
    ax.set_title('p={}'.format(str(p)))

    if true_spectral_norm is None:
        return sm, sh, hard, so, al
    return true, sm, sh, hard, so, al


def average_relative_dict(ls_err_dict, true_norm_dict):
    relative_err_ls = []
    for err_dict in ls_err_dict:
        relative_err = {}
        for key, value in err_dict.items():
            relative_err[key] = err_dict[key]/true_norm_dict[key]
        relative_err_ls.append(relative_err)
    return average_errs_dict(relative_err_ls, relative_err_ls[0].keys())



def graphics_help(result, ax, p, relative=False):

    errs_dict_sm = result['raw_error']['sm']
    errs_dict_sh = result['raw_error']['sh']
    errs_dict_th = result['raw_error']['th']
    errs_dict_so = result['raw_error']['so']
    errs_dict_al = result['raw_error']['al']

    true_spectral_norm = result['raw_error']['true']
    if relative:
        err_dict_al = average_relative_dict(errs_dict_al, true_spectral_norm)
        err_dict_th = average_relative_dict(errs_dict_th, true_spectral_norm)
        err_dict_so = average_relative_dict(errs_dict_so, true_spectral_norm)
        err_dict_sh = average_relative_dict(errs_dict_sh, true_spectral_norm)
        err_dict_sm = average_relative_dict(errs_dict_sm, true_spectral_norm)
    else:
        err_dict_al = average_errs_dict(errs_dict_al, true_spectral_norm.keys())
        err_dict_th = average_errs_dict(errs_dict_th, true_spectral_norm.keys())
        err_dict_so = average_errs_dict(errs_dict_so, true_spectral_norm.keys())
        err_dict_sh = average_errs_dict(errs_dict_sh, true_spectral_norm.keys())
        err_dict_sm = average_errs_dict(errs_dict_sm, true_spectral_norm.keys())

    if relative:
        return plot_err_curve(ax, err_dict_sm, err_dict_sh, err_dict_th, err_dict_so, err_dict_al, p,
                   graphics=False)
    else:
        return plot_err_curve(ax, err_dict_sm, err_dict_sh, err_dict_th, err_dict_so, err_dict_al, p,
                              true_spectral_norm, graphics=True)


def graphics(result, num_obs, gen_mode='ho', model_mode='ma', relative=False):
    fig, axes = plt.subplots(1, 3)
    ind = 0
    true=None; sm=None; sh=None; hard=None; so=None; al=None
    for p in p_values:
        var_name = '_'.join([gen_mode, str(p)])
        if relative:
            sm, sh, hard, so, al = graphics_help(result[var_name], axes[ind], p, relative)
        else:
            true, sm, sh, hard, so, al = graphics_help(result[var_name], axes[ind], p, relative)
        ind+=1

    plt.legend((sm, sh, hard, so, al),
            ('sm', 'sh', 'hard', 'la', 'al'), loc=9, bbox_to_anchor=(0, -0.2))

    plt.tight_layout()
    plt.savefig(os.path.join('result', model_mode+'_'+str(num_obs)+'_'+gen_mode+'_'+'error_curve.pdf'))
    plt.close('all')


def mise(self_dict):
    return np.mean(list(self_dict.values()))



def display_results(num_obs, model_mode):
    result_name = '_'.join([model_mode, 'result', str(num_obs)])
    print(result_name)
    result = load_result(result_name)
    for gen_mode in ['ho', 'he']:
        graphics(result, num_obs, gen_mode,  model_mode, relative=False)



if __name__ == "__main__":
    display_results(num_obs=100, model_mode='ma')
    display_results(num_obs=100, model_mode='var')
    display_results(num_obs=200, model_mode='ma')
    display_results(num_obs=200, model_mode = 'var')
    display_results(num_obs=400, model_mode='ma')
    display_results(num_obs=400, model_mode='var')
    display_results(num_obs=600, model_mode='ma')
    display_results(num_obs=600, model_mode='var')


