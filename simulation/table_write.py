import os
import sys
import pickle
from subprocess import call
RES_DIR='./result'
TABLE_DIR='table'

p_values=  [12, 24,48, 96]
n_values = [100, 200, 400, 600]

'''
\begin{table}[p]
\def~{\hphantom{0}}
\tbl{RMISE (in $10^{-2}$) Under Homogeneous Setting}{%
\begin{tabular}{l@{\hskip 0.4in}ccccc}
%\\
 \\
 & Smoothed   & Shrinkage& Hard Threshold  & Lasso & Adaptive Lasso\\
 \\
 VMA & & & & &\\
 p = 12 & & & & &\\
 \multicolumn{1}{r}{n = 400}  & 20.71(2.04) & 14.54(0.94) & 13.88(1.65)  & 13.24(1.33) & \textbf{12.02}(1.36)\\
  \multicolumn{1}{r}{n = 800}  & 13.57(0.85) & 10.60(0.54) & \textbf{6.19}(0.96) & 8.48(0.88) & 6.26(0.85)\\
 p = 48 & & & & &  \\
 \multicolumn{1}{r}{n = 400}  & 81.15(2.77) & 25.18(0.35) & 24.39(0.79) & 20.77(0.91) &\textbf{18.27}(0.80)\\
  \multicolumn{1}{r}{n = 800}  & 54.72(1.47) &21.65(2.35) &11.58(0.57)  & 13.44(0.51) &\textbf{9.26}(0.48)\\
 p = 96 & & & & &  \\
 \multicolumn{1}{r}{n = 400}  & 161.76(3.87)& 28.88(0.27) & 29.05(0.47) & 26.02(0.55) & \textbf{22.74}(0.52)\\
  \multicolumn{1}{r}{n = 800}  & 108.45(1.75)&26.25(0.22) &15.53(0.48)  &16.90(0.40) &\textbf{11.87}(0.38)\\
 \\
 VAR & &  &  &  & \\
 p = 12 & & & & &  \\
 \multicolumn{1}{r}{n = 400}  &28.93(1.98) &\textbf{3.91}(0.53) &5.13(0.74) & 11.44(0.93) & 6.56(0.79) \\
  \multicolumn{1}{r}{n = 800}  & 19.36(1.05) & \textbf{3.16}(0.22)& 3.89(0.34)& 7.99(0.50) & 4.55(0.35)\\
 p = 48 & & & & &\\
 \multicolumn{1}{r}{n = 400}  &114.32(2.67)&\textbf{4.36}(0.36) &5.75(0.57)  & 16.58(0.51) & 7.97(0.47) \\
  \multicolumn{1}{r}{n = 800}  &77.64(1.59) & \textbf{3.51}(0.19)& 3.81(0.13)  &11.23(0.38) & 4.93(0.20)\\
 p = 96 & & & & & \\
 \multicolumn{1}{r}{n = 400}  & 229.78(3.59)& \textbf{4.58}(0.35)& 6.42(0.41) & 19.42(0.33) & 9.02(0.30) \\
  \multicolumn{1}{r}{n = 800}  & 154.02(2.05)& \textbf{3.58}(0.18)&3.83(0.12) & 13.25(0.28) & 5.29(0.17)\\
\end{tabular}}
\label{table:rmise-homogeneous}
\begin{tabnote}
\end{tabnote}
\end{table}
'''

'''
\small
\begin{table}[p]
\centering
\def~{\hphantom{0}}
\tbl{Precision, Recall, F1 Score($\%$) Under Homogeneous Setting}{%
\begin{tabular}{l@{\hskip 0.4in} ccc ccc ccc}
 & \multicolumn{3}{c}{Hard Thresholding}  & \multicolumn{3}{c}{Lasso} & \multicolumn{3}{c}{Adaptive Lasso}\\
 VMA & precision & recall & F1 & precision & recall & F1 & precision & recall & F1\\
 p = 12 & & & & & & & & & \\
 \multicolumn{1}{r}{n = 400}  & 96.71(0.95) & 81.15(2.27) & 86.61(1.71) & 58.74(2.91) & 99.18(0.38) & 71.68(2.30) & 88.97(2.34) &92.67(1.35) & 89.37(1.70) \\
  \multicolumn{1}{r}{n = 800}  & 98.18(0.74) & 93.80(1.26) & 95.50(0.95)& 57.71(1.86) &99.92(0.10) & 71.29(1.51) &91.90(1.38) &98.38(0.56) & 94.49(0.91) \\
 p = 48 & & & & & & & & & \\
 \multicolumn{1}{r}{n = 400}  &  99.77(0.11) & 55.17(1.59) & 70.16(1.28)& 62.37(1.95)& 96.50(0.46)& 74.73(1.45) & 95.44(0.70) & 81.43(1.21) & 87.31(0.73)\\
  \multicolumn{1}{r}{n = 800}  & 99.76(0.11) & 80.90(1.07) & 88.81(0.68)& 58.76(1.59) & 99.64(0.14) & 73.06(1.27)& 95.95(0.51) & 95.72(0.54) & 95.68(0.32) \\
 p = 96 & & & & & & & & & \\
 \multicolumn{1}{r}{n = 400}  & 99.94(0.04) & 45.19(0.83) & 61.80(0.76) & 68.30(1.00)& 92.65(0.59) & 77.94(0.65) &97.15(0.40) & 72.54(1.08) & 82.58(0.69)\\
  \multicolumn{1}{r}{n = 800}  & 99.93(0.04) & 71.73(0.87) & 82.90(0.60) &64.26(1.07) & 99.08(0.14) & 77.39(0.79) &97.33(0.30) & 92.28(0.52) & 94.57(0.29) \\
 VAR & precision & recall & F1 & precision & recall & F1 & precision & recall & F1\\
 p = 12 & & & & & & & & & \\
 \multicolumn{1}{r}{n = 400}  & 99.75(0.22) & 16.75(0.09) & 29(0.12)& 76.51(3.26) & 28.11(1.73) & 39.20(1.70) & 97.39(1.09)&17.58(0.36) & 29.66(0.44) \\
  \multicolumn{1}{r}{n = 800}  & 99.78(0.15) & 16.79(0.07)& 28.73(0.09) & 76.97(2.61) & 30.29(1.84)& 41.48(1.74) & 98.22(0.63) & 17.56(0.26) & 29.68(0.34)\\
 p = 48 & & & & & & & & &\\
 \multicolumn{1}{r}{n = 400}  & 99.95(0.07) & 16.50(0.06) & 28.32(0.08) & 69.38(1.75) & 18.62(0.39)& 29.04(0.48)& 98.33(0.55)&16.75(0.07) & 28.62(0.10) \\
  \multicolumn{1}{r}{n = 800}  & 99.98(0.03) & 16.66(0.01)&28.56(0.01) & 68.86(1.53) &19.09(0.33) & 29.57(0.42)& 99.31(0.25)& 16.73(0.03) &28.63(0.04)\\
 p = 96 & & & & & & & & & \\
 \multicolumn{1}{r}{n = 400}  & 99.98(0.03)& 16.37(0.04)& 28.13(0.07)& 73.93(1.32) & 17.42(0.12) &28.07(0.19) &98.95(0.25) &16.68(0.03) & 16.68(0.04) \\
  \multicolumn{1}{r}{n = 800}  & 100.00(0.01) &16.65(0.01) & 28.55(0.01)&73.82(1.32) & 17.67(0.12) & 28.38(0.18)& 99.61(0.16)& 16.69(0.01) & 28.59(0.02) \\
\end{tabular}}
\label{table:precision-homogeneous}
\end{table}
'''


def load_result(result_file_name ):
    print(os.path.join(RES_DIR, result_file_name))
    with open(os.path.join(RES_DIR, result_file_name), 'rb') as handle:
        res = pickle.load(handle)
    return res


def test_structure():
    result_name = '_'.join(['ma', 'result', str(400)])
    print(result_name)
    res = load_result(result_name)
    print(list(res.keys()))
    print("=======")
    print(list(res['ho_12'].keys()))
    print("=========")
    print(list(res['ho_12']['error'].keys()))


def write_rmise_header_tail(file_name, model_type='ho'):
    call('rm -f temp', shell=True)
    if model_type == 'ho':
        call('touch temp && cat rmise_header_ho >> temp && cat {0} >> temp && mv temp {0}'.format(file_name), shell=True)
        call('touch temp && cat {0} >> temp && cat rmise_tail_ho >> temp  && mv temp {0}'.format(file_name), shell=True)
    else:
        call('touch temp && cat rmise_header_he >> temp && cat {0} >> temp && mv temp {0}'.format(file_name), shell=True)
        call('touch temp && cat {0} >> temp && cat rmise_tail_he >> temp  && mv temp {0}'.format(file_name), shell=True)


def write_three_metric_header_tail(file_name, model_type='ho'):
    call('rm -f temp', shell=True)
    if model_type == 'ho':
        call('touch temp && cat three_metric_header_ho >> temp && cat {0} >> temp && mv temp {0}'.format(file_name), shell=True)
        call('touch temp && cat {0} >> temp && cat three_metric_tail_ho >> temp  && mv temp {0}'.format(file_name), shell=True)
    else:
        call('touch temp && cat three_metric_header_he >> temp && cat {0} >> temp && mv temp {0}'.format(file_name), shell=True)
        call('touch temp && cat {0} >> temp && cat three_metric_tail_he >> temp  && mv temp {0}'.format(file_name), shell=True)


def tuple_2_string(my_tuple):
    return str(round(my_tuple[0]*100,2))+'('+str(round(my_tuple[1]*100,2))+')'



def extract_array_result(result, model_type, p):
    sub_result = result[model_type + '_' + str(p)]
    sm = '&' + tuple_2_string(sub_result['relative_error']['sm'])
    sh = '&' + tuple_2_string(sub_result['relative_error']['sh'])
    th = '&' + tuple_2_string(sub_result['relative_error']['th'])
    so = '&' + tuple_2_string(sub_result['relative_error']['so'])
    al = '&' + tuple_2_string(sub_result['relative_error']['al'])
    return sm+sh+th+so+al




def write_vma_rmise(file_handle, model_type='ho'):

    try:
        result_file_name = 'ma_result_100'
        result_100 = load_result(result_file_name)
    except:
        result_100 = None

    try:
        result_file_name = 'ma_result_200'
        result_200 = load_result(result_file_name)
    except:
        result_200 = None

    try:
        result_file_name = 'ma_result_400'
        result_400 = load_result(result_file_name)
    except:
        result_400 = None

    try:
        result_file_name = 'ma_result_600'
        result_600 = load_result(result_file_name)
    except:
        result_600 = None
    file_handle.write('VMA & & & & &\\\\\n')
    for p in p_values:
        file_handle.write('p = {} & & & & &\\\\\n'.format(p))
        for n in n_values:
            result = eval('result_'+str(n))
            if result is not None:
                file_handle.write('\\multicolumn{{1}}{{r}}{{n = {0}}}'.format(str(n)))
                file_handle.write(extract_array_result(result, model_type, p)+'\\\\\n')



def write_var_rmise(file_handle, model_type='ho'):

    try:
        result_file_name = 'var_result_100'
        result_100 = load_result(result_file_name)
    except:
        result_100 = None


    try:
        result_file_name = 'var_result_200'
        result_200 = load_result(result_file_name)
    except:
        result_200 = None

    try:
        result_file_name = 'var_result_400'
        result_400 = load_result(result_file_name)
    except:
        result_400 = None

    try:
        result_file_name = 'var_result_600'
        result_600 = load_result(result_file_name)
    except:
        result_600 = None

    file_handle.write('VAR & & & & &\\\\\n')
    for p in p_values:
        file_handle.write('p = {} & & & & &\\\\\n'.format(p))
        for n in n_values:
            result = eval('result_' + str(n))
            if result is not None:
                file_handle.write('\\multicolumn{{1}}{{r}}{{n = {0}}}'.format(str(n)))
                file_handle.write(extract_array_result(result, model_type, p) + '\\\\\n')


def write_rmise_table(model_type='ho'):
    file_name = 'rmise_'+model_type+'_heavy_tail_table.tex'
    with open(os.path.join(RES_DIR, 'table', file_name), 'w') as table_handle:
        write_vma_rmise(table_handle, model_type)
        write_var_rmise(table_handle, model_type)
    write_rmise_header_tail(os.path.join(RES_DIR, 'table', file_name), model_type= model_type)




def extract_three_metric_array(result, model_type, p):
    sub_result = result[model_type + '_' + str(p)]
    th = []; so = []; al = []
    for threshold_type in ['th', 'so', 'al']:
        for metric_type in ['precision', 'recall', 'F1']:
            elem = tuple_2_string(sub_result[metric_type][threshold_type])
            exec('{0}.append(\'{1}\')'.format(threshold_type, '&'+elem))
    return ''.join(th)+''.join(so)+''.join(al)



def write_vma_three_metrics(file_handle, model_type='ho'):
    try:
        result_file_name = 'ma_result_100'
        result_100 = load_result(result_file_name)
    except:
        result_100 = None

    try:
        result_file_name = 'ma_result_200'
        result_200 = load_result(result_file_name)
    except:
        result_200 = None

    try:
        result_file_name = 'ma_result_400'
        result_400 = load_result(result_file_name)
    except:
        result_400 = None

    try:
        result_file_name = 'ma_result_600'
        result_600 = load_result(result_file_name)
    except:
        result_600 = None
    file_handle.write('VMA & precision & recall & F1 & precision & recall & F1 & precision & recall & F1\\\\\n')
    for p in p_values:
        file_handle.write(' p = {} & & & & & & & & & \\\\\n'.format(str(p)))
        for n in n_values:
            result = eval('result_' + str(n))
            if result is not None:
                file_handle.write('\\multicolumn{{1}}{{r}}{{n = {0}}}'.format(str(n)))
                file_handle.write(extract_three_metric_array(result, model_type, p)+'\\\\\n')




def write_var_three_metrics(file_handle, model_type='ho'):

    try:
        result_file_name = 'var_result_100'
        result_100 = load_result(result_file_name)
    except:
        result_100 = None


    try:
        result_file_name = 'var_result_200'
        result_200 = load_result(result_file_name)
    except:
        result_200 = None

    try:
        result_file_name = 'var_result_400'
        result_400 = load_result(result_file_name)
    except:
        result_400 = None

    try:
        result_file_name = 'var_result_600'
        result_600 = load_result(result_file_name)
    except:
        result_600 = None

    file_handle.write('VAR & precision & recall & F1 & precision & recall & F1 & precision & recall & F1\\\\\n')
    for p in p_values:
        file_handle.write(' p = {} & & & & & & & & & \\\\\n'.format(str(p)))
        for n in n_values:
            result = eval('result_' + str(n))
            if result is not None:
                file_handle.write('\\multicolumn{{1}}{{r}}{{n = {0}}}'.format(str(n)))
                file_handle.write(extract_three_metric_array(result, model_type, p)+'\\\\\n')


def write_three_metric_table(model_type = 'ho'):
    file_name = 'three_metric_' + model_type + '_heavy_tail_table.tex'
    with open(os.path.join(RES_DIR, 'table', file_name), 'w') as table_handle:
        write_vma_three_metrics(table_handle, model_type)
        write_var_three_metrics(table_handle, model_type)
    write_three_metric_header_tail(os.path.join(RES_DIR, 'table', file_name), model_type=model_type)


if __name__ == "__main__":
    write_rmise_table(model_type='ho')
    #write_rmise_table(model_type='he')
    write_three_metric_table(model_type='ho')
    #write_three_metric_table(model_type='he')
