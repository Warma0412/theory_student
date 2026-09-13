import sys
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from matplotlib import pyplot as plt
import torch
import time
from benchmark_utils import gen_data_multi_dim, load_data_multi_dim, train_model, distribution_check
from benchmark_utils import two_sample_tests, plot_dist_random_two_entries, random_single_dim_check, fdr_exp

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--d", type=int, help="input data dimension", default=100)
parser.add_argument("--n", type=int, help="sample size", default=200)

parser.add_argument("--method", type=str, default='rank', help="Local rank. Necessary for using the torch.distributed.launch utility.")
parser.add_argument("--distType", type=str, help="Random seed.", default="GaussianMixtureAR1")
parser.add_argument("--working_dir", type=str, help="Directory for saving models.", default='./results/')
parser.add_argument("--appendix", type=str, default='', help="appendix to the file name")
parser.add_argument("--yConfig", type=str, default='linear!scale', help="y_dist_config")
parser.add_argument("--allRegisters", type=bool, default=False, help="if register more info")
parser.add_argument("--FSModel", type=str, default='ridge', help="options: [ridge, lasso, ols, rf, deeppink]")


# temporal configurations for rebuttal
parser.add_argument("--org_decor_coeff", type=float, default=0.0, help="for element-wise decorrelation")
parser.add_argument("--betaRx", type=float, default=30.0, help="for IRM part")
parser.add_argument("--n_swapper", type=int, default=2, help="number of swappers")
parser.add_argument("--swapper_pen_coeff", type=float, default=1.0, help="number of swappers")
parser.add_argument("--destroy_decor_coef", type=float, default=10.0, help="swc penalty")
parser.add_argument("--linear_hidden_size", type=int, default=512, help="linear_hidden_size for the transformer")
parser.add_argument("--depth", type=int, default=8, help="layer depth for the transformer")
parser.add_argument("--decor_coeff", type=float, default=2.0, help="overall scalar for decorrelation coeffs")




argv = parser.parse_args()

n = argv.n
d = argv.d
print(n, d)
method=argv.method
distType=argv.distType 
working_dir = argv.working_dir
y_cond_mean=argv.yConfig.split('!')[0]
y_coeff_dist=argv.yConfig.split('!')[1]
print(y_cond_mean, y_coeff_dist)
if distType not in ['mnist', 'ibd', 'ibd_semi', 'rna_semi']:
    gen_data_multi_dim(distType, n, d, working_dir, reset=False, save=True)
    
xTrain, xTestFull, corr_g, second_order = load_data_multi_dim(distType, n, d, working_dir)

if any([item in method for item in ['sw', 'deepkf', 'second', 'ddlk', 'wgan', 'gan', 'rank']]):
    if 'sw' in method:
        print('New Config SW')
        args_dict_sw = {'n_swapper': argv.n_swapper,
         'tau': 0.2,
         'lr_s': 1e-3,
         'lr_g': 1e-5,
         'dropout': 0.1,
         'num_epoch': 200,
         'batch_size': 64,
         'destroy_decor_coef': argv.destroy_decor_coef,
         'gsw_coeff': 1.,
         'decor_coeff': argv.decor_coeff,
         'betaRx': argv.betaRx,
         'swapper_steps': 3,
         'swapper_pen_coeff': argv.swapper_pen_coeff,
         'org_decor_coeff': argv.org_decor_coeff,
         'learn_dist': False,
         'test_ratio': 0.2,
         'SWC': True,
         'linear_hidden_size': argv.linear_hidden_size,
         'depth': argv.depth}
        print(args_dict_sw)
    else:
        args_dict_sw = None
        
    start_time = time.time()
    xTrain, xTr_tilde, machine = train_model(xTrain, d, n, distType=distType, 
                                             method=method, corr_g=corr_g,
                                             second_order=second_order,
                                             args_dict_sw=args_dict_sw)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Function took {elapsed_time} seconds to run.")


else:
    machine = '' # tmp solution bypass assert

_ = fdr_exp(xTestFull, n = n, d = d, distType=distType, 
            working_dir = working_dir,
            complex_y_model=False, method=method, machine=machine,
            nominal_fdr=0.1, true_feature_n=20, n_repeats=600 if distType not in ['mnist', 'ibd'] else 30,
            lasso_coeff=1.,lambda_val_list=[0.4, 0.45, 0.5], signal_amplitude_vec=[3, 5, 10, 15, 20, 25, 30],
            reset=True, appendix=argv.appendix, y_cond_mean=y_cond_mean, y_coeff_dist=y_coeff_dist, all_registers=argv.allRegisters,
           fs_model=argv.FSModel)

print('end')

