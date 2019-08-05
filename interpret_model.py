import utilities.inspect_checkpoint as chkp
import argparse, re
import pandas as pd
import numpy as np
from itertools import combinations
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path")

args = parser.parse_args()

checkpoint_path = Path(args.checkpoint_path)
if not re.search(r"model\.ckpt-\d+$", str(checkpoint_path)):
    if re.search(r"model\.ckpt.*$", str(checkpoint_path)):
        raise ValueError(
            "Invalid checkpoint path: \n{}\n\n".format(checkpoint_path)+
            "To specify a model file, you (only) need to include the number index behind the suffix \".ckpt\", for example:\n"
            "path/to/checkpoint/model.ckpt-888\n\n"
            "You may also give the folder containing the ckpt model files, which will read the latest checkpoint in that folder, for example:\n"
            "path/to/checkpoint"
        )
    else: # Find the checkpoint with the maximum index
        checkpoint_dict = dict()
        for subitem in checkpoint_path.iterdir():
            reg_match = re.search(r"(model\.ckpt-)(\d+)(.*)$", str(subitem))
            if reg_match:
                index = int(reg_match.group(2))
                checkpoint_dict[index] = reg_match.group(1) + reg_match.group(2)
        checkpoint_path = checkpoint_path / checkpoint_dict[max(checkpoint_dict)]
print("===> Path to checkpoint: {}".format(checkpoint_path))

tensor_dict = chkp.load_tensors_in_checkpoint_file(str(checkpoint_path), tensor_name='', all_tensors=True)
print(tensor_dict["reduction_p/dim_reduction_kernel"])
#Equal to highlight_score: if P: F = loadmat(file_name)['reduction_p_dim_reduction_kernel']
print(tensor_dict["reduction_n/dim_reduction_kernel"]) #
#Equal to highlight_score: else: F = loadmat(file_name)['reduction_n_dim_reduction_kernel']
regions = pd.read_csv("dataset/HCP/region_function.csv",header=None) ## region_file_path
def highlight_score(region_name_1,region_name_2,regions,F):
    '''regions: file that stores the regions, F the matrix got after trained'''
    F[F<0] = 0 # in the training, we used relu to filter out the negative values
    S=0
    if region_name_1 == region_name_2:
        F_red = F[regions[1]==region_name_1,:]
        L = np.shape(F_red)[0]**2/2
    else:
        F_red_1 = F[regions[1]==region_name_1,:]
        F_red_2 = F[regions[1]==region_name_2,:]
        L = np.shape(F_red_1)[0]*np.shape(F_red_2)[0]/2
    for i in range(0,np.shape(F)[1]):
        for j in range(i+1,np.shape(F)[1]):
            if region_name_1==region_name_2:
                out_prod = np.outer(F_red[:,i],F_red[:,j])
                np.fill_diagonal(out_prod,0)
                S += out_prod
            else:
                S += np.outer(F_red_1[:,i],F_red_2[:,j])+np.outer(F_red_1[:,j],F_red_2[:,i])
    return np.sum(S)/L
region_list = regions[1].unique()
dict_s = {}
for i in region_list:
        v = highlight_score(i, i, regions, tensor_dict["reduction_p/dim_reduction_kernel"]) #### you can change to negative matrix using reduction_n/dim_reduction_kernel
        dict_s[i] = v
sorted_same_reg = sorted(dict_s,key=dict_s.get,reverse=True)
with open("within_regions.txt",'w') as f:
    for item in sorted_same_reg:
        f.write(str(item)+'\n')
dict_d = {}
for i in combinations(region_list,2):
        v = highlight_score(i[0], i[1], regions, tensor_dict["reduction_p/dim_reduction_kernel"])
        dict_d[i] = v
sorted_cross_reg = sorted(dict_d,key=dict_d.get,reverse=True)
with open("across_regions.txt",'w') as f:
    for item in sorted_cross_reg:
        f.write(str(item)+'\n')

    


