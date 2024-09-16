import numpy as np
import torch
import os
import matplotlib.pyplot as plt

global_address = '/home/zxr/Documents/Github/CLVE/result'


similarity_all = []
similarity_diff_all = []

for file in os.listdir(global_address):
    if 'diff' not in file:
        similarity_tmp = np.load(os.path.join(global_address, file), allow_pickle=True)
        similarity_all.append(similarity_tmp)
    else:
        similarity_diff_tmp = np.load(os.path.join(global_address, file), allow_pickle=True)
        similarity_diff_all.append(similarity_diff_tmp)



similarity = np.concatenate(similarity_all, axis=0)
similarity_diff = np.concatenate(similarity_diff_all, axis=0)

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 6))
axes[0].hist(similarity, 12, edgecolor='black')
axes[0].set_xlim(40,65)
axes[0].set_xlabel('Similarity')
axes[0].set_ylabel('Counts')
axes[0].set_title('Image in Same Class')
axes[1].hist(similarity_diff, 80, edgecolor='black')
axes[1].set_xlim(40,65)
axes[1].set_xlabel('Similarity')
axes[1].set_ylabel('Counts')
axes[1].set_title('Image in Different Class')

fig.subplots_adjust(
    # left=0,right=0,top=0,bottom=0,
                    # wspace=0,
                    hspace=0.4)
plt.savefig('similarity_result.png')
# print('a')



