import numpy as np
import os
import cv2
from PIL import Image



global_address = '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/data/vision_data'

class0_ls = []
class1_ls = []
class2_ls = []
class3_ls = []

for file in os.listdir(global_address):
    if 'npy' in file:
        rgbd3_dict = np.load(os.path.join(global_address, file), allow_pickle=True).item()
        img3 = rgbd3_dict['img_ls']
        depth3 = rgbd3_dict['depth_ls']
        rgbd3 = np.concatenate([img3, depth3.reshape(depth3.shape[0], depth3.shape[1], depth3.shape[2], 1)],axis=-1)
        class3_ls.append(rgbd3)
    else:
        rgbd0_dict = np.load(os.path.join(global_address, file, 'class0.npy'), allow_pickle=True).item()
        rgbd1_dict = np.load(os.path.join(global_address, file, 'class1.npy'), allow_pickle=True).item()
        rgbd2_dict = np.load(os.path.join(global_address, file, 'class2.npy'), allow_pickle=True).item()

        img0 = rgbd0_dict['img_ls']
        depth0 = rgbd0_dict['depth_ls']
        rgbd0 = np.concatenate([img0, depth0.reshape(depth0.shape[0], depth0.shape[1], depth0.shape[2], 1)],axis=-1)
        class0_ls.append(rgbd0)

        img1 = rgbd1_dict['img_ls']
        depth1 = rgbd1_dict['depth_ls']
        rgbd1 = np.concatenate([img1, depth1.reshape(depth1.shape[0], depth1.shape[1], depth1.shape[2], 1)],axis=-1)
        class1_ls.append(rgbd1)

        img2 = rgbd2_dict['img_ls']
        depth2 = rgbd2_dict['depth_ls']
        rgbd2 = np.concatenate([img2, depth2.reshape(depth2.shape[0], depth2.shape[1], depth2.shape[2], 1)],axis=-1)
        class2_ls.append(rgbd2)

class0 = np.concatenate(class0_ls,axis=0)
class1 = np.concatenate(class1_ls,axis=0)
class2 = np.concatenate(class2_ls,axis=0)
class3 = np.concatenate(class3_ls,axis=0)

# np.save(os.path.join('/home/zxr/Documents/Github/DP_ur5e_open_door/CLVE','processed_data/class0'),class0)
# np.save(os.path.join('/home/zxr/Documents/Github/DP_ur5e_open_door/CLVE','processed_data/class1'),class1)
# np.save(os.path.join('/home/zxr/Documents/Github/DP_ur5e_open_door/CLVE','processed_data/class2'),class2)
# np.save(os.path.join('/home/zxr/Documents/Github/DP_ur5e_open_door/CLVE','processed_data/class3'),class3)

all_data = np.concatenate([class0, class1, class2, class3],axis=0)

for i in range(len(all_data)):
    img = all_data[i][:,:,:-1]
    depth = all_data[i][:,:,-1]
    Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)).save(os.path.join('/home/zxr/Documents/Github/CLVE/processed_data/img','%d.png' %i))
    Image.fromarray(depth, 'L').save(os.path.join('/home/zxr/Documents/Github/CLVE/processed_data/depth','%d.png' %i))

    print('a')



