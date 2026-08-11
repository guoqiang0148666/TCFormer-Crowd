import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''
try:
    shanghaiAtrain_path = '/root/autodl-tmp/guo/datasets/ShanghaiTech/part_A_final/train_data/images_crop_224c/'
    shanghaiAtest_path = '/root/autodl-tmp/guo/datasets/ShanghaiTech/part_A_final/test_data/images_crop_224c/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train_224c.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test_224c.npy', test_list)

    print("generate ShanghaiA image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiA datasets path is wrong. Please check you path.")


try:
    shanghaiBtrain_path = '/root/autodl-tmp/guo/datasets/ShanghaiTech/part_B_final/train_data/images_crop_224c/'
    shanghaiBtest_path = '/root/autodl-tmp/guo/datasets/ShanghaiTech/part_B_final/test_data/images_crop_224c/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train_224c.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test_224c.npy', test_list)
    print("Generate ShanghaiB image list successfully", len(train_list), len(test_list))
except:
    print("The ShanghaiB datasets path is wrong. Please check your path.")
