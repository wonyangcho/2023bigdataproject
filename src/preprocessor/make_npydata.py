import os
import numpy as np

if not os.path.exists('./npydata'):
    os.makedirs('./npydata')


'''please set your dataset path'''
try:
    shanghaiAtrain_path = '/work/wycho/project/2023bigdataproject/data/ShanghaiTech/part_A_final/train_data/images_crop/'
    shanghaiAtest_path = '/work/wycho/project/2023bigdataproject/data/ShanghaiTech/part_A_final/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiAtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiAtrain_path + filename)

    train_list.sort()
    np.save('./npydata/ShanghaiA_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiAtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiAtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiA_test.npy', test_list)

    print("generate ShanghaiA image list successfully",
          len(train_list), len(test_list))
except Exception as e:
    print(f"The ShanghaiA dataset error {e}")


try:
    shanghaiBtrain_path = '/work/wycho/project/2023bigdataproject/data/ShanghaiTech/part_B_final/train_data/images_crop/'
    shanghaiBtest_path = '/work/wycho/project/2023bigdataproject/data/ShanghaiTech/part_B_final/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(shanghaiBtrain_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(shanghaiBtrain_path + filename)
    train_list.sort()
    np.save('./npydata/ShanghaiB_train.npy', train_list)

    test_list = []
    for filename in os.listdir(shanghaiBtest_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(shanghaiBtest_path + filename)
    test_list.sort()
    np.save('./npydata/ShanghaiB_test.npy', test_list)
    print("Generate ShanghaiB image list successfully",
          len(train_list), len(test_list))
except Exception as e:
    print(f"The ShanghaiB dataset error {e}")


try:
    qnrf_train_path = '/work/wycho/project/2023bigdataproject/data/UCF-QNRF_ECCV18/train_data/images_crop/'
    qnrf_test_path = '/work/wycho/project/2023bigdataproject/data/UCF-QNRF_ECCV18/test_data/images_crop/'

    train_list = []
    for filename in os.listdir(qnrf_train_path):
        if filename.split('.')[1] == 'jpg':
            train_list.append(qnrf_train_path + filename)
            print(qnrf_train_path + filename)

    train_list.sort()
    np.save('./npydata/qnrf_train.npy', train_list)

    test_list = []
    for filename in os.listdir(qnrf_test_path):
        if filename.split('.')[1] == 'jpg':
            test_list.append(qnrf_test_path + filename)
    test_list.sort()
    np.save('./npydata/qnrf_test.npy', test_list)

    print("generate qnrf image list successfully",
          len(train_list), len(test_list))
except Exception as e:
    print(f"The qnrf dataset error {e}")
