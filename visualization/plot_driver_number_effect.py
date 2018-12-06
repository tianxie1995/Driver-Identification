import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import initiate.config as config
from RDclassify import RandomForest

'''
To generate the needed matrix for ploting, the model func
def model():
    1. should return the pcorr_list: the list of accuracy for all experiments
    2. should do randomly selecting driver and train-test spliting inside each
        loop
'''


def random_choose_drivers(Data_dict, driver_num):
    '''
    generate data by randomly selecting certain number of drivers
    :param Data_dict: the dict of driver data, the key should be the driver id
    :param driver_num: the number of driver being randomly selected
    retrun
    :dict with selected driver datas
    '''
    selected_driver_list = random.sample(Data_dict.keys(), driver_num)
    out_Data_dict = {}
    for s in selected_driver_list:
        out_Data_dict[s] = Data_dict[s]
    return out_Data_dict


def generate_drive_accuracy_matrix(dict_org, num_of_exp_per_driverNum=100):
    '''
    generate drive_accuracy_matrix by randomly running model on difference driver number
    :param dict_org: the original dict of driver data, the key should be the driver id
    :param num_of_exp_per_driverNum: the number of experiments per diver number
    retrun
    :drive_acc_mat with data needed to plot
    '''
    drive_acc_mat = np.array([])
    for i in range(2, 12):
        pcorr_list = RandomForest(
            dict_org,  experiment_num=num_of_exp_per_driverNum, random_select_driver=True, driver_num=i)
        local_drive_acc_mat = np.hstack(
            (np.ones((num_of_exp_per_driverNum, 1))*i, np.array(pcorr_list).reshape(-1, 1)))
        print('local_drive_acc_mat.shape', local_drive_acc_mat.shape)
        if drive_acc_mat.shape[0] == 0:
            drive_acc_mat = np.array(local_drive_acc_mat)
        else:
            drive_acc_mat = np.vstack((drive_acc_mat, local_drive_acc_mat))
    return drive_acc_mat


def plot_driver_number_effect(drive_acc_mat):
    '''
    plot effect of changing driver number on average accuracy
    :param drive_acc_mat
    '''
    sns.set_context('talk')
    plt.figure(figsize=(10, 10))
    ax = sns.lineplot(x=drive_acc_mat[:, 0], y=drive_acc_mat[:, 1])
    plt.xticks(range(2, 12))
    plt.xlabel('Number of Classes(Drivers)')
    plt.ylabel('Accuracy')
    plt.title(
        'Accuracy with increasing number of drivers with 100 experiments', fontweight='bold')
    save_figure_path = os.path.join(
        config.figures_folder, config.driver_number_casestudy_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "casestudy" + ".png"))
    plt.close()
