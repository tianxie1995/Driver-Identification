import os
import json
import numpy as np
import matplotlib.pyplot as plt
import initiate.config as config
import random


def load_json_file(folder, file_name):
    file_name = os.path.join(folder, file_name)

    with open(file_name, "r") as json_file:
        travel_distance_frequency_dict = json.load(json_file)
    return travel_distance_frequency_dict


def driving_date_figure(driving_time_dict):
    """
    Show How many days each driver drives their vehicle. Used to
    determine train/test data ratio
    :return:
    """
    driver_num = len(driving_time_dict.keys())
    driver_idx = 0
    driving_date_matrix = None
    for driver_id in driving_time_dict.keys():
        local_driving_dict = driving_time_dict[driver_id]
        for time_string in local_driving_dict.keys():

            if driving_date_matrix is None:
                driving_date_matrix = np.zeros((driver_num, 30))

            date_idx = int(time_string.split("-")[-1]) - 1
            driving_date_matrix[driver_idx, date_idx] = 1
        driver_idx += 1

    # plt.figure(figsize=[6, 7])
    plt.imshow(driving_date_matrix, cmap="gray_r")
    plt.colorbar()
    plt.title("Driving date display")
    plt.xlabel("Day in a month")
    plt.ylabel("Driver sequence")
    save_figure_path = os.path.join(config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "driving date test" + ".png"))
    plt.close()


def data_process_for_SVD_classify(link_level_dict, driving_time_dict):
    """
    Construct each driver's daily driving data as a vector for SVD classify
    :param link_level_dict:
    :return:
    """
    # construct total_link_set
    total_link_set = []
    for driver_id in link_level_dict.keys():
        local_driving_dict = link_level_dict[driver_id]
        for time_string in local_driving_dict.keys():
            local_driver_link_dict = local_driving_dict[time_string]
            for link_string in local_driver_link_dict.keys():
                local_link = link_string
                if local_link not in total_link_set:
                    total_link_set.append(link_string)

    # construct svd_dict
    SVD_dict = {}
    for driver_id in link_level_dict.keys():
        local_driving_dict = link_level_dict[driver_id]
        if driver_id not in SVD_dict.keys():
            SVD_dict[driver_id] = {}
        for date in local_driving_dict.keys():
            local_driver_link_dict = local_driving_dict[date]
            if date not in SVD_dict[driver_id].keys():
                SVD_dict[driver_id][date] = []
            for link_set_string in total_link_set:
                if link_set_string in local_driver_link_dict.keys():
                    distance = local_driver_link_dict[link_set_string]['distance']
                    SVD_dict[driver_id][date].append(distance)
                else:
                    SVD_dict[driver_id][date].append(0.0)
    for driver_id in driving_time_dict.keys():
        if driver_id not in SVD_dict.keys():
            print("Error! driver id not in SVD_dict. driving_time_dict not match with link_level_dict!")
            exit()
        local_driving_dict = driving_time_dict[driver_id]
        svd_driver_dict = SVD_dict[driver_id]
        for date in local_driving_dict.keys():
            local_driving_time_list = local_driving_dict[date]
            # some driver may have driving time but don't have link info
            # *************  this problem needs further discuss *******************
            if date not in svd_driver_dict.keys():
                # print(date)
                # print(driver_id)
                # print("Error! driver date not in SVD_dict. driving_time_dict not match with link_level_dict!")
                # exit()
                zero_distance_in_all_link_set = [0.0] * len(total_link_set)
                SVD_dict[driver_id][date] = zero_distance_in_all_link_set
            for travel_time in local_driving_time_list:
                SVD_dict[driver_id][date].append(travel_time)
    # print((len(SVD_dict['10125']['2016-6-22'])))
    return SVD_dict


def Nearest_Subspace_algorithm(SVD_dict, plot_singluar_value=False, ktrain=10,
                               experiment_num=100, plot_correction_flag=True, save_correction_csv=True,
                               num_test_data=2, special_tag=""):
    """

    :param SVD_dict:
    :param plot_singluar_value:
    :param ktrain:
    :param experiment_num:
    :param plot_correction_flag:
    :return:
    """
    # show how many data per driver have
    # for driver_id in SVD_dict.keys():
    #     local_driving_dict = SVD_dict[driver_id]
    #     print(driver_id, len(local_driving_dict))

    # delete drivers that have limited data
    # SVD_dict.pop('10150', None)
    # SVD_dict.pop('10160', None)
    # SVD_dict.pop('501', None)
    # SVD_dict.pop('502', None)
    # SVD_dict.pop('552', None)
    # SVD_dict.pop('551', None)
    # SVD_dict.pop('10125', None)

    # Construct driver_id--label dict and driver_num
    driver_num = len(SVD_dict.keys())
    driver_id2label_dict = {}
    count = 0
    for driver_id in SVD_dict.keys():
        driver_id2label_dict[driver_id] = count
        count += 1

    # Construct SVD matrix
    data_list = []
    label_list = []
    for driver_id in SVD_dict.keys():
        local_driving_dict = SVD_dict[driver_id]
        for date in local_driving_dict.keys():
            sample_driving_data = local_driving_dict[date]
            data_list.append(sample_driving_data)
            label_list.append(driver_id2label_dict[driver_id])
    svd_matrix = (np.array(data_list)).transpose()
    U, sigma, V = np.linalg.svd(svd_matrix, full_matrices=False)

    # Do scaling for feature
    svd_matrix = scaling2(svd_matrix)

    # print(label_list)

    # singular value plot
    if plot_singluar_value is True:
        plot_singluar_value_figure(sigma)

    # Nearest ss
    pcorr_list = []
    correction_table1 = np.zeros((num_test_data * driver_num, experiment_num))
    correction_table2 = np.zeros((num_test_data * driver_num, experiment_num))
    for iteration in range(experiment_num):
        # do lots of times
        # divid svd_matrix into train matrix and test matrix
        svd_matrix_train, svd_matrix_test, label_train, label_test = \
            divide_data_2_train_test(SVD_dict, svd_matrix, label_list, num_test_data)
        # Find nearest sub-space
        train_U = learn_nearest_ss(SVD_dict, svd_matrix_train, label_train, ktrain)
        # Classify nearest ss
        predicted_label = classify_nearest_ss(train_U, svd_matrix_test)
        # Calculate correct percentage
        pcorr = pcorrect(predicted_label, label_test)
        pcorr_list.append(pcorr)
        # print("predicted_label: ", predicted_label)
        # print("true label: ", label_test)
        # print(pcorr)

        # construct correction table1 and 2
        tmp1 = []
        tmp2 = []
        for i in range(len(predicted_label)):
            if predicted_label[i] == label_test[i]:
                tmp1.append(1)
                tmp2.append(-1)
            else:
                tmp1.append(0)
                tmp2.append(predicted_label[i])
        correction_table1[:, iteration] = tmp1
        correction_table2[:, iteration] = tmp2

    # plot correction table
    if plot_correction_flag is True:
        plot_correction_table(correction_table1, num_test_data, special_tag)
    # plot accuracy for specific driver
    # calculate_accuracy_for_specific_driver(correction_table1)
    # save correction csv version 2
    if save_correction_csv is True:
        save_correction_table_to_csv(correction_table2, num_test_data, special_tag)

    # plot accuracy figure
    plot_accuracy_figure(pcorr_list, num_test_data, experiment_num, special_tag)


def scaling1(svd_matrix):
    """
    scaling for each feature by variance
    :param svd_matrix: (N* dimension) data
    :return: scaling for each feature to have same variance 1
    """
    [N, d] = svd_matrix.shape
    # centering
    m = np.mean(svd_matrix, axis=0)
    svd_matrix = svd_matrix - m

    # scaling
    for dim in range(d):
        # local_list = svd_matrix[:, dim]
        local_list = np.reshape(svd_matrix[:, dim], (N, 1))
        square_sum = sum(map(lambda x: pow(x, 2), local_list))
        var = pow((1 / (N - 1)) * square_sum, 0.5)
        # print(var)
        svd_matrix[:, dim] = list(map(lambda x: x / var, local_list))
    return svd_matrix


def scaling2(svd_matrix):
    """
    scaling for each feature to [0,1]
    :param svd_matrix: (N* dimension) data
    :return: scaling for each feature to have same variance 1
    """
    [N, d] = svd_matrix.shape
    # centering
    m = np.mean(svd_matrix, axis=0)
    svd_matrix = svd_matrix - m

    # scaling
    for dim in range(d):
        # local_list = svd_matrix[:, dim]
        local_list = svd_matrix[:, dim]
        max_l = max(local_list)
        min_l = min(local_list)
        diff = max_l - min_l
        for i in range(N):
            svd_matrix[i, dim] = (svd_matrix[i, dim] - min_l) / diff
    return svd_matrix


def scaling3(svd_matrix):
    """
    scaling for each data rather than feature
    :param svd_matrix: (N* dimension) data
    :return: scaling for each data to have same variance 1
    """
    [N, d] = svd_matrix.shape
    # centering
    m = np.mean(svd_matrix, axis=0)
    svd_matrix = svd_matrix - m

    # scaling
    for num in range(N):
        # local_list = svd_matrix[:, dim]
        local_list = svd_matrix[num, :]
        square_sum = sum(map(lambda x: pow(x, 2), local_list))
        var = pow((1 / (N - 1)) * square_sum, 0.5)
        # print(var)
        svd_matrix[num, :] = list(map(lambda x: x / var, local_list))
    return svd_matrix


def calculate_accuracy_for_specific_driver(correction_table1):
    acc_list = []
    x = [16, 14, 16, 24, 19, 20, 24]
    for idx in range(len(x)):
        local_correction_data = correction_table1[idx, :]
        correct_num = sum(local_correction_data)
        acc = correct_num / correction_table1.shape[1]
        acc_list.append(acc)
        print(idx, "# of data: ", x[idx], " acc: ", acc)
    together = zip(x, acc_list)
    sorted_together = sorted(together)

    list_1_sorted = [x[0] for x in sorted_together]
    list_2_sorted = [x[1] for x in sorted_together]

    plt.plot(list_1_sorted, list_2_sorted)
    plt.title("# of data vs accuracy")
    plt.xlabel("driver's # of data")
    plt.ylabel("accuracy")
    plt.show()


def save_correction_table_to_csv(correction_table, num_test_data, special_tag):
    filename = "correction table" + str(num_test_data) + special_tag + ".csv"

    save_figure_path = os.path.join(config.figures_folder, config.correction_table_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    path = os.path.join(save_figure_path, filename)
    np.savetxt(path, correction_table, delimiter=",",
               fmt='%10.5f')


def plot_accuracy_figure(pcorr_list, num_test_data, experiment_num, special_tag):
    mean_accuracy = [sum(pcorr_list) / len(pcorr_list)] * (len(pcorr_list) + 40)
    print("mean accuracy", mean_accuracy[0])
    x = [i for i in np.arange(0, experiment_num, 1)]
    x1 = [i for i in np.arange(-20, experiment_num + 20, 1)]
    f1 = plt.plot(x, pcorr_list, color='k')
    f2 = plt.plot(x1, mean_accuracy, 'b--')
    plt.legend(f2, "mean accuracy")
    plt.title(
        "Accuracy in " + str(experiment_num) + " times experiment, # of test data per driver: " + str(num_test_data))
    # plt.grid(linewidth=0.3)
    plt.xlabel("experiment idx")
    plt.ylabel("accuracy")
    # plt.show()
    save_figure_path = os.path.join(config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "Nearest subspace num_test_data = " + str(num_test_data)
                             + special_tag + ".png"))
    plt.close()


def plot_correction_table(correction_table, num_test_data, special_tag):
    plt.figure(figsize=[14, 4])
    plt.imshow(correction_table, cmap='gray', aspect='auto')
    plt.colorbar(ticks=range(2), label="classify correct or not")
    # plt.yticks(np.arange(driver_num))
    plt.title("1:correct 0:incorrect num_test_data = " + str(num_test_data))
    plt.xlabel("Experiment idx")
    plt.ylabel("Driver idx")
    save_figure_path = os.path.join(config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(
        os.path.join(save_figure_path, "correction table num_test_data = " + str(num_test_data) + special_tag + ".png"))
    plt.close()


def plot_singluar_value_figure(sigma):
    x = [i for i in np.arange(1, 173, 1)]
    plt.scatter(x, sigma, color='k', s=10)
    plt.title("Singular value")
    plt.grid(linewidth=0.3)
    plt.xlabel("dimension")
    plt.ylabel("sigma")
    plt.show()
    save_figure_path = os.path.join(config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "singular value for Data matrix" + ".png"))
    plt.close()


def divide_data_2_train_test(SVD_dict, svd_matrix, label_list, num_test_data):
    # divide svd_matrix into train matrix and test matrix
    driver_num = len(SVD_dict.keys())
    test_idx = []  # store the index of test data
    svd_matrix_test = None
    for i in range(driver_num):
        data_idx = [j for j, x in enumerate(label_list) if x == i]
        start_idx = data_idx[0]
        end_idx = data_idx[-1]
        idx_list = np.arange(start_idx, end_idx + 1, 1)
        # randomly choose num_test_data as test sample
        # local test idx used to append svd_matrix_test
        local_test_idx = []
        for num_id in range(num_test_data):
            chosen_test_idx = random.choice(idx_list)
            # make sure the test data not repeat
            while True:
                if chosen_test_idx not in test_idx:
                    break
                else:
                    chosen_test_idx = random.choice(idx_list)
            test_idx.append(chosen_test_idx)
            local_test_idx.append(chosen_test_idx)
        local_test_idx.sort()
        dim = svd_matrix[:, 0].shape[0]
        for num_id in range(num_test_data):
            local_data = np.reshape(svd_matrix[:, local_test_idx[num_id]], (dim, 1))
            if svd_matrix_test is None:
                svd_matrix_test = local_data
            else:
                svd_matrix_test = np.hstack((svd_matrix_test, local_data))
    # sort test_idx to ascending order
    test_idx.sort()
    svd_matrix_train = np.delete(svd_matrix, test_idx, 1)
    label_train = label_list[:]
    label_test = []
    for x in test_idx[::-1]:
        label_test.append(label_list[x])
        label_train = label_train[:x] + label_train[x + 1:]
    label_test = label_test[::-1]
    return svd_matrix_train, svd_matrix_test, label_train, label_test


def learn_nearest_ss(SVD_dict, svd_matrix, label_list, ktrain):
    """

    :param SVD_dict:
    :param svd_matrix: train data matrix
    :param label_list: train data label list
    :param ktrain: number of singular value to choose
    :return: U
    """
    driver_num = len(SVD_dict.keys())
    driving_data_num_list = []

    for i in range(driver_num):
        local_num_data = len([j for j, x in enumerate(label_list) if x == i])
        driving_data_num_list.append(local_num_data)
    min_data_num = min(driving_data_num_list)
    if min_data_num < ktrain:
        print("some drivers data less than ktrain")
        # ****** Actually this maybe not enough, because ********
        # local_svd_matrix maybe not full rank, need further discuss
        # ktrain = min_data_num - 1
        ktrain = min_data_num
    print("k = ", str(ktrain))
    print("driving_data_num_train: ", driving_data_num_list)

    train_U = None
    for i in range(driver_num):
        data_idx = [j for j, x in enumerate(label_list) if x == i]
        start_idx = data_idx[0]
        end_idx = data_idx[-1]
        local_svd_matrix = svd_matrix[:, start_idx:end_idx + 1]
        U, sigma, V = np.linalg.svd(local_svd_matrix, full_matrices=False)
        # print(U.shape)
        U_k = U[:, 0:ktrain]
        # print(U_k.shape)
        if train_U is None:
            train_U = U_k
        else:
            train_U = np.dstack((train_U, U_k))
        # print(train_U.shape)
    # print(train_U[:, :, 0])
    return train_U


def classify_nearest_ss(train_U, test_matrix):
    # p is # of test data, d is # of classes
    p = test_matrix.shape[1]
    d = train_U.shape[2]
    err = np.zeros((d, p))
    for j in range(d):
        Uj = train_U[:, :, j]
        local_diff_square = np.square(test_matrix - Uj @
                                      (Uj.transpose() @ test_matrix))
        err[j, :] = np.sum(local_diff_square, axis=0)

    predicted_label = np.argmin(err, axis=0)
    return predicted_label


def pcorrect(predicted_label, test_label):
    count = 0
    for i in range(len(predicted_label)):
        predict = predicted_label[i]
        real = test_label[i]
        if predict == real:
            count += 1
        else:
            continue
    percent = count / len(predicted_label)
    return percent


def SVD2classify(folder):
    # load the travel distance and frequency data
    # travel_distance_frequency_dict = load_json_file(folder, config.grid_travel_info)
    # print("Now plot the travel distance and frequency figures...")
    # # generate_grid_travel_figures(travel_distance_frequency_dict)
    #
    # # load the driving time data
    driving_time_dict = load_json_file(folder, config.driving_time_info)
    # print(driving_time_dict['10125'].keys())

    # load the link-level data
    link_level_dict = load_json_file(folder, config.link_level)
    # print(link_level_dict['10125'].keys())

    # generate driving date figure
    # driving_date_figure(driving_time_dict)

    SVD_dict = data_process_for_SVD_classify(link_level_dict, driving_time_dict)
    Nearest_Subspace_algorithm(SVD_dict, plot_singluar_value=False,
                               ktrain=7, experiment_num=1000, plot_correction_flag=True,
                               save_correction_csv=True, num_test_data=1,
                               special_tag="after scaling 2")


if __name__ == '__main__':
    SVD2classify("ann_arbor")
