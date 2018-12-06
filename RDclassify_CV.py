import os
import json
import numpy as np
import matplotlib.pyplot as plt
import initiate.config as config
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


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
    save_figure_path = os.path.join(
        config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path, "driving date test" + ".png"))
    plt.close()


def data_process_for_classify(link_level_dict, driving_time_dict):
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

    # construct rd_dict
    RD_dict = {}
    for driver_id in link_level_dict.keys():
        local_driving_dict = link_level_dict[driver_id]
        if driver_id not in RD_dict.keys():
            RD_dict[driver_id] = {}
        for date in local_driving_dict.keys():
            local_driver_link_dict = local_driving_dict[date]
            if date not in RD_dict[driver_id].keys():
                RD_dict[driver_id][date] = []
            for link_set_string in total_link_set:
                if link_set_string in local_driver_link_dict.keys():
                    distance = local_driver_link_dict[link_set_string]['distance']
                    RD_dict[driver_id][date].append(distance)
                else:
                    RD_dict[driver_id][date].append(0.0)
    for driver_id in driving_time_dict.keys():
        if driver_id not in RD_dict.keys():
            print(
                "Error! driver id not in RD_dict. driving_time_dict not match with link_level_dict!")
            exit()
        local_driving_dict = driving_time_dict[driver_id]
        svd_driver_dict = RD_dict[driver_id]
        for date in local_driving_dict.keys():
            local_driving_time_list = local_driving_dict[date]
            # some driver may have driving time but don't have link info
            # *************  this problem needs further discuss *******************
            if date not in svd_driver_dict.keys():
                # print(date)
                # print(driver_id)
                # print("Error! driver date not in RD_dict. driving_time_dict not match with link_level_dict!")
                # exit()
                zero_distance_in_all_link_set = [0.0] * len(total_link_set)
                RD_dict[driver_id][date] = zero_distance_in_all_link_set
            for travel_time in local_driving_time_list:
                RD_dict[driver_id][date].append(travel_time)
    return RD_dict


def RandomForest(RD_dict, experiment_num=100, plot_correction_flag=True):
    """

    :param RD_dict:
    :param experiment_num:
    :param plot_correction_flag:
    :return:
    """
    # Construct driver_id--label dict and driver_num
    driver_num = len(RD_dict.keys())
    driver_id2label_dict = {}
    count = 0
    for driver_id in RD_dict.keys():
        driver_id2label_dict[driver_id] = count
        count += 1

    # Construct SVD matrix
    data_list = []
    label_list = []
    for driver_id in RD_dict.keys():
        local_driving_dict = RD_dict[driver_id]
        for date in local_driving_dict.keys():
            sample_driving_data = local_driving_dict[date]
            data_list.append(sample_driving_data)
            label_list.append(driver_id2label_dict[driver_id])
    svd_matrix = (np.array(data_list)).transpose()
    Sigma = np.cov(svd_matrix.T, rowvar=False)
    D, V = np.linalg.eig(Sigma)
    k = 200
    projecting_v = V[:, np.argsort(D)[::-1][:k]]
    print('projecting_v.shape:', projecting_v.shape)
    #svd_matrix = svd_matrix.T.dot(projecting_v).T
    svd_matrix_train, svd_matrix_test, label_train, label_test = \
        divide_data_2_train_test(RD_dict, svd_matrix, label_list)
    # Set the parameters by cross-validation
    tuned_parameters = {'criterion': ['gini'],
                        'n_estimators': np.arange(5, 51, 5),
                        'max_depth': np.arange(16, 40, 5),
                        'min_samples_split': np.arange(2, 11, 2),
                        'random_state': [0]
                        }

    clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=10)
    clf.fit(svd_matrix_train.transpose(), label_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = label_test, clf.predict(svd_matrix_test.transpose())
    print(classification_report(y_true, y_pred))
    print()


def test(correction_table):
    np.savetxt("correction tabel RD.csv", correction_table, delimiter=",",
               fmt='%10.5f')


def plot_accuracy_figure(pcorr_list):
    mean_accuracy = [sum(pcorr_list) / len(pcorr_list)] * \
        (len(pcorr_list) + 10)
    print("mean accuracy", mean_accuracy[0])
    x = [i for i in np.arange(0, 100, 1)]
    x1 = [i for i in np.arange(-5, 105, 1)]
    f1 = plt.plot(x, pcorr_list, color='k')
    f2 = plt.plot(x1, mean_accuracy, 'b--')
    plt.legend(f2, "mean accuracy")
    plt.title("Accuracy in 100 times experiment, # of test data: 11")
    # plt.grid(linewidth=0.3)
    plt.xlabel("experiment idx")
    plt.ylabel("accuracy")
    # plt.show()
    save_figure_path = os.path.join(
        config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path,
                             "Random Forest 11 test data " + ".png"))
    plt.close()


def plot_correction_table(correction_table):
    plt.figure(figsize=[14, 4])
    plt.imshow(correction_table, cmap='gray', aspect='auto')
    plt.colorbar(ticks=range(2), label="classify correct or not")
    plt.title("1:correct 0:incorrect")
    plt.xlabel("Experiment idx")
    plt.ylabel("Driver idx")
    save_figure_path = os.path.join(
        config.figures_folder, config.Learn2classify_test_folder)
    if not os.path.exists(save_figure_path):
        os.mkdir(save_figure_path)
    plt.savefig(os.path.join(save_figure_path,
                             "Random Forest correction table" + ".png"))
    plt.close()


def divide_data_2_train_test(RD_dict, svd_matrix, label_list):
    # divid svd_matrix into train matrix and test matrix
    driver_num = len(RD_dict.keys())
    test_idx = []  # store the index of test data
    svd_matrix_test = None
    for i in range(driver_num):
        data_idx = [j for j, x in enumerate(label_list) if x == i]
        start_idx = data_idx[0]
        end_idx = data_idx[-1]
        idx_list = np.arange(start_idx, end_idx + 1, 1)
        # randomly choose one data as test sample
        chosen_test_idx = random.choice(idx_list)
        test_idx.append(chosen_test_idx)
        dim = svd_matrix[:, 0].shape[0]
        local_data = np.reshape(svd_matrix[:, chosen_test_idx], (dim, 1))
        if svd_matrix_test is None:
            svd_matrix_test = local_data
        else:
            svd_matrix_test = np.hstack((svd_matrix_test, local_data))
    svd_matrix_train = np.delete(svd_matrix, test_idx, 1)
    label_train = label_list[:]
    label_test = []
    for x in test_idx[::-1]:
        label_test.append(label_list[x])
        label_train = label_train[:x] + label_train[x + 1:]
    label_test = label_test[::-1]
    return svd_matrix_train, svd_matrix_test, label_train, label_test


def predict_RandomForest(classifier, test_matrix):
    predicted_label = classifier.predict(test_matrix)
    print(predicted_label)
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


def RDclassify(folder):
    # load the travel distance and frequency data
    # travel_distance_frequency_dict = load_json_file(folder, config.grid_travel_info)
    # print("Now plot the travel distance and frequency figures...")
    # # generate_grid_travel_figures(travel_distance_frequency_dict)
    #
    # # load the driving time data
    all_driving_time_dict = load_json_file(folder, config.driving_time_info)
    # print(driving_time_dict['10125'].keys())

    # load the link-level data
    all_link_level_dict = load_json_file(folder, config.link_level)
    # print(link_level_dict['10125'].keys())
    driving_time_dict = {}
    link_level_dict = {}
    for key in all_driving_time_dict.keys():
        if len(all_driving_time_dict[key].keys()) > 10:
            driving_time_dict[key] = all_driving_time_dict[key]
            link_level_dict[key] = all_link_level_dict[key]

    RD_dict = data_process_for_classify(
        link_level_dict, driving_time_dict)
    RandomForest(RD_dict,  experiment_num=100)


if __name__ == '__main__':
    RDclassify("ann_arbor")
