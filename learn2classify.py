import os
import json
import numpy as np
import matplotlib.pyplot as plt
import initiate.config as config


def load_json_file(folder, file_name):
    file_name = os.path.join(folder, file_name)

    with open(file_name, "r") as json_file:
        travel_distance_frequency_dict = json.load(json_file)
    return travel_distance_frequency_dict


def generate_grid_travel_figures(travel_distance_frequency_dict):
    travel_distance_dict = travel_distance_frequency_dict["travel_distance"]
    travel_frequency_dict = travel_distance_frequency_dict["frequency"]

    for driver_id in travel_distance_dict.keys():
        local_travel_distance_dict = travel_distance_dict[driver_id]
        local_travel_frequency_dict = travel_frequency_dict[driver_id]

        driver_distance_matrix = None
        driver_frequency_matrix = None

        for time_string in local_travel_distance_dict.keys():
            local_travel_distance_matrix = np.array(local_travel_distance_dict[time_string])
            local_travel_frequency_matrix = np.array(local_travel_frequency_dict[time_string])

            if driver_distance_matrix is None:
                driver_distance_matrix = local_travel_distance_matrix
                driver_frequency_matrix = local_travel_frequency_matrix
            else:
                driver_frequency_matrix += local_travel_frequency_matrix
                driver_distance_matrix += local_travel_distance_matrix
        plt.figure(figsize=[12, 6])
        plt.subplot(121)
        plt.imshow(driver_distance_matrix, cmap="gray")
        plt.title("Travel distance matrix (m)")
        plt.xlabel("longitude index")
        plt.ylabel("latitude index")

        plt.subplot(122)
        plt.imshow(driver_frequency_matrix, cmap="gray")
        plt.title("Travel frequency matrix")
        plt.xlabel("longitude index")
        plt.ylabel("latitude index")
        plt.suptitle("Driver " + str(driver_id))

        save_figure_path = os.path.join(config.figures_folder, config.grid_figures)
        if not os.path.exists(save_figure_path):
            os.mkdir(save_figure_path)
        plt.savefig(os.path.join(save_figure_path, str(driver_id) + ".png"))
        plt.close()


def generate_driving_time_histogram(driving_time_dict):
    for driver_id in driving_time_dict.keys():
        driver_matrix = None
        local_driving_dict = driving_time_dict[driver_id]
        for time_string in local_driving_dict.keys():
            local_driving_vec = local_driving_dict[time_string]

            if driver_matrix is None:
                length = len(local_driving_vec)
                driver_matrix = np.zeros((30, length))

            date_idx = int(time_string.split("-")[-1]) - 1
            driver_matrix[date_idx, :] = np.array(local_driving_vec)
        plt.figure(figsize=[6, 7])
        plt.imshow(driver_matrix, cmap="gray")
        plt.title("Driver id: " + str(driver_id))
        plt.xlabel("Time in a day")
        plt.ylabel("Day in a month")
        save_figure_path = os.path.join(config.figures_folder, config.driving_time_folder)
        if not os.path.exists(save_figure_path):
            os.mkdir(save_figure_path)
        plt.savefig(os.path.join(save_figure_path, str(driver_id) + ".png"))
        plt.close()


def learn2classify(folder):
    # load the travel distance and frequency data
    travel_distance_frequency_dict = load_json_file(folder, config.grid_travel_info)
    print("Now plot the travel distance and frequency figures...")
    # generate_grid_travel_figures(travel_distance_frequency_dict)

    # load the driving time data
    driving_time_dict = load_json_file(folder, config.driving_time_info)
    print("Now plot the driving time histogram of drivers...")
    # generate_driving_time_histogram(driving_time_dict)

    # load the link-level data
    link_level_dict = load_json_file(folder, config.link_level)
    print(link_level_dict.keys())


if __name__ == '__main__':
    learn2classify("ann_arbor")

