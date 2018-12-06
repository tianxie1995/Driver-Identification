import numpy as np
from tools.time_converter import timestamp2datetime
from tqdm import tqdm
from map_match.traj_process import get_distance


def generate_driving_heat_map(drivers, block_range, lat_grids=20, lon_grids=20, interval=15):
    """

    :param drivers:
    :param block_range:
    :param lat_grids:
    :param lon_grids:
    :param interval:
    :return:
    """
    latitude_interval = (block_range[1] - block_range[0]) / lat_grids
    longitude_interval = (block_range[3] - block_range[2]) / lon_grids

    time_frequency_matrix_dict = {}
    travel_distance_matrix_dict = {}

    for driver_id in tqdm(drivers.keys()):
        # iteration for trips of the same drivers
        driver = drivers[driver_id]
        if not (driver_id in time_frequency_matrix_dict.keys()):
            time_frequency_matrix_dict[driver_id] = {}
            travel_distance_matrix_dict[driver_id] = {}

        for trip_id in driver.trips.keys():
            trajectory_dict = driver.trips[trip_id].get_gps_time_stamp(interval)
            if trajectory_dict is None:
                continue

            latitudes = trajectory_dict["latitudes"]
            longitudes = trajectory_dict["longitudes"]
            time_stamp = trajectory_dict["timestamp"]

            for idx in range(len(latitudes) - 1):
                local_latitude = latitudes[idx]
                local_longitude = longitudes[idx]
                local_time_stamp = time_stamp[idx]

                next_latitude = latitudes[idx + 1]
                next_longitude = longitudes[idx + 1]

                date_time = timestamp2datetime(local_time_stamp / 1000.0)
                time_string = "-".join([str(val) for val in date_time[0:3]])

                if not (time_string in time_frequency_matrix_dict[driver_id].keys()):
                    # print("initiate", time_string)
                    time_frequency_matrix_dict[driver_id][time_string] = \
                        np.zeros((lat_grids, lon_grids))
                    travel_distance_matrix_dict[driver_id][time_string] =\
                        np.zeros((lat_grids, lon_grids))
                latitude_idx = int((local_latitude - block_range[0]) / latitude_interval)
                longitude_idx = int((local_longitude - block_range[2]) / longitude_interval)

                if 0 <= latitude_idx <= (lat_grids - 1):
                    if 0 <= longitude_idx <= (lon_grids - 1):
                        time_frequency_matrix_dict[driver_id][time_string][latitude_idx, longitude_idx] += 1
                        travel_distance_matrix_dict[driver_id][time_string][latitude_idx, longitude_idx] +=\
                            get_distance(local_latitude, next_latitude, local_longitude, next_longitude)

    # convert the matrix to list
    for driver_id in travel_distance_matrix_dict.keys():
        local_distance_dict = travel_distance_matrix_dict[driver_id]
        local_frequency_dict = time_frequency_matrix_dict[driver_id]

        for time_string in local_distance_dict:
            local_distance_matrix = local_distance_dict[time_string].tolist()
            local_frequency_matrix = local_frequency_dict[time_string].tolist()
            travel_distance_matrix_dict[driver_id][time_string] = local_distance_matrix
            time_frequency_matrix_dict[driver_id][time_string] = local_frequency_matrix
    travel_distance_and_frequency_dict = {"travel_distance": travel_distance_matrix_dict,
                                          "frequency": time_frequency_matrix_dict}
    return travel_distance_and_frequency_dict


