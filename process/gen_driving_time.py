import numpy as np

from tqdm import tqdm
from tools.time_converter import timestamp2datetime


def generate_driving_time_histogram(drivers, period=1, interval=15):
    """

    :param drivers:
    :param period: unit: 1h
    :param interval:
    :return:
    """
    time_intervals = int(24.0 / period)
    driving_time_dict = {}

    for driver_id in tqdm(drivers.keys()):
        # iteration for trips of the same drivers
        driver = drivers[driver_id]

        if not (driver_id in driving_time_dict.keys()):
            driving_time_dict[driver_id] = {}

        for trip_id in driver.trips.keys():
            trajectory_dict = driver.trips[trip_id].get_gps_time_stamp(interval)
            timestamp_list = trajectory_dict["timestamp"]

            for timestamp in timestamp_list:
                datetime = timestamp2datetime(timestamp / 1000.0)
                time_string = "-".join([str(val) for val in datetime[0:3]])

                if not (time_string in driving_time_dict[driver_id].keys()):
                    driving_time_dict[driver_id][time_string] = np.zeros(time_intervals)

                hour_in_day = datetime[3] + datetime[4] / 60.0
                time_index = int(hour_in_day / period)

                if 0 <= time_index <= (time_intervals - 1):
                    driving_time_dict[driver_id][time_string][time_index] += (interval * 0.1 / 3600.0)
                else:
                    print("some error occur when calculating the driving time histogram!")
                    exit()

    # convert the original data to list
    for driver_id in driving_time_dict.keys():
        local_driving_dict = driving_time_dict[driver_id]
        for time_string in local_driving_dict.keys():
            local_driving_vec = local_driving_dict[time_string].tolist()
            driving_time_dict[driver_id][time_string] = local_driving_vec

    return driving_time_dict
