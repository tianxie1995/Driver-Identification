import os
from tqdm import tqdm
from initiate import config
import classes.drivers as driver_cls


def load_raw_drivers_data(folder):
    """
    load the raw drivers infos and save them in dict(cls:driver)
    :param folder:
    :return:
    dict(driver_id: class::Driver)
    """
    # load file from the original file
    file_name = os.path.join(folder, config.trajs_file)
    raw_trajs_file = open(file_name, "r")
    raw_lines = raw_trajs_file.readlines()
    raw_trajs_file.close()

    drivers_dict = {}
    for raw_line in tqdm(raw_lines[1:]):
        split_infos = raw_line.split(",")
        driver_id = int(split_infos[0])
        trip_id = int(split_infos[1])
        gps_time = int(split_infos[4])
        latitude = float(split_infos[5])
        longitude = float(split_infos[6])
        altitude = float(split_infos[7])

        if not (driver_id in drivers_dict.keys()):
            driver = driver_cls.Driver(driver_id)
            drivers_dict[driver_id] = driver

        gps_point = driver_cls.GPSPoint(gps_time, latitude, longitude, altitude)
        drivers_dict[driver_id].add_gps_point(trip_id, gps_point)
    return drivers_dict


if __name__ == '__main__':
    print(load_raw_drivers_data("ann_arbor")[501].trips)


