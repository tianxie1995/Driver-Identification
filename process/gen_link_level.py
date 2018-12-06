from tqdm import tqdm
from tools.time_converter import timestamp2datetime
from map_match.traj_process import get_distance


def generate_link_level_travel_info(drivers):
    link_level_dict = {}

    for driver_id in tqdm(drivers.keys()):
        # iteration for trips of the same drivers
        driver = drivers[driver_id]

        if not (driver_id in link_level_dict.keys()):
            link_level_dict[driver_id] = {}

        for trip_id in driver.trips.keys():
            trajectory_dict = driver.trips[trip_id].get_gps_time_stamp(30)
            time_stamp = trajectory_dict["timestamp"][0]
            date_time = timestamp2datetime(time_stamp / 1000.0)
            time_string = "-".join([str(val) for val in date_time[0:3]])

            if not (time_string in link_level_dict[driver_id].keys()):
                link_level_dict[driver_id][time_string] = {}

            map_match_info = driver.trips[trip_id].map_matching
            if map_match_info is None:
                continue
            latitude_list = map_match_info["points"]["latitudes"]
            longitude_list = map_match_info["points"]["longitudes"]
            belonged_links = map_match_info["links"]

            for idx in range(len(belonged_links) - 1):
                link_id = belonged_links[idx]
                if not (link_id in link_level_dict[driver_id][time_string]):
                    link_level_dict[driver_id][time_string][link_id] = {"distance": 0, "number": 0}

                local_latitude = latitude_list[idx]
                local_longitude = longitude_list[idx]

                next_latitude = latitude_list[idx + 1]
                next_longitude = longitude_list[idx + 1]

                link_level_dict[driver_id][time_string][link_id]["number"] += 1
                link_level_dict[driver_id][time_string][link_id]["distance"] += \
                    get_distance(local_latitude, next_latitude, local_longitude, next_longitude)

    return link_level_dict
