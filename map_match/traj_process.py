import numpy as np
from math import sqrt, cos, pi
import matplotlib.pyplot as plt


def get_distance(latitude1, latitude2, longitude1, longitude2):
    """
    get the distance between two points
    :param latitude1:
    :param latitude2:
    :param longitude1:
    :param longitude2:
    :return:
    """
    distance_interval = sqrt(pow((longitude1 - longitude2) * pi / 180, 2) *
                             pow(cos(latitude1 / 180 * pi), 2) +
                             pow((latitude1 - latitude2) * pi / 180, 2))
    distance_interval *= 6371000

    return distance_interval


def rough_filter_link_for_gps_point(latitude, longitude, link_range, threshold=30):
    """
    roughly filter the link!
    :param latitude:
    :param longitude:
    :param link_range:
    :param threshold:
    :return:
    """
    if link_range[0] <= latitude <= link_range[1]:
        lat_inc = 0
    elif latitude <= link_range[0]:
        lat_inc = link_range[0] - latitude
    else:
        lat_inc = latitude - link_range[1]

    if link_range[2] <= longitude <= link_range[3]:
        lon_inc = 0
    elif longitude <= link_range[2]:
        lon_inc = link_range[2] - longitude
    else:
        lon_inc = longitude - link_range[3]

    dis_lower_bound = get_distance(latitude, latitude + lat_inc,
                                   longitude, longitude + lon_inc)
    # print(lat_inc, lon_inc, dis_lower_bound)
    if dis_lower_bound <= threshold:
        return True
    else:
        return False


def get_min_distance_to_link(latitude, longitude, link_latitudes, link_longitudes):
    """

    :param latitude:
    :param longitude:
    :param link_latitudes:
    :param link_longitudes:
    :return:
    """
    min_distance = None
    min_lat_lon = []
    project_idx = 0
    for idx in range(len(link_latitudes) - 1):
        local_lat = link_latitudes[idx]
        local_lon = link_longitudes[idx]
        next_lat = link_latitudes[idx + 1]
        next_lon = link_longitudes[idx + 1]

        link_vec1 = np.mat([local_lon * cos(latitude / 180 * np.pi), local_lat]).T
        link_vec2 = np.mat([next_lon * cos(latitude / 180 * np.pi), next_lat]).T
        traj_vec = np.mat([longitude * cos(latitude / 180 * np.pi), latitude]).T

        unit_vec = (link_vec2 - link_vec1) / np.linalg.norm(link_vec2 - link_vec1)
        proportion = ((traj_vec - link_vec1).T * unit_vec)[0, 0] / np.linalg.norm(link_vec1 - link_vec2)

        if proportion >= 1:
            min_point_lat = next_lat
            min_point_lon = next_lon
        elif proportion <= 0:
            min_point_lat = local_lat
            min_point_lon = local_lon
        else:
            min_point_lat = local_lat + proportion * (next_lat - local_lat)
            min_point_lon = local_lon + proportion * (next_lon - local_lon)

        local_min_distance = get_distance(min_point_lat, latitude, min_point_lon, longitude)
        if min_distance is None:
            min_lat_lon = [min_point_lat, min_point_lon]
            min_distance = local_min_distance
            project_idx = idx
        if local_min_distance < min_distance:
            min_lat_lon = [min_point_lat, min_point_lon]
            min_distance = local_min_distance
            project_idx = idx

    return min_distance, min_lat_lon, project_idx


def get_trajs_within_box(lats, lons, lat_min, lat_max, lon_min, lon_max):
    """

    :param lats:
    :param lons:
    :param lat_min:
    :param lat_max:
    :param lon_min:
    :param lon_max:
    :return:
    """

    new_lats = []
    new_lons = []

    for idx in range(len(lats)):
        lat = lats[idx]
        lon = lons[idx]

        if lat_min < lat < lat_max:
            if lon_min < lon < lon_max:
                new_lats.append(lat)
                new_lons.append(lon)

    return new_lats, new_lons

