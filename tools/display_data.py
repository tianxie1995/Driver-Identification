import os
import matplotlib.pyplot as plt
from initiate import config


def display_raw_data(zone, drivers):
    """
    display the original data
    :param zone:
    :param drivers:
    :return:
    """
    # draw the map
    nodes = zone.nodes
    links = zone.links

    nodes_lats = []
    nodes_lons = []
    for node_id in nodes.keys():
        node = nodes[node_id]
        lat = node.latitude
        lon = node.longitude
        nodes_lats.append(lat)
        nodes_lons.append(lon)

    links_landmarks = []
    for link_id in links.keys():
        link = links[link_id]
        nodes_id_list = link.nodes_list

        lats = []
        lons = []
        for node_id in nodes_id_list:
            node = nodes[node_id]
            lat = node.latitude
            lon = node.longitude
            lats.append(lat)
            lons.append(lon)
        links_landmarks.append([lons, lats])

    trajs_lats = []
    trajs_lons = []
    for driver_id in drivers.keys():
        driver = drivers[driver_id]
        for trip_id in driver.trips.keys():
            trip = driver.trips[trip_id].gps_points
            for gps_point in trip:
                trajs_lats.append(gps_point.latitude)
                trajs_lons.append(gps_point.longitude)

    plt.figure(dpi=300)
    # plot the link
    for link_landmarks in links_landmarks:
        plt.plot(link_landmarks[0], link_landmarks[1], color=[0.1, 0.1, 0.1], linewidth=0.4, alpha=0.8)

    plt.plot(trajs_lons, trajs_lats, "b.", markersize=0.8, alpha=0.2)
    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.xlim([-83.76, -83.67])
    plt.ylim([42.26, 42.32])
    plt.show()
    plt.savefig(os.path.join(config.figures_folder, "raw_input.png"))
    plt.close()


def draw_single_drivers(zone, drivers):
    # draw the map
    nodes = zone.nodes
    links = zone.links

    nodes_lats = []
    nodes_lons = []
    for node_id in nodes.keys():
        node = nodes[node_id]
        lat = node.latitude
        lon = node.longitude
        nodes_lats.append(lat)
        nodes_lons.append(lon)

    links_landmarks = []
    for link_id in links.keys():
        link = links[link_id]
        nodes_id_list = link.nodes_list

        lats = []
        lons = []
        for node_id in nodes_id_list:
            node = nodes[node_id]
            lat = node.latitude
            lon = node.longitude
            lats.append(lat)
            lons.append(lon)
        links_landmarks.append([lons, lats])

    for driver_id in drivers.keys():
        trajs_lats = []
        trajs_lons = []
        driver = drivers[driver_id]
        for trip_id in driver.trips.keys():
            trip = driver.trips[trip_id]
            trajs_dict = trip.get_gps_time_stamp(30)
            trajs_lats += trajs_dict["latitudes"]
            trajs_lons += trajs_dict["longitudes"]
        # print(len(trajs_lons))

        plt.figure(dpi=200)
        # plot the link
        for link_landmarks in links_landmarks:
            plt.plot(link_landmarks[0], link_landmarks[1], color=[0.1, 0.1, 0.1], linewidth=0.4, alpha=0.8)

        plt.plot(trajs_lons, trajs_lats, "b.", markersize=2, alpha=0.3)
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.xlim([-83.76, -83.67])
        plt.ylim([42.26, 42.32])
        # plt.show()

        save_path = os.path.join(config.figures_folder, config.single_drivers)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        plt.savefig(os.path.join(save_path, "driver" + str(driver_id) + ".png"))
        plt.close()
