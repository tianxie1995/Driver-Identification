import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import map_match.traj_process as traj_process
import initiate.config as config


def map_match(zone, drivers, block_range, interval=30, draw_fig=None):
    """

    :param zone:
    :param drivers:
    :param block_range:
    :param interval:
    :param draw_fig:
    :return:
    """
    links_dict = zone.get_links_landmarks_dict()

    # iteration for drivers
    print(drivers.keys())
    for driver_id in tqdm(drivers.keys()):
        # iteration for trips of the same drivers
        driver = drivers[driver_id]
        for trip_id in driver.trips.keys():
            trip_gps_points = driver.trips[trip_id].gps_points
            trajs_lats = []
            trajs_lons = []
            for gps_point in trip_gps_points:
                trajs_lats.append(gps_point.latitude)
                trajs_lons.append(gps_point.longitude)

            # ignore the trajectories that are not within the scale of the block!
            [block_lat_min, block_lat_max, block_lon_min, block_lon_max] = block_range

            # get the trajs within the box
            trajs_lats, trajs_lons = \
                traj_process.get_trajs_within_box(trajs_lats, trajs_lons, block_lat_min,
                                                  block_lat_max, block_lon_min, block_lon_max)
            if len(trajs_lats) < 10:
                continue

            trajs_lons_max = np.max(trajs_lons)
            trajs_lons_min = np.min(trajs_lons)
            trajs_lats_max = np.max(trajs_lats)
            trajs_lats_min = np.min(trajs_lats)

            # select the link that are possible interfere with the trajs
            links_set = {}
            link_block_dict = {}
            for link_id in links_dict:
                link_landmarks = links_dict[link_id]
                link_latitudes = link_landmarks["latitudes"]
                link_longitudes = link_landmarks["longitudes"]

                link_min_lat = np.min(link_latitudes)
                link_max_lat = np.max(link_latitudes)
                link_min_lon = np.min(link_longitudes)
                link_max_lon = np.max(link_longitudes)

                buffer = 100.0 / 6371000

                if link_max_lat < trajs_lats_min - buffer:
                    continue
                if link_max_lon < trajs_lons_min - buffer:
                    continue
                if link_min_lat > trajs_lats_max + buffer:
                    continue
                if link_min_lon > trajs_lons_max + buffer:
                    continue
                links_set[link_id] = link_landmarks
                link_block_dict[link_id] = \
                    [link_min_lat, link_max_lat, link_min_lon, link_max_lon]

            # generate the table of the minimum links to the traj points
            # change the interval of the trajectory points
            trajs_lats = trajs_lats[::interval]
            trajs_lons = trajs_lons[::interval]

            # variables for the following dynamic programming
            projected_points_list = []
            minimum_distance_list = []
            point_idx_in_link = []
            traj_link_id_list = []

            for traj_idx in range(len(trajs_lats)):
                traj_lat = trajs_lats[traj_idx]
                traj_lon = trajs_lons[traj_idx]

                link_id_list = []
                min_distance_list = []
                project_points = []
                project_points_idx = []

                for link_id in links_set.keys():
                    link_landmarks = links_set[link_id]
                    link_latitudes = link_landmarks["latitudes"]
                    link_longitudes = link_landmarks["longitudes"]

                    link_range = link_block_dict[link_id]

                    # first rough filter
                    if not traj_process.rough_filter_link_for_gps_point(traj_lat, traj_lon, link_range):
                        continue

                    min_distance, min_point, point_idx = \
                        traj_process.get_min_distance_to_link(traj_lat, traj_lon,
                                                              link_latitudes, link_longitudes)

                    # ignore the link which the distance is larger than a threshold
                    if min_distance > 20:
                        continue
                    min_distance_list.append(min_distance)
                    link_id_list.append(link_id)
                    project_points.append(min_point)
                    project_points_idx.append(point_idx)

                projected_points_list.append(project_points)
                minimum_distance_list.append(min_distance_list)
                traj_link_id_list.append(link_id_list)
                point_idx_in_link.append(project_points_idx)

            if draw_fig is None:
                trajs_id = None
            else:
                trajs_id = str(driver_id) + "_" + str(trip_id)
            map_matching_dict = \
                match_traj_using_dynamic_programming([trajs_lats, trajs_lons],
                                                     minimum_distance_list, traj_link_id_list,
                                                     links_set, projected_points_list, zone,
                                                     point_idx_in_link, trajs_id)
            drivers[driver_id].trips[trip_id].map_matching = map_matching_dict

    return drivers


def match_traj_using_dynamic_programming(trajs_points,
                                         min_distance_list,
                                         min_links_list,
                                         links_info_list,
                                         projected_points_list,
                                         zone, point_idx_in_link, trajs_id=None):
    # dynamic programming part
    path_topology = []
    nodes_performance_index = []
    total_len = len(min_distance_list)
    for idx in range(total_len):
        idx_end = total_len - idx - 1
        num_of_candidates = len(min_distance_list[idx_end])

        # initiate the node probability according to the distance
        node_prob_list = []
        for jdx in range(num_of_candidates):
            local_distance = min_distance_list[idx_end][jdx]
            local_prob = get_state_possibility(local_distance)
            node_prob_list.append(local_prob)

        # initiate the start node
        if len(path_topology) == 0:
            if num_of_candidates != 0:
                start_node = []
                for jdx in range(num_of_candidates):
                    start_node.append([jdx])
                path_topology.append(start_node)
                nodes_performance_index.append(node_prob_list)
            else:
                path_topology.append([[-1]])
                nodes_performance_index.append([1])
        else:
            last_topology = path_topology[idx - 1]
            last_node_performances = nodes_performance_index[idx - 1]
            path_topology.append([])

            if num_of_candidates == 1:
                destination_idx = int(np.argmax(last_node_performances))
                path_topology[idx].append([0, last_topology[destination_idx][0]])

                # restart the dynamic programming
                nodes_performance_index.append([1])
            elif num_of_candidates < 1:
                # if there is no candidate, then ignore this...
                destination_idx = int(np.argmax(last_node_performances))
                path_topology[idx].append([-1, last_topology[int(destination_idx)][0]])

                # restart the dynamic programming
                nodes_performance_index.append([1])
            else:
                nodes_performance_list = []
                for jdx in range(num_of_candidates):
                    origin_link_id = min_links_list[idx_end][jdx]
                    local_point = projected_points_list[idx_end][jdx]
                    local_point_idx = point_idx_in_link[idx_end][jdx]
                    node_prob = node_prob_list[jdx]

                    if len(last_topology) == 1:
                        path_topology[idx].append([jdx, last_topology[0][0]])
                        nodes_performance_list.append(node_prob)
                        continue

                    transfer_prop_list = []
                    node_idx = 0
                    for topology_pair in last_topology:
                        destination_idx = topology_pair[0]
                        if destination_idx < 0:
                            print(last_topology)
                            print("Map matching error!")
                            exit()
                        destination_link_id = min_links_list[idx_end + 1][destination_idx]
                        destination_point = projected_points_list[idx_end + 1][destination_idx]
                        destination_point_idx = point_idx_in_link[idx_end + 1][destination_idx]

                        detour_index = get_transfer_detour_index(zone, origin_link_id, destination_link_id,
                                                                 local_point, local_point_idx,
                                                                 destination_point, destination_point_idx)
                        if detour_index < 1.0:
                            detour_index = 1.0
                        if detour_index > 5.0:
                            detour_index = 5.0

                        transfer_prop = get_transfer_possibility(detour_index)
                        transfer_prop *= last_node_performances[node_idx]
                        node_idx += 1
                        transfer_prop_list.append(transfer_prop)

                    best_transfer = np.max(transfer_prop_list)
                    best_performance_idx = int(np.argmax(transfer_prop_list))
                    path_topology[idx].append([jdx, last_topology[best_performance_idx][0]])
                    nodes_performance_list.append(best_transfer * node_prob)

                # normalize the nodes performance index before save it...
                best_performance = np.max(nodes_performance_list)
                nodes_performance_list = [val / best_performance for val in nodes_performance_list]
                nodes_performance_index.append(nodes_performance_list)

        # print("index=", idx, "Number of candidates:", num_of_candidates)
        # print("Nodes performance:", nodes_performance_index)
        # print("Nodes topology:", path_topology)
        # print("=========================")

    # select the path!
    connected_path_index = []
    reverse_path_topology = path_topology[::-1]
    for idx in range(len(reverse_path_topology) - 1):
        local_layer = reverse_path_topology[idx]
        if len(connected_path_index) == 0:
            connected_path_index.append(local_layer[0][0])
            connected_path_index.append(local_layer[0][1])
        else:
            start_node_idx = connected_path_index[-1]
            chain_connect = False
            for node_pair in local_layer:
                if node_pair[0] == start_node_idx:
                    connected_path_index.append(node_pair[1])
                    chain_connect = True
                    break
            if chain_connect is False:
                print("Chain not connected")
                exit()

    # # plot the path topology! this will be a cool figure maybe!
    # plt.figure()
    # layer_idx = 0
    # for layer in path_topology[1:]:
    #     for node_pair in layer:
    #         plt.plot([layer_idx, layer_idx + 1], [node_pair[1], node_pair[0]], "k.-")
    #     layer_idx += 1
    #
    # plt.plot(connected_path_index[::-1], "r.-")
    # plt.xlabel("Index of time stamp")
    # plt.ylabel("Index of candidate links")
    # plt.show()

    results_projected_points = []
    results_min_links = []
    for idx in range(len(connected_path_index)):
        point_index = connected_path_index[idx]
        if point_index < 0:
            continue
        else:
            results_projected_points.append(projected_points_list[idx][point_index])
            results_min_links.append(min_links_list[idx][point_index])

    results_projected_latitudes = [val[0] for val in results_projected_points]
    results_projected_longitudes = [val[1] for val in results_projected_points]

    map_matching_dict = {"points": {"latitudes": results_projected_latitudes,
                                    "longitudes": results_projected_longitudes},
                         "links": results_min_links}

    if trajs_id is not None:
        # create the path
        save_figures_path = os.path.join(config.figures_folder, config.map_matching)
        if not os.path.exists(save_figures_path):
            os.mkdir(save_figures_path)

        plt.figure(dpi=100)
        plt.plot(trajs_points[1], trajs_points[0], "b.-", markersize=8)
        for link_id in links_info_list:
            link_info = links_info_list[link_id]
            link_latitudes = link_info["latitudes"]
            link_longitudes = link_info["longitudes"]

            if link_id in set(results_min_links):
                plt.plot(link_longitudes, link_latitudes, "c-", linewidth=1.5, alpha=0.5)
            else:
                plt.plot(link_longitudes, link_latitudes, "k-", linewidth=1.5, alpha=0.5)

        for idx in range(len(projected_points_list)):
            traj_lat = trajs_points[0][idx]
            traj_lon = trajs_points[1][idx]

            for project_point in projected_points_list[idx]:
                local_lat = project_point[0]
                local_lon = project_point[1]

                plt.plot(local_lon, local_lat, "g.", markersize=8, alpha=0.5)
                plt.plot([local_lon, traj_lon], [local_lat, traj_lat], "k--", linewidth=0.3)

        plt.plot(results_projected_longitudes, results_projected_latitudes, "r*", markersize=8)
        buffer = 3.0 / 6371
        plt.xlim([np.min(trajs_points[1]) - buffer, np.max(trajs_points[1]) + buffer])
        plt.ylim([np.min(trajs_points[0]) - buffer, np.max(trajs_points[0]) + buffer])
        plt.xlabel("longitude")
        plt.ylabel("latitude")
        plt.savefig(os.path.join(save_figures_path, trajs_id + ".png"))
        plt.show()
        plt.close()
    return map_matching_dict


def get_state_possibility(distance, sigma_z=20):
    """

    :param distance:
    :param sigma_z:
    :return:
    """
    prop = 1 / np.sqrt(2 * np.pi) / sigma_z * np.exp(-0.5 * pow(distance / sigma_z, 2))
    return prop


def get_transfer_possibility(detour_index, beta=2):
    prop = 1 / beta * np.exp(- detour_index / beta)
    return prop


def get_transfer_detour_index(zone, origin_link_id, dest_link_id,
                              origin_point, origin_point_idx,
                              dest_point, dest_point_idx):
    distance_between_points = traj_process.get_distance(origin_point[0], dest_point[0],
                                                        origin_point[1], dest_point[1])
    if distance_between_points < 1:
        return 1.0

    if origin_link_id == dest_link_id:
        links = zone.links
        link = links[origin_link_id]
        link_landmarks = zone.get_link_landmarks(origin_link_id)
        link_oneway = False
        if "oneway" in link.tags:
            if link.tags["oneway"] == "yes":
                link_oneway = True
            else:
                link_oneway = False
        travel_distance = get_travel_distance_between_points(origin_point, origin_point_idx,
                                                             dest_point, dest_point_idx,
                                                             link_landmarks, link_oneway)
        if travel_distance < 0:
            return 5
        else:
            return travel_distance / distance_between_points
    else:
        links = zone.links

        origin_link = links[origin_link_id]
        destination_link = links[dest_link_id]

        origin_link_landmarks = zone.get_link_landmarks(origin_link_id)
        destination_link_landmarks = zone.get_link_landmarks(dest_link_id)

        inlink_oneway = False
        outlink_oneway = False
        if "oneway" in origin_link.tags:
            if origin_link.tags["oneway"] == "yes":
                inlink_oneway = True
            else:
                inlink_oneway = False

        if "oneway" in destination_link.tags:
            if destination_link.tags["oneway"] == "yes":
                outlink_oneway = True
            else:
                outlink_oneway = False

        if inlink_oneway is False:
            inlink_direction = "both"
        else:
            inlink_direction = "forward"

        if outlink_oneway is False:
            outlink_direction = "both"
        else:
            outlink_direction = "backward"

        common_nodes_id = zone.get_joint_nodes_id(origin_link_id, dest_link_id)
        if common_nodes_id is None:
            return 5

        travel_dis_list = []
        for common_node_id in common_nodes_id:
            dis1 = get_travel_distance(origin_point, origin_link_landmarks,
                                       origin_point_idx, common_node_id, inlink_direction)
            dis2 = get_travel_distance(dest_point, destination_link_landmarks,
                                       dest_point_idx, common_node_id, outlink_direction)
            travel_distance = dis1 + dis2
            travel_dis_list.append(travel_distance)

        travel_distance = np.min(travel_dis_list)
        detour_prop = travel_distance / distance_between_points

        # plt.figure()
        # plt.plot(origin_link_landmarks["longitudes"], origin_link_landmarks["latitudes"], "b.-")
        # plt.plot(destination_link_landmarks["longitudes"], destination_link_landmarks["latitudes"], "g.-")
        #
        # plt.plot(origin_point[1], origin_point[0], "r+")
        # plt.plot(dest_point[1], dest_point[0], "r*")
        # plt.title("Detour: " + str(np.round(detour_prop, 4)))
        # plt.show()
        return detour_prop


def get_travel_distance(point, link, point_idx, node_id, direction="both"):
    """
    get the travel distance between one traj point to the joint point of links
    this is....I think...a little difficult
    :param point:
    :param link:
    :param point_idx:
    :param node_id:
    :param direction:
    :return:
    """
    nodes_id_list = link["nodes_id"]
    link_latitudes = link["latitudes"]
    link_longitudes = link["longitudes"]

    node_idx = 0
    for temp_node_id in nodes_id_list:
        if temp_node_id == node_id:
            break
        node_idx += 1

    travel_distance = 0
    if node_idx <= point_idx:
        if direction == "forward":
            left_iters = len(nodes_id_list) - node_idx - 2
            for idx in range(left_iters):
                real_idx = idx + node_idx
                local_lat = link_latitudes[real_idx]
                local_lon = link_longitudes[real_idx]

                next_lat = link_latitudes[real_idx + 1]
                next_lon = link_longitudes[real_idx + 1]
                local_distance = traj_process.get_distance(local_lat, next_lat,
                                                           local_lon, next_lon)
                travel_distance += local_distance
        else:
            left_iters = len(nodes_id_list) - node_idx - 2
            for idx in range(left_iters):
                real_idx = idx + node_idx
                local_lat = link_latitudes[real_idx]
                local_lon = link_longitudes[real_idx]

                if real_idx == point_idx:
                    local_distance = traj_process.get_distance(local_lat, point[0],
                                                               local_lon, point[1])
                    travel_distance += local_distance
                    break
                else:
                    next_lat = link_latitudes[real_idx + 1]
                    next_lon = link_longitudes[real_idx + 1]
                    local_distance = traj_process.get_distance(local_lat, next_lat,
                                                               local_lon, next_lon)
                    travel_distance += local_distance
    else:
        if direction == "backward":
            left_iters = len(nodes_id_list) - node_idx - 2
            for idx in range(left_iters):
                real_idx = idx + node_idx
                local_lat = link_latitudes[real_idx]
                local_lon = link_longitudes[real_idx]

                next_lat = link_latitudes[real_idx + 1]
                next_lon = link_longitudes[real_idx + 1]
                local_distance = traj_process.get_distance(local_lat, next_lat,
                                                           local_lon, next_lon)
                travel_distance += local_distance
        else:
            left_iters = node_idx - 1
            for idx in range(left_iters):
                real_idx = node_idx - idx
                local_lat = link_latitudes[real_idx]
                local_lon = link_longitudes[real_idx]

                if real_idx == point_idx + 1:
                    local_distance = traj_process.get_distance(local_lat, point[0],
                                                               local_lon, point[1])
                    travel_distance += local_distance
                    break
                else:
                    next_lat = link_latitudes[real_idx - 1]
                    next_lon = link_longitudes[real_idx - 1]
                    local_distance = traj_process.get_distance(local_lat, next_lat,
                                                               local_lon, next_lon)
                    travel_distance += local_distance

    # plt.title("distance:" + str(np.round(travel_distance, 2)) + "m")
    # plt.show()
    return travel_distance


def get_travel_distance_between_points(org_point, org_point_idx,
                                       des_point, des_point_idx,
                                       link, oneway):
    """
    get the travel distance between two trajectory points within the same link
    this function has not been tested yet!
    :param org_point:
    :param org_point_idx:
    :param des_point:
    :param des_point_idx:
    :param link:
    :param oneway:
    :return:
    """
    if org_point_idx == des_point_idx:
        return traj_process.get_distance(org_point[0], des_point[0],
                                         org_point[1], des_point[1])
    # if the original point is behind the destination point, switch the pos or...
    elif org_point_idx > des_point_idx:
        if oneway:
            return -1
        else:
            pass
        temp_point = org_point
        temp_point_idx = org_point_idx

        org_point = des_point
        org_point_idx = des_point_idx
        des_point = temp_point
        des_point_idx = temp_point_idx
    else:
        pass

    link_latitudes = link["latitudes"]
    link_longitudes = link["longitudes"]

    travel_distance = 0
    for idx in range(len(link_latitudes)):
        if (idx < org_point_idx) or (idx > des_point_idx):
            continue

        if idx == org_point_idx:
            travel_distance += traj_process.get_distance(org_point[0], link_latitudes[idx + 1],
                                                         org_point[1], link_longitudes[idx + 1])
        elif idx == des_point_idx:
            travel_distance += traj_process.get_distance(link_latitudes[idx], des_point[0],
                                                         link_longitudes[idx], des_point[1])
        else:
            travel_distance += traj_process.get_distance(link_latitudes[idx], link_latitudes[idx + 1],
                                                         link_longitudes[idx], link_longitudes[idx + 1])

    return travel_distance
