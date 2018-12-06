class Node(object):
    def __init__(self, node_id, latitude, longitude, tags=None):
        self.node_id = node_id
        self.latitude = latitude
        self.longitude = longitude
        self.tags = tags


class Link(object):
    def __init__(self, link_id, nodes_list, tags=None):
        self.link_id = link_id
        self.nodes_list = nodes_list
        self.tags = tags


class Zone(object):
    def __init__(self, zone_id, nodes, links):
        self.zone_id = zone_id
        self.nodes = nodes
        self.links = links

    def get_links_landmarks_dict(self):
        links_dict = {}
        for link_id in self.links:
            links_dict[link_id] = self.get_link_landmarks(link_id)
        return links_dict

    def get_link_landmarks(self, link_id):
        link = self.links[link_id]
        nodes_list = link.nodes_list

        lats = []
        lons = []

        for node_id in nodes_list:
            node = self.nodes[node_id]
            node_lat = node.latitude
            node_lon = node.longitude

            lats.append(node_lat)
            lons.append(node_lon)
        return {"nodes_id": nodes_list, "latitudes": lats, "longitudes": lons}

    def get_joint_nodes_id(self, link_id1, link_id2):
        nodes_list1 = self.links[link_id1].nodes_list
        nodes_list2 = self.links[link_id2].nodes_list

        common_nodes = []
        for node_id in nodes_list1:
            if node_id in nodes_list2:
                common_nodes.append(node_id)

        if len(common_nodes) == 0:
            return None
        else:
            return common_nodes
