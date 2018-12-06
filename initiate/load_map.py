import os
import json
from initiate import config
import classes.map_classes as map_cls


def load_raw_map_info(folder):
    map_file_name = os.path.join(folder, config.map_json)

    with open(map_file_name, encoding="utf8") as map_json_file:
        map_json_info = json.load(map_json_file)

    elements = map_json_info["elements"]

    nodes_dict = {}
    links_dict = {}
    for element in elements:
        type_of_element = element["type"]

        if type_of_element == "node":
            id_of_element = element["id"]
            lat = element["lat"]
            lon = element["lon"]
            node = map_cls.Node(id_of_element, lat, lon)

            if "tags" in element.keys():
                node.tags = element["tags"]

            if not (id_of_element in nodes_dict.keys()):
                nodes_dict[id_of_element] = node

        elif type_of_element == "way":
            id_of_element = element["id"]
            nodes = element["nodes"]
            link = map_cls.Link(id_of_element, nodes)

            if "tags" in element.keys():
                link.tags = element["tags"]

            if not (id_of_element in links_dict.keys()):
                links_dict[id_of_element] = link
    zone = map_cls.Zone(0, nodes_dict, links_dict)
    return zone

