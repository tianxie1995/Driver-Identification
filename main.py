import os
import pickle
import json
import initiate.config as config

from initiate.load_trajs import load_raw_drivers_data
from initiate.load_map import load_raw_map_info
from tools.display_data import draw_single_drivers
from map_match import map_match
from process.gen_heatmap import generate_driving_heat_map
from process.gen_driving_time import generate_driving_time_histogram
from process.gen_link_level import generate_link_level_travel_info


def init_output_folder():
    if not os.path.exists(config.figures_folder):
        os.mkdir(config.figures_folder)


def main(folder, reload=False):
    # initiate the output folder
    init_output_folder()

    # initiate the block range
    block_range = [42.26, 42.32, -83.76, -83.67]

    if (reload is True) or (not os.path.exists(os.path.join(folder, config.temp_save))):
        # load the map and driver
        print("Load the original data...")
        zone = load_raw_map_info(folder)
        drivers = load_raw_drivers_data(folder)

        # run this line to see the original data and basic structure of the data
        draw_single_drivers(zone, drivers)

        # map-matching
        print("Map-matching...")
        drivers = map_match.map_match(zone, drivers, block_range=block_range)

        # save data!
        with open(os.path.join(folder, config.temp_save), "wb") as temp_save_file:
            current_info = {"network": zone, "drivers": drivers}
            pickle.dump(current_info, temp_save_file)
        print("save the temp_save.pickle complete!")
    else:
        print("Temp saving file already exists!")
        print("Load the temp_save.pickle...")
        with open(os.path.join(folder, config.temp_save), "rb") as temp_save_file:
            temp_save_file = pickle.load(temp_save_file)

        drivers = temp_save_file["drivers"]
        zone = temp_save_file["network"]
        print("There are totally", len(drivers), "drivers!")

    print("Start to process the data...")
    # generate the heat map data!
    print("Generate the heat map data!")
    travel_distance_and_frequency_dict = generate_driving_heat_map(drivers, block_range)
    print(type(travel_distance_and_frequency_dict))
    with open(os.path.join(folder, config.grid_travel_info), "w") as grid_json_file:
        grid_json_file.write(json.dumps(travel_distance_and_frequency_dict))

    # generate the link-level daily travel distance
    print("Generate the link-level daily travel distance")
    link_level_dict = generate_link_level_travel_info(drivers)
    with open(os.path.join(folder, config.link_level), "w") as link_level_file:
        link_level_file.write(json.dumps(link_level_dict))

    # generate the daily travel time histogram
    print("Generate the daily driving period")
    driving_time_dict = generate_driving_time_histogram(drivers)
    with open(os.path.join(folder, config.driving_time_info), "w") as driving_time_file:
        driving_time_file.write(json.dumps(driving_time_dict))


if __name__ == '__main__':
    main("ann_arbor", True)

