class Driver(object):
    def __init__(self, driver_id, trips=None):
        self.driver_id = driver_id
        self.trips = trips

    def add_gps_point(self, trip_id, gps_point):
        if self.trips is None:
            self.trips = {}

        if not (trip_id in self.trips.keys()):
            trip = Trip(trip_id)
            self.trips[trip_id] = trip

        self.trips[trip_id].gps_points.append(gps_point)


class Trip(object):
    def __init__(self, trip_id, gps_points=None, map_matching=None):
        self.trip_id = trip_id
        if gps_points is None:
            self.gps_points = []
        self.map_matching = map_matching

    def get_gps_time_stamp(self, interval=5):
        if self.gps_points is None:
            return None

        latitude_list = []
        longitude_list = []
        time_stamp_list = []
        for gps_point in self.gps_points[::interval]:
            latitude_list.append(gps_point.latitude)
            longitude_list.append(gps_point.longitude)
            time_stamp_list.append(gps_point.timestamp)
        trajectory_dict = {"latitudes": latitude_list,
                           "longitudes": longitude_list,
                           "timestamp": time_stamp_list}
        return trajectory_dict


class GPSPoint(object):
    def __init__(self, time_stamp, latitude, longitude, altitude):
        self.timestamp = time_stamp
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude

