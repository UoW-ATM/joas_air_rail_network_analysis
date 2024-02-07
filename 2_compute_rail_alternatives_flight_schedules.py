import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import pickle
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import pandas as pd
from datetime import timedelta
from libs.general_tools import haversine, create_folder_if_not_exist
from libs.rail_itineraries_computation import read_data, join_trips_routes_stops, trains_on_date, direct_services_to

"""
This code computes the possible rail alternatives for the flights.
It uses Renfe GTFS data to find the train timetables and routes possibles.
It checks which rail stations are close to the airports (within a defined max_distance_rail_air in km)
then, it computes for each rail station routes to the others with direct trains. This can be only on
routes that are served by air (i.e., only checking possible stations between stations that are at the
origin and destination of flights) or all origins vs all destinations. The second option allows then to 
find direct trains where indirect flights might be possible.
Note that as for each airport there are many rail stations close by, the number of combination can be high
if all options are computed it can be a few hours of computation.
It checks the trains on the days of the flight schedules.
"""

###################
# DATA/PARAMETERS #
###################

# Data files
f_flight_schedules = './output/air/schedules_1st_week_0523_all_airlines_reduced.parquet'
f_rail_stations = './data/renfe/renfe_mid_long/stops.txt'
folder_renfe_data = './data/renfe/renfe_mid_long/'
f_renfe_data_preloaded = './output/rail/rail_data_times_format.pickle'

# Parameters set manually
country_of_interest = "LE"
list_airports_avoid = ['LEIB', 'LEPA', 'LEMH']  # Avoid airports in Balearic islands

max_distance_rail_air = 25  # km
compute_all_trains = True  # False will only compute on the o-d pairs with flights, true on all possible o-d pairs

# Output files
f_output_folder = "./output/rail/"
f_dict_rail_stations_airport = "./output/rail/dict_rail_stations_airports.pickle"
f_dict_direct_rail_options = "./output/rail/dict_direct_rail_options_replacement.pickle"
f_dict_direct_rail_options_purged = "./output/rail/dict_direct_rail_purged.pickle"
f_dict_rail_stations_airport_used = "./output/rail/dict_rail_stations_used.pickle"
f_df_rail_purged = './output/rail/df_rail_purged.csv'
f_df_rail_purged_parquet = './output/rail/df_rail_purged.parquet'

f_output_folder_log = "./output/log/"
f_log = f_output_folder_log+'2_log_compute_rail_alternatives.txt'

# Do plot of stations
do_plots = True
extent_plot = [-10, 5, 25, 44]
output_figures_folder = './output/figs/'  # for figures in general
output_fig_path = output_figures_folder + 'rail_stations.png'

# Create output folders if not exist
create_folder_if_not_exist(f_output_folder)
create_folder_if_not_exist(f_output_folder_log)
if do_plots:
    create_folder_if_not_exist(output_figures_folder)


#####################
# CONFIGURE LOGGING #
#####################

logging.basicConfig(filename=f_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

##################
# FUNCTIONS USED #
##################


def purge_rail_options(rail_od):
    """
    Purge rail options to avoid same train from o-d counted from intermediate stops in o and/or d
    """

    trips_od = set()
    trips_kept = {}
    noptions = 0
    noptions_not_kept = 0
    for ods, trips in rail_od.items():
        for t in trips:
            noptions += 1
            if t[0]['trip_short_name'] in trips_od:
                # Already have this
                swap = False
                t_considering = t[0]
                t_there = trips_kept[t[0]['trip_short_name']][0]

                same_origin = (t_considering['stop_start'] == t_there['stop_start'])
                same_destination = (t_considering['stop_end'] == t_there['stop_end'])

                if same_origin and same_destination:
                    # Shouldn't happen but if it does, keep faster
                    swap = t_considering['leg_time'] < t_there['leg_time']

                elif same_origin:
                    if dict_usage_stops[t_considering['stop_end']] == dict_usage_stops[t_there['stop_end']]:
                        # Both destinations used the same, keep faster
                        swap = t_considering['leg_time'] < t_there['leg_time']
                    else:
                        swap = dict_usage_stops[t_considering['stop_end']] > dict_usage_stops[t_there['stop_end']]

                elif same_destination:
                    if dict_usage_stops[t_considering['stop_start']] == dict_usage_stops[t_there['stop_start']]:
                        # Both origin used the same, keep faster
                        swap = t_considering['leg_time'] < t_there['leg_time']
                    else:
                        swap = dict_usage_stops[t_considering['stop_start']] > dict_usage_stops[t_there['stop_start']]

                else:
                    # Both origin and destination are different
                    how_much_end = dict_usage_stops[t_considering['stop_end']] - dict_usage_stops[t_there['stop_end']]
                    how_much_start = dict_usage_stops[t_considering['stop_start']] - dict_usage_stops[
                        t_there['stop_start']]

                    if how_much_end + how_much_start == 0:
                        # both start and end use the same
                        # keep fastest
                        swap = t_considering['leg_time'] < t_there['leg_time']
                    else:
                        swap = (how_much_end + how_much_start) > 0

                if swap:
                    trips_kept[t[0]['trip_short_name']] = t

                noptions_not_kept += 1

            else:
                trips_kept[t[0]['trip_short_name']] = t
                trips_od.add(t[0]['trip_short_name'])

    return trips_kept


def flatten_dict(d, parent_keys=(), sep='_'):
    """
    Recursively flattens a nested dictionary.
    """
    items = []
    for k, v in d.items():
        new_keys = parent_keys + (k,)
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_keys, sep=sep).items())
        else:
            items.append((new_keys, v))
    return dict(items)


########################################
# PROCESSING COMPUTE RAIL ALTERNATIVES #
########################################

####
# READ DATA/CLEAN AND PREPARE
####

logging.info("\n----------\nREADING DATA\n----------\n")

# Read flight schedules and rotations
ds = pd.read_parquet(f_flight_schedules)

ds['day_sobt'] = ds['sobt'].dt.date
ds['day_sibt'] = ds['sibt'].dt.date

# Read rail stations
rail_stations = pd.read_csv(f_rail_stations)

# Get all airports
all_airports = set(ds.departure_used).union(set(ds.arrival_used))

# As rail in Spain consider only stations within LE
airports_spain = set()
for a in all_airports:
    if a.startswith(country_of_interest) and a not in list_airports_avoid:
        airports_spain.add(a)

logging.info("\n----------\nCOMPUTE ROUTES WITHIN COUNTRY\n----------\n")
# Routes within spain (o-d pairs)
routes_within_spain = ds[ds.departure_used.isin(airports_spain) & ds.arrival_used.isin(airports_spain)][
    ['departure_used', 'arrival_used',
     'dep_lat', 'dep_lon', 'arr_lat', 'arr_lon']].drop_duplicates().copy()

logging.info("Routes within %s: %d", country_of_interest, len(routes_within_spain))

logging.info('Airports in %s: %d', country_of_interest, len(airports_spain))

logging.info("\n----------\nREADING RAIL DATA\n----------\n")
# Read rail data
try:
    with open(f_renfe_data_preloaded, "rb") as f:
        agency, calendar, calendar_dates, routes, stops_times, stops, trips = pickle.load(f)
except FileNotFoundError:
    agency, calendar, calendar_dates, routes, stops_times, stops, trips = read_data(folder_path=folder_renfe_data)
    with open(f_renfe_data_preloaded, "wb") as f:
        pickle.dump((agency, calendar, calendar_dates, routes, stops_times, stops, trips), f)

###
# COMPUTE AIRPORT-RAIL STATION DISTANCE MATRIX
###
logging.info("\n----------\nCOMPUTE DISTANCE MATRIX BETWEEN AIRPORTS AND RAIL STATIONS\n----------\n")

# Create a DataFrame with empty columns
additional_df = pd.DataFrame({col: [None] * len(rail_stations) for col in airports_spain})

# Create rail airport distance matrix
rail_airport_matrix = pd.concat([rail_stations, additional_df], axis=1)

# Dictionary of airport with their lat, lon
dict_airp_latlon = ds[['departure_used', 'dep_lat', 'dep_lon']].drop_duplicates().rename(
    columns={'dep_lat': 'lat', 'dep_lon': 'lon'}).set_index('departure_used')[['lat', 'lon']].to_dict(orient='index')
dict_airp_latlon.update(ds[['arrival_used', 'arr_lat', 'arr_lon']].drop_duplicates().rename(
    columns={'arr_lat': 'lat', 'arr_lon': 'lon'}).set_index('arrival_used')[['lat', 'lon']].to_dict(orient='index'))

# Compute distance matrix
for i in rail_airport_matrix.index:
    for a in airports_spain:
        rail_airport_matrix.loc[i, a] = (
            haversine(rail_airport_matrix.loc[i].stop_lon, rail_airport_matrix.loc[i].stop_lat,
                      dict_airp_latlon[a]['lon'], dict_airp_latlon[a]['lat']))


logging.info("\n----------\nSTATIONS CLOSE TO AIRPORTS\n----------\n")
# Dictionary of rail stations close to airports
dict_closest_rail = {}
for a in airports_spain:
    rail_airport_matrix[a] = pd.to_numeric(rail_airport_matrix[a], errors='coerce')
    min_index = rail_airport_matrix[a].idxmin()
    if rail_airport_matrix.loc[min_index][a] < max_distance_rail_air:
        dict_closest_rail[a] = rail_airport_matrix.loc[min_index]['stop_id']

# Dictionary rail stations closest to threshold to airport
dict_rail_stations_airports = {}
for a in airports_spain:
    dict_rail_stations_airports[a] = rail_airport_matrix[
        rail_airport_matrix[a] < max_distance_rail_air].stop_id.to_list()

# Number of stations close to airports
n_stations_airport = 0
logging.info("\nAirports and number of stations closest to threshold")
for k, v in dict_rail_stations_airports.items():
    logging.info(str(k) + " " + str(len(v)))
    n_stations_airport += len(v)
logging.info("Avg rail stations per airport: %d",  n_stations_airport / len(dict_rail_stations_airports.keys()))

# Save dictionary of rail stations
with open(f_dict_rail_stations_airport, "wb") as f:
    pickle.dump(dict_rail_stations_airports, f)

###
# COMPUTE TRAIN ALTERNATIVES
###

logging.info("\n----------\nCOMPUTE TRAIN ALTERNATIVES BETWEEN AIRPORTS\n----------\n")

# Train alternatives to compute if only considering routes which have flights
train_alternatives_to_compute = {}
for i, r in routes_within_spain.iterrows():
    train_alternatives_to_compute[(r.departure_used, r.arrival_used)] = (dict_rail_stations_airports[r.departure_used],
                                                                         dict_rail_stations_airports[r.arrival_used])

logging.info('If only routes with flights, number o-d pairs to compute rail: %d', len(train_alternatives_to_compute))

# Train alternatives to compute considering all o-d pairs
train_alternatives_to_compute_all = {}
for d in set(routes_within_spain.departure_used):
    for a in set(routes_within_spain.arrival_used):
        if (d != a) and ((d, a)):  # not in train_alternatives_to_compute.keys()):
            train_alternatives_to_compute_all[(d, a)] = (dict_rail_stations_airports[d], dict_rail_stations_airports[a])

logging.info('If all o-d pairs to compute, number of o-d pairs to compute rail: %d',
             len(train_alternatives_to_compute_all))

if compute_all_trains:
    logging.info("Computing all trains")
    trains_to_compute = train_alternatives_to_compute_all
else:
    logging.info("Computing only trains with have direct flight")
    trains_to_compute = train_alternatives_to_compute

# Build trail trips with stops
rail_trips_full_w_stops = join_trips_routes_stops(trips, routes, calendar, stops, stops_times)

# Get trains on date of flights
date_range = [min(min(ds.day_sobt), min(ds.day_sibt)) + timedelta(days=x) for x in
              range((max(max(ds.day_sobt), max(ds.day_sibt)) - min(min(ds.day_sobt), min(ds.day_sibt))).days + 1)]

rail_trips_on_day = {}

for d in date_range:
    rail_trips_on_day[d.strftime('%Y%m%d')] = trains_on_date(rail_trips_full_w_stops, d.strftime('%Y%m%d'))

# Record the start time
start_time = time.time()

number_check = 0
from_time = '00:00:00'
max_n_connections = 0
dict_direct_rail_options_replacement = {}
# for k,v in train_alternatives_to_compute.items():
for k, v in trains_to_compute.items():
    if k[0] != k[1]:
        # Avoid circular flights
        for d in date_range:
            logging.info('Computing rail for %s, %s', k, d)
            for stop_origin in v[0]:
                for stop_destination in v[1]:
                    # Find direct rail services between origin and destination on that day from 00:00
                    rail_services = direct_services_to(stop_start=stop_origin,
                                                       stop_end=stop_destination,
                                                       trips_all=rail_trips_on_day[d.strftime('%Y%m%d')],
                                                       date_time_start=pd.to_datetime(from_time,
                                                                                      format='%H:%M:%S'))  # max_connections = max_n_connections)
                    number_check += 1
                    if rail_services is not None:
                        dict_services_day_odr = {(stop_origin, stop_destination): rail_services}
                        dict_day = dict_direct_rail_options_replacement.get(d.strftime('%Y%m%d'), {})
                        dict_od = dict_day.get(k, {})
                        dict_day[k] = dict_od
                        dict_day[k].update(dict_services_day_odr)
                        dict_direct_rail_options_replacement[d.strftime('%Y%m%d')] = dict_day

logging.info('Checked total of rail:` %d', number_check)
# Record the end time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Convert elapsed time to hours, minutes, and seconds
hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)

# Print the elapsed time in hh:mm:ss format
logging.info('The computation rail options took %s to run', f'{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}')

###
# Save rail options
###
with open(f_dict_direct_rail_options, "wb") as f:
    pickle.dump(dict_direct_rail_options_replacement, f)


###
# PURGE RAIL OPTIONS, AVOID HAVING MULTIPLE ITINERARIES BETWEEN O-D PAIR WHICH ARE SAME TRAIN
# (E.G. SANTS-GIR OR PASSEIG-GRACIA-GIR)
###
logging.info("\n----------\nPURGE RAIL OPTIONS\n----------\n")

# Purge rail options
# to avoid using trains on same routes in given o-d pair

dict_usage_stops = stops_times.groupby(['stop_id']).count()['trip_id'].to_dict()

dict_direct_rail_purged = {}
for d, od in dict_direct_rail_options_replacement.items():
    logging.info('Purging day %s', d)
    dict_od_day = {}
    for odr, tr in od.items():
        dict_od_day[odr] = purge_rail_options(tr)
    dict_direct_rail_purged[d] = dict_od_day

# Save purged options
with open(f_dict_direct_rail_options_purged, "wb") as f:
    pickle.dump(dict_direct_rail_purged, f)

logging.info("\n----------\nCONVERT RAIL OPTIONS TO DATAFRAME\n----------\n")
# Change structure to end with dict_rail_stations_used per o-d pair
flat_dict_direct_rail_purged = flatten_dict(dict_direct_rail_purged)

# Convert to DataFrame
df = pd.DataFrame.from_dict(flat_dict_direct_rail_purged, orient='index').reset_index()

# Split the index into separate columns
df[['day', 'o_d', 'trip_short_name']] = pd.DataFrame(df['index'].tolist(), index=df.index)

# Drop the 'index' column
df = df.drop(columns=['index'])

# Pivot the DataFrame to get the desired structure
df_direct_rail_purged = pd.concat([df.drop([0], axis=1), pd.json_normalize(df[0]).drop(['trip_short_name'], axis=1)],
                                  axis=1)
df_direct_rail_purged['tid'] = df_direct_rail_purged.index

# Compute dict_rail_used (after purged)
dict_rail_stations_used = {}
for o_d in df_direct_rail_purged.o_d.drop_duplicates():
    origin = o_d[0]
    destination = o_d[1]
    lorig = dict_rail_stations_used.get(origin, set())
    ldest = dict_rail_stations_used.get(destination, set())
    lorig = lorig.union(set(df_direct_rail_purged[df_direct_rail_purged.o_d == o_d].stop_start))
    ldest = ldest.union(set(df_direct_rail_purged[df_direct_rail_purged.o_d == o_d].stop_end))
    dict_rail_stations_used[origin] = lorig
    dict_rail_stations_used[destination] = ldest

###
# SAVE RESULTS
###

# Save stations used
with open(f_dict_rail_stations_airport_used, "wb") as f:
    pickle.dump(dict_rail_stations_used, f)

# Save rail purged per o-d pair
df_direct_rail_purged.to_csv(f_df_rail_purged)
df_direct_rail_purged.to_parquet(f_df_rail_purged_parquet)

# Do plot of rail stations
if do_plots:

    plt.figure(figsize=(20, 10))
    plt.interactive(False)

    ax = plt.axes(projection=ccrs.EuroPP())

    ax.add_feature(cf.COASTLINE, alpha=0.4)
    ax.add_feature(cf.BORDERS, alpha=0.4)

    ax.set_global()

    for a in airports_spain:
        plt.plot(dict_airp_latlon[a]['lon'], dict_airp_latlon[a]['lat'], 'bx', transform=ccrs.Geodetic())
        plt.text(dict_airp_latlon[a]['lon'], dict_airp_latlon[a]['lat'], a, transform=ccrs.Geodetic())

        rail_airport_matrix[a] = pd.to_numeric(rail_airport_matrix[a], errors='coerce')

        for index, re in rail_airport_matrix[rail_airport_matrix[a] < max_distance_rail_air].iterrows():
            plt.plot(re['stop_lon'], re['stop_lat'],
                     'gx', transform=ccrs.Geodetic())

        min_index = rail_airport_matrix[a].idxmin()
        if rail_airport_matrix.loc[min_index][a] < max_distance_rail_air:
            plt.plot(rail_airport_matrix.loc[min_index]['stop_lon'], rail_airport_matrix.loc[min_index]['stop_lat'],
                     'rx', transform=ccrs.Geodetic())

        for index, r in routes_within_spain.iterrows():
            plt.plot([r['dep_lon'], r['arr_lon']], [r['dep_lat'], r['arr_lat']], alpha=1, transform=ccrs.Geodetic())

    ax.set_extent(extent_plot)

    ax.coastlines()

    plt.savefig(output_fig_path,
                transparent=False,
                facecolor='white',
                bbox_inches='tight')

    plt.close()


# Close logging file
logging.shutdown()
