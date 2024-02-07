import logging
import pickle
import math
import os
import shutil
import time
import pandas as pd
from datetime import datetime, timedelta
from libs.general_tools import (haversine, date_range, save_pickle, save_csv_parquet,
                                zip_and_delete_folder, create_folder_if_not_exist)

"""
This code is the main one computing the impact of the rail bans.
- Reads the data
- Computes the connections which are valid with some parameters (to avoid non-operational itinearies)
- Computes the itineraries that are possible per day and applies the different bans to them
    - For each day a folder (then zipped) is produced containing information on the flights and trains
    - it also includes for each ban time the flights that would be left, the rail that would be used.
"""

start_time_computation_all = time.time()

###################
# DATA/PARAMETERS #
###################

# flights of interest set by hand in ds_interest with LE, and GC

# Data files
f_flight_schedules = './output/air/schedules_1st_week_0523_all_airlines_reduced.parquet'
f_airport_static = './data/airport_static.csv'
f_dict_rail_stations_airport_used = './output/rail/dict_rail_stations_used.pickle'
f_renfe_data_preloaded = './output/rail/rail_data_times_format.pickle'
f_df_rail_purged_parquet = './output/rail/df_rail_purged.parquet'

# Output files
folder_save = './output/multi/'
f_connectivity_parameters_info = folder_save + "airlines_info.pickle"
f_set_connection_not_valid = folder_save + "set_connection_not_valid.pickle"
f_set_connection_valid = folder_save + "set_connection_valid.pickle"
f_output_folder_log = './output/log/'
f_log = f_output_folder_log+'3_log_process_rail_air.txt'

# Create output folders if not exist
create_folder_if_not_exist(folder_save)
create_folder_if_not_exist(f_output_folder_log)

recompute_connection_valid = True  # If True, which connections are valid will be recomputed, if False, use pickled

# Parameters set manually

# Airlines alliances to allow connection
# In Spain only IBE so OneWorld - limitation, ds only has IBE, IBS, ANE, AEA, RYR, VLG (missing BWA for connections out)
dict_alliance = {'IBE': 'one',
                 'IBS': 'one',
                 'ANE': 'one'}

# Manually check rail stations used in key locations (LEMD, LEBL) to estimate connecting times
# print(stops[stops.stop_id.isin(dict_rail_stations_used.get('LEMD'))][['stop_id','stop_name']])
# print(stops[stops.stop_id.isin(dict_rail_stations_used.get('LEBL'))][['stop_id','stop_name']])

# Specific mct rail
dict_mct_rail_air = {17000: timedelta(minutes=math.ceil(22 + 15 / 2 + 25 + 30)),
                     # Chamartin 22 minutes train, 15 minutes time between trains, 25 minutes walking extra, 45 K2G
                     60000: timedelta(minutes=math.ceil(45 + 15 / 2 + 25 + 30)),
                     # Atocha 45 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 45 KG2
                     10000: timedelta(minutes=math.ceil(50 + 15 / 2 + 25 + 30)),
                     # Principe Pio 50 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 45 K2G
                     71801: timedelta(minutes=math.ceil(40 + 15 / 2 + 15 + 30)),
                     # Barcelona 30 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 45 K2G
                     }

dict_mct_air_rail = {17000: timedelta(minutes=math.ceil(22 + 15 / 2 + 25 + 30)),
                     # 22 minutes train, 15 minutes time between trains, 25 minutes walking extra, 25 K2G
                     60000: timedelta(minutes=math.ceil(45 + 15 / 2 + 25 + 30)),
                     # 45 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 25 KG2
                     10000: timedelta(minutes=math.ceil(50 + 15 / 2 + 25 + 30)),
                     # 50 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 25 K2G
                     71801: timedelta(minutes=math.ceil(40 + 15 / 2 + 15 + 30)),
                     # 30 minutes train/metro, 15 minutes time between trains, 25 minutes walking extra, 25 K2G
                     }

# Default mct rail and air
def_mct_rail_air = timedelta(minutes=100)
def_mct_air_rail = timedelta(minutes=60)
def_mct_air_air = timedelta(minutes=45)

# Save values airlines used
with open(f_connectivity_parameters_info, "wb") as f:
    pickle.dump((dict_alliance,
                 dict_mct_air_rail,
                 dict_mct_rail_air,
                 def_mct_air_rail,
                 def_mct_rail_air,
                 def_mct_air_air), f)

###
# Restrictions to allow o-h-d with flight.
###
# If we can go between o-d in 4h30 by train then no going there with a connection
max_time_train = pd.to_timedelta('4:30:00')
# If distance between o-d is less than 250 km then not going there with a connection
min_km_connection = 250

# BANS TO COMPUTE INFORMATION
# Define start and end times of threshold ban
start_time = pd.to_timedelta('0:00:00')
end_time = pd.to_timedelta('15:00:00')

# Define the step (15 minutes)
step = pd.to_timedelta('0:15:00')

# Define days to compute
start_date_str = "2023-05-01"
end_date_str = "2023-05-07"

#####################
# CONFIGURE LOGGING #
#####################

logging.basicConfig(filename=f_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

##################
# FUNCTIONS USED #
##################


# Add some information to direct flights
def process_direct_flights(ds_day):
    """
    Add some further information o_d, block_time, route, connection to the flight schedules dataset
    """
    # Direct flights
    ds_day_direct = ds_day.copy()
    ds_day_direct['o_d'] = ds_day_direct.apply(lambda row: (row['departure_used'], row['arrival_used']), axis=1)
    ds_day_direct['block_time'] = ds_day_direct['sibt'] - ds_day_direct['sobt']
    ds_day_direct['route'] = ds_day_direct['o_d']
    ds_day_direct['connection'] = False
    return ds_day_direct


# If two routes the same and second leg the same keep the fastest
def keep_fastest_alternative(df):
    """
    If there are different alternatives on a dataframe of options keep the one with the smaller block_time
    """
    # return df
    if len(df) > 1:
        return df.loc[df['block_time'].idxmin()]
    else:
        return df.iloc[0]


# Keep only tighter connections, purge alternatives for flight-flight
def keep_tighter_connection_from_orig_flight(df):
    return df[df.sobt_connection == df.sobt_connection.min()].copy()


def keep_tighter_departure(df):
    return df[df.sobt_orig == df.sobt_orig.min()].copy()


def purge_alternatives_flight_flight(df):
    if len(df) == 1:
        return df.copy()
    else:
        # group by first flight departing time
        df_first = df.groupby(['sobt_orig'], as_index=False).apply(keep_tighter_connection_from_orig_flight)

        # group by arrival to keep only tighter departure
        df_reduced = df_first.groupby(['sobt_connection'], as_index=False).apply(keep_tighter_departure)

        return df_reduced.reset_index(drop=True)


# Compute possible connections
def compute_possible_flight_connections(ds_day, ds_day_direct, def_mct_air_air):
    """
    Based on the schedules compute possible connections between flights
    """

    # Merge schedules with themselves considering arrival-departure to make possible connections
    ds_day_connecting = pd.merge(ds_day, ds_day, left_on='arrival_used', right_on='departure_used',
                                 suffixes=('_orig', '_connection'))

    # Filter the connections
    #ds_day_connecting[
    #    ['departure_used_orig', 'arrival_used_orig', 'departure_used_connection', 'arrival_used_connection']]
    ds_day_connecting['route'] = ds_day_connecting.apply(
        lambda row: (row['departure_used_orig'], row['departure_used_connection'], row['arrival_used_connection']),
        axis=1)
    # Invalid if middle airport is international and o-d are not
    ds_day_connecting['routes_invalid'] = ds_day_connecting.apply(lambda row:
                                                                  (row['departure_used_connection'][0:2] != row['arrival_used_connection'][0:2])
                                                                  and
                                                                  (row['departure_used_orig'][0:2] == row['arrival_used_connection'][0:2])
                                                                  ,
                                                                  axis=1)

    ds_day_connecting['o_d'] = ds_day_connecting.apply(
        lambda row: (row['departure_used_orig'], row['arrival_used_connection']), axis=1)

    # Filtering being: 
    # 1. avoid A - B - A routes
    # 2. avoid Spain (National) - International - Spain (National)
    # 3. connection being valid with restrictions from before
    # 5. same airline or within same alliance
    # 6. sibt+mct air < sobt next flight
    # 7. if more than one option keep the most 'efficient'/shorter

    ds_day_connecting = ds_day_connecting[
        (ds_day_connecting['departure_used_orig'] != ds_day_connecting['arrival_used_connection']) &
        (~ds_day_connecting['routes_invalid']) &
        (ds_day_connecting['o_d'].apply(lambda x: x not in set_connection_not_valid)) &
        ((ds_day_connecting['airline_orig'] == ds_day_connecting['airline_connection']) |
         ((ds_day_connecting['alliance_orig'] == ds_day_connecting['alliance_connection']) &
          (~ds_day_connecting['alliance_orig'].isnull()))) &
        (ds_day_connecting['sibt_orig'] + def_mct_air_air < ds_day_connecting['sobt_connection'])]

    # Compute some connection metrics
    ds_day_connecting = ds_day_connecting.copy()
    ds_day_connecting['block_time'] = ds_day_connecting['sibt_connection'] - ds_day_connecting['sobt_orig']

    ds_day_connecting = ds_day_connecting.groupby(['route', 'fid_connection']).apply(
        keep_fastest_alternative).reset_index(drop=True)

    ds_day_connecting['connection'] = True

    # Keep filtering options with connections
    # If we have direct flight instead of connecting, if the connection is more than 1.5 times direct then
    # connection not usable
    dict_fastest_direct = ds_day_direct.groupby(['o_d'])['block_time'].min().to_dict()
    ds_day_connecting['direct_flight_time'] = ds_day_connecting['o_d'].apply(lambda x: dict_fastest_direct.get(x))

    # Split connecting itineraries in itineraries with and without direct flight alternative
    ds_day_connecting_wo_direct = ds_day_connecting[ds_day_connecting['direct_flight_time'].isnull()].copy()
    ds_day_connecting_w_direct = ds_day_connecting[~ds_day_connecting['direct_flight_time'].isnull()].copy()

    # Filter options with direct if they take more than 1.5 times fastest direct alternative
    ds_day_connecting_w_direct = ds_day_connecting_w_direct[
        ds_day_connecting_w_direct['block_time'] < 1.5 * ds_day_connecting_w_direct['direct_flight_time']].copy()

    # print("After considering direct flight options",len(ds_day_connecting_w_direct)+len(ds_day_connecting_wo_direct),"itineraries with connections left")

    # We might have same o-d using different intermediate airport, if too much longer not valid either
    # Compute median time of option with connection
    dict_median_w_connection = ds_day_connecting_wo_direct.groupby(['o_d']).agg(
        median_block_time=pd.NamedAgg(column='block_time', aggfunc=lambda x: x.median() if len(x) > 0 else None)
    ).T.to_dict()

    # Filter options wo direct if they take more than 1.5 times median of alternatives
    ds_day_connecting_wo_direct = ds_day_connecting_wo_direct[
        ds_day_connecting_wo_direct['block_time'] < 1.5 * ds_day_connecting_wo_direct['o_d'].apply(
            lambda x: dict_median_w_connection[x]['median_block_time'])].copy()

    # print("After considering indirect flight options",len(ds_day_connecting_w_direct)+len(ds_day_connecting_wo_direct),"itineraries with connections left")
    # print("From",len(ds_day_connecting),"flights w connections to",len(ds_day_connecting_wo_direct)+len(ds_day_connecting_w_direct),"kept")

    ds_day_connecting = pd.concat([ds_day_connecting_wo_direct,
                                   ds_day_connecting_w_direct], ignore_index=True)

    # Add o-d legs
    ds_day_connecting['o_d_leg1'] = ds_day_connecting.apply(lambda x: (x.departure_used_orig, x.arrival_used_orig),
                                                            axis=1)
    ds_day_connecting['o_d_leg2'] = ds_day_connecting.apply(
        lambda x: (x.departure_used_connection, x.arrival_used_connection), axis=1)

    # Purge/filter to keep only most efficient connections
    ds_day_connecting['route_airlines'] = ds_day_connecting.apply(lambda x: (x.airline_orig, x.airline_connection),
                                                                  axis=1)
    ds_day_connecting = ds_day_connecting.groupby(['route', 'route_airlines'], as_index=False).apply(
        purge_alternatives_flight_flight)

    return ds_day_connecting


# Put together connecting and direct with basic o-d information
def compute_all_flight_itineraries_basic_info(ds_day_direct, ds_day_connecting):
    """
   Add basic information to flight schedules joining direct and connecting flights info
   """
    # All flight itineraries on day
    ds_day_f_it = pd.concat([ds_day_connecting[['o_d', 'block_time', 'route', 'connection']],
                             ds_day_direct[['o_d', 'block_time', 'route', 'connection']]], ignore_index=True)

    # Add origin and destination
    ds_day_f_it['origin'] = ds_day_f_it['o_d'].apply(lambda x: x[0])
    ds_day_f_it['destination'] = ds_day_f_it['o_d'].apply(lambda x: x[1])

    return ds_day_f_it


def compute_day_flights_rail_itineraries(day_computation, ds_interest, df_direct_rail_purged):
    """
    Given a day, compute flight and rail itineraries available.
    For that it adds information on the direct flights, computes possible connections between flights and the
    itineraries possible using flights. It also gets all the train itineraries available in the day.
    """
    ds_day = ds_interest[(ds_interest.day_sobt == day_computation) | (ds_interest.day_sibt == day_computation)].copy()
    dt_day = df_direct_rail_purged[df_direct_rail_purged.day == day_computation.strftime("%Y%m%d")].copy()
    logging.info("Flights in day %d", len(ds_day))
    logging.info("Rail in day %d", len(dt_day))
    logging.info("-----\n")

    ds_day_direct = process_direct_flights(ds_day)
    logging.info("Number direct itineraries flight %d", len(ds_day_direct))

    ds_day_connecting = compute_possible_flight_connections(ds_day, ds_day_direct,
                                                            def_mct_air_air)
    logging.info("Number connecting itineraries flight %d", len(ds_day_connecting))

    ds_day_f_it = compute_all_flight_itineraries_basic_info(ds_day_direct, ds_day_connecting)
    logging.info("Itineraries flight total for day %d", len(ds_day_f_it))

    logging.info("-----\n")

    # Compute rail_alternative to o-d that have connections (e.g., Zaragoza-Malaga)
    dt_baseline = dt_day[dt_day.o_d.isin(ds_day_f_it[ds_day_f_it.connection]['o_d'].drop_duplicates())].copy()

    logging.info("Train itineraries baseline %d", len(dt_baseline))

    return ds_day, ds_day_direct, ds_day_connecting, ds_day_f_it, dt_day, dt_baseline


def compute_dict_od_remove(timedelta_range, df_agg_rail_stats):
    """
    Compute for each time in a given timedelta range the o-d pairs that would be removed considering
    if there are trains that can link the o-d pair in less time than the time computed.
    The outcome is a dictionary where the key is a time (time ban) and the value is a set of o-d pairs
    that are covered by rail, i.e., banned by air.
    """
    # Compute dict_od_remove for bans analysed
    dict_od_remove = {}
    for t in timedelta_range:
        dict_od_remove[t] = set(df_agg_rail_stats[df_agg_rail_stats.min_leg_time < t].o_d)

    return dict_od_remove


def compute_ban(ds_day_connecting, ds_day_direct, dt_day, dict_mct_rail_air, dict_mct_air_rail, def_mct_rail_air,
                def_mct_air_rail, od_remove, airline_connections_used_air=None):
    """
    Main function to compute the impact of the air ban.
    Computes the impact of the ban given a set of o-d pairs to remove. It removes those flights and then check if the
    itineraries that are not possible anymore can be done by rail (direct rails).
    """

    # COMPUTE ITINERARIES THAT ARE KEPT FROM DIRECT AND CONNECTING FLIGHTS

    # print(len(od_remove))

    ds_day_connecting_ban = ds_day_connecting[
        (~ds_day_connecting.o_d_leg1.isin(od_remove)) & (~ds_day_connecting.o_d_leg2.isin(od_remove))].copy()
    ds_day_connecting_ban_leg1_removed = ds_day_connecting[(ds_day_connecting.o_d_leg1.isin(od_remove))].copy()
    ds_day_connecting_ban_leg2_removed = ds_day_connecting[(ds_day_connecting.o_d_leg2.isin(od_remove))].copy()

    ds_day_direct_ban = ds_day_direct[~ds_day_direct.o_d.isin(od_remove)].copy()
    ds_day_direct_ban_removed = ds_day_direct[ds_day_direct.o_d.isin(od_remove)].copy()

    set_od_pair_f_removed = set(ds_day_connecting_ban_leg1_removed['o_d_leg1']).union(
        set(ds_day_connecting_ban_leg2_removed['o_d_leg2'])).union(set(ds_day_direct_ban_removed['o_d']))

    # COMPUTE CONNECTIONS WITH RAIL

    df_rail_ban = dt_day[(dt_day.o_d.isin(set_od_pair_f_removed))].copy()

    # FIRST IF RAIL IS LEG2

    # Merge flights with rail
    ds_follow_up_rail = pd.merge(ds_day_direct_ban, df_rail_ban, left_on='arrival_used', right_on='departure_used',
                                 suffixes=('_orig', '_train_l2'))

    if len(ds_follow_up_rail) > 0:
        # Filter the connections
        ds_follow_up_rail['mcr'] = ds_follow_up_rail['stop_start'].apply(
            lambda x: dict_mct_air_rail.get(x, def_mct_air_rail))

        ds_follow_up_rail = ds_follow_up_rail[
            ds_follow_up_rail['sibt'] + ds_follow_up_rail['mcr'] < ds_follow_up_rail.apply(
                lambda x: datetime.combine(x.day_sibt, x.departure_time.time()), axis=1)]

        ds_follow_up_rail['route'] = ds_follow_up_rail.apply(
            lambda x: (x.departure_used_orig, x.arrival_used_orig, x.arrival_used_train_l2), axis=1)

        ds_follow_up_rail['block_time'] = ds_follow_up_rail.apply(
            lambda x: datetime.combine(x.day_sibt, x.arrival_time.time()) - x.sobt, axis=1)

        # If two routes the same and second leg the same keep the fastest
        ds_follow_up_rail = ds_follow_up_rail.groupby(['airline', 'route', 'arrival_time']).apply(
            keep_fastest_alternative).reset_index(drop=True)

        # previous gives fastest considering same arrival time, but now we could have same route
        # getting different trains, which wouldn't be good either:
        # for example Flight A - Rail B, or Flight A - Rail C, both would be kept as rail is different
        # (different arrival time)
        # so now we need to check if for same route there is same departure keep fastest

        ds_follow_up_rail = ds_follow_up_rail.groupby(['airline', 'route', 'sobt']).apply(
            keep_fastest_alternative).reset_index(drop=True)

        ds_follow_up_rail['airline_route'] = ds_follow_up_rail.apply(lambda x: (x.airline, x.route), axis=1)

        if airline_connections_used_air is not None:
            # Filter rail-air, air-rail to keep only routes offered by airlines before
            # print("----")
            # print("AR",len(ds_follow_up_rail))
            ds_follow_up_rail = ds_follow_up_rail[
                ds_follow_up_rail.airline_route.isin(airline_connections_used_air)].copy()
            # print("AR",len(ds_follow_up_rail))

    # SECOND IF RAIL IS LEG1

    ds_first_rail = pd.merge(ds_day_direct_ban, df_rail_ban, left_on='departure_used', right_on='arrival_used',
                             suffixes=('_orig', '_train_l1'))

    if len(ds_first_rail) > 0:
        # Filter the connections
        ds_first_rail['mcr'] = ds_first_rail['stop_end'].apply(lambda x: dict_mct_rail_air.get(x, def_mct_rail_air))

        ds_first_rail = ds_first_rail[ds_first_rail['sobt'] - ds_first_rail['mcr']
                                      > ds_first_rail.apply(
            lambda x: datetime.combine(x.day_sobt, x.arrival_time.time()), axis=1)]

        ds_first_rail['route'] = ds_first_rail.apply(
            lambda x: (x.departure_used_train_l1, x.departure_used_orig, x.arrival_used_orig), axis=1)

        ds_first_rail['block_time'] = ds_first_rail.apply(
            lambda x: x.sibt - datetime.combine(x.day_sobt, x.departure_time.time()), axis=1)

        # If two routes the same and second leg the same keep the fastest
        ds_first_rail = ds_first_rail.groupby(['airline', 'route', 'sobt']).apply(keep_fastest_alternative).reset_index(
            drop=True)
        # As before now I need to check if there's same train linking with two subsequent flights, keep fastest
        ds_follow_up_rail = ds_follow_up_rail.groupby(['airline', 'route', 'arrival_time']).apply(
            keep_fastest_alternative).reset_index(drop=True)

        ds_first_rail['airline_route'] = ds_first_rail.apply(lambda x: (x.airline, x.route), axis=1)

        if airline_connections_used_air is not None:
            # Filter rail-air, air-rail to keep only routes offered by airlines before
            # print("RA",len(ds_first_rail))
            ds_first_rail = ds_first_rail[ds_first_rail.airline_route.isin(airline_connections_used_air)].copy()
            # print("RA",len(ds_first_rail))
            # print("---")

    return ds_day_connecting_ban, ds_day_direct_ban, ds_follow_up_rail, ds_first_rail, df_rail_ban


##########################################
# PROCESSING AIR RAIL AS FUNCTION OF BAN #
##########################################

####
# READ DATA/CLEAN AND PREPARE
####

logging.info("\n----------\nREAD RAIL DATA\n----------\n")

# Read flight schedules and rotations
ds = pd.read_parquet(f_flight_schedules)

# Give id to each flight
ds['fid'] = ds.index

# Read dict rail stations used per airport
with open(f_dict_rail_stations_airport_used, "rb") as f:
    dict_rail_stations_used = pickle.load(f)

# Read rail data
with open(f_renfe_data_preloaded, "rb") as f:
    agency, calendar, calendar_dates, routes, stops_times, stops, trips = pickle.load(f)

# Read rail data
df_direct_rail_purged = pd.read_parquet(f_df_rail_purged_parquet)
df_direct_rail_purged['o_d'] = df_direct_rail_purged['o_d'].apply(lambda x: tuple(x))
df_direct_rail_purged['departure_used'] = df_direct_rail_purged['o_d'].apply(lambda x: x[0])
df_direct_rail_purged['arrival_used'] = df_direct_rail_purged['o_d'].apply(lambda x: x[1])


####
# COMPUTE RAIL STATISTICS NEEDED FOR BAN
####

logging.info("\n----------\nCOMPUTE RAIL STATISTICS NEEDED FOR BAN IMPLEMENTATION\n----------\n")

df_agg_rail_stats = df_direct_rail_purged.groupby(['day', 'o_d']).agg(
    number_of_elements=pd.NamedAgg(column='departure_time', aggfunc='size'),
    first_departure_time=pd.NamedAgg(column='departure_time', aggfunc='first'),
    last_departure_time=pd.NamedAgg(column='departure_time', aggfunc='last'),
    first_arrival_time=pd.NamedAgg(column='arrival_time', aggfunc='first'),
    last_arrival_time=pd.NamedAgg(column='arrival_time', aggfunc='last'),
    min_leg_time=pd.NamedAgg(column='leg_time', aggfunc='min'),
    max_leg_time=pd.NamedAgg(column='leg_time', aggfunc='max'),
    median_leg_time=pd.NamedAgg(column='leg_time', aggfunc=lambda x: x.median() if len(x) > 0 else None),
    mean_leg_time=pd.NamedAgg(column='leg_time', aggfunc=lambda x: x.mean() if len(x) > 0 else None)
).reset_index()

####
# FILTER FLIGHTS OF INTEREST
####

logging.info("\n----------\nFILTER FLIGHTS OF INTEREST\n----------\n")

# Compute some parameters on flights
ds['country_departure'] = ds['departure_used'].apply(lambda x: x[0:2])
ds['country_arrival'] = ds['arrival_used'].apply(lambda x: x[0:2])
ds['day_sobt'] = ds['sobt'].apply(lambda x: x.date())
ds['day_sibt'] = ds['sibt'].apply(lambda x: x.date())
ds['alliance'] = ds['airline'].apply(lambda x: dict_alliance.get(x))

# ds_interest == all flights coming/going/within Spain (and Canary Islands) and avoid circular flights
#              and avoid flights to/from NULL (gcd_km is NULL), remove LEBL-LEGE (outlier)
ds_interest = ds[((ds.departure_used.str.startswith('LE')) |
                  (ds.arrival_used.str.startswith('LE')) |
                  (ds.arrival_used.str.startswith('GC')) |
                  (ds.arrival_used.str.startswith('GC'))) &
                 (ds.arrival_used != ds.departure_used) &
                 ~((ds.arrival_used == 'LEGE') & (ds.departure_used == 'LEBL')) &
                 (~ds.gcd_km.isnull())
                 ].copy()

ds_interest['o_d'] = ds_interest.apply(lambda row: (row['departure_used'], row['arrival_used']), axis=1)

# ds includes all flights airlines considered, e.g. all flights of RYR
logging.info("Flights origin: %d, Flights interest: %d", len(ds), len(ds_interest))

####
# COMPUTE CONNECTIONS WHICH ARE VALID
####

logging.info("\n----------\nCOMPUTE CONNECTIONS VALID\n----------\n")

# Compute if connections are valid
# This takes a bit to computes as it checks all airports against all airports
# (save in pickle to next time can be avoided)

# Compute which o-d pairs are valid for routes which links them via a connection

d_airports = pd.read_csv(f_airport_static)

set_connection_not_valid = None
set_connection_valid = None
if not recompute_connection_valid:
    try:
        with open(f_set_connection_not_valid, "rb") as f:
            set_connection_not_valid = pickle.load(f)
        with open(f_set_connection_valid, "rb") as f:
            set_connection_valid = pickle.load(f)
    except FileNotFoundError:
        pass

if set_connection_not_valid is None:
    logging.info("Computing connections valid and no valid between o-d pairs")

    set_connection_not_valid = set()
    set_connection_valid = set()

    for a1 in set(ds_interest.departure_used).union(set(ds_interest.arrival_used)):
        for a2 in set(ds_interest.departure_used).union(set(ds_interest.arrival_used)):

            # Check if train between cities
            if len(df_direct_rail_purged[df_direct_rail_purged.o_d == (a1, a2)]) > 0:
                # We have train between those cities
                train_time = df_direct_rail_purged[df_direct_rail_purged.o_d == (a1, a2)].leg_time.min()
            else:
                train_time = pd.to_timedelta('24:00:00')

            # If o_d are the same
            # or o_d has a direct train faster than max_time_train
            # or o_d has a direct flight
            # or o_d are closer than min_km_connection
            if (a1 == a2) or train_time < max_time_train or len(ds_interest[ds_interest.o_d == (a1, a2)]) > 0:
                set_connection_not_valid.add((a1, a2))

            else:
                # check distance done here so that distance only computed when needed

                dist_km = haversine(d_airports[d_airports.ICAO == a1].iloc[0].lon,
                                    d_airports[d_airports.ICAO == a1].iloc[0].lat,
                                    d_airports[d_airports.ICAO == a2].iloc[0].lon,
                                    d_airports[d_airports.ICAO == a2].iloc[0].lat)
                if dist_km < min_km_connection:
                    set_connection_not_valid.add((a1, a2))
                else:
                    set_connection_valid.add((a1, a2))

    with open(f_set_connection_not_valid, "wb") as f:
        pickle.dump(set_connection_not_valid, f)
    with open(f_set_connection_valid, "wb") as f:
        pickle.dump(set_connection_valid, f)

logging.info("Connections valid %d, Connections not valid %d", len(set_connection_valid), len(set_connection_not_valid))


####
# COMPUTE ITINERARIES POSSIBLE PER DAY AND APPLY BAN
####

logging.info("\n----------\nAPPLY BAN FOR A SET OF DAYS\n----------\n")

# Create a range of Timedelta objects
timedelta_range = pd.timedelta_range(start=start_time, end=end_time, freq=step)

# Compute dictionary of o-d get banned as a function of ban. This is independent on day
dict_od_remove = compute_dict_od_remove(timedelta_range, df_agg_rail_stats)

start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

for day_computation in date_range(start_date, end_date):

    start_time_computation_day = time.time()

    # Define day of computation
    day_computation = day_computation.date()

    logging.info("Computing ban for day: "+str(day_computation))

    # Compute flights and trains itineraries on day
    ds_day, ds_day_direct, ds_day_connecting, ds_day_f_it, dt_day, dt_baseline = compute_day_flights_rail_itineraries(
        day_computation, ds_interest, df_direct_rail_purged)

    # Save baseline values
    file_folder = folder_save + day_computation.strftime("%Y%m%d")
    # Create folder for output (if exist delete it firs)
    if os.path.exists(file_folder):
        shutil.rmtree(file_folder)
    os.makedirs(file_folder)

    save_csv_parquet(dt_baseline, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_it_trains_baseline')
    save_csv_parquet(ds_day, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_ds_day')
    save_csv_parquet(ds_day_direct, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'ds_day_direct')
    save_csv_parquet(ds_day_connecting,
                     file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'ds_day_connecting')
    save_csv_parquet(ds_day_f_it, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'ds_day_f_it')
    save_csv_parquet(dt_day, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'dt_day')
    save_csv_parquet(dt_baseline, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'dt_baseline')
    save_pickle(dict_od_remove, file_folder + "/" + day_computation.strftime("%Y%m%d") + '_' + 'dict_od_remove')

    # Apply threshold ban

    # Create a range of Timedelta objects
    timedelta_range = pd.timedelta_range(start=start_time, end=end_time, freq=step)

    # removed previous to avoid doing bans that don't remove extra flights
    removed_previous = None

    # airlines connections used in air itineraries
    airline_connections_used_air = set(ds_day_connecting.apply(lambda x: (x.airline_orig, x.route), axis=1)).union(
        ds_day_connecting.apply(lambda x: (x.airline_connection, x.route), axis=1))

    for ban_time in dict_od_remove.keys():

        od_remove = dict_od_remove.get(ban_time)

        if removed_previous != od_remove:
            removed_previous = od_remove

            file_name_header = file_folder + "/" + day_computation.strftime("%Y%m%d") + "_{:02}{:02}".format(
                int(ban_time.total_seconds() / 60 // 60), int(ban_time.total_seconds() / 60 % 60)) + "_"

            # Compute flights, rails itineraries with ban
            ds_day_f_connecting, ds_day_direct_ban, ds_ar, ds_ra, ds_r = \
                compute_ban(ds_day_connecting, ds_day_direct, dt_day,
                            dict_mct_rail_air, dict_mct_air_rail, def_mct_rail_air, def_mct_air_rail,
                            od_remove, airline_connections_used_air=airline_connections_used_air)

            # Standardise origin/destination
            ds_rb = dt_baseline.copy()

            # Keep only from the baseline trains the ones that are removed from f-f
            ds_rb = ds_rb.merge(ds_day_f_connecting[['o_d', 'airline_orig']], on='o_d', how='left')
            ds_rb = ds_rb[ds_rb.airline_orig.isnull()]
            ds_rb = ds_rb.drop('airline_orig', axis=1)

            if len(ds_day_direct_ban) > 0:
                ds_day_direct_ban['origin'] = ds_day_direct_ban['route'].apply(lambda x: x[0])
                ds_day_direct_ban['destination'] = ds_day_direct_ban['route'].apply(lambda x: x[-1])
            else:
                ds_day_direct_ban['origin'] = None
                ds_day_direct_ban['destination'] = None

            if len(ds_day_f_connecting) > 0:
                ds_day_f_connecting['origin'] = ds_day_f_connecting['route'].apply(lambda x: x[0])
                ds_day_f_connecting['destination'] = ds_day_f_connecting['route'].apply(lambda x: x[-1])
            else:
                ds_day_f_connecting['origin'] = None
                ds_day_f_connecting['destination'] = None

            if len(ds_ra) > 0:
                ds_ra['origin'] = ds_ra['route'].apply(lambda x: x[0])
                ds_ra['destination'] = ds_ra['route'].apply(lambda x: x[-1])
            else:
                ds_ra['origin'] = None
                ds_ra['destination'] = None

            if len(ds_ar) > 0:
                ds_ar['origin'] = ds_ar['route'].apply(lambda x: x[0])
                ds_ar['destination'] = ds_ar['route'].apply(lambda x: x[-1])
            else:
                ds_ar['origin'] = None
                ds_ar['destination'] = None

            ds_r['origin'] = ds_r['departure_used']
            ds_r['destination'] = ds_r['arrival_used']
            ds_rb['origin'] = ds_rb['departure_used']
            ds_rb['destination'] = ds_rb['arrival_used']

            if len(ds_r) > 0:
                ds_r['route'] = ds_r.apply(lambda x: (x.origin, x.destination), axis=1)
            else:
                ds_r['route'] = None

            if len(ds_rb) > 0:
                ds_rb['route'] = ds_rb.apply(lambda x: (x.origin, x.destination), axis=1)
            else:
                ds_rb['route'] = None

            # o-d legs
            ds_day_direct_ban['o_d_leg1'] = ds_day_direct_ban['o_d']
            ds_day_direct_ban['o_d_leg2'] = None
            ds_ar['o_d_leg1'] = ds_ar['o_d_orig']
            ds_ar['o_d_leg2'] = ds_ar['o_d_train_l2']
            ds_ra['o_d_leg1'] = ds_ra['o_d_train_l1']
            ds_ra['o_d_leg2'] = ds_ra['o_d_orig']
            ds_r['o_d_leg1'] = ds_r['route']
            ds_r['o_d_leg2'] = None
            ds_rb['o_d_leg1'] = ds_rb['route']
            ds_rb['o_d_leg2'] = None

            # Add airlines used
            if len(ds_day_f_connecting) > 0:
                ds_day_f_connecting['route_airlines'] = ds_day_f_connecting.apply(
                    lambda x: (x.airline_orig, x.airline_connection), axis=1)
            else:
                ds_day_f_connecting['route_airlines'] = None

            ds_day_direct_ban['route_airlines'] = ds_day_direct_ban['airline'].apply(lambda x: (x))

            if len(ds_ar) > 0:
                ds_ar['route_airlines'] = ds_ar.apply(lambda x: (x.airline, x.route_short_name), axis=1)
                ds_ar['route_stations'] = ds_ar.apply(lambda x: (x.stop_start, x.stop_end), axis=1)
            else:
                ds_ar['route_airlines'] = None
                ds_ar['route_stations'] = None
            if len(ds_ra) > 0:
                ds_ra['route_airlines'] = ds_ra.apply(lambda x: (x.route_short_name, x.airline), axis=1)
                ds_ra['route_stations'] = ds_ra.apply(lambda x: (x.stop_start, x.stop_end), axis=1)
            else:
                ds_ra['route_airlines'] = None
                ds_ra['route_stations'] = None
            if len(ds_r) > 0:
                ds_r['route_stations'] = ds_r.apply(lambda x: (x.stop_start, x.stop_end), axis=1)
                ds_r['route_airlines'] = ds_r.apply(lambda x: (x.route_short_name), axis=1)
            else:
                ds_r['route_stations'] = None
                ds_r['route_airlines'] = None
            if len(ds_rb) > 0:
                ds_rb['route_stations'] = ds_rb.apply(lambda x: (x.stop_start, x.stop_end), axis=1)
                ds_rb['route_airlines'] = ds_rb.apply(lambda x: (x.route_short_name), axis=1)
            else:
                ds_rb['route_stations'] = None
                ds_rb['route_airlines'] = None

            # Add info type of itinerary
            if len(ds_day_direct_ban) > 0:
                ds_day_direct_ban['type'] = 'flight'
            else:
                ds_day_direct_ban['type'] = None

            if len(ds_day_f_connecting) > 0:
                ds_day_f_connecting['type'] = 'flight_flight'
            else:
                ds_day_f_connecting['type'] = None

            if len(ds_ra) > 0:
                ds_ra['type'] = 'rail_flight'
            else:
                ds_ra['type'] = None

            if len(ds_ar) > 0:
                ds_ar['type'] = 'flight_rail'
            else:
                ds_ar['type'] = None

            if len(ds_r) > 0:
                ds_r['type'] = 'rail'
                ds_r['block_time'] = ds_r.arrival_time - ds_r.departure_time
            else:
                ds_r['type'] = None
                ds_r['block_time'] = None
            if len(ds_rb) > 0:
                ds_rb['type'] = 'rail'
                ds_rb['block_time'] = ds_rb.arrival_time - ds_rb.departure_time
            else:
                ds_rb['type'] = None
                ds_rb['block_time'] = None

            if len(ds_ar) > 0:
                ds_ar['airline_route'] = ds_ar.airline_route.apply(lambda x: [x[0]] + list(x[1]))
            if len(ds_ra) > 0:
                ds_ra['airline_route'] = ds_ra.airline_route.apply(lambda x: [x[0]] + list(x[1]))

            # Save results
            save_pickle(od_remove, file_name_header + 'od_remove')
            save_csv_parquet(ds_day_f_connecting, file_name_header + 'ds_day_f_connecting')
            save_csv_parquet(ds_day_direct_ban, file_name_header + 'ds_day_direct_ban')
            save_csv_parquet(ds_ar, file_name_header + 'ds_ar')
            save_csv_parquet(ds_ra, file_name_header + 'ds_ra')
            save_csv_parquet(ds_r, file_name_header + 'ds_rail_ban')
            save_csv_parquet(ds_rb, file_name_header + 'ds_rb')

    zip_and_delete_folder(file_folder)

    end_time = time.time()
    elapsed_time = end_time - start_time_computation_day
    minutes, seconds = divmod(elapsed_time, 60)

    logging.info("\nDay %s done in: %d minutes %d seconds", day_computation, round(minutes), round(seconds))
    logging.info("********************\n")

end_time = time.time()
elapsed_time = end_time - start_time_computation_all
minutes, seconds = divmod(elapsed_time, 60)

logging.info("Total computation done in: %d minutes %d seconds", round(minutes), round(seconds))

# Close logging file
logging.shutdown()
