import logging
import pandas as pd
import numpy as np
from datetime import datetime
from collections import Counter
from libs.general_tools import haversine, create_folder_if_not_exist

"""
This code estimates the flight schedules from the OpenSky historic flight_data4 table.
As it is in the context of multimodal trips within a country, it filters the flights to keep only
the ones from the airlines operating in Spain of interest. The code then fixes the airports
of origin/destination to ensure that rotations make sense. In some cases first or last airports
might be NULL and that cannot be fixed.

The code will print its progress along the way with some statistics.

Some things can be manually adjusted once the data is read/processed, e.g. list of airlines of interest once
the information on the airlines operating in target country are find.

It then saves the schedules into a csv and parquet files.

Input:
  - flights_data4 files
  - aircraftDatabase file from OpenSky with information on aircraft frames based on icao24 transponder id
  - airport_static file with information on lat-lon of airports
  - fixed_airports file containing relationship between airport icao 'wrong' and correct ICAO id.

Output:
    - airlines_within_country file with information on number of flights airlines have in target country
    - airport_static_multimodality_country file with information on which airports are kept in the end after filtering
      and fixing rotations
    - schedules_xxx_all_columns file with new schedules with all columns
    - schedules_xxx_reduced files (in csv and parquet) with information needed from schedules kept.
"""

####################
# DATA/PARAMETERS #
###################

country_of_interest = 'LE'

# Data files
f_flight_data_opensky = './data/flights_data4/year=2023/month=05/1st_week_0523.csv'
f_aircraft_db_opensky = './data/aircraftDatabase.csv'
f_airport_static = './data/airport_static.csv'
f_fixed_airports = './data/manual_fixed_airports.csv'

# Airlines of interest for the study (airlines with flights within Spain)
airlines_of_interest = ['VLG', 'RYR', 'ANE', 'AEA', 'IBE', 'IBS']  # Removed 'SWT' most flights are in Balearic islands

# Parameters set manually used to estimate take-off, landing times; and then AOBT and AIBT
climb_speed = 610  # m/min --> this could be improved with values per ac type
descend_speed = 460  # m/min --> this could be improved with values per ac type
taxi_out_avg = 20  # --> this could be improved with values per airport
taxi_in_avg = 10  # --> this could be improved with values per airport


# Output files
f_output_folder = './output/air/'
f_all_output_csv = './output/air/schedules_1st_week_0523_all_airlines_all_columns.csv'
f_reduced_output_csv = './output/air/schedules_1st_week_0523_all_airlines_reduced.csv'
f_reduced_output_parquet = './output/air/schedules_1st_week_0523_all_airlines_reduced.parquet'
f_airlines_country_of_interest = './output/air/airlines_within_'+country_of_interest+'.csv'

f_output_folder_log = "./output/log/"
f_log = (f_output_folder_log+'/1_log_generate_schedules.txt')

# Create output folders if not exist
create_folder_if_not_exist(f_output_folder)
create_folder_if_not_exist(f_output_folder_log)

#####################
# CONFIGURE LOGGING #
#####################

logging.basicConfig(filename=f_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')


##################
# FUNCTIONS USED #
##################

# Process rotations per airline
def compute_rotations(df):
    df = df.copy()
    logging.info("Number flights of airline %s: %d", df.airline.iloc[0], len(df))
    logging.info("Number of ac of airline %s: %d", df.airline.iloc[0], len(df.icao24.drop_duplicates()))

    # For each ac follow it through the day and identify which rotations are 'wrong'
    # They will either miss a leg or have airport of departure, arrival wrong

    # First assume departure and arrival are as estimated
    df['departure'] = df['departure_used']
    df['arrival'] = df['arrival_used']
    
    # Sort the dataframe by ac and firstseen
    df = df.sort_values(['icao24', 'firstseen'])

    # Add column to highlight rotation mismatch with respect to previous flight
    df['previous_rotation_mismatch'] = df['departure'] != df['arrival'].shift()
    df['rotation_previous_time'] = df['firstseen'] - df['lastseen'].shift()

    # Remove previous_rotation_mismatch and previous time for first flight of a given ac type
    df.loc[df['icao24'] != df['icao24'].shift(), 'previous_rotation_mismatch'] = False
    df.loc[df['icao24'] != df['icao24'].shift(), 'rotation_previous_time'] = None

    logging.info('Aircraft with wrong rotation over: %d - %f %%',
                 len(df.loc[df.previous_rotation_mismatch].icao24.drop_duplicates()),
                 round(100*(len(df.loc[df.previous_rotation_mismatch].icao24.drop_duplicates()) /
                            len(df.icao24.drop_duplicates())), 2))
    
    logging.info("---\n")
    
    return df    


def process_rotations_mismatch(group, dict_avg_times, dict_airports_used, threshold_same_airport=80,
                               minimum_time_ground=50*60, minimum_fly_time_to_add=60*60):
    """
    Function to process flights with 'wrong' rotations (process by ac id at a time)
    'Fix' the wrong rotations
    1. If destination and origin of rotation are very close (less than a threshold_same_airport), then assume
       they were mislabelled. Change them to be the same:  Choose airport with more operations
    2. If destination and origin of rotation are very far away:
       consider adding an extra segment (if time allows) between those two airports. If not fix as in 1.
    3. If one of them is None, consider previous rotation?
    """
    
    # group = group.copy()
    
    # group = group.sort_values(['firstseen']).reset_index(drop=True)
    
    # the function will return three list:
    # - arrivals that need to be changed
    # - departures that need to be changed
    # - new rotations/flights that need to be added
    list_replace_arrival = []
    list_replace_departure = []
    list_add = []

    # There is only one flight for that ac and we don't have the arrival
    if len(group) == 1 and group.iloc[0].arrival == 'NULL':
        idx = 0
        # replace the NULL arrival for a 'new' NULL.
        list_replace_arrival.append({'icao24': group.iloc[idx].icao24,
                                     'idx': idx,
                                     'row_id': group.iloc[idx].name,
                                     'arrival_orig': group.iloc[idx].arrival,
                                     'arrival_new': 'NULL_'+str(group.iloc[idx].icao24)+'_'+str(group.iloc[idx].name)})

    # Now process all flights in the group 
    for idx in range(len(group)):
        # If the first flight of the rotations departs from NULL
        if idx == 0 and group.iloc[idx].departure == 'NULL':
            if len(group) == 2:
                # Only a return trip which is missing first origin
                # We are going to assume comes from same place
                if group.iloc[idx+1].arrival != 'NULL':
                    list_replace_departure.append({'icao24': group.iloc[idx].icao24,
                                                   'idx': idx,
                                                   'row_id': group.iloc[idx].name,
                                                   'departure_orig': group.iloc[idx].departure,
                                                   'departure_new': group.iloc[idx+1].arrival})
                else:
                    # Both are NULL so keep NULL in arrival and departure
                    list_replace_arrival.append({'icao24': group.iloc[idx].icao24,
                                                 'idx': idx+1,
                                                 'row_id': group.iloc[idx+1].name,
                                                 'arrival_orig': group.iloc[idx+1].arrival,
                                                 'arrival_new': 'NULL_'+str(group.iloc[idx].icao24)+'_'+str(group.iloc[idx].name)})
                    list_replace_departure.append({'icao24': group.iloc[idx].icao24,
                                                   'idx': idx,
                                                   'row_id': group.iloc[idx].name,
                                                   'departure_orig': group.iloc[idx].departure,
                                                   'departure_new': 'NULL_'+str(group.iloc[idx].icao24)+'_'+str(group.iloc[idx].name)})
            else:
                # We don't know from where it came, different assumptions could be done, for now just keep NULL
                list_replace_departure.append({'icao24': group.iloc[idx].icao24,
                                               'idx': idx,
                                               'row_id': group.iloc[idx].name,
                                               'departure_orig': group.iloc[idx].departure,
                                               'departure_new': 'NULL_'+str(group.iloc[idx].icao24)+'_'+str(group.iloc[idx].name)})

        # We are looking at second flight onward   
        if idx > 0:
            # Get information of current departure and previous arrival
            current_row = group.iloc[idx]
            prev_row = group.iloc[idx - 1]
            arrival_prev = prev_row.arrival
            departure_curr = current_row.departure

            if (idx == len(group)-1) and (current_row.arrival == 'NULL'):
                # This is the final flight of the sequence, and we arrive to NULL
                if (len(group) == 2) and (prev_row.departure != 'NULL'):
                    # We only have two trips, assume return if departure previous is not NULL
                    list_replace_arrival.append({'icao24': current_row.icao24,
                                                 'idx': idx,
                                                 'row_id': current_row.name,
                                                 'arrival_orig': current_row.arrival,
                                                 'arrival_new': prev_row.departure})
                elif len(group) > 2:
                    # The other option (2 flights and previous depature NULL already considered in idx==0)
                    list_replace_arrival.append({'icao24': current_row.icao24,
                                                 'idx': idx,
                                                 'row_id': current_row.name,
                                                 'arrival_orig': current_row.arrival,
                                                 'arrival_new': 'NULL_'+str(current_row.icao24)+'_'+str(current_row.name)})

            if current_row['previous_rotation_mismatch'] or (arrival_prev == 'NULL') or (departure_curr == 'NULL'):
                # The previous rotation arrival doesn't match the departure of this row
                # And/or the previous arrival was NULL or the current departure is NULL                

                if (arrival_prev == 'NULL') or (departure_curr == 'NULL'):
                    
                    if (arrival_prev == 'NULL') and (departure_curr == 'NULL'):
                        # Both arrival and departure are null
                        list_replace_arrival.append({'icao24': current_row.icao24,
                                                     'idx': idx-1,
                                                     'row_id': prev_row.name,
                                                     'arrival_orig': arrival_prev,
                                                     'arrival_new': 'NULL_'+str(current_row.icao24)+'_'+str(prev_row.name)})
                        list_replace_departure.append({'icao24': current_row.icao24,
                                                       'idx': idx,
                                                       'row_id': current_row.name,
                                                       'departure_orig': departure_curr,
                                                       'departure_new': 'NULL_'+str(current_row.icao24)+'_'+str(prev_row.name)})

                    elif arrival_prev == 'NULL':
                        # Only arrival is null, replace arrival previous for departure current
                        list_replace_arrival.append({'icao24': current_row.icao24,
                                                     'idx': idx-1,
                                                     'row_id': prev_row.name,
                                                     'arrival_orig': arrival_prev,
                                                     'arrival_new': departure_curr})

                    else:
                        # Only departure is null, replace departure for arrival previous
                        list_replace_departure.append({'icao24': current_row.icao24,
                                                       'idx': idx,
                                                       'row_id': current_row.name,
                                                       'departure_orig': departure_curr,
                                                       'departure_new': arrival_prev})

                else:
                    # We are not dealing with NULL but with airports not matching
                    # previous arriving to A, current departing from B
                   
                    # Get lat,lon of two airports to compute how far apart they are between them
                    lat_arrival = df_airport_st[df_airport_st.ICAO == arrival_prev].lat
                    lon_arrival = df_airport_st[df_airport_st.ICAO == arrival_prev].lon
                    lat_departure = df_airport_st[df_airport_st.ICAO == departure_curr].lat
                    lon_departure = df_airport_st[df_airport_st.ICAO == departure_curr].lon
                   
                    distance_km = None
                    selected_airport = None

                    check_fit_flight = False
                    keep_most_used_airport = False

                    if (len(lat_arrival) == len(lat_departure)) and (len(lat_arrival) > 0):
                        # We have lat arrival and departure (implicit then lon too)
                        # Compute distance between them,
                        # We are checking the len as if lat doesn't exit it would be NAN and len==0

                        # Distance between airports
                        distance_km = haversine(lon_arrival.iloc[0], lat_arrival.iloc[0],
                                                lon_departure.iloc[0], lat_departure.iloc[0])

                        # The two airports are very close so a mismatch on their name
                        # change arrival or departure depending on which airport is overall used the most
                        if distance_km < threshold_same_airport:
                            keep_most_used_airport = True
                        else:
                            # The distance between the two airports is too large. It could still be a mismatch
                            check_fit_flight = True

                    else:
                        # We have two airports, but they are not in the list of airports, so we don't have
                        # a distance between them, so check if we can fit a flight
                        check_fit_flight = True

                    if check_fit_flight:
                        # Check if between the rotations there is enought time to fit another flight
                        # going between A and B

                        avg_time_between_wrong_airports = dict_avg_times.get(arrival_prev+"_"+departure_curr)
                        if ((avg_time_between_wrong_airports is not None) and
                                (avg_time_between_wrong_airports >= minimum_fly_time_to_add)):
                            # Historically some flights have gone between those two airports
                            # Check if enough time available to add the flight
                            time_between_rotation = current_row.firstseen - prev_row.lastseen
                            # 2 minimum turnaround time as we need to do the one to go to the
                            # intermediate airport and the one to the final one
                            if time_between_rotation > (avg_time_between_wrong_airports + 2 * minimum_time_ground):
                                list_add.append({
                                             'icao24': current_row['icao24'],
                                             'airline': current_row['airline'],
                                             'departure': prev_row['arrival'],
                                             'arrival': current_row['departure'],
                                             'firstseen': prev_row['lastseen']+minimum_time_ground,
                                             'firstseen_datetime': pd.to_datetime(prev_row['lastseen']+minimum_time_ground, unit='s'),
                                             'firstseen_date': pd.to_datetime(prev_row['lastseen']+minimum_time_ground, unit='s').date(),
                                             'lastseen': prev_row['lastseen']+minimum_time_ground+avg_time_between_wrong_airports,
                                             'lastseen_datetime': pd.to_datetime(prev_row['lastseen']+minimum_time_ground+avg_time_between_wrong_airports, unit='s'),
                                             'lastseen_date': pd.to_datetime(prev_row['lastseen']+minimum_time_ground+avg_time_between_wrong_airports, unit='s').date(),
                                             'added_trip': True})
                            else:
                                keep_most_used_airport = True

                        else:
                            # We don't have any flight going between those two airports. So keep the 
                            # one with more usage
                            keep_most_used_airport = True

                    if keep_most_used_airport:
                        arrival_used = dict_airports_used.get(arrival_prev, 1)
                        departure_used = dict_airports_used.get(departure_curr, 1)
                        if arrival_used > departure_used:
                            # previous flight should arrive to this instead
                            list_replace_departure.append({'idx': idx,
                                                           'row_id': current_row.name,
                                                           'departure_orig': departure_curr,
                                                           'departure_new': arrival_prev})
                            selected_airport = arrival_prev
                        else:
                            selected_airport = departure_curr
                            list_replace_arrival.append({'idx': idx-1,
                                                         'row_id': prev_row.name,
                                                         'arrival_orig': arrival_prev,
                                                         'arrival_new': departure_curr})

    return {'list_replace_arrival': list_replace_arrival,
            'list_replace_departure': list_replace_departure,
            'list_add_flights': list_add}


def fix_rotations(df_fa, dict_avg_times, dict_airports_used, threshold_same_airport=80,
                  minimum_time_ground=50 * 60, minimum_fly_time_to_add=60 * 60):

    # threshold_same_airport - km
    # minimum_time_ground - sec - At least 50 minutes on ground to add a new flight
    # minimum_fly_time_to_add - sec - At least 1h flight to add, otherwise probably airports too close and mislabelled
    
    # Get the information/data to process the wrong rotations
    changes_to_apply_fix_rotations = df_fa.groupby('icao24',
                                                   group_keys=False).apply(process_rotations_mismatch,
                                                                           dict_avg_times=dict_avg_times,
                                                                           dict_airports_used=dict_airports_used,
                                                                           threshold_same_airport=threshold_same_airport,
                                                                           minimum_time_ground=minimum_time_ground,
                                                                           minimum_fly_time_to_add=minimum_fly_time_to_add)
    return changes_to_apply_fix_rotations

#################################
# PROCESSING DATA FIX ROTATIONS #
#################################

####
# READ DATA/CLEAN AND PREPARE
####

# Read data
df_flight = pd.read_csv(f_flight_data_opensky)
df_airport_st = pd.read_csv(f_airport_static)
df_aircraft = pd.read_csv(f_aircraft_db_opensky)

# Clean df_flight removing rows without callsign
df_f = df_flight[~df_flight['callsign'].isnull()].copy()

# Print info flights read
logging.info('%d, flights read', len(df_flight))
logging.info('%d, flights with callsign', len(df_f))
logging.info('%d, flights removed (%f%%)', len(df_flight)-len(df_f), 100*round((len(df_flight)-len(df_f))/len(df_flight), 3))

# Add airline column using first 3 letters of callsign as a proxy
df_f['airline'] = df_f['callsign'].apply(lambda x: x[0:3])

# Add country origin/destination as first two letters ICAO code
df_f['country_orig'] = df_f['estdepartureairport'].str[0:2]
df_f['country_dest'] = df_f['estarrivalairport'].str[0:2]

# Save airlines with flights within country_of_interest
df_n_flights = \
    df_f[(df_f['country_orig'] == country_of_interest) & (df_f['country_dest'] == country_of_interest)].groupby(
        ['airline']).count()['icao24'].reset_index().sort_values(['icao24'], ascending=False)
df_n_flights['perc'] = 100 * df_n_flights['icao24'] / df_n_flights['icao24'].sum()
df_n_flights['cum_perc'] = df_n_flights.perc.cumsum()
df_n_flights.to_csv(f_airlines_country_of_interest)

logging.info("\nTop 10 airlines with most flights within" + str(country_of_interest))
logging.info(str(df_n_flights.iloc[0:10]))
logging.info("\nThis below as part of the analysis of Spain/Fix if needed as a function of needs")
logging.info("SWT airline has most of its flights within Balearic Islands, so not of interest" +
      "Only " + str(
          len(df_f[(df_f['country_orig'] == 'LE') & (df_f['country_dest'] == 'LE') & (df_f['airline'] == 'SWT')
                   & (~df_f.estdepartureairport.isin(['LEPA', 'LEIB', 'LEMH']))]))
             + " flights outside Balearic Islands in Spain")

# Use previous information to determine airlines_of_interest (defined in input part)

# Filter by airline
df_fa = df_f[(df_f['airline'].isin(airlines_of_interest))].copy()

logging.info('%d flights of airlines interest (%f%%)', len(df_fa), 100*round(len(df_fa)/len(df_f), 3))

# Add date of first, last seen
df_fa['firstseen_datetime'] = pd.to_datetime(df_fa['firstseen'], unit='s')
df_fa['lastseen_datetime'] = pd.to_datetime(df_fa['lastseen'], unit='s')
df_fa['firstseen_date'] = df_fa['firstseen_datetime'].dt.date
df_fa['lastseen_date'] = df_fa['lastseen_datetime'].dt.date

# Dictionary of manually fixed airports
dict_manual_fixing_airports = pd.read_csv(f_fixed_airports).set_index('icao_orig')['icao_fixed'].to_dict()

df_fa[['estarrivalairport', 'estdepartureairport']] = df_fa[['estarrivalairport', 'estdepartureairport']].fillna(value="NULL")
df_fa['departure_used'] = df_fa['estdepartureairport'].apply(lambda x: dict_manual_fixing_airports.get(x, x))
df_fa['arrival_used'] = df_fa['estarrivalairport'].apply(lambda x: dict_manual_fixing_airports.get(x, x))
df_fa['departure'] = df_fa['departure_used']
df_fa['arrival'] = df_fa['arrival_used']

# Compute duration of flights seen and dictionary of average travelling time between origin-destination seen
df_fa['durationseen'] = df_fa['lastseen'] - df_fa['firstseen']
df_fa['est_departure_arrivalairports'] = df_fa['departure_used']+"_"+df_fa['arrival_used']
dict_avg_times = df_fa.groupby('est_departure_arrivalairports')['durationseen'].mean().to_dict()

# Compute how many flights departing/arriving airports
# How many flights (departing and arrival) are estimated used by airports
counter_airports_used = Counter(list(df_fa.departure)+list(df_fa.arrival))
dict_airports_used = dict(counter_airports_used)

# Compute rotations
df_fa.sort_values(['icao24', 'firstseen'], inplace=True)
df_fa_g = df_fa.groupby('airline').apply(compute_rotations).reset_index(drop=True)
df_fa_g.sort_values(['icao24', 'firstseen'], inplace=True)

####
# FIX ROTATIONS
####

df_fa_g['added_trip'] = False

df_fa_g = df_fa_g.sort_values(['icao24', 'firstseen']).reset_index(drop=True)

lra = df_fa_g.groupby('airline', group_keys=False).apply(fix_rotations,
                                                         dict_avg_times=dict_avg_times,
                                                         dict_airports_used=dict_airports_used)  # .reset_index(drop=True)

# Apply changes/ replace of arrivals, departures and new flights

df_fa_g['departure_used'] = df_fa_g['departure']
df_fa_g['arrival_used'] = df_fa_g['arrival']    
    
num_departure_changes = {}
num_arrival_changes = {}
num_added_changes = {}
ac_changed = {}
for c in lra:
    for ra in c['list_replace_arrival']:
        airline = df_fa_g.loc[ra['row_id']].airline
        ac_changed[airline] = ac_changed.get(airline, set())
        ac_changed[airline].add(df_fa_g.loc[ra['row_id']].icao24)
        num_arrival_changes[airline] = num_arrival_changes.get(airline, 0) + 1
        
        df_fa_g.loc[ra['row_id'], 'arrival_used'] = ra['arrival_new']

    for rd in c['list_replace_departure']:
        airline = df_fa_g.loc[rd['row_id']].airline
        ac_changed[airline] = ac_changed.get(airline, set())
        ac_changed[airline].add(df_fa_g.loc[rd['row_id']].icao24)
        num_departure_changes[airline] = num_departure_changes.get(airline, 0) + 1

        df_fa_g.loc[rd['row_id'], 'departure_used'] = rd['departure_new']
        
    for a in c['list_add_flights']:
        airline = a['airline']
        ac_changed[airline] = ac_changed.get(airline, set())
        ac_changed[airline].add(a['icao24'])
        num_added_changes[airline] = num_added_changes.get(airline, 0) + 1

        # Concatenate new rows with previous
        df_add = pd.DataFrame([a])
        df_add['arrival_used'] = df_add['arrival']
        df_add['departure_used'] = df_add['departure']

        df_fa_g = pd.concat([df_fa_g, df_add], ignore_index=True)
        
# Sort the result
df_fa_g.sort_values(['airline', 'icao24', 'firstseen'], inplace=True)

for k, v in ac_changed.items():
    logging.info('Airline %s', k)
    logging.info("We have modified form %d aircraft", len(v))
    logging.info("We have added %d flights", num_added_changes.get(k, 0))
    logging.info("We have modified departure %d flights", num_departure_changes.get(k, 0))
    logging.info("We have modified arrival %d flights", num_arrival_changes.get(k, 0),)
    logging.info("---\n")

####
# PROCESS ROTATIONS / DISTANCE FLOWN FIX BLOCK TIMES
####

# Merge with airports lat, lon
df_fa_g = df_fa_g.merge(df_airport_st[['ICAO', 'lat', 'lon']],
                        left_on='departure_used', right_on='ICAO', how='left').rename(columns={'lat': 'dep_lat', 'lon': 'dep_lon'}).drop('ICAO', axis=1)
df_fa_g = df_fa_g.merge(df_airport_st[['ICAO', 'lat', 'lon']],
                        left_on='arrival_used', right_on='ICAO', how='left').rename(columns={'lat': 'arr_lat', 'lon': 'arr_lon'}).drop('ICAO', axis=1)

# Identify airports arrival which are not in list of airpots with lat,lon
# Might need to be fixed in manual fixed airports
# Should be zero, so if some airports are here they need to be modified --> added to f_fixed_airports
df_airports_missing_arr = df_fa_g[df_fa_g.arr_lat.isnull() & (df_fa_g.arrival != 'NULL')].arrival_used.drop_duplicates()

logging.info("%d arrival airports missing", len(df_airports_missing_arr))
if len(df_airports_missing_arr) > 0:
    logging.info(df_airports_missing_arr.to_string())

# Identify airports departure which are not in list of airports with lat,lon
# Might need to be fixed in manual fixed airports
# Should be zero, so if some airports are here they need to be modified --> added to f_fixed_airports
df_airports_missing_dep = df_fa_g[df_fa_g.dep_lat.isnull() & (df_fa_g.departure != 'NULL')].departure_used.drop_duplicates()

logging.info("%d departure airports missing", len(df_airports_missing_dep))
if len(df_airports_missing_dep) > 0:
    logging.info(df_airports_missing_dep.to_string())

# Compute GCD between airports origin-destination in km
df_fa_g['gcd_km'] = df_fa_g.apply(lambda x: haversine(x.dep_lon, x.dep_lat, x.arr_lon, x.arr_lat), axis=1)

# Merge now with the database of aircraft types to have the ac_type per flight

df_ac = df_aircraft[['icao24', 'registration', 'model', 'typecode', 'operatoricao']]

df_fa_g = df_fa_g.merge(df_ac, left_on='icao24', right_on='icao24', how='left')

# Aircraft missing
logging.info("Ac type missing for ADS-B registration and average distance of a/c used")
logging.info(df_fa_g[df_fa_g['typecode'].isnull()].groupby(['icao24'])['gcd_km'].mean().to_string())

# Estimate takeoff and landing based on average climb and descend speeds and distance (line) from first/last seen and
# airport estimated originally
df_fa_g['extra_climb_time'] = round(60*df_fa_g['estdepartureairportvertdistance'] / climb_speed)
df_fa_g['extra_descend_time'] = round(60*df_fa_g['estarrivalairportvertdistance'] / descend_speed)
df_fa_g['extra_climb_time'] = df_fa_g['extra_climb_time'].apply(lambda x: x if not np.isnan(x) else 0)
df_fa_g['extra_descend_time'] = df_fa_g['extra_descend_time'].apply(lambda x: x if not np.isnan(x) else 0)
df_fa_g['takeoff_time'] = df_fa_g['firstseen']-df_fa_g['extra_climb_time']
df_fa_g['landing_time'] = df_fa_g['lastseen']+df_fa_g['extra_descend_time']

# Set some average taxi-in taxi-out times... to estimate AOBT and AIBT
df_fa_g['aobt'] = df_fa_g['takeoff_time'] - taxi_out_avg * 60
df_fa_g['aibt'] = df_fa_g['landing_time'] + taxi_in_avg * 60

# Estimate SOBT and SIBT as same as AOBT and AIBT in UTC form
df_fa_g['sobt'] = df_fa_g['aobt'].apply(lambda x: datetime.utcfromtimestamp(x) if not np.isnan(x) else x)
df_fa_g['sibt'] = df_fa_g['aibt'].apply(lambda x: datetime.utcfromtimestamp(x) if not np.isnan(x) else x)


#########################
# SAVE OUTPUT SCHEDULES #
#########################
df_fa_g.to_csv(f_all_output_csv, index=True)
df_fa_g[['airline', 'icao24', 'departure_used', 'arrival_used', 'sobt', 'sibt', 'callsign', 'typecode', 'model',
         'gcd_km', 'dep_lat', 'dep_lon', 'arr_lat', 'arr_lon']].to_csv(f_reduced_output_csv, index=False)
df_fa_g[['airline', 'icao24', 'departure_used', 'arrival_used', 'sobt', 'sibt', 'callsign', 'typecode', 'model',
         'gcd_km', 'dep_lat', 'dep_lon', 'arr_lat', 'arr_lon']].to_parquet(f_reduced_output_parquet, index=False)

# Close logging file
logging.shutdown()
