import pandas as pd
import datetime

"""
Basic library for direct train computation from GTFS data (from Renfe)
"""

def read_data(folder_path='.'):
    """
    Read rail data from folder_path and transform times in datetime format
    """
    def times_in_datetime_format(calendar, calendar_dates, stops_times):
        calendar.start_date = pd.to_datetime(calendar.start_date, format='%Y%m%d')
        calendar.end_date = pd.to_datetime(calendar.end_date, format='%Y%m%d')
        calendar_dates.date = pd.to_datetime(calendar_dates.date, format='%Y%m%d')

        # All this because arrival and departure time can be 24 and datetime doesn't allow that
        def custom_to_datetime(date):
            # If the time is 24, set it to 0 and increment day by 1
            if int(date.split(":")[0]) >= 24:
                str_time = str(int(date.split(":")[0])-24)+":"+date.split(":")[1]+":"+date.split(":")[2]
                return pd.to_datetime(str_time, format='%H:%M:%S') + pd.Timedelta(days=1)
            else:
                return pd.to_datetime(date, format='%H:%M:%S')

        stops_times['arrival_time'] = stops_times['arrival_time'].apply(custom_to_datetime)  
        stops_times['departure_time'] = stops_times['departure_time'].apply(custom_to_datetime)  

    agency = pd.read_csv(folder_path+'/agency.txt')
    calendar = pd.read_csv(folder_path+'/calendar.txt')
    calendar_dates = pd.read_csv(folder_path+'/calendar_dates.txt')
    routes = pd.read_csv(folder_path+'/routes.txt')
    stops_times = pd.read_csv(folder_path+'/stop_times.txt')
    stops = pd.read_csv(folder_path+'/stops.txt')
    trips = pd.read_csv(folder_path+'/trips.txt')

    times_in_datetime_format(calendar, calendar_dates, stops_times)

    return agency, calendar, calendar_dates, routes, stops_times, stops, trips


def join_trips_routes_stops(trips, routes, calendar, stops, stops_times):
    # Join trips with routes, calendar, service and stops
    trips_full = trips.merge(routes, on='route_id').merge(calendar, on='service_id')
    trips_full_w_stops = trips_full.merge(stops_times, on='trip_id').merge(stops, on='stop_id')
    return trips_full_w_stops


def extract_stops(trip):
    return trip.sort_values(['stop_sequence'])['stop_id'].to_list()


def is_contained(stops, df):
    set_stops = set(stops)
    for other_stops in df[0]:
        if set_stops.issubset(set(other_stops)) and set_stops != set(other_stops):
            return True
    return False


def keep_long_route_same_trip(trip):
    df_trips = pd.DataFrame(trip.groupby(['trip_id']).apply(extract_stops)).reset_index()
    return df_trips[~df_trips[0].apply(is_contained, df=df_trips)]
    

def trains_on_date(trips_full_w_stops, date_interest, remove_itineraries_overlap=True):
    # Filter trains operating in date of interest
    date_interest = pd.to_datetime(date_interest, format='%Y%m%d')
    day_week_interest = date_interest.strftime('%A').lower()

    trips_on_day = trips_full_w_stops[(trips_full_w_stops.start_date <= date_interest) &
                                      (trips_full_w_stops.end_date >= date_interest)]

    # keep trains running on the day of the week of the day of interest
    trips_on_day = trips_on_day[trips_on_day[day_week_interest] == 1]

    if remove_itineraries_overlap:
        # Remove itineraries which share stations and times
        trips_ids_keep = trips_on_day.groupby(['trip_short_name'],
                                              group_keys=False).apply(keep_long_route_same_trip)['trip_id'].to_list()
        trips_on_day = trips_on_day[trips_on_day['trip_id'].isin(trips_ids_keep)].reset_index(drop=True)

    return trips_on_day


# Function to check if an itinerary (group of stops) pass through all the
# stops in stops_in_route int the right order
def check_stop_sequence(itinerary, stops_in_route):
    # get the stop ids and stop sequences for the itinerary
    stop_ids = itinerary['stop_id'].tolist()
        
    # check if all stops_in_route are within stops_ids of the itinerary
    if all(x in stop_ids for x in stops_in_route):
        # if all of them are there, now check if they are in order
        # create a dictionary of stops and their sequence
        dict_stops_seqs = itinerary.set_index('stop_id')['stop_sequence'].to_dict()
        
        # check that all the stops_in_route have sequence lower than next stop_in_route
        return all(dict_stops_seqs[stops_in_route[index]] < dict_stops_seqs[stops_in_route[index+1]] for
                   index in range(len(stops_in_route)-1))
    
    else:
        # Not all of them are there so return False
        return False


def get_trip_after_stop(itinerary, stop):
    return itinerary[itinerary.stop_sequence >= (itinerary[itinerary.stop_id == stop].stop_sequence.iloc[0])]


def extract_start_end_trip(trip, stop_start, stop_end):
    trip_id = trip['trip_id'].iloc[0]
    dep_time = trip[trip['stop_id'] == stop_start].departure_time.iloc[0]
    arr_time = trip[trip['stop_id'] == stop_end].arrival_time.iloc[0]
    stop_sequence_start = trip[trip['stop_id'] == stop_start].stop_sequence.iloc[0]
    stop_sequence_end = trip[trip['stop_id'] == stop_end].stop_sequence.iloc[0]
    route_short_name = trip['route_short_name'].iloc[0]
    route_id = trip['route_id'].iloc[0]

    trip_option = {'trip_id': trip_id,
                   'trip_short_name': trip['trip_short_name'].iloc[0],
                   'stop_start': stop_start,
                   'stop_end': stop_end,
                   'route_short_name': route_short_name,
                   'route_id': route_id,
                   'n_intermediate_stops': stop_sequence_end-stop_sequence_start-1,
                   'departure_time': dep_time,
                   'arrival_time': arr_time,
                   'leg_time': arr_time-dep_time
                   }

    return [trip_option]


def direct_services_to(stop_start, stop_end, trips_all, date_time_start=None, print_found=False):

    trips = trips_all.copy()

    if date_time_start is not None:
        # First keep only stops/trips which are after the given time_start
        connecting_time = 10
        date_time_start += datetime.timedelta(minutes=connecting_time)
        trips = trips[trips['departure_time'] >= date_time_start].copy()

    # Keep only trips which contain the starting stop
    trips = trips[trips['service_id'].isin(trips[trips['stop_id'] == stop_start].service_id)].copy()
    
    # Limit to only after stop part of the trips
    trips = trips.groupby(['trip_id'], group_keys=False).apply(get_trip_after_stop,
                                                               stop=stop_start).reset_index(drop=True)

    if len(trips) == 0:
        # No more trips available from that time and stop
        return None

    # Check if trips go to the final stop and keep those
    trips_go_stop_end_index = trips.groupby(['trip_id']).apply(check_stop_sequence,
                                                               stops_in_route=[stop_start, stop_end]).reset_index()
    trips_go_stop_end = trips[trips.trip_id.isin(trips_go_stop_end_index[trips_go_stop_end_index[0]].trip_id)].copy()
    trips_go_stop_end = trips_go_stop_end.sort_values(['trip_id', 'stop_sequence'])

    if len(trips_go_stop_end) > 0:
        # Found itineraries go to final destination
        if print_found:
            print(f"Found itineraries to final destination {stop_end} coming from {stop_start}")

        full_itineraries = trips_go_stop_end[['trip_id', 'stop_id', 'arrival_time', 'departure_time',
                                              'route_short_name', 'stop_sequence', 'trip_short_name',
                                              'route_id']].groupby(['trip_id']).apply(extract_start_end_trip,
                                                                                      stop_start=stop_start,
                                                                                      stop_end=stop_end).to_list()
    else:
        full_itineraries = None

    return full_itineraries
