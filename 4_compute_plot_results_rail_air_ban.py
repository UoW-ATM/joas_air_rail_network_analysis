import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
import pickle
import imageio
import zipfile
import shutil
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cf
from datetime import datetime, timedelta
from libs.general_tools import date_range, create_folder_if_not_exist

###################
# DATA/PARAMETERS #
###################
analyse_individual_days = True
do_rail_replacement_maps = True
analyse_route_ban_general = True
analyse_itineraries = True

country_interest = 'LE'
manually_remove_airports = ['LEIB', 'LEPA', 'LEMH']

flight_schedules_path = './output/air/schedules_1st_week_0523_all_airlines_reduced.parquet'
rail_dict_direct_path = './output/rail/dict_direct_rail_purged.pickle'
rail_data_times_path = './output/rail/rail_data_times_format.pickle'
rail_df_purged_path = './output/rail/df_rail_purged.parquet'
rail_used_emissions_path = './data/data_computed/rail_used_emissions.csv'
rail_used_emissions_additional_path = './data/data_computed/rail_used_emissions_2.csv'
routes_emissions_path = './data/data_computed/routes_emissions_v2.csv'
dict_d_it_file_path = './output/multi/dict_d_it_fixed_baseline_rail.pickle'
airline_info_path = './output/multi/airlines_info.pickle'
read_precomputed_dict_d_it = True

output_multi_folder_path = './output/multi/'
output_img_folder_path = './output/figs/img_map_replacement/'  # for images of maps of routes
output_figures_folder_path = './output/figs/'  # for figures in general
output_competition_routes_path = './output/multi/competition_routes.csv'
output_emissions_flight_rail = './output/multi/emissions_flight_rail_final.csv'
output_emissions_flight_rail_pax = './output/multi/emissions_flight_rail_pax_replaced.csv'
output_seats_replaced_rail = './output/multi/seats_replaced_rail.csv'

f_dict_d_it_analysis_pickle = './output/multi/dict_d_it_analysis.pickle'

f_output_folder_log = './output/log/'
f_log = f_output_folder_log+'4_log_results_air_rail_plots.txt'

# Create output folders if not exist
create_folder_if_not_exist(output_multi_folder_path)
create_folder_if_not_exist(output_img_folder_path)
create_folder_if_not_exist(output_figures_folder_path)
create_folder_if_not_exist(f_output_folder_log)


unzip = True
list_individual_day_analysis = ['20230503', '20230504']  # list of days to analyse

# Start and end date for range of days analysis
start_date_str = "2023-05-01"
end_date_str = "2023-05-07"

# Define start and end times of threshold ban
start_time = pd.to_timedelta('0:00:00')
end_time = pd.to_timedelta('15:00:00')

# Define the step (15 minutes)
step = pd.to_timedelta('0:15:00')

day_example_histogram_demand = '20230503'
airport_interest_histogram = 'LEMD'

airlines_order = ['AEA', 'ANE', 'IBE', 'IBS', 'RYR', 'VLG', 'All']


#####################
# CONFIGURE LOGGING #
#####################

logging.basicConfig(filename=f_log, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')

##################
# FUNCTIONS USED #
##################

def compute_routes_within_country(ds, country_interest='LE', do_print=False):
    """
    Function to filter flights and routes given a country code.
    """
    all_airports = set(ds.departure_used).union(set(ds.arrival_used))

    # As rail in country consider only airports in country_interest code
    airports_country = set()
    for a in all_airports:
        if a.startswith(country_interest) and a != 'LEIB' and a != 'LEPA' and a != 'LEMH':
            airports_country.add(a)

    if do_print:
        logging.info('Airports in '+country_interest+' : '+str(len(airports_country)))

    # Flights within country interest
    ds_country = ds[(ds.departure_used.isin(airports_country)) & (ds.arrival_used.isin(airports_country))].copy()
    ds_country = ds_country[ds_country.departure_used != ds_country.arrival_used]
    ds_country['o_d'] = ds_country.apply(lambda row: (row['departure_used'], row['arrival_used']), axis=1)

    # Remove LEBL-LEGE as it's not commercial
    ds_country = ds_country[ds_country['o_d'] != ('LEBL', 'LEGE')]

    airports_flights_within_country = set(ds_country.departure_used).union(set(ds_country.arrival_used))

    if do_print:
        logging.info("Airports within "+country_interest + ' : '+str(len(airports_flights_within_country)))

    # Compute routes within country of interest
    routes_within_country = ds_country[['departure_used', 'arrival_used', 'dep_lat', 'dep_lon',
                                        'arr_lat', 'arr_lon']].drop_duplicates().copy()

    # Avoid circular routes
    routes_within_country = routes_within_country[routes_within_country.departure_used !=
                                                  routes_within_country.arrival_used]

    routes_within_country['o_d'] = routes_within_country.apply(lambda row: (row['departure_used'], row['arrival_used']),
                                                               axis=1)
    if do_print:
        logging.info("Routes within "+country_interest+' : '+str(len(routes_within_country)))

    return ds_country, routes_within_country, airports_flights_within_country


def get_stops_train(df, stops_times, stops):
    """
    Get coordinates stops of a given train line/trip
    """
    x = df.iloc[0]
    return stops_times[(stops_times.trip_id == x.trip_id) &
                       (stops_times.stop_sequence >=
                        stops_times[(stops_times.trip_id == x.trip_id) & (
                                    stops_times.stop_id == x.stop_start)].stop_sequence.iloc[0]) &
                       (stops_times.stop_sequence <=
                        stops_times[(stops_times.trip_id == x.trip_id) & (
                                    stops_times.stop_id == x.stop_end)].stop_sequence.iloc[0])
                       ].merge(stops, on='stop_id')[['stop_lat', 'stop_lon']]


def generate_plots_routes(airports_flights_within_country, routes_within_country, dict_od_remove, dict_airp_latlon,
                          dict_rail_stations_used, stops, stops_times, dt_baseline=None, dt_baseline_lat_lon=None,
                          dict_ds_rail_ban=None, dict_ds_ff_ban=None, plot_baseline_train=True,
                          plot_replacement_trains=True, day='', plot_title=False, plot_legend=False,
                          output_folder_path='./img/'):
    """
    Plot routes used as a function of ban threshold (save in img to create gif later)
    """
    if dict_ds_rail_ban is None:
        dict_ds_rail_ban = {}
    if dict_ds_ff_ban is None:
        dict_ds_ff_ban = {}
    ds_rail_ban = None
    ds_ff_ban = None
    for ban_time in dict_od_remove.keys():
        od_remove = dict_od_remove.get(ban_time)

        mpl.rcParams.update({'font.size': 30})

        plt.figure(figsize=(25, 15))
        ax = plt.axes(projection=ccrs.EuroPP())

        ax.add_feature(cf.COASTLINE, alpha=0.4, linewidth=4)
        ax.add_feature(cf.BORDERS, alpha=0.4, linewidth=4)

        ax.set_global()

        plot_rail_used = True
        print_airport_not_used = False

        routes_ploted = set()
        for a in airports_flights_within_country:
            plt.plot(dict_airp_latlon[a]['lon'], dict_airp_latlon[a]['lat'], 'bo', markersize=18,
                     transform=ccrs.Geodetic(), linewidth=12)
            plt.text(dict_airp_latlon[a]['lon'], dict_airp_latlon[a]['lat'], a, transform=ccrs.Geodetic())

            if plot_rail_used:
                if dict_rail_stations_used.get(a) is not None:
                    plt.plot(stops[stops.stop_id.isin(dict_rail_stations_used[a])].stop_lon,
                             stops[stops.stop_id.isin(dict_rail_stations_used[a])].stop_lat,
                             'rx', transform=ccrs.Geodetic(), linewidth=12)
                elif print_airport_not_used:
                    logging.info("Airport which station is not used "+str(a))

        for index, r in routes_within_country[~routes_within_country.o_d.isin(od_remove)].iterrows():
            if ((r.departure_used, r.arrival_used) not in routes_ploted and
                    (r.arrival_used, r.departure_used) not in routes_ploted):
                plt.plot([r['dep_lon'], r['arr_lon']], [r['dep_lat'], r['arr_lat']], 'b', alpha=0.5,
                         transform=ccrs.Geodetic(), linewidth=8)
                routes_ploted.add((r.departure_used, r.arrival_used))

        # baseline train
        rail_plotted = set()
        if plot_baseline_train:
            ds_ff_ban = dict_ds_ff_ban.get(ban_time, ds_ff_ban)
            # Compute trips on rail that are not available on ff options
            dt_baseline_missing = dt_baseline.merge(ds_ff_ban[['o_d', 'airline_orig']], on='o_d', how='left')
            dt_baseline_missing = dt_baseline_missing[dt_baseline_missing.airline_orig.isnull()]

            dt_baseline_latlon_needed = dt_baseline_lat_lon.merge(dt_baseline_missing[['trip_id']].drop_duplicates(),
                                                                  on='trip_id')

            for tt in dt_baseline_latlon_needed.trip_id.drop_duplicates():

                dt_to_plot_rail = dt_baseline_latlon_needed[dt_baseline_latlon_needed.trip_id == tt]

                for r in range(len(dt_to_plot_rail) - 1):
                    to_plot_lon = [dt_to_plot_rail.iloc[r].stop_lon, dt_to_plot_rail.iloc[r + 1].stop_lon]
                    to_plot_lat = [dt_to_plot_rail.iloc[r].stop_lat, dt_to_plot_rail.iloc[r + 1].stop_lat]

                    if (tuple(to_plot_lon), tuple(to_plot_lat)) not in rail_plotted:
                        rail_plotted.add((tuple(to_plot_lon), tuple(to_plot_lat)))

                        plt.plot(to_plot_lon, to_plot_lat,
                                 ':g', transform=ccrs.Geodetic(), linewidth=12)

                plt.plot(dt_to_plot_rail['stop_lon'].iloc[0],
                         dt_baseline_latlon_needed[dt_baseline_latlon_needed.trip_id == tt]['stop_lat'].iloc[0],
                         'go', markersize=18, transform=ccrs.Geodetic(), linewidth=12)
                plt.plot(dt_to_plot_rail['stop_lon'].iloc[-1],
                         dt_baseline_latlon_needed[dt_baseline_latlon_needed.trip_id == tt]['stop_lat'].iloc[-1],
                         'go', markersize=18, transform=ccrs.Geodetic(), linewidth=12)

        # Replacement trains
        if plot_replacement_trains:
            ds_rail_ban = dict_ds_rail_ban.get(ban_time, ds_rail_ban)

            if ds_rail_ban is not None:
                # Filter groups based on the custom function

                ds_rail_ban_lat_lon = ds_rail_ban.groupby('trip_id').apply(get_stops_train, stops_times=stops_times,
                                                                           stops=stops).reset_index().drop(['level_1'],
                                                                                                           axis=1)

                seen_before = set()

                def is_unique(group):
                    if ('_'.join(group.stop_lat.astype(str)), '_'.join(group.stop_lon.astype(str))) in seen_before:
                        return False
                    seen_before.add(('_'.join(group.stop_lat.astype(str)), '_'.join(group.stop_lon.astype(str))))
                    return True

                ds_rail_ban_lat_lon = ds_rail_ban_lat_lon.groupby('trip_id').filter(is_unique)

                for tt in ds_rail_ban_lat_lon.trip_id.drop_duplicates():

                    dt_to_plot_rail = ds_rail_ban_lat_lon[ds_rail_ban_lat_lon.trip_id == tt]

                    for r in range(len(dt_to_plot_rail) - 1):
                        to_plot_lon = [dt_to_plot_rail.iloc[r].stop_lon, dt_to_plot_rail.iloc[r + 1].stop_lon]
                        to_plot_lat = [dt_to_plot_rail.iloc[r].stop_lat, dt_to_plot_rail.iloc[r + 1].stop_lat]

                        if (tuple(to_plot_lon), tuple(to_plot_lat)) not in rail_plotted:
                            rail_plotted.add((tuple(to_plot_lon), tuple(to_plot_lat)))

                            plt.plot(to_plot_lon, to_plot_lat,
                                     ':g', transform=ccrs.Geodetic(), linewidth=12)

                    plt.plot(ds_rail_ban_lat_lon[ds_rail_ban_lat_lon.trip_id == tt]['stop_lon'].iloc[0],
                             ds_rail_ban_lat_lon[ds_rail_ban_lat_lon.trip_id == tt]['stop_lat'].iloc[0],
                             'go', markersize=10, transform=ccrs.Geodetic())
                    plt.plot(ds_rail_ban_lat_lon[ds_rail_ban_lat_lon.trip_id == tt]['stop_lon'].iloc[-1],
                             ds_rail_ban_lat_lon[ds_rail_ban_lat_lon.trip_id == tt]['stop_lat'].iloc[-1],
                             'go', markersize=10, transform=ccrs.Geodetic())

        ax.set_extent([-10, 5, 25, 44])

        ax.coastlines()

        if plot_title:
            plt.title('Ban flights with rail <= ' + str(ban_time).split()[-1][:5])

        if plot_legend:
            legend_labels = ['Flights', 'Rail']
            legend_colors = ['blue', 'green']
            legend_elements = [plt.Line2D([0], [0], linewidth=8, color=color, label=label, markersize=10) for
                               color, label in zip(legend_colors, legend_labels)]

            plt.legend(handles=legend_elements)

        plt.savefig(f'{output_folder_path}img_{day}_{ban_time}.png',
                    transparent=False,
                    facecolor='white',
                    bbox_inches='tight')

        plt.close()


def compute_statistics_num_flight_routes_mean_flight_time(ds_country, dict_od_remove):
    # Number flights per arline
    dict_flights_airlines_ban = {}
    for ban_time in dict_od_remove.keys():
        od_remove = dict_od_remove.get(ban_time)
        ds_kept = ds_country[~ds_country.o_d.isin(od_remove)].groupby(['airline']).count()['icao24']

        for a in ds_kept.keys():
            dict_ab = dict_flights_airlines_ban.get(ban_time, {})
            dict_ab[a] = ds_kept[a]
            dict_flights_airlines_ban[ban_time] = dict_ab
        dict_ab['All'] = ds_kept.sum()
        dict_flights_airlines_ban[ban_time] = dict_ab

    # Number routes per arline
    dict_routes_airlines_ban = {}
    for ban_time in dict_od_remove.keys():
        od_remove = dict_od_remove.get(ban_time)
        ds_kept = ds_country[~ds_country.o_d.isin(od_remove)].groupby(['airline'])['o_d'].nunique()
        ds_removed = ds_country[ds_country.o_d.isin(od_remove)]['o_d'].nunique()

        for a in ds_kept.keys():
            dict_ab = dict_routes_airlines_ban.get(ban_time, {})
            dict_ab[a] = ds_kept[a]
            dict_routes_airlines_ban[ban_time] = dict_ab
        dict_ab['All'] = sum(ds_kept)
        dict_ab['Rail'] = ds_removed
        dict_routes_airlines_ban[ban_time] = dict_ab

    # Mean flying time inside ban per airline
    ds_country['block_time'] = ds_country['sibt'] - ds_country['sobt']
    dict_avg_flight_airlines_ban = {}
    for ban_time in dict_od_remove.keys():
        od_remove = dict_od_remove.get(ban_time)
        ds_kept = ds_country[ds_country.o_d.isin(od_remove)].groupby(['airline']).agg(
            mean_block_time=pd.NamedAgg(column='block_time', aggfunc=lambda x: x.mean() if len(x) > 0 else None)
        ).reset_index()

        if len(ds_kept) > 0:
            for a in ds_kept['airline']:
                dict_ab = dict_avg_flight_airlines_ban.get(ban_time, {})
                dict_ab[a] = ds_kept[ds_kept.airline == a].mean_block_time.iloc[0]
                dict_avg_flight_airlines_ban[ban_time] = dict_ab
            if len(ds_kept) > 1:
                dict_ab['All'] = ds_country[ds_country.o_d.isin(od_remove)]['block_time'].mean()
            dict_avg_flight_airlines_ban[ban_time] = dict_ab

    return dict_flights_airlines_ban, dict_routes_airlines_ban, dict_avg_flight_airlines_ban


def extract_key_values_dict_flight_routes(dict_flight_routes_airlines_ban):
    # Extract keys and values
    keys = list(dict_flight_routes_airlines_ban.keys())
    values = {}
    i = 0
    for k, v in dict_flight_routes_airlines_ban.items():
        if i == 0:
            all_keys = list(v.keys())
            i += 1
        for a, n in v.items():
            lv = values.get(a, [])
            lv += [n]
            values[a] = lv

        for m in all_keys:
            if m not in v.keys():
                values[m] += [0]
    return keys, values

def extract_flights_used_flight_flight(df):
    df_o = df.copy()
    df_c = df.copy()

    df_o['route'] = df_o['route'].apply(lambda x: (x[0], x[1]))
    df_c['route'] = df_c['route'].apply(lambda x: (x[1], x[2]))
    df_o['sobt'] = df_o['sobt_orig']
    df_c['sobt'] = df_c['sobt_connection']
    df_o['sibt'] = df_o['sibt_orig']
    df_c['sibt'] = df_c['sibt_connection']
    df_o['route_airlines'] = df_o['route_airlines'].apply(lambda x: x[0])
    df_c['route_airlines'] = df_c['route_airlines'].apply(lambda x: x[1])
    df_o['typecode'] = df_o['typecode_orig']
    df_c['typecode'] = df_c['typecode_connection']

    df_c = pd.concat(
        [df_o[['route', 'sobt', 'sibt', 'route_airlines', 'typecode']],
         df_c[['route', 'sobt', 'sibt', 'route_airlines', 'typecode']]])

    df_c[['origin', 'destination']] = pd.DataFrame(df_c['route'].tolist(), index=df_c.index)

    return df_c


def extract_flight_rail_flight(df):
    df_c = df.copy()
    df_c['route'] = df_c['route'].apply(lambda x: (x[1], x[2]))
    df_c['sobt'] = df_c['sobt']
    df_c['route_airlines'] = df_c['route_airlines'].apply(lambda x: x[1])
    df_c['typecode'] = df_c['typecode']

    return df_c


def extract_flight_flight_rail(df):
    df_c = df.copy()
    df_c['route'] = df_c['route'].apply(lambda x: (x[0], x[1]))
    df_c['sobt'] = df_c['sobt']
    df_c['route_airlines'] = df_c['route_airlines'].apply(lambda x: x[0])
    df_c['typecode'] = df_c['typecode']

    return df_c


def extract_rail_rail_flight(df):
    df_c = df.copy()
    df_c['route'] = df_c['route'].apply(lambda x: (x[0], x[1]))
    df_c['trip_id'] = df_c['trip_id']
    df_c['route_stations'] = df_c['route_stations']
    df_c['arrival_time'] = df_c['arrival_time']

    return df_c


def extract_rail_flight_rail(df):
    df_c = df.copy()
    df_c['route'] = df_c['route'].apply(lambda x: (x[1], x[2]))
    df_c['trip_id'] = df_c['trip_id']
    df_c['train_type'] = df_c['route_airlines'].apply(lambda x: x[1])
    df_c['route_stations'] = df_c['route_stations']
    df_c['departure_time'] = df_c['departure_time']
    df_c['arrival_time'] = df_c['arrival_time']

    return df_c


def extract_flights(df):
    df = pd.concat([df[df.type == 'flight'],
                    extract_flights_used_flight_flight(df[df.type == 'flight_flight']),
                    extract_flight_rail_flight(df[df.type == 'rail_flight']),
                    extract_flight_flight_rail(df[df.type == 'flight_rail'])
                    ])[['route', 'sibt', 'sobt', 'route_airlines', 'typecode']].drop_duplicates().copy()

    return df


def extract_rails(df):
    df = pd.concat([df[df.type == 'rail'].rename(columns={'route_airlines': 'train_type'}),
                    extract_rail_rail_flight(df[df.type == 'rail_flight']),
                    extract_rail_flight_rail(df[df.type == 'flight_rail'])
                    ])[['trip_id', 'train_type', 'route', 'route_stations', 'arrival_time',
                        'departure_time']].drop_duplicates().copy()

    return df


def plot_histogram_demand(df_, airport_interest, day_example, departure_times=None, arrival_times=None,
                          save_ending='', save_folder_path='./figs/', df_orig=None, ylim_max=10, yfigmax=3):
    if departure_times is None:
        departure_times = df_[(df_.origin == airport_interest) & (df_.type == 'flight')].sobt
    if arrival_times is None:
        arrival_times = df_[(df_.destination == airport_interest) & (df_.type == 'flight')].sibt

    if df_orig is not None:
        departure_time_orig = df_orig[(df_orig.origin == airport_interest) & (df_orig.type == 'flight')].sobt
        arrival_times_orig = df_orig[
            (df_orig.destination == airport_interest) & (df_orig.type == 'flight')].sibt

    departure_numerical = [mdates.date2num(dt) for dt in departure_times]
    arrival_numerical = [mdates.date2num(dt) for dt in arrival_times]

    # Create histogram bins every 30 minutes starting at the o'clock hour
    bins = np.arange(min(departure_numerical), max(arrival_numerical) + 1800, 1800)

    binwidth = timedelta(minutes=30)
    # Plot the stacked histogram
    mpl.rcParams.update({'font.size': 30})
    plt.subplots(figsize=(15, yfigmax))
    if df_orig is None:
        plt.hist([departure_times, arrival_times], stacked=True,
                 bins=np.arange(datetime.strptime(day_example + " 03:00", '%Y%m%d %H:%M'),
                                datetime.strptime(day_example + " 22:00", '%Y%m%d %H:%M')
                                + binwidth, binwidth), edgecolor='black', label=['Departure', 'Arrivals'],
                 alpha=0.7)
    else:
        plt.hist([departure_time_orig, arrival_times_orig, departure_times, arrival_times], stacked=True,
                 bins=np.arange(datetime.strptime(day_example + " 03:00", '%Y%m%d %H:%M'),
                                datetime.strptime(day_example + " 22:00", '%Y%m%d %H:%M')
                                + binwidth, binwidth), edgecolor='black', label=['Departure', 'Arrivals'],
                 alpha=0.7)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Format to show only time
    plt.xticks(rotation=45)
    plt.ylim(0, ylim_max)  # Adjust the values based on your data
    plt.xlabel('Time')
    plt.ylabel('Demand')
    plt.legend()
    plt.savefig(save_folder_path+'demand_' + airport_interest + '_' + day_example + '_' + save_ending + '.png',
                bbox_inches='tight')


def compute_statistics_flights(df, routes_emissions, country_interest, manually_remove_airports=None):
    if manually_remove_airports is None:
        manually_remove_airports = []

    statistics = {}

    statistics['number_flights'] = len(df)
    statistics['mean_block_time'] = (df['sibt'] - df['sobt']).mean()
    statistics['median_block_time'] = (df['sibt'] - df['sobt']).median()

    df['origin'] = df['route'].str[0]
    df['destination'] = df['route'].str[1]

    df_country = df[df.origin.str.startswith(country_interest) & df.destination.str.startswith(country_interest)]
    df_country_mr = df_country[~df_country['origin'].isin(manually_remove_airports) &
                            ~df_country['destination'].isin(manually_remove_airports)]

    statistics['number_flights_country_mr'] = len(df_country_mr)
    statistics['mean_block_time_country_mr'] = (df_country_mr['sibt'] - df_country_mr['sobt']).mean()
    statistics['median_block_time_country_mr'] = (df_country_mr['sibt'] - df_country_mr['sobt']).median()

    statistics['number_flights_country_no_mr'] = len(df_country) - len(df_country_mr)

    df_f_e = df_country_mr.merge(routes_emissions[['route', 'typecode', 'seats',
                                                   'CO2_g_ASK', 'CO2_kg_pax', 'CO2_t_flight']],
                                 on=['route', 'typecode'], how='left')

    statistics['emissions_CO2_t_flights_country_mr'] = df_f_e.CO2_t_flight.sum()
    statistics['seats_flights_country_mr'] = df_f_e.seats.sum()
    statistics['emissions_CO2_t_per_seat_country_mr'] = statistics['emissions_CO2_t_flights_country_mr'] / \
                                                            statistics['seats_flights_country_mr']

    return statistics


def compute_statistics_rails(df, rail_emissions, dict_f_seats, d, b):
    statistics = {}

    statistics['number_rails'] = len(df)

    df_emissions_seats = df.merge(rail_emissions, on=['route_stations', 'train_type'], how='left')

    # Code commented below used to identify trains for which the seats/emissions were missing
    trains_missing_emissions_seats = df_emissions_seats[df_emissions_seats.seats.isnull()][
        ['train_type', 'route', 'route_stations']].drop_duplicates()
    if len(trains_missing_emissions_seats) > 0:
        logging.info(str(trains_missing_emissions_seats))

    #trains_missing_emissions_seats.to_parquet('./trains_missing_seats/' + d + '_' + b + '.parquet')

    statistics['seats_rail'] = df_emissions_seats.seats.sum()
    statistics['emissions_CO2_rail'] = (df_emissions_seats['CO2kg/PAX'] * df_emissions_seats['seats']).sum() / 1000

    df_emissions_seats_agg = df_emissions_seats.groupby(['route']).agg(
        mean_co2pax=pd.NamedAgg(column='CO2kg/PAX', aggfunc=lambda x: x.mean() if len(x) > 0 else None)
    ).reset_index()

    df_emissions_seats_agg['pax_replaced'] = df_emissions_seats_agg['route'].apply(lambda x: dict_f_seats.get(x, 0))

    # routes_with_repaced_pax = set(df_emissions_seats_agg[df_emissions_seats_agg['pax_replaced']>0].route)

    statistics['seats_replaced'] = df_emissions_seats_agg['pax_replaced'].sum()

    statistics['emissions_CO2_rail_pax_replaced'] = (df_emissions_seats_agg['pax_replaced'] * df_emissions_seats_agg[
        'mean_co2pax']).sum() / 1000

    statistics['mean_rail_block_time'] = (df['arrival_time'] - df['departure_time']).mean()

    # statistics['mean_rail_replacing_block_time'] = (df[df.route.isin(routes_with_repaced_pax)]['arrival_time']-
    #                                                df[df.route.isin(routes_with_repaced_pax)]['departure_time']).mean()

    # statistics['emissions_CO2_t_per_seat_rail'] = statistics['emissions_CO2_rail']/statistics['seats_rail']

    return statistics


# Plots
def plot_results(dict_results, pl, dict_label={}, ylabel=None, stack=False, figpath=None):
    x_labels = [td[0:2] + ':' + td[2:4] for td in dict_results['bans']]

    mpl.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 5))
    if stack:
        # Create a plot
        l_plot = []
        lab_plot = []
        for p in pl:
            l_plot += [dict_results[p]]
            lab_plot += [dict_label.get(p, p)]

        plt.stackplot(x_labels, np.array(l_plot), labels=lab_plot, alpha=0.5)

    else:
        # Create a plot
        for p in pl:
            y = dict_results[p]

            if type(dict_results[p][-1]) is pd.Timedelta:
                y = [z.total_seconds() / 60 if z != None else z for z in y]

            plt.plot(x_labels, y, linestyle='-', linewidth=4, label=dict_label.get(p, p), markersize=8)

    if type(dict_results[p][-1]) is pd.Timedelta:
        # Formatting y-axis to show hours and minutes
        plt.gca().yaxis.set_major_formatter(lambda x, _: '{:02}:{:02}'.format(int(x) // 60, int(x) % 60))

    # Set labels and title
    plt.xlabel('Ban time (hh:mm)')

    if ylabel is not None:
        plt.ylabel(ylabel)

    # Rotate x-axis labels for better visibility
    plt.xticks(np.arange(0, len(x_labels), step=4), rotation=45, ha='center')
    plt.legend()
    if figpath is not None:
        plt.savefig(figpath, bbox_inches='tight')
    else:
        plt.show()


def plot_results_time_types(dict_results, pl, dict_label={}, dict_linestyle={}, reset_colour=[], dict_colour={},
                            ylabel=None, stack=False, figpath=None):
    x_labels = [td[0:2] + ':' + td[2:4] for td in dict_results['bans']]

    mpl.rcParams.update({'font.size': 13})
    plt.figure(figsize=(10, 5))

    if stack:
        # Create a plot
        l_plot = []
        lab_plot = []
        for p in pl:
            l_plot += [dict_results[p]]
            lab_plot += [dict_label.get(p, p)]

        plt.stackplot(x_labels, np.array(l_plot), labels=lab_plot, alpha=0.5)

        plt.legend(pl)


    else:
        # Create a plot
        for p in pl:
            y = list(dict_results[p])
            y1 = y
            if type(y[-1]) is pd.Timedelta:
                y = [z.total_seconds() / 60 if not pd.isna(z) else None for z in y]

            y = [z if not pd.isna(z) else None for z in y]

            if type(y[-1]) is pd.Timestamp:
                y1 = [pd.Timedelta(hours=time.hour, minutes=time.minute) if time is not None else None for time in y]
                y = [td.total_seconds() / 60 if td is not None else None for td in y1]

            if p in reset_colour:
                plt.gca().set_prop_cycle(None)

            if dict_colour.get(p) is not None:
                plt.plot(x_labels, y, linestyle=dict_linestyle.get(p, '-'), linewidth=4,
                         color=dict_colour.get(p), label=dict_label.get(p, p), markersize=8)
            else:
                plt.plot(x_labels, y, linestyle=dict_linestyle.get(p, '-'), linewidth=4, label=dict_label.get(p, p),
                         markersize=8)

            if type(y1[-1]) is pd.Timedelta:
                # Formatting y-axis to show hours and minutes
                plt.gca().yaxis.set_major_formatter(lambda x, _: '{:02}:{:02}'.format(int(x) // 60, int(x) % 60))

        plt.legend()

    # Set labels and title
    plt.xlabel('Ban time (hh:mm)')
    if ylabel is not None:
        plt.ylabel(ylabel)

    # Rotate x-axis labels for better visibility
    plt.xticks(np.arange(0, len(x_labels), step=4), rotation=45, ha='center')

    if figpath is not None:
        plt.savefig(figpath, bbox_inches='tight')
    else:
        plt.show()


# Compute statistics itineraries possible
def compute_statistics_itineraries(df):
    statistics = {}

    statistics['number_flight'] = len(df[df.type == 'flight'])
    statistics['number_rail'] = len(df[df.type == 'rail'])
    statistics['number_flight_flight'] = len(df[df.type == 'flight_flight'])
    statistics['number_rail_flight'] = len(df[df.type == 'rail_flight'])
    statistics['number_flight_rail'] = len(df[df.type == 'flight_rail'])
    statistics['number_multimodal'] = statistics['number_flight_rail'] + statistics['number_rail_flight']
    statistics['number_all'] = len(df)

    statistics['mean_block_time'] = df[df.type == 'flight']['block_time'].mean()
    statistics['mean_block_rail'] = df[df.type == 'rail']['block_time'].mean()
    statistics['mean_block_flight_flight'] = df[df.type == 'flight_flight']['block_time'].mean()
    statistics['mean_block_rail_flight'] = df[df.type == 'rail_flight']['block_time'].mean()
    statistics['mean_block_flight_rail'] = df[df.type == 'flight_rail']['block_time'].mean()
    statistics['mean_block_multimodal'] = df[(df.type == 'rail_flight') | (df.type == 'flight_rail')][
        'block_time'].mean()
    statistics['mean_block_all'] = df['block_time'].mean()

    if len(df[df.type == 'flight_flight']) > 0:
        statistics['median_connecting_time_flight_flight'] = (
                    df[df.type == 'flight_flight']['sobt_connection'] - df[df.type == 'flight_flight'][
                'sibt_orig']).median()
        statistics['mean_connecting_time_flight_flight'] = (
                    df[df.type == 'flight_flight']['sobt_connection'] - df[df.type == 'flight_flight'][
                'sibt_orig']).mean()
    else:
        statistics['median_connecting_time_flight_flight'] = None
        statistics['mean_connecting_time_flight_flight'] = None
    if len(df[df.type == 'rail_flight']) > 0:
        statistics['median_connecting_time_rail_flight'] = pd.Timedelta(seconds=(df[df.type == 'rail_flight']['sobt'].
                                                                                 apply(
            lambda x: x.time().hour * 3600 + x.time().minute * 60)
                                                                                 - df[df.type == 'rail_flight'][
                                                                                     'arrival_time']
                                                                                 .apply(
                    lambda x: x.time().hour * 3600 + x.time().minute * 60)).median())
        statistics['mean_connecting_time_rail_flight'] = pd.Timedelta(seconds=(df[df.type == 'rail_flight']['sobt'].
                                                                               apply(
            lambda x: x.time().hour * 3600 + x.time().minute * 60)
                                                                               - df[df.type == 'rail_flight'][
                                                                                   'arrival_time']
                                                                               .apply(
                    lambda x: x.time().hour * 3600 + x.time().minute * 60)).mean())
    else:
        statistics['median_connecting_time_rail_flight'] = None
        statistics['mean_connecting_time_rail_flight'] = None
    if len(df[df.type == 'flight_rail']) > 0:
        statistics['median_connecting_time_flight_rail'] = pd.Timedelta(
            seconds=((df[df.type == 'flight_rail']['departure_time']
                      .apply(lambda x: x.time().hour * 3600 + x.time().minute * 60))
                     - df[df.type == 'flight_rail']['sibt']
                     .apply(lambda x: x.time().hour * 3600 + x.time().minute * 60)).median())
        statistics['mean_connecting_time_flight_rail'] = pd.Timedelta(
            seconds=((df[df.type == 'flight_rail']['departure_time']
                      .apply(lambda x: x.time().hour * 3600 + x.time().minute * 60))
                     - df[df.type == 'flight_rail']['sibt']
                     .apply(lambda x: x.time().hour * 3600 + x.time().minute * 60)).mean())
    else:
        statistics['median_connecting_time_flight_rail'] = None
        statistics['mean_connecting_time_flight_rail'] = None

    statistics['first_flight_rail_flight'] = df[df.type == 'rail_flight'].sobt.min()
    statistics['last_flight_rail_flight'] = df[df.type == 'rail_flight'].sobt.max()
    statistics['first_rail_rail_flight'] = df[df.type == 'rail_flight'].departure_time.min()
    statistics['last_rail_rail_flight'] = df[df.type == 'rail_flight'].departure_time.max()

    statistics['first_flight_flight_rail'] = df[df.type == 'flight_rail'].sobt.min()
    statistics['last_flight_flight_rail'] = df[df.type == 'flight_rail'].sobt.max()
    statistics['first_rail_flight_rail'] = df[df.type == 'flight_rail'].departure_time.min()
    statistics['last_rail_flight_rail'] = df[df.type == 'flight_rail'].departure_time.max()

    return statistics


if __name__ == '__main__':

    logging.info("\n----------\nREAD DATA\n----------\n")

    # Read all flight schedules
    ds = pd.read_parquet(flight_schedules_path)

    # Give each flight an id
    ds['fid'] = ds.index

    ds['day_sobt'] = ds['sobt'].apply(lambda x: x.date())
    ds['day_sibt'] = ds['sibt'].apply(lambda x: x.date())

    # Real rail data
    with open(rail_dict_direct_path, "rb") as f:
        dict_rail_stations_used = pickle.load(f)

    with open(rail_data_times_path, "rb") as f:
        agency, calendar, calendar_dates, routes, stops_times, stops, trips = pickle.load(f)

    # Dictionary with latlon airports
    dict_airp_latlon = ds[['departure_used', 'dep_lat', 'dep_lon']].drop_duplicates().rename(
        columns={'dep_lat': 'lat', 'dep_lon': 'lon'}).set_index('departure_used')[['lat', 'lon']].to_dict(
        orient='index')
    dict_airp_latlon.update(ds[['arrival_used', 'arr_lat', 'arr_lon']].drop_duplicates().rename(
        columns={'arr_lat': 'lat', 'arr_lon': 'lon'}).set_index('arrival_used')[['lat', 'lon']].to_dict(orient='index'))

    # Compute flights, routes and airports within country of interest
    ds_country, routes_within_country, airports_within_country = compute_routes_within_country(ds,
                                                                                               country_interest,
                                                                                               do_print=True)

    if analyse_individual_days:
        logging.info("\n----------\nANALYSE SPECIFIC DAYS\n----------\n")

        for day_analysis in list_individual_day_analysis:
            logging.info("Doing analysis of day"+str(day_analysis))

            if unzip:
                with zipfile.ZipFile(output_multi_folder_path + day_analysis + '.zip', 'r') as zip_ref:
                    zip_ref.extractall(output_multi_folder_path + day_analysis + '/')

            # This dict_od_remove should be the same as before but as stored in the day
            # Dictionary of o-d pairs remove as a function of ban for a given day
            with open(output_multi_folder_path + day_analysis + "/" + day_analysis +
                      "_dict_od_remove.pickle", "rb") as f:
                dict_od_remove = pickle.load(f)

            # Get route of trains lat_lon
            dt_baseline = pd.read_parquet(
                output_multi_folder_path + day_analysis + '/' + day_analysis + '_it_trains_baseline.parquet')
            dt_baseline.o_d = dt_baseline.o_d.apply(lambda x: (x[0], x[1]))

            dt_baseline_lat_lon = dt_baseline.groupby('trip_id').apply(get_stops_train, stops_times=stops_times,
                                                                       stops=stops).reset_index().drop(['level_1'],
                                                                                                       axis=1)

            # Filter groups based on the custom function
            seen_before = set()

            def is_unique(group):
                if ('_'.join(group.stop_lat.astype(str)), '_'.join(group.stop_lon.astype(str))) in seen_before:
                    return False
                seen_before.add(('_'.join(group.stop_lat.astype(str)), '_'.join(group.stop_lon.astype(str))))
                return True

            dt_baseline_lat_lon = dt_baseline_lat_lon.groupby('trip_id').filter(is_unique)
            dt_baseline_lat_lon = dt_baseline_lat_lon.merge(dt_baseline[['o_d', 'trip_id']], on='trip_id')

            # Load data on rails used when ban in place
            dict_ds_rail_ban = {}
            dict_ds_ff_ban = {}
            for ban_time in dict_od_remove.keys():
                hours, remainder = divmod(ban_time.seconds, 3600)
                minutes, _ = divmod(remainder, 60)
                ban_time_str = '{:02}{:02}'.format(hours, minutes)

                try:
                    ds_rail_ban = pd.read_parquet(
                        output_multi_folder_path + day_analysis + '/' + day_analysis + '_' + ban_time_str +
                        '_ds_rail_ban.parquet')
                    if len(ds_rail_ban) > 0:
                        dict_ds_rail_ban[ban_time] = ds_rail_ban
                except FileNotFoundError:
                    pass

                try:
                    ds_ff_ban = pd.read_parquet(
                        output_multi_folder_path + day_analysis + '/' + day_analysis + '_' + ban_time_str +
                        '_ds_day_f_connecting.parquet')
                    if len(ds_ff_ban) > 0:
                        ds_ff_ban.o_d = ds_ff_ban.o_d.apply(lambda x: (x[0], x[1]))
                        dict_ds_ff_ban[ban_time] = ds_ff_ban
                except FileNotFoundError:
                    pass

            ds_day = pd.read_parquet(output_multi_folder_path + day_analysis + '/' + day_analysis + "_ds_day.parquet")
            ds_country_day, routes_within_country_day, airports_flights_within_country_day = compute_routes_within_country(ds_day,
                                                                                                                           country_interest)

            if do_rail_replacement_maps:
                # Generate plots routes
                # Only rail makes it day specific
                generate_plots_routes(airports_flights_within_country=airports_flights_within_country_day,
                                      routes_within_country=routes_within_country_day,
                                      dict_od_remove=dict_od_remove, dict_airp_latlon=dict_airp_latlon,
                                      dict_rail_stations_used=dict_rail_stations_used, stops=stops,
                                      stops_times=stops_times, dt_baseline=dt_baseline,
                                      dt_baseline_lat_lon=dt_baseline_lat_lon, dict_ds_rail_ban=dict_ds_rail_ban,
                                      dict_ds_ff_ban=dict_ds_ff_ban, plot_baseline_train=True,
                                      plot_replacement_trains=True, day=day_analysis, plot_title=False,
                                      plot_legend=True, output_folder_path=output_img_folder_path)

                # Create gif from images generated of ban
                # And save all of the single frames into an array called frames.
                frames = []

                for ban_time in dict_od_remove.keys():
                    image = imageio.v2.imread(f'{output_img_folder_path}img_{day_analysis}_{ban_time}.png')
                    frames.append(image)

                imageio.mimsave(f'{output_img_folder_path}ban_rail_routes_{day_analysis}.gif',  # output gif
                                frames,  # array of input frames
                                duration=(20000 * 1 / 50),  # optional: duration in ms
                                loop=0x7fff * 2 + 1)  # how many loops the gif does (0x7fff * 2 + 1 is the max, in theory not passing this shouuld do infinite loop but does not seems to be working...)


            if unzip:
                # Delete extracted folder after finishing processing it
                shutil.rmtree(output_multi_folder_path + day_analysis + '/')

    if analyse_route_ban_general:
        logging.info("\n----------\nANALYSE ROUTE BAN IN GENERAL\n----------\n")
        # Read rail data
        df_direct_rail_purged = pd.read_parquet(rail_df_purged_path)
        df_direct_rail_purged['o_d'] = df_direct_rail_purged['o_d'].apply(lambda x: tuple(x))
        df_direct_rail_purged['departure_used'] = df_direct_rail_purged['o_d'].apply(lambda x: x[0])
        df_direct_rail_purged['arrival_used'] = df_direct_rail_purged['o_d'].apply(lambda x: x[1])

        # Add emissions and seats information
        rail_emissions = pd.read_csv(rail_used_emissions_path, sep=';')
        rail_emissions['route_stations'] = rail_emissions.apply(
            lambda x: (x['stop_origin_id'], x['stop_destination_id']), axis=1)

        df_direct_rail_purged['route_stations'] = df_direct_rail_purged.apply(
            lambda x: (x['stop_start'], x['stop_end']), axis=1)

        df_direct_rail_purged = df_direct_rail_purged.merge(rail_emissions,
                                                            right_on=['route_stations', 'train_type'],
                                                            left_on=['route_stations', 'route_short_name'], how='left')

        df_agg_rail_stats = df_direct_rail_purged.groupby(['day', 'o_d']).agg(
            number_of_elements=pd.NamedAgg(column='departure_time', aggfunc='size'),
            first_departure_time=pd.NamedAgg(column='departure_time', aggfunc='first'),
            last_departure_time=pd.NamedAgg(column='departure_time', aggfunc='last'),
            first_arrival_time=pd.NamedAgg(column='arrival_time', aggfunc='first'),
            last_arrival_time=pd.NamedAgg(column='arrival_time', aggfunc='last'),
            min_leg_time=pd.NamedAgg(column='leg_time', aggfunc='min'),
            max_leg_time=pd.NamedAgg(column='leg_time', aggfunc='max'),
            median_leg_time=pd.NamedAgg(column='leg_time', aggfunc=lambda x: x.median() if len(x) > 0 else None),
            mean_leg_time=pd.NamedAgg(column='leg_time', aggfunc=lambda x: x.mean() if len(x) > 0 else None),
            mean_co2pax=pd.NamedAgg(column='CO2kg/PAX', aggfunc=lambda x: x.mean() if len(x) > 0 else None)
        ).reset_index()


        # Create a range of Timedelta objects
        timedelta_range = pd.timedelta_range(start=start_time, end=end_time, freq=step)

        dict_od_remove = {}
        for t in timedelta_range:
            dict_od_remove[t] = set(df_agg_rail_stats[df_agg_rail_stats.min_leg_time < t].o_d)

        logging.info("\n----------\nROUTES BANNED AS A FUNCTION OF BAN THRESHOLD\n----------\n")

        # Count routes that are kept as a function of time ban
        dict_ban_number_routes = {}
        for ban_time in dict_od_remove.keys():
            od_remove = dict_od_remove.get(ban_time)
            dict_ban_number_routes[ban_time] = len(routes_within_country[~routes_within_country.o_d.isin(od_remove)])

        # Plot routes as a function of threshold
        # Extract keys and values
        keys = list(dict_ban_number_routes.keys())
        values = list(dict_ban_number_routes.values())

        # Convert Timedelta objects to hours and minutes for the x-axis labels
        x_labels = [str(td).split()[-1][:5] for td in keys]

        mpl.rcParams.update({'font.size': 13})

        plt.figure(figsize=(10, 4))
        # Create a plot
        plt.plot(x_labels, values, marker='o', linestyle='-', linewidth=4, markersize=8)

        # Set labels and title
        plt.xlabel('Ban time (hh:mm)')
        plt.ylabel('Number routes')
        # plt.title('Number routes as a function of rail ban time')

        # Rotate x-axis labels for better visibility
        plt.xticks(np.arange(0, len(x_labels), step=4), rotation=45, ha='center')

        # Save the plot
        plt.savefig(output_figures_folder_path+'routes_operated_function_ban_all_days.png', bbox_inches='tight')

        logging.info("\n----------\nCOMPETITION ON ROUTES\n----------\n")
        ds_country.groupby('o_d')['airline'].agg(unique_count='nunique').reset_index().to_csv(output_competition_routes_path)

        logging.info("\n----------\nNUMBER FLIGHTS/ROUTES PER AIRLINE AS FUNCTION OF BAN\n----------\n")

        ##############
        # All values #
        ##############

        dict_flights_airlines_ban = {}
        for ban_time in dict_od_remove.keys():
            od_remove = dict_od_remove.get(ban_time)
            ds_kept = ds_country[~ds_country.o_d.isin(od_remove)].groupby(['airline']).count()['icao24']

            for a in ds_kept.keys():
                dict_ab = dict_flights_airlines_ban.get(ban_time, {})
                dict_ab[a] = ds_kept[a]
                dict_flights_airlines_ban[ban_time] = dict_ab
            dict_ab['All'] = ds_kept.sum()
            dict_flights_airlines_ban[ban_time] = dict_ab

        dict_flights_airlines_ban, dict_routes_airlines_ban, dict_avg_flight_airlines_ban = compute_statistics_num_flight_routes_mean_flight_time(
            ds_country, dict_od_remove)

        # keys, values = plot_flight_routes_airlines_ban(dict_flights_airlines_ban,
        #                                                output_figures_folder_path+'flights_per_airline_ban_all_days',
        #                                                title=None,  # 'Number flights as a function of rail ban time',
        #                                                show=False)
        # keys, values = plot_flight_routes_airlines_ban(dict_routes_airlines_ban,
        #                                                output_figures_folder_path+'routes_per_airline_ban_all_days',
        #                                                title=None,  # 'Number routes as a function of rail ban time'
        #                                                show=False)

        ##################
        # Average values #
        ##################
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        dict_flight_values = {}
        dict_routes_values = {}

        for day_computation in date_range(start_date, end_date):
            day_computation = day_computation.date()

            ds_country_day = ds_country[
                (ds_country.day_sobt == day_computation) | (ds_country.day_sibt == day_computation)].copy()

            dict_flights_airlines_ban, dict_routes_airlines_ban, dict_avg_flight_airlines_ban = compute_statistics_num_flight_routes_mean_flight_time(
                ds_country_day, dict_od_remove)
            keys, values = extract_key_values_dict_flight_routes(dict_flights_airlines_ban)

            dict_flight_values[day_computation] = (keys, values)

            keys, values = extract_key_values_dict_flight_routes(dict_routes_airlines_ban)

            dict_routes_values[day_computation] = (keys, values)

        hours_bans = keys

        averaged_flights_values = {}

        for key, (list_keys, values_dict) in dict_flight_values.items():
            for airline, values_list in values_dict.items():
                if airline not in averaged_flights_values:
                    averaged_flights_values[airline] = [0] * len(values_list)

                for i, value in enumerate(values_list):
                    averaged_flights_values[airline][i] += value

        # Calculate the average
        for key, values in averaged_flights_values.items():
            averaged_flights_values[key] = [x / len(dict_flight_values) for x in values]

        averaged_routes_values = {}

        for key, (list_keys, values_dict) in dict_routes_values.items():
            for airline, values_list in values_dict.items():
                if airline not in averaged_routes_values:
                    averaged_routes_values[airline] = [0] * len(values_list)

                for i, value in enumerate(values_list):
                    averaged_routes_values[airline][i] += value

        # Calculate the average
        for key, values in averaged_routes_values.items():
            averaged_routes_values[key] = [x / len(dict_routes_values) for x in values]

        # Plot avg routes stacked
        x_labels = [str(td).split()[-1][:5] for td in hours_bans]

        mpl.rcParams.update({'font.size': 13})

        values_staked = averaged_routes_values.copy()
        del values_staked['All']

        mpl.rcParams.update({'font.size': 13})
        plt.figure(figsize=(10, 5))
        # Create a plot
        plt.stackplot(x_labels, np.array(list(values_staked.values())), labels=values_staked.keys(), alpha=0.5)

        # Set labels and title
        plt.xlabel('Ban time (hh:mm)')
        plt.ylabel('Average number routes')
        # plt.title('Number routes as a function of rail ban time')

        # Rotate x-axis labels for better visibility
        plt.xticks(np.arange(0, len(x_labels), step=4), rotation=45, ha='center')

        plt.legend(values_staked.keys())

        plt.savefig(output_figures_folder_path+'average_routes_per_airline_ban_stacked.png', bbox_inches='tight')

        # Save values of routes average
        pd.DataFrame(values_staked).to_csv(output_multi_folder_path+'routes_stacked_average.csv')

    if analyse_itineraries:
        logging.info("\n----------\nANALYSE ITINERARIES\n----------\n")

        logging.info("\n----------\nCOMPUTE/READ DICT_D_IT\n----------\n")
        dict_d_it = None
        if read_precomputed_dict_d_it:
            try:
                with open(dict_d_it_file_path, "rb") as f:
                    dict_d_it = pickle.load(f)
            except FileNotFoundError:
                pass

        if dict_d_it is None:
            # Process all days to read all the data previously computed and create d_it with
            # all the itineraries for a given day and ban this will be stored in dict_d_it
            dict_d_it = {}

            # Define days to compute
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

            print_skip_files = False  # Print files that are nof found and previous is used

            for day_analysis in date_range(start_date, end_date):

                day_analysis = day_analysis.strftime("%Y%m%d")

                zip_path = f'{output_multi_folder_path}{day_analysis}.zip'
                extraction_path = f'{output_multi_folder_path}{day_analysis}_temp'

                # Unzip the folder
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extraction_path)

                timedelta_range = pd.timedelta_range(start=start_time, end=end_time, freq=step)

                for ban_time in timedelta_range:
                    hours, remainder = divmod(ban_time.seconds, 3600)
                    minutes, _ = divmod(remainder, 60)
                    ban_time_str = '{:02}{:02}'.format(hours, minutes)
                    ban_time_str_p = '{:02}:{:02}'.format(hours, minutes)

                    try:
                        ds_day_direct_ban = pd.read_parquet(
                            f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_day_direct_ban.parquet')
                        ds_day_f_connecting = pd.read_parquet(
                            f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_day_f_connecting.parquet')
                        ds_day_f_connecting.o_d = ds_day_f_connecting.o_d.apply(lambda x: (x[0], x[1]))
                        ds_ra = pd.read_parquet(f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_ra.parquet')
                        ds_ar = pd.read_parquet(f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_ar.parquet')
                        ds_r = pd.read_parquet(f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_rail_ban.parquet')
                        ds_rb = pd.read_parquet(f'{extraction_path}/{day_analysis}_{ban_time_str}_ds_rb.parquet')
                        ds_rb.o_d = ds_rb.o_d.apply(lambda x: (x[0], x[1]))
                        # From all the rail baseline (ds_rb), which are also for a given day, only the ones that
                        # cannot be reached by f-f are to be kept.
                        ds_rb = ds_rb.merge(ds_day_f_connecting[['o_d', 'airline_orig']], on='o_d', how='left')
                        ds_rb = ds_rb[ds_rb.airline_orig.isnull()]

                    except FileNotFoundError as e:
                        if print_skip_files:
                            logging.info(f'File missing: {extraction_path}/{day_analysis}_{ban_time_str}_ds_day_direct_ban.parquet')
                        pass

                    # Fix tuples and check if any rail
                    any_rail = False
                    if len(ds_day_direct_ban) > 0:
                        ds_day_direct_ban['o_d'] = ds_day_direct_ban['o_d'].apply(lambda x: (x[0], x[1]))
                        ds_day_direct_ban['route'] = ds_day_direct_ban['route'].apply(lambda x: (x[0], x[1]))
                        ds_day_direct_ban['o_d_leg1'] = ds_day_direct_ban['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                    ds_day_direct_ban['route_airlines'] = ds_day_direct_ban['route_airlines'].apply(lambda x: (x))

                    if len(ds_day_f_connecting) > 0:
                        ds_day_f_connecting['o_d'] = ds_day_f_connecting['o_d'].apply(lambda x: (x[0], x[1]))
                        ds_day_f_connecting['route'] = ds_day_f_connecting['route'].apply(lambda x: (x[0], x[1], x[2]))
                        ds_day_f_connecting['o_d_leg1'] = ds_day_f_connecting['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_day_f_connecting['o_d_leg2'] = ds_day_f_connecting['o_d_leg2'].apply(lambda x: (x[0], x[1]))
                        ds_day_f_connecting['route_airlines'] = ds_day_f_connecting['route_airlines'].apply(
                            lambda x: (x[0], x[1]))

                    if len(ds_ra) > 0:
                        any_rail = True
                        ds_ra['o_d_train_l1'] = ds_ra['o_d_train_l1'].apply(lambda x: (x[0], x[1]))
                        ds_ra['o_d_orig'] = ds_ra['o_d_orig'].apply(lambda x: (x[0], x[1]))
                        ds_ra['route'] = ds_ra['route'].apply(lambda x: (x[0], x[1], x[2]))
                        ds_ra['o_d_leg1'] = ds_ra['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_ra['o_d_leg2'] = ds_ra['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_ra['route_airlines'] = ds_ra['route_airlines'].apply(lambda x: (x[0], x[1]))
                        ds_ra['route_stations'] = ds_ra['route_stations'].apply(lambda x: (x[0], x[1]))

                    if len(ds_ar) > 0:
                        any_rail = True
                        ds_ar['o_d_train_l2'] = ds_ar['o_d_train_l2'].apply(lambda x: (x[0], x[1]))
                        ds_ar['o_d_orig'] = ds_ar['o_d_orig'].apply(lambda x: (x[0], x[1]))
                        ds_ar['route'] = ds_ar['route'].apply(lambda x: (x[0], x[1], x[2]))
                        ds_ar['o_d_leg1'] = ds_ar['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_ar['o_d_leg2'] = ds_ar['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_ar['route_airlines'] = ds_ar['route_airlines'].apply(lambda x: (x[0], x[1]))
                        ds_ar['route_stations'] = ds_ar['route_stations'].apply(lambda x: (x[0], x[1]))

                    if len(ds_r) > 0:
                        any_rail = True
                        ds_r['o_d'] = ds_r['o_d'].apply(lambda x: (x[0], x[1]))
                        ds_r['route'] = ds_r['route'].apply(lambda x: (x[0], x[1]))
                        ds_r['o_d_leg1'] = ds_r['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_r['o_d_leg2'] = ds_r['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_r['route_stations'] = ds_r['route_stations'].apply(lambda x: (x[0], x[1]))
                    ds_r['route_airlines'] = ds_r['route_airlines'].apply(lambda x: (x))

                    if len(ds_rb) > 0:
                        any_rail = True
                        ds_rb['o_d'] = ds_rb['o_d'].apply(lambda x: (x[0], x[1]))
                        ds_rb['route'] = ds_rb['route'].apply(lambda x: (x[0], x[1]))
                        ds_rb['o_d_leg1'] = ds_rb['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_rb['o_d_leg2'] = ds_rb['o_d_leg1'].apply(lambda x: (x[0], x[1]))
                        ds_rb['route_stations'] = ds_rb['route_stations'].apply(lambda x: (x[0], x[1]))
                    else:
                        ds_rb['route_stations'] = None
                    ds_rb['route_airlines'] = ds_rb['route_airlines'].apply(lambda x: (x))

                    # Concat on all itineraries
                    #columns_keep = ['origin','destination','region_orig','region_destination','block_time',
                    #    'route','route_airlines','type']
                    columns_keep = ['origin', 'destination', 'block_time',
                                    'route', 'route_airlines', 'type']

                    lconcat = []
                    if len(ds_day_direct_ban) > 0:
                        lconcat += [ds_day_direct_ban[columns_keep + ['typecode', 'sobt', 'sibt']].reset_index(drop=True)]
                    if len(ds_day_f_connecting) > 0:
                        lconcat += [ds_day_f_connecting[
                                        columns_keep + ['typecode_orig', 'typecode_connection', 'sobt_orig', 'sibt_orig',
                                                        'sobt_connection', 'sibt_connection']].reset_index(drop=True)]
                    if len(ds_ra) > 0:
                        lconcat += [ds_ra[columns_keep + ['trip_id', 'sobt', 'sibt', 'typecode', 'route_stations',
                                                          'departure_time', 'arrival_time']].reset_index(drop=True)]
                    if len(ds_ar) > 0:
                        lconcat += [ds_ar[columns_keep + ['trip_id', 'sobt', 'sibt', 'typecode', 'route_stations',
                                                          'departure_time', 'arrival_time']].reset_index(drop=True)]
                    if len(ds_r) > 0:
                        lconcat += [ds_r[columns_keep + ['trip_id', 'route_stations', 'departure_time',
                                                         'arrival_time']].reset_index(drop=True)]
                    if len(ds_rb) > 0:
                        lconcat += [ds_rb[columns_keep + ['trip_id', 'route_stations', 'departure_time',
                                                          'arrival_time']].reset_index(drop=True)]

                    d_it = pd.concat(lconcat, ignore_index=True)

                    if not any_rail:
                        for col in ['trip_id', 'route_stations', 'departure_time', 'arrival_time']:
                            d_it[col] = None

                    dict_it_bans = dict_d_it.get(day_analysis)
                    if dict_it_bans is None:
                        dict_it_bans = {ban_time_str: d_it}
                    else:
                        dict_it_bans[ban_time_str] = d_it

                    dict_d_it[day_analysis] = dict_it_bans

                # Delete the extracted folder
                shutil.rmtree(extraction_path)

            with open(dict_d_it_file_path, "wb") as f:
                pickle.dump(dict_d_it, f)

        logging.info("\n----------\nPROCESS DICTIONARY OF DATA-BAN TO COMPUTE LIST OF FLIGHTS, "
              "RAILS AND METRICS ON ITINERARIES\n----------\n")
        ###############################################
        # Number of flights and rail services per day #
        ###############################################

        ds_per_day = ds_country.groupby(['day_sobt']).count().reset_index()[['day_sobt', 'airline']]

        ds_per_day = ds_per_day[ds_per_day['airline'] > 10]

        num_rail = []
        for d in ds_per_day.day_sobt:
            num_rail += [
                len(dict_d_it[d.strftime('%Y%m%d')]['1500'][dict_d_it[d.strftime('%Y%m%d')]['1500'].type == 'rail'])]

        logging.info('mean number flights: '+str(ds_per_day['airline'].mean()))
        logging.info('mean number rail services: '+str(sum(num_rail) / len(num_rail)))

        logging.info("\n----------\nHISTOGRAM OF DEMAND\n----------\n")
        # Number of runway usage at airport_interest_histogram nominal and with max ban for one day

        df_ = dict_d_it[day_example_histogram_demand]['0000']
        plot_histogram_demand(df_, airport_interest_histogram, day_example_histogram_demand, save_ending='0000',
                              save_folder_path=output_figures_folder_path, ylim_max=30, yfigmax=10)
        df_ = dict_d_it[day_example_histogram_demand]['1500']
        df_orig_ = df_.copy()
        plot_histogram_demand(df_, airport_interest_histogram, day_example_histogram_demand, save_ending='1500',
                              save_folder_path=output_figures_folder_path, ylim_max=30, yfigmax=10)

        df_ = dict_d_it[day_example_histogram_demand]['0000']
        departure_times_orig = df_[(df_.origin == airport_interest_histogram) & (df_.type == 'flight')].sobt
        arrival_times_orig = df_[(df_.destination == airport_interest_histogram) & (df_.type == 'flight')].sibt
        df_ = dict_d_it[day_example_histogram_demand]['1500']
        departure_times_ban = df_[(df_.origin == airport_interest_histogram) & (df_.type == 'flight')].sobt
        arrival_times_ban = df_[(df_.destination == airport_interest_histogram) & (df_.type == 'flight')].sibt
        departure_times_removed = [elem for elem in list(departure_times_orig) if elem not in list(departure_times_ban)]
        arrival_times_removed = [elem for elem in list(arrival_times_orig) if elem not in list(arrival_times_ban)]
        plot_histogram_demand(df_, airport_interest_histogram, day_example_histogram_demand,
                              departure_times=departure_times_removed, arrival_times=arrival_times_removed,
                              save_ending='diff_0000_15000',
                              save_folder_path=output_figures_folder_path, ylim_max=10)

        logging.info("\n----------\nEMISSIONS AND STATISTICS COMPUTATION\n----------\n")
        # Read routes/air emissions
        routes_emissions = pd.read_csv(routes_emissions_path)
        routes_emissions_inv = routes_emissions.copy()
        routes_emissions['route'] = routes_emissions.apply(lambda x: (x['departure_used'], x['arrival_used']), axis=1)
        routes_emissions_inv['route'] = routes_emissions.apply(lambda x: (x['arrival_used'], x['departure_used']),
                                                               axis=1)
        routes_emissions = pd.concat([routes_emissions, routes_emissions_inv])

        # Read rail emissions
        rail_emissions = pd.read_csv(rail_used_emissions_path, sep=';')
        rail_emissions['route_stations'] = rail_emissions.apply(
            lambda x: (x['stop_origin_id'], x['stop_destination_id']), axis=1)

        rail_emissions = rail_emissions.drop(['Unnamed: 0'], axis=1)
        rail_emissions_additional = pd.read_csv(rail_used_emissions_additional_path, sep=';')
        rail_emissions_additional = rail_emissions_additional.rename(columns={'station_orig_id': 'stop_origin_id',
                                                                              'station_dest_id': 'stop_destination_id',
                                                                              'kgCO2': 'CO2kg/PAX',
                                                                              'stop_name_dest': 'stop_destination_name',
                                                                              'stop_name_orig': 'stop_orig_name'}).drop(
            ['Unnamed: 0'], axis=1)
        rail_emissions_additional['route_stations'] = rail_emissions_additional.apply(lambda x:
                                                                                      (x.stop_origin_id,
                                                                                       x.stop_destination_id), axis=1)
        rail_emissions = pd.concat([rail_emissions, rail_emissions_additional])

        # Read airline info
        # with open(airline_info_path, "rb") as f:
        #    (dict_alliance, dict_mct_rail, def_mct_rail, def_mct_air) = pickle.load(f)

        #######################################
        # Compute statistics flights and rail #
        #######################################

        # compute matrix of flight and rail per day and ban
        dict_d_fr = {}
        for d in dict_d_it.keys():
            dict_ban_in_day = {}
            i = 0
            for b in dict_d_it[d].keys():
                df = dict_d_it[d][b]
                df_flights = extract_flights(df)
                df_rails = extract_rails(df)

                if i == 0:
                    # first time in this day, i.e, ban=0000 so all flights available, compute seats per o-d pair
                    f_orig = df_flights.copy()
                    f_orig = f_orig.merge(routes_emissions[['route', 'typecode', 'seats',
                                                            'CO2_g_ASK', 'CO2_kg_pax', 'CO2_t_flight']],
                                          on=['route', 'typecode'], how='left')

                    dict_f_seats = f_orig.groupby('route')['seats'].sum().to_dict()
                    i += 1

                # Computes statistics we want to plot/analyse later
                statistics_df_flights = compute_statistics_flights(df_flights, routes_emissions,
                                                                   country_interest, manually_remove_airports)

                statistics_df_rails = compute_statistics_rails(df_rails, rail_emissions, dict_f_seats, d, b)

                dict_ban_in_day[b] = {'df_flights': df_flights,
                                      'statistics_df_flights': statistics_df_flights,
                                      'df_rails': df_rails,
                                      'statistics_df_rails': statistics_df_rails
                                      }

            dict_d_fr[d] = dict_ban_in_day

        ######################################################
        # Aggregate results and create list across ban times #
        ######################################################

        results_dict = {}

        for d in dict_d_fr.keys():
            results_array = {}
            for b in dict_d_it[d].keys():
                bans = results_array.get('bans', [])
                bans += [b]
                results_array['bans'] = bans
                for s in ['statistics_df_flights', 'statistics_df_rails']:
                    for k in dict_d_fr[d][b][s].keys():
                        ks = results_array.get(k, [])
                        ks += [dict_d_fr[d][b][s][k]]
                        results_array[k] = ks

            results_dict[d] = results_array


        # Compute average of all day

        def average_timedeltas(timedeltas):
            if len(timedeltas) == 0:
                return None
            total_seconds = sum(td.total_seconds() for td in timedeltas)
            return pd.Timedelta(seconds=total_seconds / len(timedeltas))


        def average_numeric(nums):
            return sum(nums) / len(nums)


        # Get the unique keys (e.g., 'res1', 'res2')
        res_keys = set(key for values in results_dict.values() for key in values)
        res_keys.discard('bans')

        # Create a dictionary to store the averages
        average_dict = {}

        # Compute the average for all keys
        for res_key in res_keys:
            all_values = [values[res_key] for values in results_dict.values() if res_key in values]

            if all_values:
                if any(isinstance(val, timedelta) for sublist in all_values for val in sublist):
                    # Compute the average for 'res1' (timedeltas)
                    averages = [average_timedeltas(
                        [values[i] for values in all_values if i < len(values) and not pd.isna(values[i])])
                                for i in range(max(map(len, all_values)))]
                else:
                    # Compute the average for numeric values (e.g., 'res2')
                    averages = [average_numeric([values[i] for values in all_values if i < len(values)])
                                for i in range(max(map(len, all_values)))]

                # Store the averages in the result dictionary
                average_dict[res_key] = averages

        average_dict['bans'] = results_dict[list(results_dict.keys())[0]]['bans']


        average_dict['emissions_CO2_t_flights_rail_pax_replaced'] = \
            [co2f + co2r for co2f, co2r in zip(average_dict['emissions_CO2_t_flights_country_mr'],
                                               average_dict['emissions_CO2_rail_pax_replaced'])]

        average_dict['emissions_CO2_t_saved'] = \
            [average_dict['emissions_CO2_t_flights_country_mr'][0] - co2f - co2r for co2f, co2r in
             zip(average_dict['emissions_CO2_t_flights_country_mr'],
                 average_dict['emissions_CO2_rail_pax_replaced'])]

        average_dict['emissions_CO2_t_saved_marginal_extra_ban'] = \
            [co2tp - co2t for co2t, co2tp in
             zip(average_dict['emissions_CO2_t_saved'], average_dict['emissions_CO2_t_saved'][1:])]
        average_dict['emissions_CO2_t_saved_marginal_extra_ban'] = [None] + average_dict[
            'emissions_CO2_t_saved_marginal_extra_ban']

        # Plot results emissions
        plot_results(average_dict, ['emissions_CO2_t_flights_country_mr', 'emissions_CO2_rail_pax_replaced',
                                    'emissions_CO2_t_saved',
                                    ],
                     dict_label={'emissions_CO2_t_flights_country_mr': '$CO_2$ flights Peninsular Spain',
                                 'emissions_CO2_rail_pax_replaced': '$CO_2$ for passengers rail replacement',
                                 'emissions_CO2_rail': '$CO_2$ total rail',
                                 'emissions_CO2_t_saved': '$CO_2$ saved'},
                     ylabel='Tonnes $CO_2$',
                     figpath=output_figures_folder_path+'emissions_shifted_air_rail_final.png')

        # Save some outputs
        pd.DataFrame({
            'bans': average_dict['bans'],
            'emissions_CO2_t_flights_country_mr': average_dict['emissions_CO2_t_flights_country_mr'],
            'emissions_CO2_rail': average_dict['emissions_CO2_rail']
        }).to_csv(output_emissions_flight_rail)

        pd.DataFrame({
            'bans': average_dict['bans'],
            'emissions_CO2_t_flights_country_mr': average_dict['emissions_CO2_t_flights_country_mr'],
            'emissions_CO2_rail': average_dict['emissions_CO2_rail'],
            'emissions_CO2_rail_pax_replaced': average_dict['emissions_CO2_rail_pax_replaced'],
            'emissions_CO2_t_saved': average_dict['emissions_CO2_t_saved'],
        }).to_csv(output_emissions_flight_rail_pax)

        pd.DataFrame({
            'bans': average_dict['bans'],
            'seats_rail': average_dict['seats_rail'],
            'seats_replaced': average_dict['seats_replaced']
        }).to_csv(output_seats_replaced_rail)

        logging.info("\n----------\nITINERARIES POSSIBLE\n----------\n")

        # compute matrix of itineraries stats per day and ban
        dict_d_it_analysis = {}
        for d in dict_d_it.keys():
            dict_ban_in_day = {}

            for b in dict_d_it[d].keys():
                df = dict_d_it[d][b]

                # Computes statistics we want to plot/analyse later
                statistics_df_it_all = compute_statistics_itineraries(df)

                dict_stats_all = {}
                for k, v in statistics_df_it_all.items():
                    dict_stats_all[k + "_all"] = v

                # Flights within country of interest
                df = df[df.origin.str.startswith(country_interest) &
                        df.destination.str.startswith(country_interest)].copy()

                # Manually removing the Balearic islands
                df = df[~df['origin'].isin(manually_remove_airports) &
                        ~df['destination'].isin(manually_remove_airports)].copy()

                # HERE
                # df = df[~df.route.str[1].isin(manually_remove_airports)].copy()

                # Computes statistics we want to plot/analyse later
                statistics_df_it_country_interest = compute_statistics_itineraries(df)

                dict_stats_main = {}
                for k, v in statistics_df_it_country_interest.items():
                    dict_stats_main[k + "_country_interest"] = v

                dict_ban_in_day[b] = {'df_itineraries_country_interest': df,
                                      'statistics_it_all': dict_stats_all,
                                      'statistics_it_country_interest': dict_stats_main,
                                      }

            dict_d_it_analysis[d] = dict_ban_in_day

        with open(f_dict_d_it_analysis_pickle, "wb") as f:
            pickle.dump(dict_d_it_analysis, f)

        # Note that in itineraries we might have for a given day something like departing day before arriving in day
        # that's fine for flights but when doing rail-flight you might end up with rail on the day before!
        # some double counting might exit!

        # Aggregate results and create list across ban times
        results_dict = {}

        for d in dict_d_it_analysis.keys():
            results_array = {}
            for b in dict_d_it[d].keys():
                bans = results_array.get('bans', [])
                bans += [b]
                results_array['bans'] = bans
                for s in ['statistics_it_all', 'statistics_it_country_interest']:
                    for k in dict_d_it_analysis[d][b][s].keys():
                        ks = results_array.get(k, [])
                        ks += [dict_d_it_analysis[d][b][s][k]]
                        results_array[k] = ks

            results_dict[d] = results_array

        # Compute average of all day

        # Get the unique keys (e.g., 'res1', 'res2')
        res_keys = set(key for values in results_dict.values() for key in values)
        res_keys.discard('bans')

        # Create a dictionary to store the averages
        average_it_dict = {}

        # Compute the average for all keys
        for res_key in res_keys:
            all_values = [values[res_key] for values in results_dict.values() if res_key in values]

            if all_values:
                # averages = np.nanmean([pd.Series(x) for x in all_values], axis=0)#np.nanmean(all_values, axis=0)
                averages = pd.DataFrame(all_values).mean(axis=0, numeric_only=False)
                # if type(averages[0])==np.timedelta64:
                #    averages = [pd.Timedelta(x) for x in averages]
                # Store the averages in the result dictionary
                average_it_dict[res_key] = averages

        average_it_dict['bans'] = results_dict[list(results_dict.keys())[0]]['bans']

        # logging.info(str(average_it_dict['mean_block_time_country_interest'][0]))

        # Plot results
        plot_results_time_types(average_it_dict, pl=['number_flight_country_interest',
                                          'number_rail_country_interest',
                                          'number_flight_flight_country_interest',
                                          'number_rail_flight_country_interest',
                                          'number_flight_rail_country_interest',
                                          'number_multimodal_country_interest',
                                          'number_all_country_interest'],
                     dict_label={'number_flight_country_interest': 'flight',
                                 'number_rail_country_interest': 'rail',
                                 'number_flight_flight_country_interest': 'flight-flight',
                                 'number_rail_flight_country_interest': 'rail-flight',
                                 'number_flight_rail_country_interest': 'flight-rail',
                                 'number_multimodal_country_interest': 'multimodal',
                                 'number_all_country_interest': 'all'},
                     ylabel='Number itineraries possible',
                     figpath=output_figures_folder_path+'number_itinerary_country_interest_w_all.png')

        plot_results_time_types(average_it_dict, pl=['mean_block_time_country_interest',
                                          'mean_block_rail_country_interest',
                                          'mean_block_flight_flight_country_interest',
                                          'mean_block_rail_flight_country_interest',
                                          'mean_block_flight_rail_country_interest',
                                          'mean_block_multimodal_country_interest'
                                          ],
                     dict_label={'mean_block_time_country_interest': 'flight',
                                 'mean_block_rail_country_interest': 'rail',
                                 'mean_block_flight_flight_country_interest': 'flight-flight',
                                 'mean_block_rail_flight_country_interest': 'rail-flight',
                                 'mean_block_flight_rail_country_interest': 'flight-rail',
                                 'mean_block_multimodal_country_interest': 'multimodal'},
                     ylabel='Mean block time (hh:mm)',
                     figpath=output_figures_folder_path+'mean_itinerary_time_country_interest.png')

logging.shutdown()