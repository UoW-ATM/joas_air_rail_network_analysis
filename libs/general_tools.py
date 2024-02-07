import shutil
import pickle
import os
from datetime import timedelta
import numpy as np


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two point on the earth (specified in decimal degrees)
    :param lon1: longitude first point in degrees
    :param lat1: latitude fist point in degrees
    :param lon2: longitude second point in degrees
    :param lat2: latitude second point in degrees
    :return: distance between points in km
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    km = 2. * 6373. * np.arcsin(np.sqrt(np.sin(dlat / 2.) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.) ** 2))

    return km


def zip_and_delete_folder(folder_path):
    # Create a zip file with the same name as the folder
    shutil.make_archive(folder_path, 'zip', folder_path)

    # Remove the original folder
    shutil.rmtree(folder_path)


def save_csv_parquet(df, file_path):
    df.to_csv(file_path + '.csv')
    df.to_parquet(file_path + '.parquet')


def save_pickle(data, file_path):
    with open(file_path + ".pickle", "wb") as f:
        pickle.dump(data, f)


def date_range(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        yield current_date
        current_date += timedelta(days=1)


def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
