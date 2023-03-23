import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging



DEMO_DIRECTORY = '../data/us/ct/'

def get_addresses(directory=DEMO_DIRECTORY, filename=None):
    """Get address from number and street columns."""
    if directory is None:
        if filename is None:
            raise ValueError('Must specify directory or filename.')
        df = gpd.read_file(filename)
        df['address'] = df['number'].apply(str) + ' ' + df['street']
        return df['address']


    if filename is not None:
        raise ValueError('Cannot specify both directory and filename.')

    dfs = []
    directory = Path(directory)
    path = directory.glob('*.geojson')

    for file in tqdm(list(path), desc='Reading files'):
        df = gpd.read_file(file)

        if 'number' not in df.columns or 'street' not in df.columns:
            logging.warning(f'File {file} does not have number and street columns.')
            continue

        df['address'] = df['number'].apply(str) + ' ' + df['street']
        dfs.append(df['address'])

    return pd.concat(dfs).reset_index(drop=True)


def parse_filename(filename):
    """Parse directory name"""
    ## get name after data/
    filename = str(filename).split('data/')[-1]
    filename = filename.split('.geojson')[0]
    filename = filename.split('/')
    return '-'.join(filename)


def create_compressed_address_df(directory=DEMO_DIRECTORY, filename=None):
    """Get address from number and street columns."""
    if directory is None:
        if filename is None:
            raise ValueError('Must specify directory or filename.')
        df = gpd.read_file(filename)
        df['address'] = df['number'].apply(str) + ' ' + df['street']
        return df['address']


    if filename is not None:
        raise ValueError('Cannot specify both directory and filename.')

    dfs = []
    _directory = Path(directory)
    path = _directory.rglob('*.geojson')

    for file in tqdm(list(path), desc='Reading files'):
        df = gpd.read_file(file)

        if 'number' not in df.columns or 'street' not in df.columns:
            logging.warning(f'File {file} does not have number and street columns.')
            continue

        df['address']  = df['number'].apply(str) + ' ' + df['street']
        df['location'] = parse_filename(file)
        dfs.append(df[['location', 'address']])

    df = pd.concat(dfs).reset_index(drop=True)
    df.to_feather(f'../data/extracted_addresses.feather')



if __name__ == '__main__':
    FILENAME = 'data/us/ct/city_of_bridgeport-addresses-city.geojson'
    DIRECTORY = '../data/us/'

    #df = get_addresses()
    create_compressed_address_df(directory=DIRECTORY)
