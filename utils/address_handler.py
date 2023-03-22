import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging



DEMO_DIRECTORY = 'data/us/ct/'

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



if __name__ == '__main__':
    FILENAME = 'data/us/ct/city_of_bridgeport-addresses-city.geojson'
    DIRECTORY = 'data/us/ct/'

    df = get_addresses()
    print(df.head())
