# -*- coding: utf-8 -*-
#
# This script is used to preprocess Landsat 5 TM, 7 ETM+, and 8 OLI data in netCDF format derived form the Swiss Data
# Cube archive (Chatenoux et. al., 2021). Preprocessing includes filtering (fill-value & valid data range), masking
# (saturated pixels, clouds, cloud shadows & snow) using qa_band bitmasks, scaling surface reflectance values (0-1),
# and merging scenes (same day and grid). The Landsat scenes should be arranged in the following directory structure:
#
# Dir structure:    root_dir/landsat_scenes_raw/landsat version/year/landsat_scene.nc
# Example dir:      landsat_scenes_raw/landsat5/2008/LS5_TM_LEDAPS_4326_2_13_20080826100207000000.nc
#
# The script extract_landsat_data.py can be used to extract the data in order to fit the required structure above.
#
# File:             preprocess_landsat.py
# Synopsis:         python preprocess_landsat.py root_dir
#                   Ex.:
#                   python preprocess_landsat.py 'E:/'
#
# Author:
# Elias Frey, RSGB/Unibe
# Date: 02.10.2023

""" Preprocess Landsat 5 TM, 7 ETM+, and 8 OLI data from netCDF """

import os
import glob
import sys

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray
from datetime import datetime
import rasterio
from rasterio.transform import from_origin
import affine
from affine import Affine

# ______________________________________________________________________________________________________________________
# Define parameters

# Calculation
calculate_indices = False
calculate_indices_composites = True
calculate_rgb_composites = False

# Minimum threshold (NDVI) for forest
threshold_value = 0.4

# ______________________________________________________________________________________________________________________


def create_dir_structure(root_directory):
    """
    Create directory structure for preprocessed Landsat scenes
    :param root_directory: root directory
    :return: paths to directories 'landsat_scenes_raw' & 'landsat_scenes_processed'
    """
    # Directory of raw Landsat scenes to process
    landsat_scenes_dir = os.path.join(root_dir, 'landsat_scenes_processed')
    if not os.path.exists(landsat_scenes_dir):
        raise ValueError(f'Directory "landsat_scenes_raw" not found. Check script description to adapt required '
                         f'directory structure.')
    # Directory for processed Landsat scenes to create
    landsat_preprocessed_dir = os.path.join(root_dir, 'landsat_scenes_calculated')
    if not os.path.exists(landsat_preprocessed_dir):
        os.makedirs(landsat_preprocessed_dir)

    for dirpath, dirnames, _ in os.walk(landsat_scenes_dir):
        relative_path = os.path.relpath(dirpath, landsat_scenes_dir)
        new_dir_path = os.path.join(root_directory, landsat_preprocessed_dir, relative_path)
        # Copy directory structure of landsat_scenes_raw
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path, exist_ok=True)

    return landsat_scenes_dir, landsat_preprocessed_dir


def load_dataset(dataset_path, mask_coords=False, lat_coords=None, lon_coords=None):
    """
    Load Landsat netCDF scenes with xarray
    :param lon_coords:
    :param lat_coords:
    :param mask_coords:
    :param dataset_path: path to scene
    :return: xarray dataset
    """
    if mask_coords:
        dataset = xr.open_dataset(dataset_path, decode_coords='all')
        dataset = dataset.sel(latitude=lat_coords, longitude=lon_coords)
    else:
        dataset = xr.open_dataset(dataset_path, decode_coords='all')
    if dataset.time.size == 0:
        print(f'Warning: Dataset {os.path.basename(dataset_path)} is empty. Check file or input masking coordinates')

    return dataset


def save_netcdf(dataset, save_path):
    dataset = dataset.rio.write_crs("epsg:4326", inplace=True)
    dataset.to_netcdf(save_path, format='NETCDF4', mode='w',
                      encoding={var: {'zlib': True, 'complevel': 4} for var in dataset.data_vars})


def assign_crs(dataset):
    xr_dataset = dataset.rio.write_crs("epsg:4326", inplace=True)
    return xr_dataset


def save_geotiff(source_file, save_file, file_path):
    # Extract CRS and GeoTransform info from netCDF source file
    crs = source_file.crs.crs_wkt
    x_ul, we_pxw, _, y_ul, _, ns_pxh = source_file.crs.GeoTransform
    transform = from_origin(x_ul, y_ul, we_pxw, ns_pxh * -1)

    # Save the image as a GeoTIFF
    with rasterio.open(file_path, 'w', driver='GTiff',
                       height=save_file.shape[0], width=save_file.shape[1],
                       count=1, dtype=str(save_file.dtype), crs=crs,
                       transform=transform) as dst:
        dst.write(save_file, 1)


def save_rgb(rgb_ds, save_path):

    rgb_ds = assign_crs(rgb_ds)
    # Convert nan values to match RGB uint8n format (0-255)
    blue_rgb = rgb_ds.blue_median.fillna(0).squeeze(dim='time')
    green_rgb = rgb_ds.green_median.fillna(0).squeeze(dim='time')
    red_rgb = rgb_ds.red_median.fillna(0).squeeze(dim='time')

    # Stack bands along dim
    rgb_image = np.stack([red_rgb, green_rgb, blue_rgb], axis=-1)
    # Extract CRS and GeoTransform info from netCDF source file

    geotransform = rgb_ds.crs.attrs['GeoTransform']
    affine_transform = affine.Affine(geotransform[1], geotransform[2], geotransform[0],
                                     geotransform[4], geotransform[5], geotransform[3])


    # Reorder dimensions to match rasterio convention (bands, height, width)
    rgb_image = rgb_image.transpose(2, 0, 1)
    # Save the image as a GeoTIFF
    with rasterio.open(save_path, 'w', driver='GTiff',
                       height=rgb_image.shape[1], width=rgb_image.shape[2],
                       count=rgb_image.shape[0], dtype=str(rgb_image.dtype),
                       transform=affine_transform, crs=rgb_ds.crs.crs_wkt) as dst:
        dst.write(rgb_image)


def landsat_harmonization(dataset):
    """
    Function to harmonize TM, ETM+ and OLI sensor band values as of Roy et al. (2016)
    :param dataset: Landsat 5 TM or Landsat 7 ETM+ dataset
    :return: Landsat 5 TM or Landsat 7 ETM+ to OLI band values harmonized dataset
    """
    #[0.0003, 0.0088, 0.0061, 0.0412, 0.0254, 0.0172]
    #[0.8474, 0.8483, 0.9047, 0.8462, 0.8937, 0.9071]

    dataset['blue'] = (dataset['blue'] * 0.8474) + 0.0003
    dataset['green'] = (dataset['green'] * 0.8483) + 0.0088
    dataset['red'] = (dataset['red'] * 0.9047) + 0.0061
    dataset['nir'] = (dataset['nir'] * 0.8462) + 0.0412
    dataset['swir1'] = (dataset['swir1'] * 0.8937) + 0.0254
    dataset['swir2'] = (dataset['swir2'] * 0.9071) + 0.0172

    return dataset


def get_mask_coord(coord_lat, coord_lon):
    """
    Slices latitude & longitude coordinates to mask dataset --> Check min, max order!
    :param coord_lat: list of max, min latitude coordinates
    :param coord_lon: list of min, max longitude coordinates
    :return: slice range of coordinates
    """
    lat_range = slice(coord_lat[0], coord_lat[1])
    lon_range = slice(coord_lon[0], coord_lon[1])
    return lat_range, lon_range


def calculate_ndvi(dataset):
    """
    Normalized Difference Vegetation Index
    :return: ndvi
    """
    ndvi = (dataset['nir'] - dataset['red']) / (dataset['nir'] + dataset['red'])

    return ndvi


def calculate_msavi(dataset):

    msavi = (2 * dataset['nir'] + 1 - np.sqrt((2 * dataset['nir'] + 1) ** 2 - 8 * (dataset['nir'] - dataset['red']))) / 2

    return msavi


def calculate_evi(dataset):
    """
    Enhanced Vegetation Index
    :param dataset:
    :return: evi
    """
    # EVI constants as in Huete et al. (2002)
    g, c1, c2, l = [2.5, 6, 7.5, 1]

    evi = g * (dataset['nir'] - dataset['red']) / (dataset['nir'] + c1 * dataset['red'] - c2 * dataset['blue'] + l)
    evi = np.clip(evi, -1, 1)
    return evi


def calculate_ndmi(dataset):
    ndmi = (dataset['nir'] - dataset['swir1']) / (dataset['nir'] + dataset['swir1'])
    return ndmi


def calculate_lai(dataset):
    """
    Leaf Area Index
    :param dataset:
    :return: lai
    """
    # LAI constants as in Boegh et al. (2002)
    lai = 3.618 * dataset['evi'] - 0.118
    return lai


def calculate_gci(dataset):
    """
    Green Chlorophyll Vegetation Index from Gitelson et al. (2003)
    :param dataset:
    :return:
    """
    gci = (dataset['nir'] / dataset['green']) - 1
    return gci


def calculate_tasseled_cap(dataset):
    # As of Kauth & Thomas (1976)
    brightness = 0.3037 * dataset['blue'] + 0.2793 * dataset['green'] + 0.4743 * dataset['red'] + 0.5585 * \
                 dataset['nir'] + 0.5082 * dataset['swir1'] + 0.1863 * dataset['swir2']
    greenness = - 0.2848 * dataset['blue'] - 0.2435 * dataset['green'] - 0.5436 * dataset['red'] + 0.7243 * \
                 dataset['nir'] + 0.0840 * dataset['swir1'] - 0.1800 * dataset['swir2']
    wetness = 0.1509 * dataset['blue'] + 0.1973 * dataset['green'] + 0.3279 * dataset['red'] + 0.3406 * \
                 dataset['nir'] - 0.7112 * dataset['swir1'] - 0.4572 * dataset['swir2']

    return brightness, greenness, wetness


def get_indices(scenes_calculated, rgb_composite=True):
    """
    Load template dataset and list its indices
    :param scenes_calculated: List of scenes (paths)
    :param rgb_composite: Option to include blue, green, red to calculate mean RGB image
    :return: Template dataset containing only calculated indices; list of indices
    """
    # Load template dataset to read available indices
    indices_tmpl = load_dataset(dataset_path=scenes_calculated[0], mask_coords=False)
    # Keep blue, green, red band
    if rgb_composite:
        indices_tmpl = indices_tmpl.drop_vars(['nir', 'swir1', 'swir2'])
    # Drop all non-VI bands
    else:
        indices_tmpl = indices_tmpl.drop_vars(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    # List all variables
    indices = [v for v in indices_tmpl.data_vars]
    template_dataset = indices_tmpl.drop_vars(indices)

    return template_dataset, indices


def calculate_mean(template_dataset, indices_variables, curr_index):
    """
    Calculates mean indices of all input datasets across time dimension
    :param template_dataset: Dataset sering as template
    :param indices_variables: List of all mean layers
    :param curr_index: Index selected for calculation
    :return: Dataset with calculated mean index in new assigned variable
    """
    # Create mean variable
    mean_variable = xr.concat(indices_variables, dim='time').mean(dim='time')  # dimension to distinguish mean values
    # Create a new dataset with the mean variable
    new_var_name = f'{curr_index}_mean'
    mean_dataset = xr.Dataset({new_var_name: mean_variable})
    mean_dataset = mean_dataset.expand_dims(time=template_dataset.time)
    template_ds[new_var_name] = mean_dataset[new_var_name]

    return template_ds


def calculate_stat(dataset, indices_variables, curr_index, operation):
    """
    Calculates mean indices of all input datasets across time dimension
    :param dataset: Dataset sering as template
    :param indices_variables: List of all mean layers
    :param curr_index: Index selected for calculation
    :param operation: Statistical operation (mean, median, max)
    :return: Dataset with calculated mean index in new assigned variable
    """
    # Create mean variable
    if operation == 'mean':
        stat_variable = xr.concat(indices_variables, dim='time').mean(dim='time')
    elif operation == 'median':
        stat_variable = xr.concat(indices_variables, dim='time').median(dim='time')
    elif operation == 'max':
        stat_variable = xr.concat(indices_variables, dim='time').max(dim='time')
    else:
        raise ValueError(f'Operation <{operation}> not in [mean, median, max].')
    # Create a new dataset with the mean variable
    new_var_name = f'{curr_index}_{operation}'
    stat_dataset = xr.Dataset({new_var_name: stat_variable})
    stat_dataset = stat_dataset.expand_dims(time=dataset.time)
    dataset[new_var_name] = stat_dataset[new_var_name]

    return dataset


def calculate_max(template_dataset, indices_variables, curr_index):
    """
    Calculates mean indices of all input datasets across time dimension
    :param template_dataset: Dataset sering as template
    :param indices_variables: List of all mean layers
    :param curr_index: Index selected for calculation
    :return: Dataset with calculated mean index in new assigned variable
    """
    # Create mean variable
    max_variable = xr.concat(indices_variables, dim='time').max(dim='time')  # dimension to distinguish max values
    # Create a new dataset with the mean variable
    new_var_name = f'{curr_index}_max'
    mean_dataset = xr.Dataset({new_var_name: max_variable})
    mean_dataset = mean_dataset.expand_dims(time=template_dataset.time)
    template_ds[new_var_name] = mean_dataset[new_var_name]

    return template_ds


def create_dataset(template_dataset, time=None):
    """
    Create new dataset based on the coordinates of a template dataset
    :param template_dataset: Template xarray dataset
    :param time: time value in string format (E.g. 2011-01-01)
    :return: New xarray dataset
    """
    if time:
        # datetime_str = f"2011-01-01 00:00:00"
        time_datetime = pd.to_datetime(time)
        time_dim = [pd.to_datetime(time_datetime)]
    else:
        time_dim = template_dataset['time']

    new_dataset = xr.Dataset(
        coords={
            "time": time_dim,
            "latitude": template_dataset["latitude"],
            "longitude": template_dataset["longitude"],
            'crs': template_dataset['crs']
        }
    )

    new_dataset = new_dataset[["time", "latitude", "longitude", 'crs']]
    for var_name in template_dataset.data_vars:
        new_dataset[var_name] = (["time", "latitude", "longitude"], template_dataset[var_name].values)

    return new_dataset


def update_attrs(dataset, attributes):
    """
    Add or update attributes of xarray dataset
    :param dataset: Xarray dataset
    :param attributes: Dictionary of attributes
    :return: Xarray dataset with assigned attributes
    """
    for key, value in attributes.items():
        dataset.attrs[key] = value

    return dataset


if __name__ == '__main__':
    print(f'Starting preprocessing_landsat.py')
    # Check command line arguments input
    try:
        # root_dir = os.path.normpath(sys.argv[1])
        root_dir = r'E:/'
    except IndexError:
        raise ValueError('ERROR: root dir as command line argument. Synopsis:>>> python preprocess_landsat.py root_dir')

    # Create directory structure
    landsat_raw_scenes_dir, landsat_processed_scenes_dir = create_dir_structure(root_directory=root_dir)
    print(f'Set up directory structure successfully')
    """
    Loop through Landsat directory
    """
    # Landsat dirs
    for landsat_dir in os.listdir(landsat_raw_scenes_dir):
        if landsat_dir == 'landsat7':
            # Year dirs
            year_tot = len(os.listdir(os.path.join(landsat_raw_scenes_dir, landsat_dir)))
            year_counter = 0
            for year_dir in os.listdir(os.path.join(landsat_raw_scenes_dir, landsat_dir)):
                year_counter += 1
                scenes_path = glob.glob(os.path.join(landsat_raw_scenes_dir, landsat_dir, year_dir, "*.nc"))
                landsat_version = int(os.path.basename(scenes_path[0]).split('_')[0][2:3])
                print(f'Processing Landsat{landsat_version} data of year {year_dir} ({year_counter}/{year_tot})...')
                # Single/pair scenes dictionary

                if calculate_indices:
                    ind_count = 0
                    for scene in scenes_path:
                        ind_count += 1
                        scene_name = os.path.basename(scene)
                        #lat_coord, lon_coord = get_mask_coord([47.7038, 46.0], [6.8, 8.5185])
                        landsat_scene = load_dataset(dataset_path=scene, mask_coords=False)
                        # Apply Landsat 5 TM and 7 ETM+ harmonization to match OLI sensor values
                        #if landsat_dir == 'landsat5' or landsat_dir == 'landsat7':
                            #landsat_scene = landsat_harmonization(dataset=landsat_scene)
                        # Calculate indices
                        landsat_scene['ndvi'] = calculate_ndvi(dataset=landsat_scene)
                        landsat_scene['msavi'] = calculate_msavi(dataset=landsat_scene)
                        landsat_scene['evi'] = calculate_evi(dataset=landsat_scene)
                        landsat_scene['ndmi'] = calculate_ndmi(dataset=landsat_scene)
                        landsat_scene['lai'] = calculate_lai(dataset=landsat_scene)
                        landsat_scene['gci'] = calculate_gci(dataset=landsat_scene)
                        landsat_scene['tc_brightness'], landsat_scene['tc_greenness'], landsat_scene['tc_wetness'] = \
                            calculate_tasseled_cap(dataset=landsat_scene)

                        # Save dataset
                        landsat_indices = create_dataset(template_dataset=landsat_scene, time=None)
                        landsat_indices = update_attrs(dataset=landsat_indices, attributes={
                            'title': f'Landsat {str(landsat_dir)} calculated vegetation indices',
                            'date_created': pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S.%f'),
                            'processing_script': 'https://github.com/freyelias',
                            'references': 'master thesis',
                            'source_of_raw_data': f'Swiss Data Cube; Landsat {str(landsat_dir)} Analysis Ready Data (netCDF4)',
                        })
                        dataset_basename = scene_name.split('.')[0] + '_calc.nc'
                        dataset_save_path = os.path.join(landsat_processed_scenes_dir, landsat_dir, year_dir, dataset_basename)
                        save_netcdf(dataset=landsat_indices, save_path=dataset_save_path)
                        print(f'Scene ({ind_count}/{len(scenes_path)}) saved as netCDF')

                if calculate_indices_composites:
                    # List for mean variables
                    # List of scenes with calculated indices
                    calc_scenes = glob.glob(os.path.join(landsat_processed_scenes_dir, landsat_dir, year_dir, "*_calc.nc"))
                    template_ds, indices_list = get_indices(scenes_calculated=calc_scenes,
                                                            rgb_composite=calculate_rgb_composites)
                    inc_count = 0

                    for index in indices_list:
                        inc_count += 1
                        idx_vars = []

                        for calc_scene in calc_scenes:
                            scene = load_dataset(dataset_path=calc_scene, mask_coords=False)
                            if index == 'blue' or index == 'green' or index == 'red':
                                idx_var = scene[index]
                            else:
                                idx_var = scene[index]#.where(scene['ndvi'] >= threshold_value)
                            # Calculate the mean
                            idx_vars.append(idx_var)

                        if index == 'blue' or index == 'green' or index == 'red':
                            template_ds = calculate_stat(dataset=template_ds,
                                                         indices_variables=idx_vars,
                                                         curr_index=index,
                                                         operation='median')
                        else:
                            template_ds = calculate_stat(dataset=template_ds,
                                                         indices_variables=idx_vars,
                                                         curr_index=index,
                                                         operation='max')

                        print(f'Index {index} mean calculated ({inc_count}/{len(indices_list)})')
                    scene_name = os.path.basename(calc_scenes[0])
                    name_parts = scene_name.split('_')
                    if calculate_rgb_composites:
                        rgb_basename = '_'.join(name_parts[:6])
                        rgb_basename += f'_{year_dir}_RGB.tif'
                        rgb_save_path = os.path.join(landsat_processed_scenes_dir, landsat_dir, year_dir, rgb_basename)
                        save_rgb(rgb_ds=template_ds, save_path=rgb_save_path)
                        template_ds = template_ds.drop_vars(['blue_median', 'green_median', 'red_median'])
                    landsat_mean = create_dataset(template_dataset=template_ds, time=str(year_dir))
                    landsat_mean = update_attrs(dataset=landsat_mean, attributes={
                        'title': f'Landsat {str(landsat_dir)} extended summer max of precalculated vegetation indices',
                        'date_created': pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'processing_script': 'https://github.com/freyelias',
                        'references': 'master thesis',
                        'source_of_raw_data': f'Swiss Data Cube; Landsat {str(landsat_dir)} Analysis Ready Data (netCDF4)',
                        })
                    dataset_basename = '_'.join(name_parts[:6])
                    dataset_basename += f'_{year_dir}_max.nc'
                    dataset_save_path = os.path.join(landsat_processed_scenes_dir, landsat_dir, year_dir, dataset_basename)
                    save_netcdf(dataset=landsat_mean, save_path=dataset_save_path)

                    print(f'Scene saved as netCDF')
                print(f'Landsat{landsat_version} {year_dir} done!')
        else:
            print(f'SKIPPING {landsat_dir}, ALREADY PROCESSED -- REMOVE HARDCODED PART!!')
