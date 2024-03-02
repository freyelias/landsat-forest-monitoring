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


def create_dir_structure(root_directory):
    """
    Create directory structure for preprocessed Landsat scenes
    :param root_directory: root directory
    :return: paths to directories 'landsat_scenes_raw' & 'landsat_scenes_processed'
    """
    # Directory of raw Landsat scenes to process
    landsat_scenes_dir = os.path.join(root_dir, 'landsat_scenes_raw')
    if not os.path.exists(landsat_scenes_dir):
        raise ValueError(f'Directory "landsat_scenes_raw" not found. Check script description to adapt required '
                         f'directory structure.')
    # Directory for processed Landsat scenes to create
    landsat_preprocessed_dir = os.path.join(root_dir, 'landsat_scenes_processed')
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


def get_bitmasks(bitmask_type, ls_version):
    """
    Bitmasks for Landsat surface reflectance and quality assessment bands
    :param ls_version: Landsat version number
    :param bitmask_type: type of bitmask
    :return: bitmask value to encrypt band mask
    """
    if ls_version == 5 or ls_version == 7:
        bitmasks = {
            'pixel_qa_fill': 1 << 0,  # Bit 0: fill
            'pixel_qa_cloud': 1 << 5,  # Bit 5: clouds
            'pixel_qa_cloud_shadow': 1 << 3,  # Bit 3: cloud shadow
            'pixel_qa_snow': 1 << 4,  # Bit 4: snow
            'radsat_qa_fill': 1 << 0,  # Bit 0: fill
            'radsat_qa_blue': 1 << 1,  # Bit 1: blue saturated
            'radsat_qa_green': 1 << 2,  # Bit 2: green saturated
            'radsat_qa_red': 1 << 3,  # Bit 3: red saturated
            'radsat_qa_nir': 1 << 4,  # Bit 4: nir saturated
            'radsat_qa_swir1': 1 << 5,  # Bit 5: swir1 saturated
            'radsat_qa_swir2': 1 << 7,  # Bit 7: swir2 saturated
        }
    elif ls_version == 8:
        bitmasks = {
            'pixel_qa_fill': 1 << 0,  # Bit 0: fill
            'pixel_qa_cloud': 1 << 5,  # Bit 5: clouds
            'pixel_qa_cloud_shadow': 1 << 3,  # Bit 3: cloud shadow
            'pixel_qa_snow': 1 << 4,  # Bit 4: snow
            'radsat_qa_fill': 1 << 0,  # Bit 0: fill
            'radsat_qa_blue': 1 << 2,  # Bit 1: blue saturated
            'radsat_qa_green': 1 << 3,  # Bit 2: green saturated
            'radsat_qa_red': 1 << 4,  # Bit 3: red saturated
            'radsat_qa_nir': 1 << 5,  # Bit 4: nir saturated
            'radsat_qa_swir1': 1 << 6,  # Bit 5: swir1 saturated
            'radsat_qa_swir2': 1 << 7,  # Bit 7: swir2 saturated
        }
    else:
        raise ValueError(f'Wrong landsat_version: {ls_version}')

    bitmask_value = bitmasks[bitmask_type]
    return bitmask_value


def get_qa_bands(dataset, ls_version):
    """
    Load valid quality assessment bands
    :param ls_version: Landsat version number
    :param dataset: xarray dataset
    :return: xarray band array
    """
    pix_qa = dataset['pixel_qa']
    sat_qa = dataset['radsat_qa']
    if ls_version == 5 or ls_version == 7:
        pix_qa_valid = pix_qa.where((pix_qa >= 1) & (pix_qa <= 255)).astype('uint16')
        sat_qa_valid = sat_qa.where((sat_qa >= 0) & (sat_qa <= 255)).astype('uint8')
    elif ls_version == 8:
        pix_qa_valid = pix_qa.where((pix_qa >= 1) & (pix_qa <= 2047)).astype('uint16')
        sat_qa_valid = sat_qa.where((sat_qa >= 0) & (sat_qa <= 4095)).astype('uint16')
    else:
        raise ValueError(f'Wrong landsat_version: {ls_version}')

    return pix_qa_valid, sat_qa_valid


def get_qa_masks(dataset, ls_version):
    """
    Get mask layer combined from each QA band
    :param dataset: Landsat scene xarray dataset
    :param ls_version: Landsat version number
    :return: combined mask layer for given dataset
    """
    pix_qa, rad_qa = get_qa_bands(dataset=dataset, ls_version=ls_version)

    # Apply bitwise operations to extract the masks
    pix_fill = pix_qa & get_bitmasks(bitmask_type='pixel_qa_fill', ls_version=ls_version)
    cloud_masked = pix_qa & get_bitmasks(bitmask_type='pixel_qa_cloud', ls_version=ls_version)
    cloud_shadow_masked = pix_qa & get_bitmasks(bitmask_type='pixel_qa_cloud_shadow', ls_version=ls_version)
    snow_masked = pix_qa & get_bitmasks(bitmask_type='pixel_qa_snow', ls_version=ls_version)  # Snow disabled

    rs_fill = rad_qa & get_bitmasks(bitmask_type='radsat_qa_fill', ls_version=ls_version)
    blue_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_blue', ls_version=ls_version)
    green_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_green', ls_version=ls_version)
    red_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_red', ls_version=ls_version)
    nir_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_nir', ls_version=ls_version)
    swir1_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_swir1', ls_version=ls_version)
    swir2_sat = rad_qa & get_bitmasks(bitmask_type='radsat_qa_swir2', ls_version=ls_version)

    combined_mask = (pix_fill > 0) | (cloud_masked > 0) | (cloud_shadow_masked > 0) | (snow_masked > 0) | \
                    (rs_fill > 0) | (blue_sat > 0) | (green_sat > 0) | (red_sat > 0) | (nir_sat > 0) | \
                    (swir1_sat > 0) | (swir2_sat > 0)

    return combined_mask


def mask_band(sr_band, combined_mask):
    """
    Filter out fill-value, valid data range, scale
    :param sr_band: surface reflectance band
    :param combined_mask: extracted qa masks
    :return: filtered, masked and scaled xr DataArray band
    """
    sr_band_valid = sr_band.where(sr_band != -9999, np.nan)
    sr_band_scaled = sr_band_valid.where((sr_band_valid >= 0) & (sr_band_valid <= 10000)) * 0.0001
    sr_band_masked = sr_band_scaled.where(~combined_mask)

    return sr_band_masked


def mask_dataset(dataset, combined_mask, ls_version):
    """
    Add masked bands to dataset
    :param ls_version: Landsat version number
    :param dataset: source dataset (netCDF)
    :param combined_mask: qa mask
    :return: dataset with masked bands
    """
    blue_band = mask_band(dataset['blue'], combined_mask)
    green_band = mask_band(dataset['green'], combined_mask)
    red_band = mask_band(dataset['red'], combined_mask)
    nir_band = mask_band(dataset['nir'], combined_mask)
    swir1_band = mask_band(dataset['swir1'], combined_mask)
    swir2_band = mask_band(dataset['swir2'], combined_mask)

    dataset['blue'] = blue_band
    dataset['green'] = green_band
    dataset['red'] = red_band
    dataset['nir'] = nir_band
    dataset['swir1'] = swir1_band
    dataset['swir2'] = swir2_band

    if ls_version == 5 or ls_version == 7:
        dataset = dataset.drop_vars(['pixel_qa', 'radsat_qa', 'cloud_qa', 'dataset'])
    elif ls_version == 8:
        dataset = dataset.drop_vars(['coastal_aerosol', 'pixel_qa', 'radsat_qa', 'aerosol_qa', 'dataset'])
    else:
        raise ValueError(f'Wrong landsat_version: {ls_version}')

    return dataset


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


def mosaic_scenes(scene1, scene2):
    """
    Merge landsat scenes of same datetime and average pixel values of overlapping areas
    :param scene2:
    :param scene1:
    :return: merged mosaic dataset with same shape
    """
    bands = [i for i in scene1.data_vars]
    scene_mosaic = scene1.copy(deep=True)

    for band in bands:
        scene1_band = scene1[band]
        scene2_band = scene2[band]
        scene1_nan = np.isnan(scene1_band.values)
        scene2_nan = np.isnan(scene2_band.values)

        scenes_combined = np.where(
            scene1_nan, scene2_band.values, np.where(
                scene2_nan, scene1_band.values, (scene1_band.values + scene2_band.values) / 2))

        scene_mosaic[band].values = scenes_combined

    return scene_mosaic


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
        # Year dirs
        year_tot = len(os.listdir(os.path.join(landsat_raw_scenes_dir, landsat_dir)))
        year_counter = 0
        for year_dir in os.listdir(os.path.join(landsat_raw_scenes_dir, landsat_dir)):
            year_counter += 1
            scenes_path = glob.glob(os.path.join(landsat_raw_scenes_dir, landsat_dir, year_dir, "*.nc"))
            print(scenes_path)
            landsat_version = int(os.path.basename(scenes_path[0]).split('_')[0][2:3])
            print(f'Processing Landsat{landsat_version} data of year {year_dir} ({year_counter}/{year_tot})...')
            # Single/pair scenes dictionary
            datetime_dict = {}

            for scene in scenes_path:
                scene_name = os.path.basename(scene)
                # Extract datetime info
                dataset_ts_date = scene_name.split('.')[0].split('_')[6][:-12]
                dataset_datetime = datetime.strptime(dataset_ts_date, '%Y%m%d')
                # List preprocessed scenes with same datetime
                scene_ex = glob.glob(os.path.join(
                    landsat_processed_scenes_dir, landsat_dir, year_dir, f"*{dataset_ts_date}*.nc"))
                # Check if scene already preprocessed
                if scene_ex:
                    pass
                else:
                    # Select seasonal range (M, J, J, A, S)
                    if 5 <= dataset_datetime.month <= 9:
                        # Create datetime dict to distinguish single and pair scenes
                        if dataset_datetime in datetime_dict:
                            datetime_dict[dataset_datetime].append(scene)
                        else:
                            datetime_dict[dataset_datetime] = [scene]

            # Merge follow-up scenes or proceed when single scene
            for scene_date, scenes in datetime_dict.items():
                # Follow-up scenes to merge first
                if len(scenes) == 2:
                    lat_coord, lon_coord = get_mask_coord([47.7038, 46.0], [6.8, 8.5185])
                    scene_1 = load_dataset(dataset_path=scenes[0], mask_coords=True,
                                           lat_coords=lat_coord, lon_coords=lon_coord)
                    sc1_mask = get_qa_masks(dataset=scene_1, ls_version=landsat_version)
                    sc1_masked = mask_dataset(dataset=scene_1, combined_mask=sc1_mask, ls_version=landsat_version)
                    scene_2 = load_dataset(dataset_path=scenes[1], mask_coords=True,
                                           lat_coords=lat_coord, lon_coords=lon_coord)
                    sc2_mask = get_qa_masks(dataset=scene_2, ls_version=landsat_version)
                    sc2_masked = mask_dataset(dataset=scene_2, combined_mask=sc2_mask, ls_version=landsat_version)
                    landsat_dataset = mosaic_scenes(scene1=sc1_masked, scene2=sc2_masked)
                    # Save memory
                    scene_1.close()
                    scene_2.close()
                    print(f'Scenes merged to mosaic and masked')

                # No follow-up scenes to merge
                elif len(scenes) == 1:
                    lat_coord, lon_coord = get_mask_coord([47.7038, 46.0], [6.8, 8.5185])
                    scene_1 = load_dataset(dataset_path=scenes[0], mask_coords=True,
                                           lat_coords=lat_coord, lon_coords=lon_coord)
                    sc1_mask = get_qa_masks(dataset=scene_1, ls_version=landsat_version)
                    landsat_dataset = mask_dataset(dataset=scene_1, combined_mask=sc1_mask, ls_version=landsat_version)
                    print(f'Scene masked')
                else:
                    raise ValueError(f'Error in number of scenes to merge: {len(scenes)}')

                # Save dataset
                new_landsat_dataset = create_dataset(template_dataset=landsat_dataset, time=None)
                new_landsat_dataset = update_attrs(dataset=new_landsat_dataset, attributes={
                    'title': f'Landsat {str(landsat_dir)} composite mosaic over Canton of Bern',
                    'date_created': pd.Timestamp.today().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    'processing_script': 'https://github.com/freyelias',
                    'references': 'master thesis',
                    'source_of_raw_data': f'Swiss Data Cube; Landsat {str(landsat_dir)} Analysis Ready Data (netCDF4)',
                    })
                dataset_basename = os.path.basename(scenes[0]).split('.')[0] + '_proc.nc'
                dataset_save_path = os.path.join(landsat_processed_scenes_dir, landsat_dir, year_dir, dataset_basename)
                save_netcdf(dataset=new_landsat_dataset, save_path=dataset_save_path)
                print(f'Scene saved as netCDF')
            print(f'Landsat{landsat_version} {year_dir} done!')



