import os
import glob
import json
import pickle

import numpy as np
from scipy.interpolate import interp1d

from astropy.io import fits
from astropy import units as u
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord

def pickle_to_file(data, fname):
    fh = open(fname, 'wb')
    pickle.dump(data, fh)
    fh.close()

def pickle_from_file(fname):
    fh = open(fname, 'rb')
    data = pickle.load(fh)
    fh.close()
    return data

def get_rms_coverage(rms_image):
    image = fits.open(rms_image)[0]
    rms_data = image.data.flatten()

    rms_range = np.logspace(np.log10(np.nanmin(rms_data)), np.log10(np.nanmax(rms_data)), 100)
    coverage = [np.sum([rms_data < rms])/np.count_nonzero(~np.isnan(rms_data)) for rms in rms_range]

    # Define a spline and interpolate the values
    data_spline = interp1d(rms_range, coverage, bounds_error=False, fill_value=(0,1))
    return data_spline, rms_range, coverage

def get_ext_data(sf_dirs, rms_correction=False):
    '''
    Get completeness data
    '''
    individual_completeness = dict()

    sigma_lim = np.array([0,2.5])
    flux_bins = np.logspace(0,2.5,50)
    size_bins = np.logspace(0,1.3,50)

    all_full_counts = np.zeros((49,49))
    all_match_counts = np.zeros((49,49))
    for sd in sf_dirs:
        im_id = os.path.basename(sd).split('_')[0]

        catalog_files = glob.glob(os.path.join(sd,'completeness',f'*_sim_*_ext_catalog.fits'))
        catalogs = []
        for cat_file in catalog_files:
            catalog = Table.read(cat_file)
            catalogs.append(catalog)
        full_cat = vstack(catalogs)

        # Determine distance from center
        header = dict([x.split(' = ') for x in catalog.meta['comments'][41:]])
        pointing_center = SkyCoord(header['CRVAL1'],header['CRVAL2'], unit='deg')
        all_sources = SkyCoord(full_cat['RA_1'], full_cat['DEC_1'], unit='deg')
        sep = pointing_center.separation(all_sources)

        # Get image beam
        bmaj = float(header['BMAJ'])*3600
        bmin = float(header['BMIN'])*3600

        # Select only sources at certain distance
        if rms_correction:
            filename = os.path.join(sd, im_id+'_bdsfcat_rms_radial.json')
            with open(filename, 'r') as f:
                radialprofile = json.load(f)

            func = interp1d(radialprofile['dist'], radialprofile['rms']/np.nanmin(radialprofile['rms']), fill_value=1)
            flux_correction = func(sep.degree)
        else:
            full_cat = full_cat[np.logical_and(sep > 0.1 * u.degree, sep < 0.4 * u.degree)]
            flux_correction = 1

        full_flux = (10**full_cat['i_1285'])/flux_correction
        coverage_func, rms_range, coverage = get_rms_coverage(os.path.join(sd,im_id+'_rms.fits'))
        sigma_20 = rms_range[np.argmin(np.abs(np.array(coverage)-0.2))]
        full_flux /= sigma_20

        source_sizes = (full_cat['major_axis']*full_cat['minor_axis']+bmaj*bmin)/(bmaj*bmin)
        full_counts , _, _ = np.histogram2d(full_flux, source_sizes, bins=(flux_bins,size_bins))
        matches = full_cat['idx'] < 5000
        match_counts, _, _ = np.histogram2d(full_flux[matches], source_sizes[matches], bins=(flux_bins,size_bins))

        all_full_counts += full_counts
        all_match_counts += match_counts

        individual_completeness[im_id] = match_counts/full_counts

    all_full_counts[all_full_counts == 0] = 1
    full_completeness = all_match_counts/all_full_counts

    return full_completeness, individual_completeness, flux_bins, size_bins

def get_point_data(sf_dirs, rms_correction=False):
    '''
    Get data from catalogs
    '''
    individual_completeness = dict()

    sigma_lim = np.array([0,2.5])
    flux_bins = np.logspace(0,2.5,100)

    all_full_counts = np.zeros(99)
    all_match_counts = np.zeros(99)
    for sd in sf_dirs:
        im_id = os.path.basename(sd).split('_')[0]

        catalog_files = glob.glob(os.path.join(sd,'completeness',f'*_sim_*_point_catalog.fits'))
        catalogs = []
        for cat_file in catalog_files:
            catalog = Table.read(cat_file)
            catalogs.append(catalog)
        full_cat = vstack(catalogs)

        # Determine distance from center
        header = dict([x.split(' = ') for x in catalog.meta['comments'][41:]])
        pointing_center = SkyCoord(header['CRVAL1'],header['CRVAL2'], unit='deg')
        all_sources = SkyCoord(full_cat['RA_1'], full_cat['DEC_1'], unit='deg')
        sep = pointing_center.separation(all_sources)

        # Select only sources at certain distance
        if rms_correction:
            filename = os.path.join(sd, im_id+'_bdsfcat_rms_radial.json')
            with open(filename, 'r') as f:
                radialprofile = json.load(f)

            func = interp1d(radialprofile['dist'], radialprofile['rms']/np.nanmin(radialprofile['rms']), fill_value=1)
            flux_correction = func(sep.degree)
        else:
            full_cat = full_cat[np.logical_and(sep > 0.1 * u.degree, sep < 0.4 * u.degree)]
            flux_correction = 1

        full_flux = (10**full_cat['i_1285'])/flux_correction
        coverage_func, rms_range, coverage = get_rms_coverage(os.path.join(sd,im_id+'_rms.fits'))
        sigma_20 = rms_range[np.argmin(np.abs(np.array(coverage)-0.2))]
        full_flux /= sigma_20

        full_counts , _ = np.histogram(full_flux, bins=flux_bins)
        matches = full_cat['idx'] < 5000
        match_counts, _ = np.histogram(full_flux[matches], bins=flux_bins)

        all_full_counts += full_counts
        all_match_counts += match_counts

        individual_completeness[im_id] = match_counts/full_counts

    full_completeness = all_match_counts/all_full_counts

    return full_completeness, individual_completeness, flux_bins