import glob
import json
import ast
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, Column
from astropy.io import fits
import astropy.units as u

from argparse import ArgumentParser
from pathlib import Path

import bdsf

plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

plt.rcParams.update({'font.size': 14})

def invert_image(infile, outfile):
    image = fits.open(infile)
    image[0].data *= -1
    image.writeto(outfile, overwrite=True)

def run_bdsf(image, rms_map, mean_map):
    '''
    Run PyBDSF on an image

    Keyword arguments:
    image      -- Name of image
    rms_map    -- Name of rms image
    mean_map   -- Name of mean image
    '''
    imname = os.path.basename(image).rsplit('.',1)[0]
    outcatalog = imname+'_bdsfcat.fits'

    path = Path(__file__).parent / 'parsets/bdsf_args.json'
    with open(path) as f:
        args_dict = json.load(f)

    # Make sure tuples are correctly parsed
    if 'rms_box' in args_dict['process_image']:
        rms_box_opt = args_dict['process_image']['rms_box']
        if rms_box_opt is not None:
            args_dict['process_image']['rms_box'] = ast.literal_eval(rms_box_opt)

    img = bdsf.process_image(image, **args_dict['process_image'],
                             rmsmean_map_filename=[mean_map,rms_map])
    img.write_catalog(outfile=outcatalog, format='fits', 
                      **args_dict['write_catalog'])

    return outcatalog

def transform_cat(catalog):
    '''
    Add distances from pointing center for analysis
    '''
    header = dict([x.split(' = ') for x in catalog.meta['comments'][5:]])

    path = Path(__file__).parent / 'parsets/bdsf_args.json'
    with open(path) as f:
        args_dict = json.load(f)

    pointing_center = SkyCoord(float(header['CRVAL1'])*u.degree,
                               float(header['CRVAL2'])*u.degree)
    pointing_name = ['PT-'+header['OBJECT'].replace("'","")] * len(catalog)

    source_coord = SkyCoord([source['RA'] for source in catalog],
                            [source['DEC'] for source in catalog],
                            unit=(u.deg,u.deg))

    sep = pointing_center.separation(source_coord)

    # Add columns at appropriate indices
    col_a = Column(pointing_name, name='Pointing_id')
    col_b = Column(sep, name='Sep_PC')
    catalog.add_columns([col_a, col_b],
                         indexes=[0,6])

    return catalog


def main():

    parser = new_argument_parser()
    args = parser.parse_args()

    input_image = args.image
    clean_up = args.clean_up

    im = input_image.rsplit('.',1)[0]
    # Invert image
    im_in = input_image
    im_out = im+'_inverse.fits'
    invert_image(im_in, im_out)

    # Invert mean image for pybdsf
    mean_in = im+'_mean.fits'
    mean_out = im+'_inverse_mean.fits'
    invert_image(mean_in, mean_out)

    out_catalog = run_bdsf(im_out, im+'_rms.fits',
                           im+'_inverse_mean.fits')

    try:
        inverse_catalog = Table.read(out_catalog)
        inverse_catalog = transform_cat(inverse_catalog)
    except FileNotFoundError:
        print('File not found, most likely no sources have been found in the image')

    # Clean up
    if clean_up:
        os.system(f'rm -f {out_catalog}')
        os.system(f'rm -f {im_out}.pybdsf.log')
    os.system(f'rm -f {im_out}')
    os.system(f'rm -f {mean_out}')

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("image",
                        help="Input image.")
    parser.add_argument("--clean_up", action='store_true',
                        help="""Clean up output for individual images 
                                leaving only the combined catalog""")
    return parser

if __name__ == '__main__':
    main()