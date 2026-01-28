import numpy as np

from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack, Column
from astropy.io import fits
import astropy.units as u

import matplotlib.pyplot as plt
from utils import helpers

def plot_purity(inverse_catalog, full_catalog):
    '''
    Plot purity measures

    Keyword arguments:
    inverse_catalog -- Filename of catalog with source from inverted images
    full_catalog    -- Full catalog of sources from regular images
    '''
    output_dir = os.path.dirname(inverse_catalog)
    inverse = Table.read(inverse_catalog)
    full = Table.read(full_catalog)

    inverse_groups = inverse.group_by('Pointing_id')
    full_groups = full.group_by('Pointing_id')

    full_mask = []
    for i, group in enumerate(full_groups.groups.keys):
        pointing_full = full[full['Pointing_id'] == group['Pointing_id']]
        pointing_inverse = inverse[inverse['Pointing_id'] == group['Pointing_id']]

        bright_idx = np.argpartition(-pointing_full['Peak_flux'], 10)[:10]
        idx = flag_artifacts(pointing_full[bright_idx], pointing_inverse)

        mask = np.ones(len(pointing_inverse), dtype=bool)
        mask[idx] = False

        full_mask.append(mask)

    full_mask = np.concatenate(full_mask)
    reduced_inverse = inverse[full_mask]

    print(f'Found {len(inverse)-len(reduced_inverse)} sources close to bright sources, of {len(inverse)} total sources')

    # Plot as a function of flux

    # Define flux bins
    flux_bins = np.logspace(-4,-1,20)
    flux_full = full['Total_flux']
    flux_inverse = inverse['Total_flux']
    flux_reduced_inverse = reduced_inverse['Total_flux']

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n_full, _, _ = ax2.hist(flux_full, bins=flux_bins, edgecolor='k', facecolor='none')
    n_inverse, _, _ = ax2.hist(flux_inverse, bins=flux_bins, edgecolor='k', facecolor='k', alpha=0.5)
    n_reduced_inverse, _ = np.histogram(flux_reduced_inverse, bins=flux_bins)

    ax1.plot((flux_bins[1:] + flux_bins[:-1]) / 2, n_inverse/n_full, color='crimson', linestyle=':')
    ax1.plot((flux_bins[1:] + flux_bins[:-1]) / 2, n_reduced_inverse/n_full, color='crimson')

    ax1.set_xscale('log')
    ax1.set_xlabel('Flux density (Jy)')
    ax1.set_ylabel('Fraction')
    ax2.set_ylabel('Counts')

    ax1.set_xlim(flux_bins[0], flux_bins[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'purity_flux.png'), dpi=300)
    plt.close()

    # Plot as a function of distance
    sep_bins = np.arange(0, 1.2, 0.1)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    n_full, _, _ = ax2.hist(full['Sep_PC'], bins=sep_bins, edgecolor='k', facecolor='none')
    n_inverse, _, _ = ax2.hist(inverse['Sep_PC'], bins=sep_bins, edgecolor='k', facecolor='k', alpha=0.5)
    n_reduced_inverse, _ = np.histogram(reduced_inverse['Sep_PC'], bins=sep_bins)

    ax1.plot((sep_bins[1:] + sep_bins[:-1]) / 2, n_inverse/n_full, color='crimson', linestyle=':')
    ax1.plot((sep_bins[1:] + sep_bins[:-1]) / 2, n_reduced_inverse/n_full, color='crimson')
    ax1.set_xlabel('$\\rho$ (degrees)')
    ax1.set_ylabel('Fraction')
    ax2.set_ylabel('Counts')

    ax1.set_xlim(sep_bins[0], sep_bins[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'purity_sep.png'), dpi=300)
    plt.close()

    # Plot as a function of field
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    inverse_ratio = []
    reduced_inverse_ratio = []
    for i, group in enumerate(full_groups.groups.keys):
        color = next(ax2._get_lines.prop_cycler)['color']

        full_counts = len(full_groups.groups[i])
        inverse_counts = np.sum(inverse['Pointing_id'] == group['Pointing_id'])
        reduced_inverse_counts = np.sum(reduced_inverse['Pointing_id'] == group['Pointing_id'])

        ax2.bar(i, inverse_counts, edgecolor=color, facecolor=color, alpha=0.5)
        ax2.bar(i, full_counts, edgecolor=color, facecolor='none')

        inverse_ratio.append(inverse_counts/full_counts)
        reduced_inverse_ratio.append(reduced_inverse_counts/full_counts)

    ax1.plot(range(len(full_groups.groups.keys)), inverse_ratio, color='crimson', linestyle=':')
    ax1.plot(range(len(full_groups.groups.keys)), reduced_inverse_ratio, color='crimson')

    ax1.set_xticks(range(len(full_groups.groups.keys)))
    ax1.set_xticklabels([key.replace('PT-','') for key in full_groups.groups.keys['Pointing_id']], fontsize=8, rotation=45, ha='right')

    ax1.set_ylabel('Fraction')
    ax2.set_ylabel('Counts')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir,'purity_im.png'), dpi=300)
    plt.close()

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

    catalogs = args.catalogs
    full_catalog = args.full_catalog

    full_inverse = Table()
    for catalog_file in catalogs:
        try:
            inverse_catalog = Table.read(out_catalog)
            inverse_catalog = transform_cat(inverse_catalog)
            full_inverse = vstack([full_inverse,inverse_catalog])
        except FileNotFoundError:
            print('File not found, most likely no sources have been found in the image')

    if full_catalog is not None:
        plot_purity(output_cat_file, full_catalog)

def new_argument_parser():

    parser = ArgumentParser()

    parser.add_argument("catalogs",
                        help="Input catalogues of inverse images.")
    parser.add_argument("--full_catalog", default=None,
                        help="Full catalog for comparison with inverse catalog.")
    return parser

if __name__ == '__main__':
    main()