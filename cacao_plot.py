import os
import fiona
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cacao_config as CF

from osgeo import gdal
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.ticker import PercentFormatter
from descartes import PolygonPatch
from shapely.geometry import Polygon


gdal.UseExceptions()


def _get_cb_params(var):
    S1a = '#28a014'
    S1b = '#8cd746'
    S2 = '#fde725'
    S3 = '#f05005'
    N2 = '#aa0155'

    if var == 'prec':
        bounds = [0, 1200, 1400, 1600, 1800, 2000, 2500, 3500, 4400, 6000]
        colors = [N2, S3, S2, S1b, S1a, S1b, S2, S3, N2]
    elif var == 'tmax':
        bounds = [0, 28, 30, 40]
        colors = [S1a, S1b, S2]
    elif var == 'tmean':
        bounds = [0, 21, 22, 23, 25, 28, 29, 30, 40]
        colors = [N2, S3, S2, S1b, S1a, S1b, S2, N2]
    elif var == 'tmin':
        bounds = [0, 10, 13, 15, 20, 30]
        colors = [N2, S3, S2, S1b, S1a]
    else:
        raise
    return bounds, colors


def _get_bounds(ranges):
    range_lst = []
    for _range in ranges:
        for item in _range:
            if item[0] not in range_lst:
                range_lst.append(item[0])
            if item[1] not in range_lst:
                range_lst.append(item[1])
    return sorted(range_lst)


def plot(input_file, var=None, vmin=-1, title=None, label=None, linx=False, factor=1, raw=False, with_border=False, between=None):
    if not linx and var is None:
        raise ValueError(f"Var type needed for non-LINDX plot")

    ds = gdal.Open(input_file)
    if factor == 1:
        ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray())
    else:
        ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray()) * factor

    fig, ax1 = plt.subplots(1,1)

    if linx:
        if between:
            lbound, rbound = tuple(list(sorted(between)))
            ds_arr[(ds_arr <= lbound) & (ds_arr > rbound) & (ds_arr != -999)] = 0

            if lbound == 0:
                bounds = [-0.5, lbound, rbound]
            else:
                bounds = [0, lbound, rbound]

            colors = ['#d8ebb5', '#2b580c']
            cmap = LinearSegmentedColormap.from_list(
                'custom', colors, N=len(colors))
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

        else:
            # Assertive remap
            bounds = [0, 12.5, 25, 50, 75, 100]
            cmap = cm.get_cmap('RdYlGn')
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        if not raw:
            bounds, colors = _get_cb_params(var)

            cmap = LinearSegmentedColormap.from_list(
                'custom', colors, N=len(colors))
            norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        else:
            cmap = cm.get_cmap('RdYlGn')
            norm = None


    cmap.set_under(color='black')

    nrows, ncols = ds_arr.shape
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()

    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows

    extent = [x0, x1, y1, y0]

    im = ax1.imshow(ds_arr, cmap=cmap, norm=norm, vmin=vmin, extent=extent, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax1)

    fg_color = 'white'
    bg_color = 'black'

    # IMSHOW
    # ax1.axes.get_xaxis().set_visible(False)
    # ax1.axes.get_yaxis().set_visible(False)

    # set title plus title color
    ax1.set_title(title, color=fg_color)

    # set figure facecolor
    ax1.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    # for spine in im.axes.spines.values():
    #     spine.set_edgecolor(fg_color)

    # COLORBAR
    # set colorbar label plus label color
    cb.set_label(label, color=fg_color)

    # set colorbar tick color
    cb.ax.yaxis.set_tick_params(color=fg_color)

    # set colorbar edgecolor
    cb.outline.set_edgecolor(fg_color)

    # set colorbar ticklabels
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

    fig.patch.set_facecolor(bg_color)


    if with_border:
        from shapely.geometry import shape

        with fiona.open(CF.PROVINCIAL_BORDER) as shapefile:
            # for feature in shapefile:
            #     print(feature["properties"]["PROVINCE"])
            features = ((feature["geometry"], feature["properties"]["PROVINCE"]) for feature in shapefile)

            for feature, name in features:
                patch = PolygonPatch(feature, edgecolor="black", facecolor="none", linewidth=0.5, alpha=0.8)
                ax1.add_patch(patch)


    plt.tight_layout()
    plt.show()


def _get_bounds_delta(var, vmin=None, vmax=None):
    spacing = 200 if var == 'prec' else 0.5

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = [0]
    i = 0
    while True:
        i += spacing
        if i <= vmax:
            bounds.append(i)
        else:
            break
    i = 0
    while True:
        i -= spacing
        if i >= vmin:
            bounds.append(i)
        else:
            break

    return sorted(bounds)


def plot_histogram(data, var=None, title=None, label=None, vmax=None, vmin=None, vmean=None, **kwargs):

    def set_bar_labels(patches):
        # For each bar: Place a label

        for rect in patches:
            # Get X and Y placement of label from rect.
            x_value = rect.get_width()
            y_value = rect.get_y() + rect.get_height() / 2

            # Number of points between bar and label. Change to your liking.
            space = 5
            # Vertical alignment for positive values
            ha = 'left'

            # Use X value as label and format number with one decimal place
            label = x_value * 100
            if round(label, 1) == 0.0:
                continue

            label = "{:.1f}%".format(label)

            # Create annotation
            plt.annotate(
                label,                      # Use `label` as label
                (x_value, y_value),         # Place label at end of the bar
                xytext=(space, 0),          # Horizontally shift label by `space`
                textcoords="offset points", # Interpret `xytext` as offset in points
                va='center',                # Vertically center label
                ha=ha)                      # Horizontally align label differently for
                                            # positive and negative values.

    fig, ax1 = plt.subplots(1, 1)

    fig.suptitle(title)

    ax1.set_ylabel(f'Change in {label}')
    ax1.set_xlabel('Percentage')

    bins = _get_bounds_delta(var, vmin=vmin, vmax=vmax)

    plt.hist(data, bins, weights=np.ones(len(data)) / len(data), range=[vmin, vmax], orientation='horizontal')

    # LABELS
    set_bar_labels(ax1.patches)
    plt.gca().xaxis.set_major_formatter(PercentFormatter(1))

    # MEAN
    min_lim, max_xlim = plt.xlim()
    y_factor = 1.1 if var != 'prec' else 1.5
    plt.text(max_xlim*0.05, vmean * y_factor, 'Mean: {:.3f}'.format(vmean), alpha=0.5)
    plt.axhline(vmean, color='k', linestyle='dashed', linewidth=1, alpha=0.5)

    # EXTEND
    ax1.set_xlim(min_lim, max_xlim + 0.05)
    plt.gcf().subplots_adjust(left=0.15)
    plt.show()


def plot_delta(input_file, var=None, vmin=None, vmax=None, title=None, label=None, **kwargs):
    ds = gdal.Open(input_file)
    ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray())

    fig, ax1 = plt.subplots(1,1)

    cmap = cm.get_cmap('RdBu_r')
    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    bounds = _get_bounds_delta(var, vmin=vmin, vmax=vmax)

    norm = BoundaryNorm(bounds, cmap.N)

    cmap.set_under(color='black')

    nrows, ncols = ds_arr.shape
    x0, dx, _, y0, _, dy = ds.GetGeoTransform()

    x1 = x0 + dx * ncols
    y1 = y0 + dy * nrows

    extent = [x0, x1, y1, y0]

    im = ax1.imshow(ds_arr, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, extent=extent, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax1)

    fg_color = 'white'
    bg_color = 'black'

    # IMSHOW
    # ax1.axes.get_xaxis().set_visible(False)
    # ax1.axes.get_yaxis().set_visible(False)

    # set title plus title color
    ax1.set_title(title, color=fg_color)

    # set figure facecolor
    ax1.patch.set_facecolor(bg_color)

    # set tick and ticklabel color
    im.axes.tick_params(color=fg_color, labelcolor=fg_color)

    # set imshow outline
    # for spine in im.axes.spines.values():
    #     spine.set_edgecolor(fg_color)

    # COLORBAR
    # set colorbar label plus label color
    cb.set_label(label, color=fg_color)

    # set colorbar tick color
    cb.ax.yaxis.set_tick_params(color=fg_color)

    # set colorbar edgecolor
    cb.outline.set_edgecolor(fg_color)

    # set colorbar ticklabels
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=fg_color)

    fig.patch.set_facecolor(bg_color)
    plt.tight_layout()
    plt.show()
