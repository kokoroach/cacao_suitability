import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from osgeo import gdal
from matplotlib.colors import LinearSegmentedColormap


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


def plot(input_file, var=None, title=None, vmin=-0.9, label=None, linx=False, apply_factor=None):
    if not linx and var is None:
        raise ValueError(f"Var type needed for non-LINDX plot")
    
    ds = gdal.Open(input_file)
    if apply_factor == 1:
        ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray()) 
    else:
        ds_arr = np.array(ds.GetRasterBand(1).ReadAsArray()) * apply_factor

    fig, ax1 = plt.subplots(1,1)

    if linx:
        # Assertive remap
        bounds = [0, 12.5, 25, 50, 75, 100]
        cmap = mpl.cm.RdYlGn
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    else:
        bounds, colors = _get_cb_params(var)

        cmap = LinearSegmentedColormap.from_list(
            'custom', colors, N=len(colors))
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cmap.set_under(color='black')

    im = ax1.imshow(ds_arr, cmap=cmap, norm=norm, vmin=vmin, interpolation='nearest')
    cb = fig.colorbar(im, ax=ax1)

    fg_color = 'white'
    bg_color = 'black'

    # IMSHOW 
    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

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
    # fig.savefig('sasd.png', dpi=200)

