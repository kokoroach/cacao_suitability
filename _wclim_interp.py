import numpy as np
import inspect

from scipy import interpolate


def interp(value, type, l_extrap=False, r_extrap=False):
    x, is_asc = get_ranges_by_type(type, value)
    # x = [1200, 1400, 1600, 1800, 1900]
    # y = [40,   60,   85,   95,  100]
    # l_delta = 15
    # r_delta = 0
    print(x, is_asc)
    exit('asd')

    f = interpolate.interp1d(x, y, fill_value='extrapolate')

    if l_extrap and value < x[0]:
        result = f(value) - l_delta
    elif  r_extrap and value > x[-1]:
        result = f(value) + r_delta
    else:
        result = np.interp(value, x, y)

    return result

def get_ranges_by_type(type, value):
    ranges = []
    is_asc = True

    if type == 'prec':
        pivot = 1900
        if value >= pivot:
            ranges = [1900, 2000, 2500, 3500, 4400]
        else:
            ranges = [1200, 1400, 1600, 1800, 1900]
    
    return ranges, is_asc

def get_module_dir():
    _cur_file = inspect.getfile(inspect.currentframe())


      

if __name__ == "__main__":
    r = interp(1199.999, type='prec', l_extrap=True, r_extrap=True)
    print(r)