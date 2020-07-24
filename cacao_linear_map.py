# On the basis of climate conditions classification, linear interpolation function
# will be implemented to assign a score for climatic conditions (continuous quantitative values).
# This scheme is based on linear interpolation functions that map value intervals to score intervals.
# If the observed value is x and it falls into the interval [a, b] it needs to get a score y that falls
# into the interval [c, d]. The formula to calculate y is:


import pandas as pd
import numpy as np
from math import inf, isinf


# Linear Interpolation
class LinearMapper:

    def __init__(self, ctype):
        self.ctypes = ['prec', 'tmin', 'tmean', 'tmax']
        self.ratings = [(100, 95), (95, 85), (85, 60), (60, 40), (25, 0)]
        self.ranges = []

        if ctype not in self.ctypes:
            raise Exception('ctype not found for: {}'.format(ctype))

        self.ctype = ctype
        self.conversion = self.set_conversion_table()

    @staticmethod
    def _is_in_range(range, value):
        if range[1] > range[0]:
            if value >= range[0] and value <= range[1]:
                return True
            return False
        else:
            if value >= range[1] and value <= range[0]:
                return True
            return False


    def get_ranges(self):
        ranges = None

        if self.ctype == 'prec':
            ranges =  [
                [(1900, 1800), (1900, 2000)],
                [(1800, 1600), (2000, 2500)],
                [(1600, 1400), (2500, 3500)],
                [(1400, 1200), (3500, 4400)],
                [(1200, -inf), (4400, inf)],  # NOTE: < 120 or > 4400, For extrapolation 
            ]
        elif self.ctype == 'tmin':
            ranges = [
                [(inf, 20)],  # NOTE: > 20, For extrapolation 
                [(20, 15)],
                [(15, 13)],
                [(13, 10)],
                [(10, -inf)],  # NOTE: < 10, For extrapolation 
            ]
        elif self.ctype == 'tmean':
            ranges =  [
                [(26, 25), (26, 28)],
                [(25, 23), (28, 29)],
                [(23, 22), (29, 30)],
                [(22, 21), (30, inf)],  # NOTE: Linear Mapped, none existing. Handle thru delta
                [(21, -inf), (0, 0)],
            ]
        elif self.ctype == 'tmax':
            ranges = [
                [(-inf, 28)],  # NOTE: < 28, Arbitrary data
                [(28, 30)],
                [(30, inf)],  # NOTE: > 30, Arbitrary data
            ]

        return ranges


    def set_conversion_table(self):
        ranges = self.get_ranges()

        factor_dict = {}
        factor_dict['ratings'] = self.ratings[:len(ranges)]
        factor_dict['ranges'] = ranges

        return pd.DataFrame(factor_dict)


    def get_mapping_factors(self, value):
        factor = None
        range = None

        for c_index, c_col in self.conversion.iterrows():
            ranges = c_col['ranges']

            for r_index, _range in enumerate(ranges):
                if self._is_in_range(_range, value):
                    if isinf(_range[0]) or isinf(_range[1]):
                        if c_index == 0:
                            factor = self.conversion['ratings'][1]
                            range = self.conversion['ranges'][1][r_index]
                        else:
                            factor = self.conversion['ratings'][c_index-1]
                            range = self.conversion['ranges'][c_index-1][r_index]
                    else:
                        factor = self.conversion['ratings'][c_index]
                        range = self.conversion['ranges'][c_index][r_index]
                    return factor, range

        raise Exception('Mapping ratings error')


    def resolve_delta(self, value):
        if self.ctype == 'prec':
            if value < 1200 or value > 4400:
                return 15
        elif self.ctype == 'tmin':
            if value < 10:
                return 15
        elif self.ctype == 'tavg':
            if value < 21:
                return 15
            elif value > 30:
                return 35
        return 0


    def get_mapped_values(self, values):

        def get_factor_range(_range):
            if isinf(_range[0]) or isinf(_range[1]):
                if c_index == 0:
                    factor = self.conversion['ratings'][1]
                    ranges = self.conversion['ranges'][1][r_index]
                else:
                    factor = self.conversion['ratings'][c_index-1]
                    ranges = self.conversion['ranges'][c_index-1][r_index]
            else:
                factor = self.conversion['ratings'][c_index]
                ranges = self.conversion['ranges'][c_index][r_index]

            return factor, ranges


        row, col = values.shape
        result = np.ndarray(shape=(row, col), dtype=np.float32)

        for c_index, c_col in self.conversion.iterrows():
            ranges = c_col['ranges']

            for r_index, _range in enumerate(ranges):
                lbound, rbound = sorted(list(_range))

                if lbound == rbound:
                    continue
                indexes = (values >= lbound) & (values <= rbound)
                if not indexes.any():
                    continue

                factor, ranges = get_factor_range(_range)

                for y in range(row):
                    for x in range(col):
                        if not indexes[y][x]:
                            continue
                        value = values[y][x]
                        delta = self.resolve_delta(value)

                        result[y][x] = self._map_value(value, factor, ranges, delta)

        return result


    def _map_value(self, value, factor, range, delta):
        y1, y2 = factor[0], factor[1]
        x1, x2 = range[0], range[1]

        try:
            m_val = y2 + (((y1-y2)*(value-x2))/(x1-x2))
        except ZeroDivisionError:
            raise

        if delta:
            m_val -= delta

        if m_val < 0:
            m_val = 0

        return round(m_val, 2)


    def get_mapped_value(self, value):
        factor, range = self.get_mapping_factors(value)
        delta = self.resolve_delta(value)

        return self._map_value(value, factor, range, delta)


if __name__ == "__main__":
    mapper = LinearMapper('prec')
    vals = pd.Series([4400.00001, 1900.00])
    r = mapper.get_mapped_values(vals)

