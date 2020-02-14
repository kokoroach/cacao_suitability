# On the basis of climate conditions classification, linear interpolation function
# will be implemented to assign a score for climatic conditions (continuous quantitative values).
# This scheme is based on linear interpolation functions that map value intervals to score intervals.
# If the observed value is x and it falls into the interval [a, b] it needs to get a score y that falls
# into the interval [c, d]. The formula to calculate y is:


# TODO: Resolve for
# (DONE) prec,
# (TODO) PET,
# (DONE) mean_annual_temp,
# (DONE) mean_annual_maxtemp,
# (DONE) mean_annual_mintemp


import pandas as pd
from math import inf, isinf


# Linear Interpolation
class LinearMapper(object):

    def __init__(self):
        self.types = ['prec', 'tmin', 'tavg', 'tmax']
        self.factors = [(95, 100), (85, 95), (60, 85), (40, 60), (0, 25)]
        self.ranges = []
        self.delta = 15  # from delta(factor(25, 40)))

    def get_conversion_table(self, type, value):
        factor = None
        range = None
        asert_delta = False

        ranges, is_asc, has_delta = self.get_ranges_by_type(type, value)

        factor_dict = {}
        factor_dict['factors'] = self.factors[:len(ranges)]
        factor_dict['ranges'] = ranges
        
        factor_df = pd.DataFrame(factor_dict)

        items_len = len(factor_df)

        if is_asc:
            for index, row in factor_df.iterrows():
                ranges = row['ranges']
                
                if len(ranges) == 1:
                    if value >= ranges[0][0] and value <= ranges[0][1]:
                        if isinf(ranges[0][0]):
                            range = factor_df['ranges'][index+1][0][::-1]
                            factor = factor_df['factors'][index+1]
                        elif isinf(ranges[0][1]):
                            range = factor_df['ranges'][index-1][0][::-1]
                            factor = factor_df['factors'][index-1]
                        else:
                            range = ranges[0][::-1]
                            factor = row['factors']
                        break
        else:
            for index, row in factor_df.iterrows():
                ranges = row['ranges']
                
                if len(ranges) == 1:
                    if value >= ranges[0][0] and value <= ranges[0][1]:
                        if isinf(ranges[0][1]):
                            range = factor_df['ranges'][index+1][0]
                            factor = factor_df['factors'][index+1]
                        elif isinf(ranges[0][0]):
                            range = factor_df['ranges'][index-1][0]
                            factor = factor_df['factors'][index-1]
                            asert_delta = has_delta
                        else:
                            range = ranges[0]
                            factor = row['factors']
                        break

                elif len(ranges) == 2:
                    if value >= ranges[0][0] and value <= ranges[0][1]:
                        if isinf(ranges[0][1]):
                            range = factor_df['ranges'][index+1][0]
                            factor = factor_df['factors'][index+1]

                        elif isinf(ranges[0][0]):
                            range = factor_df['ranges'][index-1][0]
                            factor = factor_df['factors'][index-1]
                            asert_delta = has_delta

                        else:
                            range = ranges[0]
                            factor = row['factors']
                        break

                    elif value >= ranges[1][0] and value <= ranges[1][1]:
                        if isinf(ranges[1][0]):
                            range = factor_df['ranges'][index+1][1][::-1]
                            factor = factor_df['factors'][index+1]
                        elif isinf(ranges[1][1]):
                            range = factor_df['ranges'][index-1][1][::-1]
                            factor = factor_df['factors'][index-1]
                            asert_delta = has_delta
                        else:
                            range = ranges[1][::-1]
                            factor = row['factors']
                        break

        return factor, range, asert_delta

    def get_ranges_by_type(self, type, value):
        ranges = None
        is_asc = False
        has_delta = True

        if type == 'prec':
            ranges =  [
                [(1800, 1900), (1900, 2000)],
                [(1600, 1800), (2000, 2500)],
                [(1400, 1600), (2500, 3500)],
                [(1200, 1400), (3500, 4400)],
                [(-inf, 1200), (4400, inf)],  # NOTE: < 120 or > 4400, For extrapolation 
            ]
        elif type == 'tmin':
            ranges = [
                [(20, inf)],  # NOTE: > 20, For extrapolation 
                [(15, 20)],
                [(13, 15)],
                [(10, 13)],
                [(-inf, 10)],  # NOTE: < 10, For extrapolation 
            ]
        elif type == 'tavg':
            ranges =  [
                [(25, 26), (26, 28)],
                [(23, 25), (28, 29)],
                [(22, 23), (29, 30)],
                [(21, 22), (30, inf)],  # NOTE: Linear Mapped, none existing
                [(-inf, 21), (0, 0)],   
            ]
            self.delta += 20

        elif type == 'tmax':
            is_asc = True
            has_delta = False
            ranges = [
                [(-inf, 28)],  # NOTE: < 28, Arbitrary data
                [(28, 30)],
                [(30, inf)],  # NOTE: > 30, Arbitrary data
            ]

        return ranges, is_asc, has_delta

    def get_linear_map(self, type, value):

        if type not in self.types:
            raise Exception('type not found for: {}'.format(type))

        factor, range, assert_delta = self.get_conversion_table(type, value)

        y1, y2 = factor[0], factor[1]
        x1, x2 = range[0], range[1]

        y = y2 + (((y1-y2)*(value-x2))/(x1-x2))

        if assert_delta:
            y -= self.delta

        if y < 0:
            y = 0
        elif y > 100:
            y = 100
        
        return round(y, 2)


if __name__ == "__main__":
    mapper = LinearMapper()
    val = 1119.99999999
    r = mapper.get_linear_map('prec', val)
    print(r)
