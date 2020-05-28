
from osgeo import gdal


def get_data(path):
    gd = gdal.Open(path)
    data = gd.ReadAsArray()
    gd = None
    return data


def intersect(baseline=None, future=None):
    if not baseline:
        baseline = r''
    
    if not future:
        future = r''


    baseline = get_data(baseline)
    future = get_data(future)

    if baseline.shape != future.shape:
        raise Exception('Shapes does not match')
    
    A = (baseline >= 0)
    B = (future >= 0)

    common = []
    baseline = []
    future = []

    col, row = A.shape
    for y in range(col):
        for x in range(row):
            if A[y][x] and B[y][x]:
                common.append((x,y))
            elif A[y][x]:
                baseline.append((x,y))
            elif B[y][x]:
                future.append((x,y))
            else:
                continue
    
    return baseline, future 


# get_intersections(baseline=baseline, future=future)
