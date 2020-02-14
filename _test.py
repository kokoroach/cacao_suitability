filename = r'cropped\baseline\prec\csv\prec_mean.csv'
src_tif = r'cropped\baseline\prec\prec_01.tif'
path_out = r'cropped\baseline\prec\prec_01_test.tif'

from PIL import Image
import numpy as np
import csv
import rasterio as rio

# im = Image.open(tif)

# arr = numpy.array(im)
# arr.flatten(order='C')
# arr[(arr <= 100) & (arr != -999)] = 0
# arr[arr > 100] = 1

# numpy.savetxt("foo.csv", arr, delimiter=",")

# # print(arr)

# myFileList = createFileList('path/to/directory/')
# fileList = [tif]

# for file in fileList:
#     img_file = Image.open(file)
#     # img_file.show()

#     # get original image parameters...
#     width, height = img_file.size
#     format = img_file.format
#     mode = img_file.mode

#     # Make image Greyscale
#     # img_grey = img_file.convert('L')
#     #img_grey.save('result.png')
#     # img_grey.show()
#     # x = img_file.size[1]
#     # y = img_file.size[0]

#     # Save Greyscale values
#     value = np.asarray(img_file.getdata(), dtype=np.int).reshape((width, height))
#     value = value.flatten()
#     value[(value <= 100) & (value != -999)] = 0
#     value[value > 100] = 1

#     with open("img_pixels.csv", 'w',  newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(value)

def to_dry(src_file, path_out):
    with rio.open(src_file) as clim_data:
        clim_data_val, clim_data_meta = clim_data.read(), clim_data.meta

    clim_data = clim_data_val[0]
    clim_data[(clim_data <= 100) & (clim_data != -999)] = 1  # Note,
    clim_data[clim_data > 100] = 0  # Note,

    with rio.open(path_out, 'w', **clim_data_meta) as ff:
        ff.write(clim_data, 1)


if __name__ == "__main__":
    
    for i in range(1, 13):
        src_tif = f'cropped\\baseline\\prec\\prec_{i:02d}.tif'
        path_out = f'cropped\\baseline\\dry\\dry_{i:02d}.tif'

        to_dry(src_tif, path_out)
    

