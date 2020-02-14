import os

from string import ascii_uppercase as upc
from subprocess import Popen, PIPE

from wclim import WorldClimRaster
from config import script_dir, baseline_dir


class WorldClimRasterCalculator(WorldClimRaster):

    def __init__(self, clim_type=None, ext='tif'):
        super().__init__(clim_type, 1)
        
        self.ext = ext
        self.clim_type = clim_type


    def mean(self, src_dir, clim_type, dtype='Int32', items=12, no_value=False):
        status = True

        try:
            ext = self.ext
            dest_file = '{}_mean.{}'.format(clim_type, ext)
            out_tif = os.path.join(src_dir, dest_file)    

            raster_args = []
            add_arg = []

            for i in range(1, items + 1):
                filename = '{}_{:02d}.{}'.format(clim_type, i, ext)
                file_path = os.path.join(src_dir, filename)
                
                if not os.path.exists(file_path):
                    raise Exception('Path does not exists for: {file_path}')

                _L = upc[i-1]  # Letter
                raster_args.append('-{} {}'.format(_L, file_path))
                add_arg.append(f'{_L}*({_L}>0)')

            raster_args = ' '.join(raster_args)
            add_arg = '+'.join(add_arg)
            add_arg = '({})/{}'.format(add_arg, items)

            no_data = ''
            if no_value:
                no_data = '--NoDataValue=-999'
            
            cmd_tokens = f'python gdal_calc.py {raster_args} --type={dtype} --outfile={out_tif} --calc="{add_arg}" {no_data}'.format(
                target_dir=script_dir,
                raster_args=raster_args,
                out_tif=out_tif,
                add_arg=add_arg,
                no_data=no_data)
            
            shell = Popen(cmd_tokens, cwd=script_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)
            out, err = shell.communicate()

            if err:
                raise Exception(err)

        except Exception as err:
            print(err)
            status = False
        
        return status

    def count(self, src_dir, clim_type, dtype='Int32', items=12, no_value=False):
        status = True

        try:
            ext = self.ext
               
            raster_args = None
            add_arg = None

            for i in range(1, items + 1):
                dest_file = '{}_{:02d}_count.{}'.format(clim_type, i, ext)
                out_tif = os.path.join(src_dir, dest_file) 

                filename = '{}_{:02d}.{}'.format(clim_type, i, ext)
                file_path = os.path.join(src_dir, filename)
                
                if not os.path.exists(file_path):
                    raise Exception('Path does not exists for: {file_path}')
                
                raster_args = '-{} {}'.format('A', file_path)
                add_arg =f'A*(A<100)'
                # add_arg =f'999*(A<100)+(A<=100)*1'

                no_data = ''
                if no_value:
                    no_data = '--NoDataValue=-999'
                
                cmd_tokens = f'python gdal_calc.py {raster_args} --type={dtype} --outfile={out_tif} --calc="{add_arg}" {no_data}'.format(
                    target_dir=script_dir,
                    raster_args=raster_args,
                    out_tif=out_tif,
                    add_arg=add_arg,
                    no_data=no_data)

                shell = Popen(cmd_tokens, cwd=script_dir, stdin=PIPE, stdout=PIPE, stderr=PIPE)
                out, err = shell.communicate()

                if err:
                    raise Exception(err)
                
                exit('asd')

        except Exception as err:
            print(err)
            status = False
        
        return status


if __name__ == "__main__":
    clim_type = 'prec'

    src_dir = os.path.join(baseline_dir, clim_type)

    calc = WorldClimRasterCalculator(clim_type=clim_type)
    # calc.mean(src_dir, clim_type, dtype='Int32', no_value=True)
    calc.count(src_dir, clim_type, dtype='Int32', no_value=False)
    
