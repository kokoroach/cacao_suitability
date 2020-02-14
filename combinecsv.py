import pandas as pd
import os
import glob

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)

#extension = 'csv'

#all_filenames = [i for i in glob.glob('*.{}'.format(extension))]

# for f in all_filenames:
# 	a = pd.read_csv("output.csv")
# 	b = pd.read_csv(f)
# 	b = b.dropna(axis=1)
# 	merged = pd.merge(a,b,how='outer', on=['X', 'Y'])
# 	merged.to_csv("output.csv", index=False)

a = pd.read_csv("prec.csv")
b = pd.read_csv("tavg.csv")
b = b.dropna(axis=1)
merged1 = pd.merge(a,b,how='outer', on=['X', 'Y'])

a = pd.read_csv("tmax.csv")
b = pd.read_csv("tmin.csv")
b = b.dropna(axis=1)
merged2 = pd.merge(a,b,how='outer', on=['X', 'Y'])

merged = pd.merge(merged1,merged2,how='outer', on=['X', 'Y'])
merged.to_csv("FinalData.csv", index=False)

