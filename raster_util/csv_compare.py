
import csv
import pandas as pd


def validate(csv_1, csv_2):

    df_1 = pd.read_csv(csv_1, header=0, usecols=['x','y'])
    df_2 = pd.read_csv(csv_2, header=0, usecols=['x','y'])

    merged = pd.merge(df_1, df_2, how='outer', on=['x','y'], indicator=True)
    
    common = merged.loc[merged['_merge'] == 'both'] 
    baseline = merged.loc[merged['_merge'] == 'left_only']
    future = merged.loc[merged['_merge'] == 'right_only']
    
    out_csv = 'active_baseline_exclusive.csv'
    baseline = baseline.filter(items=['x', 'y'])
    baseline.to_csv(out_csv, index=False)

    out_csv = 'active_future_exclusinve.csv'
    future = future.filter(items=['x', 'y'])
    future.to_csv(out_csv, index=False)


if __name__ == '__main__':
    csv_1 = r''
    csv_2 = r''

    validate(csv_1, csv_2)