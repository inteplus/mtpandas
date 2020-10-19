#!/usr/bin/python3

'''Converts a dataframe from one file format to another. Currently accepting '.csv', '.csv.zip' and '.parquet'.'''


import argparse as _ap
import pandas as _pd
from mt.pandas.convert import dfload, dfsave


def main(args):
    df = dfload(args.in_df_filepath)
    dfsave(df, args.out_df_filepath)


if __name__ == '__main__':
    parser = _ap.ArgumentParser(description="Converts a dataframe from one file format to another. Currently accepting '.csv', '.csv.zip' and '.parquet'.")
    parser.add_argument('in_df_filepath', type=str,
                        help="Filepath to the input dataframe.")
    parser.add_argument('out_df_filepath', type=str,
                        help="Filepath to the output dataframe.")
    args = parser.parse_args()    
    
    main(args)
