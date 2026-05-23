"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2010 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

import numpy as np
import pandas as pd
from statadict import parse_stata_dict
from thinkstat import underride

def read_stata(dct_file, dat_file, **options):
    """Read data from a stata file.
    
    Args:
        dct_file: string file name of the dictionary file.
        dat_file: string file name of the data file.
        **options: additional options passed to pd.read_fwf.

    Returns:
        DataFrame: Stata data loaded into pandas DataFrame.
    """

    stata_dict = parse_stata_dict(dct_file)
    underride(options, compression = "gzip")
    resp = pd.read_fwf(
        dat_file,
        names = stata_dict.names,
        colspecs = stata_dict.colspecs,
        **options
    )

def read_fem_resp(dct_file = "data/2002FemResp.dct", dat_file="data/2002FemResp.dat.gz"):
    """Read the 2002 NSFG respondent file.

    Args:
        dct_file: string file name of the dictionary file.
        dat_file: string file name of the data file.

    Returns:
        DataFrame: NSFG respondent data with cleaned variables.
    """
    resp = read_stata


if __name__ == "__main__":
    main()
