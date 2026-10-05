"""
WAP.py -- the statutory WAP/LPC return guarantee rate G_t (art. 24 §3 WAP, law of
28 April 2003 as reformed in 2015).

    G = clip( round_to_25bp( WAP_SHARE * mean_24m(OLO 10Y) ), 1.75%, 3.75% )

The FSMA fixes the rate on 1 January from the average 10Y OLO yield over the 24
months preceding 1 June of the previous year, rounded to the nearest 25 bp. The
share was 65% in 2016-2017, 75% in 2018-2019 and 85% from 2020 on. On the NBB
10Y series (pension/rates/data/olo_yields.csv) the 85% rule reproduces every published rate
from 2016 to 2026 (1.75% to 2024, 2.50% for 2025 and 2026); it gives 2.75% for
2027 where the FSMA published 2.50%, i.e. one notch off -- most likely a different
reference series on the FSMA side.

Under the HORIZONTAL method (Branch 21 insurance) a contribution keeps the rate in
force when it was paid until retirement; that bookkeeping lives in the liability
ledger (pension.dynamics.HorizontalLedger), not here. This module only produces the rate.
"""

import numpy as np

WAP_SHARE  = 0.85     # share of the 24-month OLO average (85% since 2020)
WAP_FLOOR  = 0.0175   # statutory minimum
WAP_CAP    = 0.0375   # statutory maximum
WAP_STEP   = 0.0025   # rounded to the nearest 25 bp
WAP_WINDOW = 24       # months averaged
WAP_LAG    = 8        # months from the last averaged month (May) to the fixing (1 January)


def wap_formula(avg10y):
    """The statutory map from a 24-month average 10Y yield (decimal) to G (decimal)."""
    raw = WAP_SHARE * np.asarray(avg10y, float)
    return np.clip(np.round(np.round(raw / WAP_STEP) * WAP_STEP, 10), WAP_FLOOR, WAP_CAP)


def computeWAPRate(y10_monthly, start_row, n_years, months_per_year=12):
    """
    Annual WAP guarantee rate per path from a monthly 10Y yield series.

    Parameters
    ----------
    y10_monthly : (n_months, n_paths) monthly 10Y yields (decimal): observed
                  history first, then the simulated months.
    start_row   : row of the month in which model year 0 starts. Year k starts at
                  row start_row + 12k; its rate averages the WAP_WINDOW months that
                  end WAP_LAG months before that row (May, for a 1 January fixing).
                  start_row >= WAP_WINDOW + WAP_LAG - 1 makes year 0 a fixing on
                  observed data only, as it is in reality.
    n_years     : number of model years to fix a rate for

    Returns
    -------
    G : (n_years, n_paths) -- the rate fixed at the start of model year k and
        held for that year.
    """
    y = np.asarray(y10_monthly, float)
    if y.ndim == 1:
        y = y[:, None]
    G = np.empty((n_years, y.shape[1]))
    for k in range(n_years):
        end = start_row + k * months_per_year - WAP_LAG      # last month in the window
        lo = end - WAP_WINDOW + 1
        assert lo >= 0, f"year {k}: WAP window starts before the data (row {lo})"
        G[k] = wap_formula(y[lo:end + 1].mean(axis=0))
    return G
