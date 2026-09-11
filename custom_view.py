import numpy as np
from tools import *


class Respi_Rate:
    """Viewer row: distribution of the respiratory rate over time (2d) or its variability (1d)."""
    name = 'Respi_Rate'

    def __init__(self, resp_features, rate_bins_resp = np.arange(5, 30, 0.5), resp_wsize_in_mins = 4, ratio_sat = 4, units = 'bpm', plot_type = '2d'):
        self.rate_bins_resp = rate_bins_resp
        self.resp_wsize_in_mins = resp_wsize_in_mins
        self.ratio_sat = ratio_sat
        self.resp_features = resp_features
        self.units = units
        self.plot_type = plot_type
        
    def plot(self, ax, t0, t1):

        resp_features = self.resp_features
        local_resp_features = resp_features[(resp_features['inspi_date'] > t0) & (resp_features['inspi_date'] < t1)]
        
        if not local_resp_features.shape[0] == 0:
            try:
                res = get_rate_variablity(cycles = local_resp_features, 
                                        rate_bins = self.rate_bins_resp, 
                                        bin_size_min = self.resp_wsize_in_mins, 
                                        colname_date = 'inspi_date', 
                                        colname_time = 'inspi_time', 
                                        units = self.units
                                        )

                plot_variability(res, ax=ax, ratio_saturation = self.ratio_sat, plot_type = self.plot_type)
                if self.plot_type == '2d':
                    ax.set_ylim(self.rate_bins_resp[0], self.rate_bins_resp[-1])
                elif self.plot_type == '1d':
                    ax.set_ylim(0, 5)
                ax.set_ylabel(f'Respi\nrate\n[{self.units}]')
            except:
                ax.plot()
