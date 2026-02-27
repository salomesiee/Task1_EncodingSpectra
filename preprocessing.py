import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from pybaselines.whittaker import asls, airpls


class PreprocessingPipeline:
    def __init__(
            self, 
            wavenumbers=np.arange(0, 4000, 1), 
            steps=["interpolate", "snv", "baseline", "smoothing"], 
            range_cut=[0, 2000], 
            normalize=True,
            min_max_itp=False
        ):
        self.wavenumbers = wavenumbers
        self.steps = steps
        self.range_cut = range_cut
        self.min_max_itp = min_max_itp
        self.normalize = normalize

    def process(self, spectrum):
        wv, x = spectrum[:, 0], spectrum[:, 1]

        for _, step in enumerate(self.steps):
            x = self._remove_nans(x)
            
            if step == "interpolate":
                x = self._interpolate(wv, x)
                wv = self.wavenumbers                
            elif step == "snv":
                x = self._snv(x)
            elif step == "baseline":
                x = self._baseline_correction(x)
            elif step == "smoothing":
                x = self._smoothing(x)
            elif step == "airpls":
                x = self._airPLS(x)
            elif step == "cut":
                wv, x = self._cut(wv, x)
            else:
                raise ValueError(f"Unknown preprocessing step: {step}")
            
        if self.normalize:
            x = self._min_max_normalize(x)

        return x
        
    
    def _cut(self, wv, intensities):
        mask = (wv >= self.range_cut[0]) & (wv <= self.range_cut[1])
        cutted_wv, cutted_intensities = wv[mask], intensities[mask]
        self.wavenumbers = np.arange(self.range_cut[0], self.range_cut[1], 1)
        return cutted_wv, cutted_intensities
    
    def _remove_nans(self, x):
        return np.where(np.isnan(x), np.nanmin(x) , x)


    def _interpolate(self, wv, intensities):
        if np.array_equal(wv, self.wavenumbers):
            return intensities
        else:
            if self.min_max_itp: 
                min_wv = int(np.nanmin(wv))
                max_wv= int(np.nanmax(wv))
                new_wv = np.linspace(min_wv, max_wv, max_wv-min_wv)
                self.wavenumbers = new_wv

            interp_func = interp1d(
                wv, intensities,
                kind='linear', bounds_error=False, fill_value=np.min(intensities)
            )
            return interp_func(self.wavenumbers)
    
    def _snv(self, x):
        x[np.isnan(x)] = 0.0
        if np.std(x):
            return (x - np.nanmean(x)) / np.std(x)
        else:
            return (x - np.nanmean(x))

    def _baseline_correction(self, x, lam=1e5, p=0.01):
        baseline, _ = asls(x, lam=lam, p=p)
        return x - baseline

    def _smoothing(self, x, window_size=5, poly_order=3):
        return savgol_filter(x, window_length=window_size, polyorder=poly_order)
    
    def _airPLS(self, x, lam=1e7):
        baseline, _ = airpls(x, lam=lam)
        return x - baseline
    
    def _min_max_normalize(self, x): 
        x_norm = (x - x.min()) / (x.max() - x.min())
        return x_norm
