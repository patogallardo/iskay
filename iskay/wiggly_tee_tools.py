from scipy import interpolate
from iskay import weighted_median
import numpy as np
from iskay import pairwiser


def eval_wiggly_tee(zs, dTs, sigma_z, z_center,
                    gaussian_or_square='gaussian',
                    mean_or_median='mean'):
    '''Evals wiggly tee in one point z_center.
    With a population of zs and dTs, a sigam_z semiwidth.
    Weights can be gaussian or square and mean or median
    can be used to compute the estimator.'''
    ws = weighted_median.make_weights(zs, z_center, sigma_z,
                                      method=gaussian_or_square)
    wiggly_tee_of_z_center=weighted_median.compute_weighted_median_or_mean(dTs, # noqa
                      w=ws, method=mean_or_median)
    return wiggly_tee_of_z_center


def get_wiggly_tee(z, dT, sigma_z, N_in_sigma_z,
                   gaussian_or_square='gaussian',
                   mean_or_median='mean'):
    '''Returns a function that can be evaluated on z and the points
        at which it was evaluated.
    This function is a spline that is computed by
    evaluating the exact form a few times per sigma_z.
    z: redshifts
    dT: decrements
    sigma_z: semi-width of the window
    N_in_sigma_z: how many times to evaluate
    one semiwidth.
    method is specified in gaussian_or_square and mean_or_median.
    Returns: zs_to_eval, wiggly_tee and wiggly_tee_f
    Where the first two are z
    and wiggly Tee computed for
    this sample, and wiggly_tee_f is the function interpolated
    from these values'''
    zmin, zmax = z.min(), z.max()
    N_times = int(np.round((zmax-zmin)/sigma_z * N_in_sigma_z))

    zs_to_eval = np.linspace(zmin, zmax, N_times)
    wiggly_tee = np.zeros(len(zs_to_eval))

    for j in range(len(zs_to_eval)):
        wiggly_tee[j] = eval_wiggly_tee(z, dT, sigma_z, zs_to_eval[j],
                                        gaussian_or_square=gaussian_or_square,
                                        mean_or_median=mean_or_median)
    f = interpolate.interp1d(zs_to_eval, wiggly_tee, kind='cubic')
    return zs_to_eval, wiggly_tee, f


def correct_tsz_decrement(df, sigma_z, N_in_sigma=10,
                          gaussian_or_square='gaussian',
                          mean_or_median='median'):
    '''Takes dTs from aperture photometry, computes wiggly tee
    for each redshift and substracts wiggly tee.
    Results are appended in df['dT_kSZ']'''
    if 'gaussian_conventional' not in gaussian_or_square:  # new method
        print('Applying new wtee correction')
        zs_to_eval, w_tee, f_wiggly_tee = get_wiggly_tee(df.z.values,
                                                     df.dT.values,
                                                     sigma_z,
                                                     N_in_sigma,
                                                     gaussian_or_square=gaussian_or_square, # noqa
                                                     mean_or_median=mean_or_median) # noqa
        df['wiggly_tee'] = f_wiggly_tee(df.z.values)
    else:  # usual method
        assert 'gaussian_conventional' in gaussian_or_square
        print('Applying convetional wiggly tee correction')
        tzav = pairwiser.get_tzav_fast(df.dT.values, df.z.values, sigma_z)
        df['wiggly_tee'] = tzav
    df['dT_kSZ'] = df['dT'].values - df['wiggly_tee']
