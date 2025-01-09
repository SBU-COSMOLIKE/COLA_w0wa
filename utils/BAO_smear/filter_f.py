from scipy.integrate import simps
import sys
import numpy as np
import scipy
sys.path.append("./projects/lsst_y1/COLA_Emulators/Cython_Filter_Files/")
import cython_filter

def smear_bao_vec(ks, pk, pk_nw):


    Gk = [ ]    
    integral = simps(pk,ks)#trapz(ks * pk, x=np.log(ks))  #simps(pk,ks)#
    k_star_inv = (1.0/(3.0 * np.pi**2)) * integral
    for i in range(pk.shape[0]):
        Gk.append(np.exp(-0.5*k_star_inv[i] * (ks**2)))
    Gk= np.array(Gk)    
    pk_smeared = pk*Gk + pk_nw*(1.0 - Gk)
    return pk_smeared


def smooth_bao_ver2(ks, pk, n=15):
 
    def linear_extrapolation_below(x, y, xi):        
        y_extrapolated = y[:, 0].reshape(-1, 1) + (xi - x[0]) * (y[:, 1].reshape(-1, 1) - y[:, 0].reshape(-1, 1)) / (x[1] - x[0])
        return y_extrapolated

    def linear_extrapolation_above(x, y, xi):   
        y_extrapolated = y[:, -1].reshape(-1, 1) + (xi - x[-1]) * (y[:, -1].reshape(-1, 1) - y[:, -2].reshape(-1, 1)) / (x[-1] - x[-2])
        return y_extrapolated


    size_pk =pk.shape[0]

   
    dst_ks = np.linspace(1e-4, 5, 2**n) #10
    log_dst_ks = np.log(dst_ks)
    logks = np.log(ks)

    num_bins_smaller = np.sum(log_dst_ks < logks[0])
    num_bins_bigger = np.sum(log_dst_ks > logks[-1])




    inter_log_pkz= cython_filter.interp1d_openmp(np.log(ks),
                                           np.log(pk),
                                           log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                                           0) 

    if num_bins_smaller != 0:
        extrap_below = linear_extrapolation_below(log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                                   inter_log_pkz,
                                   log_dst_ks[:num_bins_smaller])
        inter_log_pkz_ = np.concatenate((extrap_below , inter_log_pkz ), axis=1) 
    else:
        inter_log_pkz_ =inter_log_pkz

    if num_bins_bigger != 0:

        extrap_above = linear_extrapolation_above(log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                               inter_log_pkz,
                               log_dst_ks[log_dst_ks.shape[0]-num_bins_bigger:])

        inter_log_pkz = np.concatenate((inter_log_pkz_ , extrap_above), axis=1)
    else:
        inter_log_pkz =inter_log_pkz_
    


    logkpk = cython_filter.logkpk_openmp(dst_ks,inter_log_pkz)


    sine_transf_logkpk = scipy.fft.dst(logkpk, type=2, workers=-1)

    indices = np.arange(len(dst_ks))
    even_is = indices[indices % 2 == 0].astype(np.float64)
    odd_is = indices[indices % 2 != 0].astype(np.float64)
    evens = (sine_transf_logkpk[:,0::2])
    odds = (sine_transf_logkpk[:,1::2]  ) 

    odds_interp = scipy.interpolate.CubicSpline(odd_is[50:150], odds[:,50:150],  axis=1).derivative(2) 
    evens_interp = scipy.interpolate.CubicSpline(even_is[50:150], evens[:,50:150],  axis=1).derivative(2)

    d2_odds_avg = (odds_interp(odd_is[50:150]) + odds_interp(odd_is[50:150]+2)  + odds_interp(odd_is[50:150]-2))/3 

    d2_evens_avg = (evens_interp(even_is[50:150]) +evens_interp(even_is[50:150]+2) +evens_interp(even_is[50:150]-2) )/3


    imin_even = 50+np.argmax(d2_evens_avg, axis = 1) -9

    imax_even = 50+np.argmin(d2_evens_avg, axis = 1)+36

    imin_odd = 50+np.argmax(d2_odds_avg, axis = 1)-9

    imax_odd = 50+np.argmin(d2_odds_avg, axis = 1)+37 

    pk_nw = cython_filter.pk_nw_( even_is,
             odd_is,
             imin_even.astype(np.int32),
             imin_odd.astype(np.int32),
             imax_even.astype(np.int32),
             imax_odd.astype(np.int32),
             evens,
             odds,
             dst_ks,
             size_pk) # idst(treated_transform, type=2, norm='ortho')



    k_highk = ks[ks > 4]
    log_ks = np.log(ks)
    k_extended = np.ascontiguousarray(np.concatenate((dst_ks[dst_ks < 4], k_highk)))
    log_k_extended = np.log(k_extended)
    num_bins_smaller = np.sum(log_ks < log_k_extended[0])

    p_highk = pk[:,ks > 4]

    pksmooth_interp = cython_filter.pksmooth_openmp( pk_nw, 
                            p_highk, 
                            k_extended, 
                            ks, 
                            dst_ks, 
                            num_bins_smaller)





    return np.exp(np.concatenate((linear_extrapolation_below(np.log(ks)[num_bins_smaller:], pksmooth_interp, np.log(ks)[:num_bins_smaller]),pksmooth_interp), axis=1))



def smooth_bao_ver1(ks, pk, n=15):
 
    def linear_extrapolation_below(x, y, xi):
        y_extrapolated = y[:, 0].reshape(-1, 1) + (xi - x[0]) * (y[:, 1].reshape(-1, 1) - y[:, 0].reshape(-1, 1)) / (x[1] - x[0])
        return y_extrapolated

    def linear_extrapolation_above(x, y, xi):
        y_extrapolated = y[:, -1].reshape(-1, 1) + (xi - x[-1]) * (y[:, -1].reshape(-1, 1) - y[:, -2].reshape(-1, 1)) / (x[-1] - x[-2])
        return y_extrapolated


    size_pk =pk.shape[0]

   
    dst_ks = np.linspace(1e-4, 5, 2**n) #10
    log_dst_ks = np.log(dst_ks)
    logks = np.log(ks)

    num_bins_smaller = np.sum(log_dst_ks < logks[0])
    num_bins_bigger = np.sum(log_dst_ks > logks[-1])




    inter_log_pkz= cython_filter.interp1d_openmp(np.log(ks),
                                           np.log(pk),
                                           log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                                           0) 

    if num_bins_smaller != 0:
        extrap_below = linear_extrapolation_below(log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                                   inter_log_pkz,
                                   log_dst_ks[:num_bins_smaller])
        inter_log_pkz_ = np.concatenate((extrap_below , inter_log_pkz ), axis=1) 
    else:
        inter_log_pkz_ =inter_log_pkz

    if num_bins_bigger != 0:

        extrap_above = linear_extrapolation_above(log_dst_ks[num_bins_smaller:log_dst_ks.shape[0]-num_bins_bigger],
                               inter_log_pkz,
                               log_dst_ks[log_dst_ks.shape[0]-num_bins_bigger:])

        inter_log_pkz = np.concatenate((inter_log_pkz_ , extrap_above), axis=1)
    else:
        inter_log_pkz =inter_log_pkz_
    


    logkpk = cython_filter.logkpk_openmp(dst_ks,inter_log_pkz)


    sine_transf_logkpk = scipy.fft.dst(logkpk, type=2, workers=-1)

    indices = np.arange(len(dst_ks))
    even_is = indices[indices % 2 == 0].astype(np.float64)
    odd_is = indices[indices % 2 != 0].astype(np.float64)
    evens = (sine_transf_logkpk[:,0::2])
    odds = (sine_transf_logkpk[:,1::2]  ) 


    d2_odds_avg =cython_filter.second_derivative( odd_is[50:150], odds[:,50:150])
    d2_evens_avg = cython_filter.second_derivative(even_is[50:150], evens[:,50:150])


    imin_even = 50+np.argmax(d2_evens_avg, axis = 1) -9

    imax_even = 50+np.argmin(d2_evens_avg, axis = 1)+36

    imin_odd = 50+np.argmax(d2_odds_avg, axis = 1)-9

    imax_odd = 50+np.argmin(d2_odds_avg, axis = 1)+37 

    pk_nw = cython_filter.pk_nw_( even_is,
             odd_is,
             imin_even.astype(np.int32),
             imin_odd.astype(np.int32),
             imax_even.astype(np.int32),
             imax_odd.astype(np.int32),
             evens,
             odds,
             dst_ks,
             size_pk) # idst(treated_transform, type=2, norm='ortho')



    k_highk = ks[ks > 4]
    log_ks = np.log(ks)
    k_extended = np.ascontiguousarray(np.concatenate((dst_ks[dst_ks < 4], k_highk)))
    log_k_extended = np.log(k_extended)
    num_bins_smaller = np.sum(log_ks < log_k_extended[0])

    p_highk = pk[:,ks > 4]

    pksmooth_interp = cython_filter.pksmooth_openmp( pk_nw, 
                            p_highk, 
                            k_extended, 
                            ks, 
                            dst_ks, 
                            num_bins_smaller)





    return np.exp(np.concatenate((linear_extrapolation_below(np.log(ks)[num_bins_smaller:], pksmooth_interp, np.log(ks)[:num_bins_smaller]),pksmooth_interp), axis=1))
