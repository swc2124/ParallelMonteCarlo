# ====================================================================
# Author                : swc21
# Date                  : 2018-03-14 11:49:48
# Project               : GitHub
# File Name             : dist_bsize_cluster
# Last Modified by      : swc21
# Last Modified time    : 2018-03-14 12:02:40
# ====================================================================
# 
#--[IMPORTS]------------------------------------------------------------------#
import numpy as np
import time

from mpi4py import MPI
from random import shuffle
#--[PROGRAM-OPTIONS]----------------------------------------------------------#
# box size params (Kpc)
box_min = 50
box_max = 300
box_step = 10
boxsizes = range(box_min, box_max, box_step)
# distance params (Mpc)
d_min = 75
d_max = 1101
d_step = (d_max-d_min)/len(boxsizes)
distances = [i/100.0 for i in range(d_min, d_max, d_step)]
# MPI params
comm = MPI.COMM_WORLD
mpisize = comm.Get_size()
rank = comm.Get_rank()
# path params
load_path1 = '/root/SHARED/Halos1/'
load_path2 = '/root/SHARED/Halos2/'
load_path3 = '/root/SHARED/Halos3/'
load_local = load_path3  # '/root/Arrays/'
save_path = '/root/SHARED/Data_out/Arrays/'
# number of projections to make for each halo
n_projections = 10
# number of jobs to go before saving array
save_interval = 5
# array params
dataarray = np.zeros((len(distances), len(boxsizes), 26))
#--[FUNCTIONS]----------------------------------------------------------------#


def rotation_matrix(ax, th):
    '''
    Input   ax      : list of two zeros and a one. ie. [0,1,0]
                    theta   : radians to rotate about the axis axs
    Output the rotation matrix to use np.dot() with
    '''
    axis = np.asarray(ax)
    theta = np.asarray(th)
    axis = axis/(np.dot(ax, ax))**2
    a = np.cos(np.divide(theta, 2.0))
    b, c, d = np.multiply(-axis, np.sin(np.divide(theta, 2.0)))
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
    return [aa+bb-cc-dd, 2.0*(bc+ad), 2.0*(bd-ac)], [2.0*(bc-ad), aa+cc-bb-dd, 2.0*(cd+ab)], [2.0*(bd+ac), 2.0*(cd-ab), aa+dd-bb-cc]


def rotate(xyz, axs, theta):
    '''
    Input   xyz     : dictionary of np.arrays
                    axs     : list of two zeros and a one. ie. [0,1,0]
                    theta   : radians to rotate about the axis axs
    Output numpy array of size three corosponding to the three new position arrays
    '''
    rot_matrix = rotation_matrix(axs, theta)
    try:
        new_xyz = np.asarray(np.dot(rot_matrix, [xyz[0], xyz[1], xyz[2]]))
    except MemoryError:
        print '     [MemoryError caught]'
        new_xyz = np.asarray(
            np.dot(rot_matrix, [xyz[0][:100], xyz[1][:100], xyz[2][:100]]))
    return [new_xyz[0], new_xyz[1], new_xyz[2]]


def kpc_box_at_distance(distance, kpc):
    '''
    Input distance in Mpc
    Output square degrees
    ((206265.0*(_kpc_/1e2)/distance*1e1)/3600)^2
    '''
    return np.square(np.divide(np.divide(206265.0*kpc, distance*1e3), 3600.0))


def calculate_nFov(kpc, d):
    '''
    Input distance in Mpc
    Output number of FOV for a box covering _kpc_ radius at input distance 
    ((square deg of box) / WFIRST square degree FOV)
    '''
    WFIRST_FOV = 0.79 * 0.43
    n_fov = round(np.divide(kpc_box_at_distance(d, kpc), WFIRST_FOV), 2)
    if n_fov < 1.0:
        return int(1)
    else:
        return n_fov


def calculate_app_mag(cutoff=26.15, t_fov=1000.0, t_exp=1000.0):
    '''
    Inputs are all defined
            --> later on this can be used to mesure differences from using 
                    longer exposures, different total times and different filters.
    Outputs apparent magnitude 
    '''
    return cutoff+2.5*np.log10(t_fov/t_exp)


def calculate_abs_mag(distance=1.0, app_mag=calculate_app_mag()):
    '''
    Input distance Mpc & apparent magnitude
    Output Absolute Magnitude limit for given distance
    '''
    return app_mag-(5.0*np.log10(distance*1e5))


def prep_stars_1(halo, keys, dlim, lpl=load_local, lp1=load_path1, lp2=load_path2, lp3=load_path3):
    '''
    Input   halo    : halo name as a string
                    keys    : list of keys to load as strings
                    dlim    : list of indices as int
                    lpl     : load local paths as a string
                    lp1     : load path 1 as a string
                    lp2     : load path 2 as a string
                    lp3     : load path 3 as a string
    Output a dictionary of numpy arrays
    '''
    if rank % 3 == 0:
        path = lp1
    elif rank % 7 == 0:
        path = lp2
    else:
        path = lp3
    if keys == ['px', 'py', 'pz']:
        arrays = []
        path = lpl
        for key in keys:
            arrays.append(np.load(path+str(halo)+'/'+str(key)+'.npy')[dlim])
        return arrays
    else:
        return np.load(path+str(halo)+'/'+str(keys[0])+'.npy')[dlim]


def load_mags(halo, mag_filter='dcmc_ks', lpl=load_local):
    '''
    Input   halo        : halo name as a string
                    mag_filter  : key for filter band to load as string * must match galaxia name
                    lpl         : load local path as a string
    Output np.array of magnitudes
    '''
    return np.load(lpl+str(halo)+'/'+mag_filter+'.npy')


def find_lims(a, b, lim_, step_=box_step):
    '''
    Input   a       : numpy array acting as the X values
                    b       : numpy array acting as the Y values
                    lim_    : box_size limit in Kpc (the size of box on it's end)
                    step_   : interval to increase box_size Kpc
    Output a list of indices for included stars
    '''
    step = step_/2.0
    lm = lim_/2.0
    alim_inr = np.logical_and(-lm < a, a < lm)
    blim_inr = np.logical_and(-lm < b, b < lm)
    inner_box = np.nonzero(np.logical_and(alim_inr, blim_inr))[0]
    alim_otr = np.logical_and(-(lm+step) < a, a < (lm+step))
    blim_otr = np.logical_and(-(lm+step) < b, b < (lm+step))
    outter_box = np.nonzero(np.logical_and(alim_otr, blim_otr))[0]
    return np.setdiff1d(outter_box, inner_box, assume_unique=True)


def find_dlims(Ks, abs_m):
    '''
    returns list of indices of stars visiable within given range
    '''
    return np.nonzero(Ks < abs_m)[0]


def mix(lst, n):
    '''
    shuffles list n times, returns shuffled list as type list
    '''
    for i in range(n):
        shuffle(lst)
    return lst


#--[OPTIONS]------------------------------------------------------------------#
# names for halo files
halos = ['halo02', 'halo05', 'halo07',
         'halo08', 'halo09', 'halo10',
         'halo12', 'halo14', 'halo15',
         'halo17', 'halo20']
# misc print
line_ = '   --------------------------'
line = '\n'+line_+'\n'
line2_ = '#############################'
line2 = '\n'+line2_+line2_+'\n'
# job counter/ticker
n_jobs_done = 1
# rotation matrix axii
r_ax1 = [0, 0, 1]
r_ax2 = [0, 1, 0]
r_ax3 = [1, 0, 0]
#--[MAIN]---------------------------------------------------------------------#
# save initial empty array
np.save(save_path+str(rank)+'array.npy', dataarray)
#--[SETUP]--------------------------------------------------------------------#
# rank 0 creates work lists for each worker
if rank == 0:
    jobs = [[k, j, distance, size] for k, distance in enumerate(
        distances) for j, size in enumerate(boxsizes)]
    mix(jobs, 2)
    cmds = [jobs[i:i + len(jobs)//mpisize+1]
            for i in range(0, len(jobs), (len(jobs)//mpisize)+1)]
    mix(cmds, 2)
else:
    cmds = None
# opening message
if rank == 0:
    print '\n', len(boxsizes), 'boxes and ', len(distances), 'distances'
    print len(cmds), 'command chunks'
    print 'jobs per chunk:'
    tot = 0
    for c in cmds:
        n = len(c)
        tot += n
        print ' --> ', n
    print tot, 'total jobs'
    print round(dataarray.nbytes/1e6, 2), 'Megabytes'
# scatter work
comm.Barrier()
work = comm.scatter(cmds, root=0)
n_jobs = len(work)*len(halos)
tot_work = len(work)
comm.Barrier()
time.sleep(rank*5)
print line, '   [RANK '+str(rank)+' STARTING]', line
#--[RUN]----------------------------------------------------------------------#
for job in work:
    #-------------------
    # create empty lists
    # running totals
    n_stars = []
    l_n_stars = []
    n_trgb_stars = []
    # mins
    teff_min = []
    age_min = []
    feh_min = []
    alpha_min = []
    # maxes
    teff_max = []
    age_max = []
    feh_max = []
    alpha_max = []
    # means
    teff_mean = []
    age_mean = []
    feh_mean = []
    alpha_mean = []
    #-------------------
    inner_mag = calculate_abs_mag(distance=job[2])
    outter_mag = calculate_abs_mag(distance=(job[2]+d_step))
    # halo operations
    for halo in halos[:2]:
        # mag cuts for (d --> d+step)
        dcmc_Ks = load_mags(halo)
        inner_set = find_dlims(dcmc_Ks, inner_mag)
        outter_set = find_dlims(dcmc_Ks, outter_mag)
        del dcmc_Ks
        # make sure there are more visable stars closer than farther
        if len(inner_set) <= len(outter_set):
            print line, '[RANK '+str(rank)+' MAG LIMIT PROBLEM]', mag_limit_inner, mag_limit_outer, line
        # limits for magnitude cut at this distance
        dlim = np.setdiff1d(inner_set, outter_set, assume_unique=True)
        # list of xyz arrays
        positions = prep_stars_1(halo, ['px', 'py', 'pz'], dlim)
        n_projections = 1
        # projection loop
        for rep in range(n_projections):
            # three different rotation angles
            theta1 = np.divide(np.multiply(
                rep*np.random.randint(90)*13*rep, np.pi), 180.0)
            theta2 = np.divide(np.multiply(
                rep*np.random.randint(90)*4*rep, np.pi), 180.0)
            theta3 = np.divide(np.multiply(
                rep*np.random.randint(90)*7*rep, np.pi), 180.0)
            # rotate coordinants 3 times
            rot_1 = rotate(positions, r_ax1, theta1)
            rot_2 = rotate(rot_1, r_ax2, theta2)
            rot_3 = rotate(rot_2, r_ax3, theta3)
            # find the included indicies of rotated stars
            lims = find_lims(rot_3[0], rot_3[1], job[3]).tolist()
            ########################
            #[END OF CONFIGURATION]#
            ########################
            #---------------------------------------------------#
            # we made it to all the way to the little square!   #
            # now... collect all the data at this configuration #
            #---------------------------------------------------#
            # number of stars for this projection
            n_stars += [len(rot_3[0][lims])]
            # log(number) of stars for this projection
            l_n_stars += [np.log10(len(rot_3[0][lims]))]
            # effective temp (log(T/Kelvin) *converted to deg Kelvin)
            temp_array = np.power(10, prep_stars_1(halo, ['teff'], dlim)[lims])
            teff_min += [np.min(temp_array)]
            teff_mean += [np.mean(temp_array)]
            teff_max += [np.max(temp_array)]
            # age (log (age/yr) *converted to Yr)
            temp_array = np.power(10, prep_stars_1(halo, ['age'], dlim)[lims])
            age_min += [np.min(temp_array)]
            age_mean += [np.mean(temp_array)]
            age_max += [np.max(temp_array)]
            # [Fe/H]
            temp_array = prep_stars_1(halo, ['feh'], dlim)[lims]
            feh_min += [np.min(temp_array)]
            feh_mean += [np.mean(temp_array)]
            feh_max += [np.max(temp_array)]
            # alpha abundances ([alpha/Fe])
            temp_array = prep_stars_1(halo, ['alpha'], dlim)[lims]
            alpha_min += [np.min(temp_array)]
            alpha_mean += [np.mean(temp_array)]
            alpha_max += [np.max(temp_array)]
            # tip of the red giant branch (TRGB)/ red clump stars
            red_giants = 0
            # compair smass to mtip
            temp_array = prep_stars_1(halo, ['smass'], dlim)[lims]
            temp_array1 = prep_stars_1(halo, ['mtip'], dlim)[lims]
            for sr, star in enumerate(temp_array):
                if star > temp_array1[sr]:
                    red_giants += 1
            n_trgb_stars += [red_giants]
            #-----------------------#
            # [END PROJECTION LOOP] #
            #-----------------------#
        # <back to halo loop>
        n_jobs -= 1
        #-----------------#
        # [END HALO LOOP] #
        #-----------------#
    # <back to job loop>
    # number of Fov
    nfov = calculate_nFov(job[3], job[2])
    # time required (hours)
    time = 1000.0*(nfov)/3600.0
    #----------------------------------------------------#
    # Load the data array with all the consolidated data #
    #----------------------------------------------------#
    # Time (hours)
    dataarray[job[0], job[1], 0] = time
    # Number of Stars (log(n_stars))
    minstars = np.min(l_n_stars)
    maxstars = np.max(l_n_stars)
    dataarray[job[0], job[1], 1] = maxstars-minstars
    dataarray[job[0], job[1], 2] = maxstars
    dataarray[job[0], job[1], 3] = np.mean(l_n_stars)
    dataarray[job[0], job[1], 4] = minstars
    # Effective Temperature (Kelvin)
    dataarray[job[0], job[1], 5] = np.mean(teff_max)-np.mean(teff_min)
    dataarray[job[0], job[1], 6] = np.mean(teff_max)
    dataarray[job[0], job[1], 7] = np.mean(teff_mean)
    dataarray[job[0], job[1], 8] = np.mean(teff_min)
    # Age (Years)
    dataarray[job[0], job[1], 9] = np.mean(age_max)-np.mean(age_min)
    dataarray[job[0], job[1], 10] = np.mean(age_max)
    dataarray[job[0], job[1], 11] = np.mean(age_mean)
    dataarray[job[0], job[1], 12] = np.mean(age_min)
    # Metallicity [Fe/H]
    dataarray[job[0], job[1], 13] = np.mean(feh_max)-np.mean(feh_min)
    dataarray[job[0], job[1], 14] = np.mean(feh_min)
    dataarray[job[0], job[1], 15] = np.mean(feh_mean)
    dataarray[job[0], job[1], 16] = np.mean(feh_max)
    # Alpha abundance [alpha/Fe]
    dataarray[job[0], job[1], 17] = np.mean(alpha_max)-np.mean(alpha_min)
    dataarray[job[0], job[1], 18] = np.mean(alpha_min)
    dataarray[job[0], job[1], 19] = np.mean(alpha_mean)
    dataarray[job[0], job[1], 20] = np.mean(alpha_max)
    # N_TRGB Stars (smass>mtip M_solar)
    dataarray[job[0], job[1], 21] = np.max(n_trgb_stars)-np.min(n_trgb_stars)
    dataarray[job[0], job[1], 22] = np.min(n_trgb_stars)
    dataarray[job[0], job[1], 23] = np.mean(n_trgb_stars)
    dataarray[job[0], job[1], 24] = np.max(n_trgb_stars)
    # N_WFIRST FoV (Int)
    dataarray[job[0], job[1], 25] = nfov
    #---------------------------------------------------------#
    # finished with array, ready to save and continue to next #
    #---------------------------------------------------------#
    # happy print statment for progress reporting
    print '    ['+str(rank)+']  ['+str(n_jobs_done)+'/'+str(tot_work)+']         d:  '+str(round(job[2], 1))+'       bs:  '+str(round(job[3], 1))+'          ['+str(round(n_jobs_done/float(tot_work)*1e2, 2))+' %]'
    # save per job interval
    if n_jobs_done % save_interval == 0:
        np.save(save_path+str(rank)+'array.npy', dataarray)
        # special print to show saved array
        print line, '     --> [', rank, '] ['+str(round(n_jobs_done/float(tot_work)*1e2, 2))+' %]', line
    # plus one on the jobdone ticker
    n_jobs_done += 1
    #----------------#
    # [END JOB LOOP] #
    #----------------#
# <back to main>
# extra save for safty
np.save(save_path+str(rank)+'array.npy', dataarray)
# end of the line print statment
print
    line2, '['+str(rank)+'][FINISHED]', line2
# process exits clean
exit(0)
