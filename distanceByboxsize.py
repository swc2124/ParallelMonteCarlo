# ====================================================================
# Author 				: swc21
# Date 					: 2018-03-14 09:42:10
# Project 				: ClusterFiles
# File Name 			: distanceByboxsize
# Last Modified by 		: swc21
# Last Modified time 	: 2018-03-14 11:00:05
# ====================================================================
#
#--[IMPORTS]------------------------------------------------------------------#

import numpy as np

from halo_functions import calculate_nFov
from halo_functions import data_array
from halo_functions import dlims_indicies
from halo_functions import final_indicies
from halo_functions import find_and_count
from halo_functions import jobs
from halo_functions import n_figures as n_slices
from halo_functions import rotate_stars
from halo_functions import save_path
from mpi4py import MPI
from time import sleep

#--[PROGRAM-OPTIONS]----------------------------------------------------------#

# MPI params
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

#--[OPTIONS]------------------------------------------------------------------#


#--[MAIN]---------------------------------------------------------------------#

if rank == 0:
    import os
    from operator import itemgetter
    from collections import OrderedDict
    failed = 0
    log_book = {}
    status = 'STARTING'
    commands = jobs(rank, comm.Get_size())
    for s, i in enumerate(commands):
        d = comm.recv(source=MPI.ANY_SOURCE)
        if d:
            failed += d[4]
            log_book[0] = (len(commands)-s, rank, name, status, failed)
            if d[3] == True:
                status = 'SAVED'
            if d[0] in log_book.keys():
                if d[4] > log_book[d[0]][-1]:
                    status = 'FAILED'
            log_book[d[0]] = (d[2], d[0], d[1], status, d[4])
            comm.send(i, dest=d[0])
            status = 'RUNNING'
            d_view = [(v, k) for k, v in log_book.iteritems()]
            d_view.sort(reverse=True)
            os.system('clear')
            for v, k in d_view:
                print v
            sleep(.01)

    status = 'CLOSED'
    for i in range(size-1):
        d = comm.recv(source=MPI.ANY_SOURCE)
        log_book[d[0]] = (d[0], d[1], d[2], status)
        comm.send('KILL', dest=d[0])
        os.system('clear')
        for i in log_book.keys():
            print log_book[i]
    log_book[0] = (rank, name, s, status)
    os.system('clear')
    for i in log_book.keys():
        print log_book[i]
    print '\nclosing program\n'
    exit(0)
else:
    k = 0
    job_log = {}
    save = False
    fail = 0
    while True:
        comm.send((rank, name, k, save, fail), 0)
        job = comm.recv(source=0)
        if job:
            if job == 'KILL':
                exit(0)
            try:
                d_lims = dlims_indicies(job[3], job[1], rank)
                px, py, pz = rotate_stars(job[3], d_lims, job[4], rank)
                lims = final_indicies(px, py, job[2])
                n_fov = calculate_nFov(job[8], job[7])
                job_log[str(job[0])] = [	job[6],
                                         find_and_count(
                                             job[3], job[5], lims, rank),
                                         1000.0*(n_fov)/3600.0,
                                         n_fov
                                         ]
                k += 1
                save = False
                if k % 100 == 0:
                    np.save(
                        save_path+'rank['+str(rank)+']_job_log.npy', job_log)
                    save = True
            except Exception, e:
                fail += 1
                with open('./'+str(name)+'_errors', 'w') as log:
                    log.write(e)
    exit(0)
#-----------------------------------------------------------------------------#
