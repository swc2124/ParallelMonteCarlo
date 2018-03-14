# ====================================================================
# Author                : swc21
# Date                  : 2018-03-14 09:43:12
# Project               : ClusterFiles
# File Name             : carlos_parallel2
# Last Modified by      : swc21
# Last Modified time    : 2018-03-14 11:45:21
# ====================================================================
# 
# carlos_parallel2.py   sol courtney, Columbia U. march 2016
import numpy as np
import datetime
import time
from mpi4py import MPI
filter_types = ['dcmc_i', 'dcmc_j', 'dcmc_h', 'dcmc_ks']
inclusion_areas = []
for i in range(0, 100, 5):
    inclusion_areas.append((i, i+5))
distances = []
for i in range(1, 100, 4):
    distances.append(i*1e5)
save_path = 'SHARED/sols_testplots/'
carlos = np.zeros((len(filter_types), len(inclusion_areas), len(distances)))


def inner_most(filter_type='dcmc_j', inclusion_area=750, distance=1e3, data=None):
    L_now = 0.0
    for star in data:
        if not inclusion_area[0] <= star[0] <= inclusion_area[1]:
            continue
        elif distance >= star[3]:
            continue
        L_now += star[1]
    return L_now


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()
if rank == 0:
    jobs = []
    for i, x in enumerate(filter_types):
        for j, y in enumerate(inclusion_areas):
            for k, z in enumerate(distances):
                jobs.append([x, y, z, i, j, k])
    chunk = len(jobs)/size+1
    work = [jobs[x:x+chunk] for x in range(0, len(jobs), chunk)]
else:
    work = None
work_list = comm.scatter(work, root=0)
if rank == 0:
    print ' '
    print ' [host]      [filter]       [area]        [distance]         [time]'
    print ' -----------------------------------------------------------------------'
runtime = datetime.datetime.now()
for i in range(0, len(work_list)):
    start = datetime.datetime.now()
    # ,  dtype='float32', mode='c', shape=(18208538, 4))
    data = np.load('/root/SHARED/sols_data/'+work_list[i][0]+'_dataArray.npy')
    carlos[work_list[i][3], work_list[i][4], work_list[i][5]] = inner_most(
        work_list[i][0], work_list[i][1], work_list[i][2], data)
    #np.save(save_path+work_list[i][0]+'_'+str(work_list[i][1])+'_'+str(work_list[i][2])+'.npy', carlos)
    end = datetime.datetime.now()
    print name, '   ', work_list[i][0], '     ', work_list[i][1], 'Kpc      ', work_list[i][2]/1e6, 'Mpc     ', end-start
np.save(save_path+str(rank)+str(name), carlos)
endtime = datetime.datetim.now()
print 'Program finished in:', endtime-runtime
