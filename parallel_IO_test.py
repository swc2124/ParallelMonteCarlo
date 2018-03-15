# ====================================================================
# Author                : swc21
# Date                  : 2018-03-14 09:42:27
# Project               : ClusterFiles
# File Name             : parallel_IO_test
# Last Modified by      : swc21
# Last Modified time    : 2018-03-14 12:03:25
# ====================================================================
# 
import datetime
import numpy as np
import time

from mpi4py import MPI

save_path = './'
load_path = 'SHARED/sols_data/'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
name = MPI.Get_processor_name()

mode = MPI.MODE_RDWR | MPI.MODE_CREATE

Filter_Types = ['dcmc_i', 'dcmc_j', 'dcmc_h', 'dcmc_ks']
Inclusion_Areas = []
for i in range(0, 300, 5):
    Inclusion_Areas.append((i, i+5))
Distances = []
for i in range(1, 100, 5):
    Distances.append(i*1e5)


def inner_most(filter_type=None, inclusion_area=None, distance=None, data=None):
    assert filter_type in ['dcmc_i', 'dcmc_j', 'dcmc_h', 'dcmc_ks']
    assert len(inclusion_area) > 1.0
    assert 0.0 <= inclusion_area[0]
    assert inclusion_area[0] < inclusion_area[1] <= 750.0
    L_now = 0.0
    for star in data:
        if not inclusion_area[0] <= star[0] <= inclusion_area[1]:
            continue
        L_now += star[1]
    return L_now


if rank == 0:
    carlos = np.zeros(
        (len(Filter_Types), len(Distances), len(Inclusion_Areas)))
    np.save(save_path+'carlos_out.npy', carlos)
    jobs = []
    for i, x in enumerate(Filter_Types):
        for k, z in enumerate(Distances):
            for j, y in enumerate(Inclusion_Areas):
                jobs.append([x, y, z, i, j, k])
    chunk = len(jobs)/size+1
    work = [jobs[x:x+chunk] for x in range(0, len(jobs), chunk)]
else:
    work = None


comm.Barrier()
work_list = comm.scatter(work, root=0)
local_filter_types = []
local_inclusion_areas = []
local_distances = []
for i in range(0, len(work_list)):
    local_filter_types.append((work_list[i][0], work_list[i][3]))
    local_inclusion_areas.append((work_list[i][1], work_list[i][4]))
    local_distances.append((work_list[i][2], work_list[i][5]))


comm.Barrier()
time.sleep(rank)
print ' --> Process:', rank, ' is on:', name, ' with', len(work_list), 'jobs'


comm.Barrier()
time.sleep(2)
comm.Barrier()
if rank == 0:
    print ''
    print '  [host]      [filter]       [area]        [distance]         [time]         [percent]'
    print ' ---------------------------------------------------------------------------------------'


comm.Barrier()
out_file = MPI.File.Open(comm, save_path+'carlos_out.npy', mode)
buffer = np.zeros((len(Filter_Types), len(Distances), len(Inclusion_Areas)))
last_filter = None
last_distance = None


comm.Barrier()
if rank == 0:
    runtime = datetime.datetime.now()


comm.Barrier()
job_counter = 0
runtime1 = datetime.datetime.now()
time.sleep(rank*0.25)
for filter_type in local_filter_types:

    if last_filter == filter_type[1]:
        print rank, 'skipped', filter_type[0]
        continue
    for distance in local_distances:
        if last_distance == distance[1]:
            print rank, 'skipped', distance[0]
            continue
        data = np.load(
            load_path+filter_type[0]+str(distance[0]/1e6)+'Mpc_dataArray.npy')
        for inclusion_area in local_inclusion_areas:
            start = datetime.datetime.now()
            x = inner_most(
                filter_type[0], inclusion_area[0], distance[0], data)
            buffer[filter_type[1], distance[1], inclusion_area[1]] = inner_most(
                filter_type[0], inclusion_area[0], distance[0], data)
            out_file.Iwrite(
                buffer[filter_type[1], distance[1], inclusion_area[1]])
            job_counter += 1
            end = datetime.datetime.now()
            out_file.Sync()
            print name, rank, '    ', filter_type[0], '     ', inclusion_area[0], 'Kpc      ', distance[0]/1e6, 'Mpc     ', end-start, ' ', job_counter, 'Jobs Done'

        out_file.Sync()
        last_distance = distance[1]

    out_file.Sync()
    last_filter = filter_type[1]

out_file.Sync()
endtime1 = datetime.datetime.now()
print 'process', rank, 'on ', name, 'has finished all', len(work_list), 'jobs in', endtime1 - runtime1
comm.Barrier()
out_file.Close()
if rank == 0:
    endtime = datetime.datetime.now()
    print 'Program Finished in', endtime-runtime
