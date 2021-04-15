"""
Script to read gauge data and calculate errors relative to the baseline run for the batch runs
"""
# ============================================================================
#      Copyright (C) 2013 Kyle Mandli <kyle@ices.utexas.edu>
#
#          Distributed under the terms of the MIT license
#                http://www.opensource.org/licenses/
#
# (post process scripts by Akshay Sriapda <as4928@columbia.edu>, 2017)
# ============================================================================


from collections import defaultdict
import numpy
import matplotlib.pyplot as plt
import os
import glob



def extract_level_data(data,var,level):
    # get gauge data at particular levels
    lev_data = []
    for i in data:
        if i[0] == level:
            lev_data.append([i[1],i[var]])
    return numpy.array(lev_data)

def plot_data(test,base,gauge,level,dir):
    # plots all the h,hu,hv, Eta vs time 
    directory = dir+'/Plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    colors = ['or','ob','oy','om','oc','ok','or','ob','-g']
    labels = ['Baseline','Level ']
    marker_size = 7
    width = 3
    plt.ioff()
    fig, ((ax1, ax2,ax3, ax4)) = plt.subplots(4)
    
    ax1.plot(base[:,1],base[:,2],'-k',linewidth = width,markerfacecolor = 'w',markersize = marker_size)
    for k in range(1,level+1):
        lev_data = extract_level_data(test,2,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax1.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax1.set_title('h')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Distance")
    
    ax2.plot(base[:,1],base[:,3],'-k',linewidth = width,markerfacecolor = 'w',markersize = marker_size)
    for k in range(1,level+1):
        lev_data = extract_level_data(test,3,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax2.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax2.set_title('hu')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Momentum")
    
    ax3.plot(base[:,1],base[:,4],'-k',linewidth = width,markerfacecolor = 'w',markersize = marker_size)
    for k in range(1,level+1):
        lev_data = extract_level_data(test,4,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax3.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax3.set_title('hv')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Momentum")

    ax4.plot(base[:,1],base[:,5],'-k',linewidth = width,markerfacecolor = 'w',markersize = marker_size,label= labels[0])
    for k in range(1,level+1):
        lev_data = extract_level_data(test,5,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax4.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size,label= labels[1]+str(k))
    ax4.set_title('Eta')
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Distance")
    
    plt.legend(bbox_to_anchor=(0, -0.2, 1., -0.2), loc=2,ncol=4, mode="expand", borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(directory+str(gauge)+'_Data.png')

def plot_error(gauge,error,level,dir):
    # plots error relative to basseline for the four unkowns
    directory = dir+'/Plots/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    colors = ['or','ob','oy','om','oc','ok','or','ob','-g']
    labels = ['Baseline','Level ']
    marker_size = 7
    width = 3
    plt.ioff()
    fig, ((ax1, ax2,ax3, ax4)) = plt.subplots(4)
    
    for k in range(1,level+1):
        lev_data = extract_level_data(error,2,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax1.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax1.set_title('h')
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error")
    
    for k in range(1,level+1):
        lev_data = extract_level_data(error,3,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax2.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax2.set_title('hu')
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Error")
    
    for k in range(1,level+1):
        lev_data = extract_level_data(error,4,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax3.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size)
    ax3.set_title('hv')
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Error")

    for k in range(1,level+1):
        lev_data = extract_level_data(error,5,k)
        if len(lev_data) > 0:
            if k == level:
                c = colors[-1]
            else:
                c = colors[k-1]
            ax4.plot(lev_data[:,0],lev_data[:,1],c,linewidth = width,markersize = marker_size,label= labels[1]+str(k))
    ax4.set_title('Eta')
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Error")
    
    plt.legend(bbox_to_anchor=(0, -0.2, 1., -0.2), loc=2,ncol=4, mode="expand", borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig(directory+str(gauge)+'_Error.png')


def norm_calc(error,level,max_quants,debug_file):
    # calculates error norm value which is normalized by the largest value in the dataset
    level_error = numpy.empty([4,3])
    
    if level == 0:
        for i in range(2,6):
            level_error[i-2,0] = numpy.linalg.norm(error[:,i]/max_quants[i-2], ord=1)
            level_error[i-2,1] = numpy.linalg.norm(error[:,i]/max_quants[i-2], ord=2)
            level_error[i-2,2] = numpy.linalg.norm(error[:,i]/max_quants[i-2], ord=numpy.inf)
    	
    else:
        
        for i in range(2,6):
            lev_error = extract_level_data(error,i,level)
	    
            if len(lev_error) == 0:
                #print 'No data found for level '+str(level)+ ' '
                level_error[i-2,0] = 0.0
                level_error[i-2,1] = 0.0
                level_error[i-2,2] = 0.0
            else:
                level_error[i-2,0] = numpy.linalg.norm(lev_error[:,1]/max_quants[i-2], ord=1)
                level_error[i-2,1] = numpy.linalg.norm(lev_error[:,1]/max_quants[i-2], ord=2)
                level_error[i-2,2] = numpy.linalg.norm(lev_error[:,1]/max_quants[i-2], ord=numpy.inf)
    
    return [level_error,debug_file]

def interpolate(test,base_0,base_1):
    # interpolates data points in the baseline for time steps not recorded
    x = test[1]
    x_0 = base_0[1]
    x_1 = base_1[1]
    
    x_ratio = (x - x_0)/(x_1 - x_0)
    base = [0,0,0,0]
    for i in range(2,6):
        base[i-2] = (x_ratio*(base_1[i] - base_0[i])) + base_0[i]
    
    return base

def error_calc(test,base,gauge,number_of_levels,dir,summary_file,output_file,debug_file):
    # Actual error calculations done for each timestep the data is recorded
    for i in range(0,len(test)):
        if test[i,1] > base[-1,1]:
            cutoff_location = i-1
            break
        else:
            cutoff_location = i

    error = numpy.empty(test.shape)
    interpolate_base = numpy.empty([cutoff_location,6])
    
    for i in range(0,cutoff_location):
        for j in range(0,len(base)):
            if test[i,1] == base[j,1]:
                interpolate_base[i] = base[j]
                break
            elif base[j,1] > test[i,1]:
                interpolate_base[i,0:2] = test[i,0:2]
                interpolate_base[i,2:6] = interpolate(test[i,:],base[j-1,:],base[j,:])
                break

    error = numpy.empty(interpolate_base.shape)
    error[:,0:2] = interpolate_base[:,0:2]
    error[:,2:6] = numpy.abs(test[:cutoff_location,2:6] - interpolate_base[:,2:6])


    max_quants = []
    for k in range(2,6):
        max_quants.append(numpy.max(base[:,k]))

    for i in range(0,int(number_of_levels)+1):
        
        [norm_error,debug_file] = norm_calc(error,i,max_quants,debug_file)
        
        if i == 0:
            output_file.write('All levels L1 Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,0])+ '\n')
            output_file.write('All levels L2 Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,1])+ '\n')
            output_file.write('All levels Infinity Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,2]) + '\n')
            summary_file.write(str(norm_error[3,0])+' '+str(norm_error[3,1])+' '+str(norm_error[3,2]))
        else:
            output_file.write('Level '+str(i)+' L1 Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,0])+ '\n')
            output_file.write('Level '+str(i)+' L2 Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,1])+ '\n')
            output_file.write('Level '+str(i)+' Infinity Norm error for gauge '+str(gauge)+' is '+str(norm_error[:,2]) + '\n')

    return [output_file,summary_file,debug_file]

def get_num_of_levels(root_path,sweep_data):
    # Uses code from Clawpack file plot_num_grids.py to calculate the number of cells used at each output time step
    for k in range(0,len(sweep_data)):
    
        output_path = root_path+'sweep_'+str(k)+'_output/fort.q*'
        num_levels = sweep_data[k][2]
        file_list = glob.glob(output_path)
        time = numpy.empty(len(file_list), dtype=float)
        num_grids = numpy.zeros((time.shape[0], num_levels), dtype=int)
        num_cells = numpy.zeros((time.shape[0], num_levels), dtype=int)
    
        for (n,path) in enumerate(file_list):
            t_path = path[:-5] + "t" + path[-4:]
            t_file = open(t_path, 'r')
            time[n] = float(t_file.readline().split()[0])
            t_file.readline()
            t_file_num_grids = int(t_file.readline().split()[0])
            t_file.close()
            q_file = open(path, 'r')
            line = "\n"
            while line != "":
                line = q_file.readline()
                if "grid_number" in line:
                    level = int(q_file.readline().split()[0])
                    num_grids[n,level-1] += 1
                    mx = int(q_file.readline().split()[0])
                    my = int(q_file.readline().split()[0])
                    num_cells[n,level-1 ] += mx * my
            q_file.close()
            if numpy.sum(num_grids[n,:]) != t_file_num_grids:
                raise Exception("Number of grids in fort.t* file and fort.q* file do not match.")
        
        numpy.savetxt('post-process-data/num_cells_run_'+str(k)+'.txt',num_cells)
    numpy.savetxt('post-process-data/output-times.txt',time)

if __name__ == "__main__":
	# point output_path to the output folder from the scenarios
	output_path = '../../scratch/Tohoku-hawaii/tohoku-hawaii-scenario/all-scenarios-1/'

	# point run_data_path to where the run details text file is located.
	# If there is no text file with the details, create an array with the deatils
	run_data_path = '../../scratch/Tohoku-hawaii/'
	sweep_data = numpy.loadtxt(run_data_path+'run-data.txt')

	# point log_path to where all the run log files are located
	log_path = '../../scratch/Tohoku-hawaii/'

	# All the post processed data will be stored in a new folder names post-process-data
	if not os.path.exists('post-process-data'):
	    os.makedirs('post-process-data')

    # summary file contains error norm and run time data, used for visualizing
	summary_file = open('post-process-data/summary-data.txt','w')
    # output file logs all the erros for each gauge for each sweep
	output_file = open('post-process-data/output-data.txt','w')
    # debug file can be used to log the parameters passed through the various functions
	debug_file = open('post-process-data/debug-data.txt','w')

	gauges_data = open(output_path+'sweep_0_output/gauges.data')
	get_num_of_levels(output_path,sweep_data)
	gauge_list = []
	count = 0
    
	for i in gauges_data:
	    l = i.strip().split(' ')
	    count += 1
	    if count >= 8 and l == ['']:
	        break
	    elif count>= 8:
	        gauge_list.append(int(l[0]))

	gauge_list.sort()
	baseline_gauges = defaultdict(list)

	for i in gauge_list:
	    
	    l = len(str(i))
	    if l < 5:
		   t = '0'*(5-l)+str(i)
	    else:
		   t = str(i)
	    baseline_gauges[i] = numpy.loadtxt(output_path+'sweep_0_output/gauge' + t +'.txt')
	    baseline_gauges[i] = numpy.array(baseline_gauges[i])

	log_file = open(output_path + 'sweep_0_log.txt')
	run_number = log_file.readline().strip().split('.')[0]
	base_log_data = open(log_path+'sweep_0.o'+run_number)

	number_of_tests = len(sweep_data) - 1

	for line in base_log_data:
	    l = line.strip().split()
	    if not l:
	        continue
	    elif l[0] == 'total':
	        base_total_cells = l[3]
	        output_file.write('Total cells for baseline run '+base_total_cells+ '\n')
	    elif l[0] == 'Total' and l[1] == 'time:':
	        base_total_time = l[2]
	        output_file.write('Total wall time for baseline run '+base_total_time+ '\n')
	    elif l[0] == 'Regridding':
	        base_regridding_time = l[1]
	        output_file.write('Regridding time for baseline run '+base_regridding_time+ '\n')

	summary_file.write(str(number_of_tests)+' '+str(len(gauge_list))+' '+ str(base_total_cells) +'\n')
	                  
	for i in range(0,number_of_tests):
	    dir = 'post-process-data/test-no-'+str(i+1)
	    if not os.path.exists(dir):
	        os.makedirs(dir)
	              
	    max_level = sweep_data[i+1][2]
	    
	    output_file.write('The Sweep number '+str(i+1)+' had the following data: '+ '\n')
	    output_file.write('Grid s`  ize: '+str(sweep_data[i+1][0])+' by '+ str(sweep_data[i+1][1])+ '\n')
	    output_file.write('The max AMR Level is '+ str(max_level)+ '\n')
	    output_file.write('The AMR Levels used are '+ str(sweep_data[i+1][3:]) +'\n')

	    log_file = open(output_path + 'sweep_'+str(i+1)+'_log.txt')
	    run_number = log_file.readline().strip().split('.')[0]
	    test_log_data = open(log_path+'sweep_'+str(i+1)+'.o'+run_number)

	    for line in test_log_data:
	        l = line.strip().split()
	        if not l:
	            continue
	        elif l[0] == 'total':
	            test_total_cells = l[3]
	            output_file.write('Total cells for test run '+base_total_cells+ '\n')
	        elif l[0] == 'Total' and l[1] == 'time:':
	            test_total_time = l[2]
	            summary_file.write('0 '+str(test_total_time)+' '+str(test_regridding_time)+' '+str(test_total_cells)+' '+str(i+1)+'\n')
	            output_file.write('Total wall time for test run '+test_total_time+ '\n')

	        elif l[0] == 'Regridding':
	            test_regridding_time = l[1]
	            output_file.write('Regridding time for test run '+test_regridding_time +'\n')

	    test_gauges = defaultdict(list)

	    for j in gauge_list:
		test_path = output_path+'sweep_'+str(i+1)+'_output/'
	    	l = len(str(j))
	    	if l < 5:
		    t = '0'*(5-l)+str(j)
	    	else:
		    t = str(j)

	        test_gauges[j] = numpy.loadtxt(test_path+'gauge' + t +'.txt')
	    
	              
	    for j in gauge_list:
	        test_gauges[j] = numpy.array(test_gauges[j])
	        summary_file.write('g ')
	        [output_file,summary_file,debug_file] = error_calc(test_gauges[j],baseline_gauges[j],j,max_level,dir,summary_file,output_file,debug_file)
	        summary_file.write('\n')
	    
	summary_file.write('b '+str(base_regridding_time)+' '+str(base_total_time)+'\n\n')
	print 'Error Analysis done'
