"""
Script to analyze and visualze gauge errors for the batch runs
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
from matplotlib import cm
import os
import math
from mpl_toolkits.mplot3d import Axes3D

def plot_summary(time,L1_error,L2_error,Inf_error,no_of_gauges,no_of_sweeps,regrid_time):
    # plots all the three norm errors for all gauges in three subplots. 
    # Colors represnt different sweeps
    # Markers represent different gauges 
    #Marker size is relative to regridding time
    sweep_labels = ['r','b','g','c','m','y','k']
    gauge_labesls = ['o','s','d','^','v','>','+','p','h','o','s']
    
    regrid_marker_fac = 5
    plot_count = 0
    fig, (ax1,ax2,ax3) = plt.subplots(3)
    #fig, (ax1) = plt.subplots(1)

    for i in range(0,no_of_gauges):
        for j in range(0,no_of_sweeps):
            if (i==0):
                ax1.plot(time[j],L1_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax2.plot(time[j],L2_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax3.plot(time[j],Inf_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],label='Scenario '+str(j+1),markersize=5+(regrid_time[j]*regrid_marker_fac))
            
            elif (j==0):
                ax1.plot(time[j],L1_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax2.plot(time[j],L2_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax3.plot(time[j],Inf_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],label='Gauge '+str(i+1),markersize=5+(regrid_time[j]*regrid_marker_fac))
            else:
                ax1.plot(time[j],L1_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax2.plot(time[j],L2_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))
                ax3.plot(time[j],Inf_error[j,i],sweep_labels[j%len(sweep_labels)]+gauge_labesls[i%len(gauge_labesls)],markersize=5+(regrid_time[j]*regrid_marker_fac))

    ax1.set_title('L1 Norm Error')
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel("L1 Norm Error")
    ax2.set_title('L2 Norm Error')
    ax2.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax2.set_ylabel("L2 Norm Error")
    ax3.set_title('Inf Norm Error')
    ax3.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax3.set_ylabel("Inf Norm Error")

    plt.legend(bbox_to_anchor=(0, -0.2, 1., -0.2), loc=2,ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(plot_path+'tohoku-error.png',bbox_inches='tight')
    
def get_avg_bound(L1_error,L2_error,Inf_error,no_of_sweeps):
    # calcualtes the max and min errors for each sweep and which gauge the max and min occurs at.
    # Also the average error over all the gauges for each sweep
    L1_info = numpy.empty([6,no_of_sweeps])
    L2_info = numpy.empty([6,no_of_sweeps])
    Inf_info = numpy.empty([6,no_of_sweeps])
    
    for j in range(0,no_of_sweeps):
        L1_info[0,j] = numpy.mean(L1_error[j,:])
        L1_info[1,j] = numpy.std(L1_error[j,:])
        L1_info[2,j] = numpy.amax(L1_error[j,:])
        L1_info[3,j] = L1_error[j,:].tolist().index(numpy.amax(L1_error[j,:]))
        L1_info[4,j] = numpy.amin(L1_error[j,:])
        L1_info[5,j] = L1_error[j,:].tolist().index(numpy.amin(L1_error[j,:]))
        
        L2_info[0,j] = numpy.mean(L2_error[j,:])
        L2_info[1,j] = numpy.std(L2_error[j,:])
        L2_info[2,j] = numpy.amax(L2_error[j,:])
        L2_info[3,j] = L2_error[j,:].tolist().index(numpy.amax(L2_error[j,:]))
        L2_info[4,j] = numpy.amin(L2_error[j,:])
        L2_info[5,j] = L2_error[j,:].tolist().index(numpy.amin(L2_error[j,:]))
    
        Inf_info[0,j] = numpy.mean(Inf_error[j,:])
        Inf_info[1,j] = numpy.std(Inf_error[j,:])
        Inf_info[2,j] = numpy.amax(Inf_error[j,:])
        Inf_info[3,j] = Inf_error[j,:].tolist().index(numpy.amax(Inf_error[j,:]))
        Inf_info[4,j] = numpy.amin(Inf_error[j,:])
        Inf_info[5,j] = Inf_error[j,:].tolist().index(numpy.amin(Inf_error[j,:]))

    highest_error_sweep = L1_info[0,:].tolist().index(numpy.amax(L1_info[0,:]))
    lowest_error_sweep = L1_info[0,:].tolist().index(numpy.amin(L1_info[0,:]))
    return L1_info,L2_info,Inf_info,lowest_error_sweep,highest_error_sweep

def plot_error_bar(time,l1_data,l2_data,inf_data,no_of_sweeps,error_bars):
    # plots the max and min errors for each sweep as a error bar plot with gauge nos 
    #fig, (ax1,ax2,ax3) = plt.subplots(3)
    fig, (ax1) = plt.subplots(1)
    
    if(error_bars):
        ax1.errorbar(time,l1_data[0,:],yerr=[numpy.subtract(l1_data[0,:],l1_data[4,:]),numpy.subtract(l1_data[2,:],l1_data[0,:])],ls='',marker='o',color='r',markersize=5)
        #ax2.errorbar(time,l2_data[0,:],yerr=[numpy.subtract(l2_data[0,:],l2_data[4,:]),numpy.subtract(l2_data[2,:],l2_data[0,:])],ls='',marker='o',color='r',markersize=5)
        #ax3.errorbar(time,inf_data[0,:],yerr=[numpy.subtract(inf_data[0,:],inf_data[4,:]),numpy.subtract(inf_data[2,:],inf_data[0,:])],ls='',marker='o',color='r',markersize=5)
    else:
        ax1.plot(time,l1_data[0,:],'w',markersize=3)
        #ax2.plot(time,l2_data[0,:],'w',markersize=3)
        #ax3.plot(time,inf_data[0,:],'w',markersize=3)
    
    for i in range(0,no_of_sweeps):
        ax1.text(time[i]+0.2,l1_data[0,i],str(i+1),fontsize = 10)
        ax1.text(time[i],l1_data[2,i]+0.01,str(l1_data[3,i]+1),fontsize = 10)
        ax1.text(time[i],l1_data[4,i]-0.01,str(l1_data[5,i]+1),fontsize = 10)
        
        #ax2.text(time[i]+0.2,l2_data[0,i],str(i+1),fontsize = 10)
        #ax2.text(time[i],l2_data[2,i]+0.1,str(l2_data[3,i]+1),fontsize = 10)
        #ax2.text(time[i],l2_data[4,i]-0.1,str(l2_data[5,i]+1),fontsize = 10)
        
        #ax3.text(time[i]+0.2,inf_data[0,i],str(i+1),fontsize = 10)
        #ax3.text(time[i],inf_data[2,i]+0.1,str(inf_data[3,i]+1),fontsize = 10)
        #ax3.text(time[i],inf_data[4,i]-0.1,str(inf_data[5,i]+1),fontsize = 10)

    ax1.set_title('Error bars with gauge IDs')
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel("L1 Norm Error")
    ax1.set_ylim([-0.1,1])
    #ax2.set_title('L2 Norm Error')
    #ax2.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    #ax2.set_ylabel("L2 Norm Error")
    #ax2.set_ylim([-0.1,1])
    #ax3.set_title('Inf Norm Error')
    #ax3.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    #ax3.set_ylabel("Inf Norm Error")
    #ax3.set_ylim([-0.1,1])

    plt.tight_layout()
    if(error_bars):
        plt.savefig(plot_path+'gauge_details_with_error_bar.png',bbox_inches='tight')
    else:
        plt.savefig(plot_path+'gauge_details_no_error_bar.png',bbox_inches='tight')
                    
def plot_error_average(time,data,no_of_sweeps,max_level,error_type):
    # plots the error average of all gauges for each sweep. error_type can handle different norms 
	colour = ['r','b','g','c','m','y']
	count = defaultdict(list)
	for i in range(number_of_sweeps):
		count[max_level[0,i]] = 0

	counter = 0

	fig, (ax1) = plt.subplots(1)
	                
	for i in range(no_of_sweeps):
		if (count[max_level[0,i]] == 0):
			ax1.plot(time[i],data[i],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
			count[max_level[0,i]] += 1
		else:
			ax1.plot(time[i],data[i],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')
	   	counter = counter + 1
	    
	   	if(counter%2 == 0 ):
	   		ax1.text(time[i],data[i]+0.05,str(i+1),fontsize = 10)
	   	else:	
	   		ax1.text(time[i],data[i]-0.075,str(i+1),fontsize = 10)
	    
	ax1.set_title(error_type+" Norm Error Averages")
	ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
	ax1.set_ylabel(error_type+" Norm Error")
	ax1.set_ylim([-0.1,1.1])
	plt.tight_layout()
	ax1.legend(loc=1)
	plt.savefig(plot_path+error_type+'-average.png',bbox_inches='tight')
	                
def plot_error_std(time,average,std,lower,upper,no_of_sweeps,error_type):
    # plots the standard deviation of errors of all gauges for each sweep. error_type can handle different norms 
    fig, (ax1) = plt.subplots(1)
    ax1.errorbar(time,average,yerr=[numpy.subtract(average,lower),numpy.subtract(upper,average)],ls='',marker='o',color='r',markersize=5)
    
    for i in range(0,no_of_sweeps):
        ax1.text(time[i],average[i],str(round(std[i],3)),fontsize=10)
        
    ax1.set_title(error_type+" Norm error bars")
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel(error_type+" Norm Error")
    ax1.set_ylim([-0.1,1.1])
    
    plt.tight_layout()
    plt.savefig(plot_path+error_type+'-standard-deviation.png',bbox_inches='tight')


def plot_number_cells(time,data,no_of_sweeps):
    # plots the total number of cells used for each sweep
    fig, (ax1) = plt.subplots(1)
    
    scale = len(str(int(max(data)))) - 1
    count = 0
    for i in range(0,no_of_sweeps):
        ax1.plot(time[i],data[i],'ow',markersize = 0.5)
        count = count + 1
        if(count%2 == 0 ):
            ax1.text(time[i],data[i]+0.005*count*10**scale,str(i+1),fontsize = 10)
        else:
            ax1.text(time[i],data[i]-0.005*count*10**scale,str(i+1),fontsize = 10)
                    
        ax1.set_title('Number of Cells')
        ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
        ax1.set_ylabel("Number of Cells")
        ax1.set_ylim([min(-0.1,min(data)-0.1),max(1.1,max(data)+0.1)])
        plt.tight_layout()
        plt.savefig(plot_path+'Number-of-Cells.png',bbox_inches='tight')
    

def plot_regrid_time(time,data,no_of_sweeps,error_type):
    #plots the regridding time used for each sweep
    fig, (ax1) = plt.subplots(1)
    
    count = 0
    for i in range(0,no_of_sweeps):
        ax1.plot(time[i],data[i],'ow',markersize = 0.5)
        count = count + 1
        if(count%2 == 0 ):
            ax1.text(time[i],data[i],str(i+1),fontsize = 10)
        else:
            ax1.text(time[i],data[i],str(i+1),fontsize = 10)
    
    ax1.set_title(error_type+' Norm Error vs Regridding time')
    ax1.set_xlabel("Regridding time")
    ax1.set_ylabel(error_type+" Norm Error")
    #ax1.set_ylim([0.1,1.1])
    
    plt.tight_layout()
    plt.savefig(plot_path+error_type+'-regrid-time.png',bbox_inches='tight')
                    
def plot_features(time,l1_data,l2_data,inf_data,no_of_gauges,no_of_sweeps,regrid_time,number_of_total_cell_updates,max_level):

    #All the error types with gauge numbers for the largest and smallest error for each sweep
    # error_bars = True, plots with error bars, if false no error bars.
    plot_error_bar(time,l1_data,l2_data,inf_data,no_of_sweeps,error_bars = True)
                    
    # Plotting the error avergaes with sweep numbers
    plot_error_average(time,l1_data[0,:],no_of_sweeps,max_level,'L1')
    plot_error_average(time,l2_data[0,:],no_of_sweeps,max_level,'L2')
    plot_error_average(time,inf_data[0,:],no_of_sweeps,max_level,'Inf')

    # Plotting error avergaes with standard deviation and error bars
    plot_error_std(time,l1_data[0,:],l1_data[1,:],l1_data[2,:],l1_data[4,:],no_of_sweeps,'L1')
    plot_error_std(time,l2_data[0,:],l2_data[1,:],l2_data[2,:],l2_data[4,:],no_of_sweeps,'L2')
    plot_error_std(time,inf_data[0,:],inf_data[1,:],inf_data[2,:],inf_data[4,:],no_of_sweeps,'Inf')

    # Plotting errors vs the regirding times for each sweep
    plot_regrid_time(regrid_time,l1_data[0,:],no_of_sweeps,'L1')
    plot_regrid_time(regrid_time,l2_data[0,:],no_of_sweeps,'L2')
    plot_regrid_time(regrid_time,inf_data[0,:],no_of_sweeps,'Inf')
    
    # Plotting the number of cells used for each sweep
    plot_number_cells(time,number_of_total_cell_updates,no_of_sweeps)


def plot_cell_ratios(output_time,number_of_sweeps,number_of_cells,refinemnt_ratios,max_level,grid_size,lowest_error_sweep,highest_error_sweep,highest_cost_sweep,lowest_cost_sweep,data,cost,time):
    # calculates the cell ratios and plots them vs various other quantities
    cell_number_ratio_sum = numpy.zeros([number_of_sweeps,len(output_time)])
    cell_number_ratio_avg = numpy.zeros([number_of_sweeps,1])
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    count = defaultdict(list)
    for i in range(number_of_sweeps):
        count[max_level[0,i]] = 0

    for i in range(0,number_of_sweeps):
        for j in range(0,len(output_time)):
            ratio = 0
            for k in range(0,int(max_level[0,i])-1):
                
                if (number_of_cells[i+1][j,k+1] == 0 and number_of_cells[i+1][j,k] == 0):
                    ratio += 0.0
                else:
                    ratio += (number_of_cells[i+1][j,k+1]) / (number_of_cells[i+1][j,k] * float(refinemnt_ratios[i,k])**2)
            cell_number_ratio_sum[i,j] = ratio
    
        if (count[max_level[0,i]] == 0):
            ax1.plot(output_time/3600,cell_number_ratio_sum[i,:],'-'+colour[int(max_level[0,i]-numpy.amin(max_level))],linewidth=2,label='Level '+str(max_level[0,i]))
            count[max_level[0,i]] += 1
        else:
            ax1.plot(output_time/3600,cell_number_ratio_sum[i,:],'-'+colour[int(max_level[0,i]-numpy.amin(max_level))],linewidth=2)

        cell_number_ratio_avg[i,0] = numpy.mean(cell_number_ratio_sum[i,:])
    numpy.savetxt(plot_path+'cell_uti_ratios.txt',cell_number_ratio_sum)
    ax1.plot(output_time/3600,cell_number_ratio_sum[lowest_error_sweep,:],'wo',markersize=5,linewidth=3,label='lowest error sweep '+str(lowest_error_sweep+1))
    ax1.plot(output_time/3600,cell_number_ratio_sum[highest_error_sweep,:],'ko',markersize=5,linewidth=3,label='highest error sweep '+str(highest_error_sweep+1))
    ax1.plot(output_time/3600,cell_number_ratio_sum[lowest_cost_sweep,:],'ws',markersize=5,linewidth=3,label='lowest cost sweep '+str(lowest_cost_sweep+1))
    ax1.plot(output_time/3600,cell_number_ratio_sum[highest_cost_sweep,:],'ks',markersize=5,linewidth=3,label='highest cost sweep '+str(highest_cost_sweep+1))

    ax1.set_title('Cell Utilization Ratios over Scenarios')
    ax1.set_xlabel("Output Times (hrs)")
    ax1.set_ylabel("Cell Utilization Ratios")
    #ax1.legend(loc=1)
    ax1.legend(bbox_to_anchor=(0, -0.05, 1., -0), loc=2,ncol=3, mode="expand", borderaxespad=0.)
    plt.tight_layout()
    plt.savefig(plot_path+'cell-ratios.png',bbox_inches='tight')

    for i in range(number_of_sweeps):
        count[max_level[0,i]] = 0
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    for i in range(0,number_of_sweeps):
    	if (count[max_level[0,i]] == 0):
    		ax1.plot(time[i],numpy.mean(cell_number_ratio_sum[i,:]),colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
    		count[max_level[0,i]] += 1
    	else:
    		ax1.plot(time[i],numpy.mean(cell_number_ratio_sum[i,:]),colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')
    	ax1.text(time[i],numpy.mean(cell_number_ratio_sum[i,:])+0.2,str(i+1))
    ax1.set_title('Average Cell Utilization Ratios vs Wall Time')
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel("Cell Utilization Ratio")
    #ax1.legend(loc=1)
    ax1.legend(loc=1)
    plt.tight_layout()
    plt.savefig(plot_path+'time-vs-cellratios.png',bbox_inches='tight')

    for i in range(number_of_sweeps):
        count[max_level[0,i]] = 0
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    for i in range(0,number_of_sweeps):
    	if (count[max_level[0,i]] == 0):
    		ax1.plot(data[0,i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
    		count[max_level[0,i]] += 1
    	else:
    		ax1.plot(data[0,i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')

    	ax1.text(data[0,i],numpy.mean(cell_number_ratio_sum[i,:])+0.2,str(i+1))
    ax1.set_title('Cell Utilization Ratios vs L1 Error')
    ax1.set_xlabel("L1 Error")
    ax1.set_ylabel("Cell Utilization Ratios")
    #ax1.legend(loc=1)
    ax1.legend(loc=1)
    plt.tight_layout()
    plt.savefig(plot_path+'error-vs-cellratios.png',bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(0,number_of_sweeps):
    	ax.scatter(time[i], data[0,i], cell_number_ratio_avg[i,0],c=colour[int(max_level[0,i]-numpy.amin(max_level))])
    ax.set_xlabel('Time')
    ax.set_ylabel(' Error')
    ax.set_zlabel('Cell Ratio')
    plt.savefig(plot_path+'cell-ratios-3d.pdf',bbox_inches='tight')

    for i in range(number_of_sweeps):
        count[max_level[0,i]] = 0
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    for i in range(0,number_of_sweeps):
    	if (count[max_level[0,i]] == 0):
    		ax1.plot(cost[i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
    		count[max_level[0,i]] += 1
    	else:
    		ax1.plot(cost[i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')

    	ax1.text(cost[i],numpy.mean(cell_number_ratio_sum[i,:])+0.2,str(i+1))
    ax1.set_title('Cost vs Cell Utilization Ratios')
    ax1.set_xlabel("Cost")
    ax1.set_ylabel("Cell Utilization Ratios")
    #ax1.legend(loc=1)
    ax1.legend(loc=1)
    plt.tight_layout()
    plt.savefig(plot_path+'cost-vs-cellratios.png',bbox_inches='tight')


    
    for i in range(number_of_sweeps):
    	count[max_level[0,i]] = 0
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    for i in range(0,number_of_sweeps):
    	if (count[max_level[0,i]] == 0):
    		ax1.loglog(time[i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
    		count[max_level[0,i]] += 1
    	else:
    		ax1.loglog(time[i],cell_number_ratio_avg[i,0],colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')

    	#ax1.text(cost[i],numpy.mean(cell_number_ratio_sum[i,:])+0.2,str(i+1))
    ax1.set_title('Log Log Plot for Cell Utilization Ratios vs Wall Time')
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel("Cell Utilization Ratios")
    #ax1.legend(loc=1)
    ax1.legend(loc=1)
    plt.tight_layout()
    plt.savefig(plot_path+'time-vs-cellratios-loglog.png',bbox_inches='tight')
    
    p = numpy.polyfit(numpy.log(time),numpy.log(cell_number_ratio_avg[:,0].tolist()),1)
    print p
    line_fit = lambda t: p[0]*t + p[1]
    for i in range(number_of_sweeps):
        count[max_level[0,i]] = 0
    fig, (ax1) = plt.subplots(1)
    colour = ['r','b','g','c','m','y']
    time_range = numpy.linspace(1.0,max(numpy.log(time)+1.0),30)
    ax1.plot(time_range,line_fit(time_range),'-k',label = str(round(p[0],2))+'t + '+str(round(p[1],2)))

    for i in range(0,number_of_sweeps):
    	if (count[max_level[0,i]] == 0):
    		ax1.plot(numpy.log(time[i]),numpy.log(cell_number_ratio_avg[i,0]),colour[int(max_level[0,i]-numpy.amin(max_level))]+'o',label='Level '+str(max_level[0,i]))
    		count[max_level[0,i]] += 1
    	else:
    		ax1.plot(numpy.log(time[i]),numpy.log(cell_number_ratio_avg[i,0]),colour[int(max_level[0,i]-numpy.amin(max_level))]+'o')
    	#ax1.text(cost[i],numpy.mean(cell_number_ratio_sum[i,:])+0.2,str(i+1))
    ax1.set_title('Line fit plot')
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)")
    ax1.set_ylabel("Cell Utilization Ratios")
    #ax1.legend(loc=1)
    ax1.legend(loc=1)
    plt.tight_layout()
    plt.savefig(plot_path+'time-vs-cellratios-log.png',bbox_inches='tight')


def plot_level_group(time,data,distinct_max_level,sweep_indices,error_type,refinemnt_ratios,pair_indices):
    # plots the sweeps using the same number of levels in different subplot 
    count = 0
    
    nrow = int(math.ceil(len(distinct_max_level)/2.0)) ; ncol = int(math.floor(len(distinct_max_level)/2.0));
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    axs = numpy.array(axs)
    count = 0
    for ax in axs.reshape(-1):
        for i in sweep_indices[distinct_max_level[count]]:
            #ax.errorbar(time[i],data[0,i],yerr=[numpy.subtract(data[0,i],data[4,i]),numpy.subtract(data[2,i],data[0,i])],ls='',marker='o',color='r',markersize=5)
            
            ax.text(time[i],data[0,i],str(i+1),fontsize = 10)
            #ax.text(time[i],data[2,i],str(data[3,i]),fontsize = 8)
            #ax.text(time[i],data[4,i],str(data[5,i]),fontsize = 8)
            #ax.text(time[i],data[0,i],str(round(data[1,i],3)),fontsize = 8)


        ax.set_title("Level "+str(int(distinct_max_level[count]))+" Groupings",fontsize = 8)
        ax.set_xlabel("Total Wall Time (%, Baseline = 100%)",fontsize = 8)
        ax.set_ylabel(error_type+" Error",fontsize = 8)
        ax.set_ylim([-0.1,1.1])
        ax.set_xlim([0,max(time)+5])
        count = count + 1
        if(count >= len(distinct_max_level)):
            break
    plt.tight_layout()
    #plt.suptitle("Level "+str(distinct_max_level[count])+" Groupings")
    plt.savefig(plot_path+error_type+'-grouping.png',bbox_inches='tight')

    # plots grouping with levels represnted by colors and also plots an arrow for the pairs of sweeps. 
    # Pairs are the sweeps with same refinement ratios but different arrangements
    fig, ax1 = plt.subplots(1)
    colour = ['r','b','g','c','m','y']

    for i in range(0,len(distinct_max_level)):
        count = 0
        pair = []
        for j in sweep_indices[distinct_max_level[i]]:
            
            if len(sweep_indices[distinct_max_level[i]]) >1:
                if(count == 0):
                    count += 1
                
                    if pair_indices[j+1]:
                        if refinemnt_ratios[j,0] > refinemnt_ratios[pair_indices[j+1][0]-1,0]:
                            ax1.plot(time[j],data[0,j],colour[i]+'s',label='Level '+str(int(distinct_max_level[i])))
                            ax1.text( time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1]+0.01,str(refinemnt_ratios[j,0]/refinemnt_ratios[j,1]))
                            ax1.arrow(time[j],data[0,j], time[pair_indices[j+1][0]-1]-time[j],data[0,pair_indices[j+1][0]-1]-data[0,j], head_width=0.03, head_length=0.25, fc='k', ec='k')
                        else:
                            ax1.plot(time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1],colour[i]+'s',label='Level '+str(int(distinct_max_level[i])))
                            ax1.arrow(time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1],time[j]-time[pair_indices[j+1][0]-1],data[0,j]-data[0,pair_indices[j+1][0]-1], head_width=0.03, head_length=0.25, fc='k', ec='k')
                            ax1.text( time[j],data[0,j]+0.01,str(refinemnt_ratios[j,0]/refinemnt_ratios[j,1]))

                        pair.append(pair_indices[j+1][0] - 1)

                elif(j not in pair):
                    if pair_indices[j+1]:
                        if refinemnt_ratios[j,0] > refinemnt_ratios[pair_indices[j+1][0]-1,0]:
                            ax1.plot(time[j],data[0,j],colour[i]+'s')
                            ax1.text( time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1]+0.01,str(refinemnt_ratios[j,0]/refinemnt_ratios[j,1]))
                            ax1.arrow(time[j],data[0,j],time[pair_indices[j+1][0]-1]-time[j],data[0,pair_indices[j+1][0]-1]-data[0,j], head_width=0.03, head_length=0.25, fc='k', ec='k')
                        else:
                            ax1.plot(time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1],colour[i]+'s',label='Level '+str(int(distinct_max_level[i])))
                            ax1.arrow(time[pair_indices[j+1][0]-1],data[0,pair_indices[j+1][0]-1],time[j]-time[pair_indices[j+1][0]-1],data[0,j]-data[0,pair_indices[j+1][0]-1], head_width=0.03, head_length=0.25, fc='k', ec='k')
                            ax1.text( time[j],data[0,j]+0.01,str(refinemnt_ratios[pair_indices[j+1][0]-1,0]/refinemnt_ratios[pair_indices[j+1][0],1]))

                        pair.append(pair_indices[j+1][0] - 1)
                    else:
                        ax1.plot(time[j],data[0,j],colour[i]+'^')
            else:
                ax1.plot(time[j],data[0,j],colour[i]+'s',label='Level '+str(int(distinct_max_level[i])))
#break
    ax1.set_title("Grouping by level")
    ax1.set_xlabel("Total Wall Time (%, Baseline = 100%)",fontsize = 12)
    ax1.set_ylabel(error_type+" Error",fontsize = 12)
    ax1.legend(loc=2)
    ax1.set_ylim([-0.1,1.1])
    ax1.set_xlim([0,max(time)+5])
    plt.tight_layout()
    plt.savefig(plot_path+error_type+'-grouping-1.png',bbox_inches='tight')


def plot_cost_objective(time,data,error_weight,time_weight,error_type):
    # calculates and pltos a cost plot. The cost calculations depend on the weights sent to the functions
    cost = numpy.zeros(numpy.shape(time))
    fig, ax1 = plt.subplots(1)
    for i in range(0,len(time)):
        cost[i] = error_weight*data[0,i] + time_weight*time[i]/100.0
        ax1.plot(i+1,cost[i],'ro',markersize=8)

    ax1.set_title("Cost Plot with error weight "+str(error_weight)+" and time weight "+str(time_weight))
    ax1.set_xlabel("Sweep Number")
    ax1.set_ylabel("Cost normalized to 1")
    plt.tight_layout()
    plt.savefig(plot_path+error_type+'-cost-plot'+str(error_weight)+','+str(time_weight)+'.png',bbox_inches='tight')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(time/100, data[0,:], cost)

    ax.set_xlabel('Time')
    ax.set_ylabel(error_type+' Error')
    ax.set_zlabel('Cost')
    plt.savefig(plot_path+error_type+'-cost-plot-3d.pdf',bbox_inches='tight')
    #plt.show()

    return cost,cost.tolist().index(max(cost)),cost.tolist().index(min(cost))

if __name__ == "__main__":
    # point run_data_path to where the run details text file is located.
    # If there is no text file with the details, create an array with the deatils
    run_data_path = '../../scratch/Tohoku-hawaii/'
    run_data = numpy.loadtxt(run_data_path+'run-data.txt')

    path = 'post-process-data/'
    data = open(path+'summary-data.txt')
    output_time = numpy.loadtxt(path+'output-times.txt')
    if not os.path.exists(path+'plots'):
        os.makedirs(path+'plots')
    plot_path = path+'plots/'

    first_line = data.readline().strip().split()
    number_of_gauges = int(first_line[1])
    number_of_sweeps = int(first_line[0])
    time_data = numpy.empty([number_of_sweeps,2])
    number_of_total_cell_updates = numpy.empty([number_of_sweeps,1])
    L1_error_data = numpy.empty([number_of_sweeps,number_of_gauges])
    L2_error_data = numpy.empty([number_of_sweeps,number_of_gauges])
    Inf_error_data = numpy.empty([number_of_sweeps,number_of_gauges])

    number_of_cells = {}
    for i in range(number_of_sweeps):
        number_of_cells[i+1] = numpy.loadtxt(path+'num_cells_run_'+str(i+1)+'.txt')

    sweep_count = -1
    gauge_count = 0

    for l in data:
        line = l.strip().split()
        if len(line) == 0:
            break

        elif line[0] == '0':
            sweep_count+=1
            time_data[sweep_count,0] = float(line[1])
            time_data[sweep_count,1] = float(line[2])
            number_of_total_cell_updates[sweep_count,0] = float(line[3])
            
        elif line[0] == 'g':
            L1_error_data[sweep_count,gauge_count] = float(line[1])
            L2_error_data[sweep_count,gauge_count] = float(line[2])
            Inf_error_data[sweep_count,gauge_count] = float(line[3])
            gauge_count+=1
            if gauge_count >= number_of_gauges:
                gauge_count = 0

        elif line[0] == 'b':
            basline_total_time = float(line[2])
            basline_regridding_time = float(line[1])

    # Normalizing the data
    time = (time_data[:,0]*100)/basline_total_time
    regridding_time = (time_data[:,1])/numpy.amax(time_data[:,1])
    L1_error_data = L1_error_data/numpy.amax(L1_error_data)
    L2_error_data = L2_error_data/numpy.amax(L2_error_data)
    Inf_error_data = Inf_error_data/numpy.amax(Inf_error_data)

    grid_size = numpy.empty([1,number_of_sweeps])
    max_level = numpy.empty([1,number_of_sweeps])
    refinemnt_ratios = []
    count = 0
    for i in range(1,number_of_sweeps+1):

        grid_size[0,count] = run_data[i,0]
        max_level[0,count] = run_data[i,2]
        refinemnt_ratios.append(run_data[i,3:])
        
        count += 1

    refinemnt_ratios = numpy.array(refinemnt_ratios)

    l1_data,l2_data,inf_data,lowest_error_sweep,highest_error_sweep = get_avg_bound(L1_error_data,L2_error_data,Inf_error_data,number_of_sweeps)
    plot_summary(time,L1_error_data,L2_error_data,Inf_error_data,number_of_gauges,number_of_sweeps,regridding_time)
    plot_features(time,l1_data,l2_data,inf_data,number_of_gauges,number_of_sweeps,regridding_time,number_of_total_cell_updates,max_level)
    cost,highest_cost_sweep,lowest_cost_sweep = plot_cost_objective(time,l1_data,0.5,0.5,'L1')

    plot_cell_ratios(output_time,number_of_sweeps,number_of_cells,refinemnt_ratios,max_level,grid_size,lowest_error_sweep,highest_error_sweep,highest_cost_sweep,lowest_cost_sweep,l1_data,cost,time)

    distinct_max_level = []
    sweep_indices = defaultdict(list)
    pair_indices = defaultdict(list)
    for i in range(0,number_of_sweeps):
        sweep_indices[max_level[0,i]].append(i)
        if max_level[0,i] not in distinct_max_level:
            distinct_max_level.append(max_level[0,i])

    for i in range(0,number_of_sweeps):
        for j in range(i+1,number_of_sweeps):
            
            if(numpy.array_equal(refinemnt_ratios[i,:max_level[0,i]-1],refinemnt_ratios[j,:max_level[0,j]-1][::-1])):
                pair_indices[i+1].append(j+1)
    plot_level_group(time,l1_data,distinct_max_level,sweep_indices,'L1',refinemnt_ratios,pair_indices)


    plot_cost_objective(time,l1_data,0.75,0.25,'L1')
    plot_cost_objective(time,l1_data,0.25,0.75,'L1')