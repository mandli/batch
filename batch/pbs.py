r"""Batch sub-classes for runs on Columbia's Yeti cluster"""

# ============================================================================
#      Copyright (C) 2013 Kyle Mandli <kyle@ices.utexas.edu>
#
#          Distributed under the terms of the MIT license
#                http://www.opensource.org/licenses/
#
# (adapted for yeti by Andrew Kaluzny <akk2141@columbia.edu>, 2015)
# ============================================================================

from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import subprocess

import batch

class Job(batch.Job):
    r"""
    Modifications to the basic :class:`batch.Job` class for Yeti runs

    """

    def __init__(self, use_v2=True, memory=4096, time=60, nodes=1,
                       omp_num_threads=1):
        r"""
        Initialize Yeti job

        See :class:`YetiJob` for full documentation
        """

        super(Job, self).__init__()

        # Add extra job parameters
        self.omp_num_threads = omp_num_threads
        ##self.mic_omp_num_threads = 1
        ##self.mic_affinity = "none"
        self.memory = memory
        self.time = time
        self.nodes = nodes
        self.use_v2 = use_v2
        
        self.group = None # needs to be set at some point


class BatchController(batch.BatchController):
    r"""
    Modifications to the basic batch controller for Yeti runs


    :Ignored Attributes:

    Due to the system setup, the following controller attributes are ignored:  

        *plot*, *terminal_output*, *wait*, *poll_interval*, *plotclaw_cmd*
    """

    def __init__(self, jobs=[], email=None, email_behavior="abe", output=None):
        r"""
        Initialize Yeti batch controller

        See :class:`YetiBatchController` for full documentation
        """

        super(BatchController, self).__init__(jobs)

        # Yeti specific execution controls
        self.email = email
        self.email_behavior = email_behavior
        self.output = output
        self.queue = None

    def run(self):
        r"""Run Yeti jobs from controller's *jobs* list.

        This run function is modified to run jobs through the slurm queue system
        and provides controls for running serial jobs (OpenMP only).

        Unless otherwise noted, the behavior of this function is identical to
        the base class :class:`BatchController`'s function.
        """

        # Run jobs
        paths = []
        for (i,job) in enumerate(self.jobs):
            # Create output directory
            data_dirname = ''.join((job.prefix,'_data'))
            output_dirname = ''.join((job.prefix,"_output"))
            plots_dirname = ''.join((job.prefix,"_plots"))
            run_script_name = ''.join((job.prefix,"_run.sh"))
            log_name = ''.join((job.prefix,"_log.txt"))

            
            if len(job.type) > 0:
                job_path = os.path.join(self.base_path,job.type,job.name)
            else:
                job_path = os.path.join(self.base_path,job.name)
            job_path = os.path.abspath(job_path)
            data_path = os.path.join(job_path,data_dirname)
            output_path = os.path.join(job_path,output_dirname)
            plots_path = os.path.join(job_path,plots_dirname)
            log_path = os.path.join(job_path,log_name)
            run_script_path = os.path.join(job_path,run_script_name)
            paths.append({'job':job_path, 'data':data_path,
                          'output':output_path, 'plots':plots_path,
                          'log':log_path})

            # Create job directory if not present
            if not os.path.exists(job_path):
                os.makedirs(job_path)

            # Clobber old data directory
            if os.path.exists(data_path):
                if not job.rundata.clawdata.restart:
                    data_files = glob.glob(os.path.join(data_path,'*.data'))
                    for data_file in data_files:
                        os.remove(data_file)
            else:
                os.mkdir(data_path)

            # Write out data
            temp_path = os.getcwd()
            os.chdir(data_path)
            job.write_data_objects()
            os.chdir(temp_path)

            # Handle restart requests
            if job.rundata.clawdata.restart:
                restart = "T"
                overwrite = "F"
            else:
                restart = "F"
                overwrite = "T"            
            
            # Construct string commands
            run_cmd = "%s %s %s %s %s %s True\n" % (self.runclaw_cmd, job.executable, output_path,
                                                      overwrite, restart, data_path)

            # Write slurm run script
            run_script = open(run_script_path, 'w')

            hours = int(job.time / 60)
            minutes = job.time % 60

            run_script.write("#!/bin/sh\n")
            run_script.write("#PBS -N %s        # Job name\n" % job.prefix)
            run_script.write("#PBS -W group_list=%s        # Group\n" % job.group)
            run_script.write("#PBS -l mem=%smb        # Memory\n" % job.memory)
            run_script.write("#PBS -l walltime=00:%s:%s:00        # Walltime\n" % (hours, minutes))
            if job.use_v2:
                run_script.write("#PBS -l nodes=%s:ppn=%s:v2         # Nodes and processers per node\n" % (job.nodes, job.omp_num_threads))
            else:
                run_script.write("#PBS -l nodes=%s:ppn=%s            # Nodes and processers per node\n" % (job.nodes, job.omp_num_threads))
            run_script.write("#PBS -V    # export env. variables to the job\n")
            if self.queue is not None:
                run_script.write("#PBS -q %s      # Requested queue\n" % self.queue)
            if self.email is not None:
                run_script.write("#PBS -M %s\n" % self.email)
                run_script.write("#PBS -m %s # email for abort, begin, end\n" % self.email_behavior)
            if self.output is not None:
                run_script.write("#PBS -o localhost:%s   # stdout\n" % self.output)
                run_script.write("#PBS -e localhost:%s   # stderr\n" % self.output)
            run_script.write("\n")
            run_script.write("# OpenMP controls\n")
            run_script.write("export OMP_NUM_THREADS=%s\n" % job.omp_num_threads)
            ## run_script.write("export MIC_ENV_PREFIX=MIC")
            ## run_script.write("export MIC_OMP_NUM_THREADS=%s\n" % job.mic_omp_num_threads)
            ## run_script.write("export MIC_KMP_AFFINITY=%s\n" % job.mic_affinity)
            run_script.write("\n")
            run_script.write("# Location of output\n")
            run_script.write(self.output)
            run_script.write("\n")
            run_script.write("# Run command\n")
            run_script.write(run_cmd)

            run_script.close()

            # Submit job to queue
            subprocess.Popen("qsub %s > %s" % (run_script_path,log_path), shell=True).wait()

        # -- All jobs have been started --
            
        return paths
