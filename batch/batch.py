r"""Simple controller for runs with GeoClaw

Includes support for multiple runs at the same time

"""
# ============================================================================
#      Copyright (C) 2013 Kyle Mandli <kyle@ices.utexas.edu>
#
#          Distributed under the terms of the MIT license
#                http://www.opensource.org/licenses/
# ============================================================================

from __future__ import print_function
from __future__ import absolute_import

import subprocess
import os
import time
import glob


class Job(object):
    r"""Base object for all jobs

    The ``type``, ``name``, and ``prefix`` attributes are used by the
    :class:`BatchController` to create the path to the output files along with
    the name of the output directories and log file.  The pattern is
    ``base_path/type/name/prefix*``.  See :class:`BatchController` for more
    information on how these are created.

    .. attribute:: type

        (string) - The top most directory that the batch output will be
        located in.  ``default = ""``.

    .. attribute:: name

        (string) - The second top most directory that the batch output will
        be located in.  ``default = ""``.

    .. attribute:: prefix

        (string) - The prefix applied to the data directory, the output
        directory, and the log file.  ``default = None``.

    .. attribute:: executable

        (string) - Name of the binary executable.  ``default = "xclaw"``.

    .. attribute:: setplot

        (string) - Name of the module containing the `setplot`
        function.  ``default = "setplot"``.

    .. attribute:: rundata

        (clawpack.clawutil.data.ClawRunData) - The data object containing all
        data objects.  By default all data objects inside of this object will
        be written out.  This attribute must be instantiated by any subclass
        and if not will raise a ValueError exception when asked to write out.

    :Initialization:

    Output:
     - (:class:`Job`) - Initialized Job object.
    """

    def __init__(self):
        r"""
        Initialize a Job object

        See :class:`Job` for full documentation
        """

        super(Job, self).__init__()

        # Base job traits
        self.type = ""
        self.name = ""
        self.setplot = "setplot"
        self.prefix = None
        self.executable = 'xclaw'

        self.rundata = None

    def __str__(self):
        output = "Job %s: %s\n" % (self.name, self.prefix)
        output += "  Setplot: %s\n" % self.setplot
        return output

    def write_data_objects(self):
        r"""
        Write out data objects contained in *rundata*

        Raises ValueError if *rundata* has not been set.

        """

        if self.rundata is None:
            raise ValueError("Must set rundata to a ClawRunData object.")
        self.rundata.write()


class BatchController(object):
    r"""Controller for Clawpack batch runs.

    Controller object that will run the set of jobs provided with the
    parameters set in the object including plotting, path creation, and
    simple process parallelism.

    .. attribute:: jobs

        (list) - List of :class:`Job` objects that will be run.

    .. attribute:: plot

        (bool) - If True each job will be plotted after it has run.
        ``default = True``

    .. attribute:: tar

        (bool) - If True will tar and gzip the plots directory.
        ``default = False``

    .. attribute:: verbose

        (bool) - If True will print to stdout the remaining jobs
        waiting to be run and how many are currently in the process queue.
        ``default = False``.

    .. attribute:: terminal_output

        (bool) - If ``paralllel`` is False, this controls where
        the output is sent.  If True then it will simply use stdout, if False
        then the usual log file will be used.  ``default = False``.

    .. attribute:: base_path

        (path) - The base path to put all output.  If the
        environment variable ``DATA_PATH`` is set than the ``base_path`` will
        be set to that.  Otherwise the current working directory (returned by
        ``os.getcwd()``) will be used.

    .. attribute:: parallel

        (bool) - If True, jobs will be run in parallel.  This means
        that jobs will be run concurrently with other jobs up to a maximum at
        one time of ``max_processes``.  Once a process completes a new one is
        started.  ``default = True``.

    .. attribute:: wait

        (bool) - If True, the method waits to return until the last job
        has completed.  If False then the method returns immediately once the
        last job has been added to the process queue.  Default is `False`.

    .. attribute:: poll_interval

        (float) - Interval to poll for the status of each
        process.  Default is `5.0` seconds.

    .. attribute:: max_processes

        (int) - The maximum number of processes that can be run
        at one time.  If the environment variable `OMP_NUM_THREADS` is set then
        this defaults to that number.  Otherwise `4` is used.

    .. attribute:: runclaw_cmd

        (string) - The string that stores the base command for the
        run command.
        ``default = "python $CLAW/clawutil/src/python/clawutil/runclaw.py"``.

    .. attribute:: plotclaw_cmd

        (string) - The string that stores the base command for the
        plotting command.
        ``default = "python $CLAW/visclaw/src/visclaw/plotclaw.py"``.

    :Initialization:

    Input:
     - *jobs* - (list) List of :class:`Job` objects to be run.

    Output:
     - (:class:`BatchController`) Initialized BatchController object

    """

    def __init__(self, jobs=[]):
        r"""Initialize a BatchController object.

        See :class:`BatchController` for full documentation

        """

        super(BatchController, self).__init__()

        # Establish controller default parameters
        # Execution controls
        self.plot = True
        self.tar = False
        self.verbose = False
        self.terminal_output = False

        # Path controls
        if 'DATA_PATH' in os.environ.keys():
            self.base_path = os.environ['DATA_PATH']
        else:
            self.base_path = os.getcwd()
        self.base_path = os.path.expanduser(self.base_path)

        # Parallel run controls
        self.parallel = True
        self.wait = False
        self.poll_interval = 5.0
        if 'OMP_NUM_THREADS' in os.environ.keys():
            self.max_processes = int(os.environ['OMP_NUM_THREADS'])
        else:
            self.max_processes = 4
        self._process_queue = []

        # Default commands for running and plotting
        self.runclaw_cmd = "python $CLAW/clawutil/src/python/clawutil/runclaw.py"
        self.plotclaw_cmd = "python $CLAW/visclaw/src/python/visclaw/plotclaw.py"

        # Add the initial jobs to the jobs list
        if not isinstance(jobs, list) and not isinstance(jobs, tuple):
            raise ValueError("Jobs must be a list or tuple.")
        self.jobs = []
        for job in jobs:
            if isinstance(job, Job):
                self.jobs.append(job)
            else:
                raise ValueError("Elements of jobs must be a Job.")

    def __str__(self):
        output = ""
        for (i, job) in enumerate(self.jobs):
            output += "====== Job #%s ============================\n" % (i)
            output += str(job) + "\n"
        return output

    def run(self, only_write_data=False):
        r"""Run jobs from controller's *jobs* list.

        For each :class:`Job` object in *jobs* create a set of paths, directory
        structures, and log files in preperation for running the commands
        constructed.  If *parallel* is True then jobs are started and added
        to a queue with a maximum of *maximum_processes*.  If *parallel* is
        False each job is run to completion before continuing.  The *wait*
        parameter controls whether the function waits for the last job to run
        before returning.

        Output:
         - *paths* - (list) List of dictionaries containing paths to the data
           constructed for each job. The dictionary has keys 'job', 'data',
           'output', 'plots', and 'log' which respectively stores the base
           directory of the job, the data, output, and plot directories, and
           the log file.

        """

        # Run jobs
        paths = []
        for (i, job) in enumerate(self.jobs):
            # Create output directory
            data_dirname = ''.join((job.prefix, '_data'))
            output_dirname = ''.join((job.prefix, "_output"))
            plots_dirname = ''.join((job.prefix, "_plots"))
            log_name = ''.join((job.prefix, "_log.txt"))

            if len(job.type) > 0:
                job_path = os.path.join(self.base_path, job.type, job.name)
            else:
                job_path = os.path.join(self.base_path, job.name)
            job_path = os.path.abspath(job_path)
            data_path = os.path.join(job_path, data_dirname)
            output_path = os.path.join(job_path, output_dirname)
            plots_path = os.path.join(job_path, plots_dirname)
            log_path = os.path.join(job_path, log_name)
            paths.append({'job': job_path, 'data': data_path,
                          'output': output_path, 'plots': plots_path,
                          'log': log_path})

            # Create job directory if not present
            if not os.path.exists(job_path):
                os.makedirs(job_path)

            # Clobber old data directory
            if os.path.exists(data_path):
                if not job.rundata.clawdata.restart:
                    data_files = glob.glob(os.path.join(data_path, "*.data"))
                    for data_file in data_files:
                        os.remove(data_file)
            else:
                os.mkdir(data_path)

            # Open and start log file
            log_file = open(log_path, 'w')
            tm = time.localtime()
            year = str(tm[0]).zfill(4)
            month = str(tm[1]).zfill(2)
            day = str(tm[2]).zfill(2)
            hour = str(tm[3]).zfill(2)
            minute = str(tm[4]).zfill(2)
            second = str(tm[5]).zfill(2)
            date = 'Started %s/%s/%s-%s:%s.%s' % (year, month, day, hour,
                                                  minute, second)
            log_file.write(date)

            # Write out data
            temp_path = os.getcwd()
            os.chdir(data_path)
            job.write_data_objects()
            os.chdir(temp_path)

            if only_write_data:
                continue

            # Handle restart requests
            if job.rundata.clawdata.restart:
                restart = "T"
                overwrite = "F"
            else:
                restart = "F"
                overwrite = "T"

            # Construct string commands
            run_cmd = "%s %s %s %s %s %s True" % (self.runclaw_cmd,
                                                  job.executable,
                                                  output_path,
                                                  overwrite,
                                                  restart,
                                                  data_path)
            if self.plot:
                plot_cmd = "%s %s %s %s" % (self.plotclaw_cmd,
                                            output_path,
                                            plots_path,
                                            job.setplot)
            tar_cmd = "tar -cvzf %s.tgz -C %s/.. %s" % (plots_path,
                                                        plots_path,
                                                        os.path.basename(
                                                            plots_path))
            cmd = run_cmd
            if self.plot:
                cmd = ";".join((cmd, plot_cmd))
                if self.tar:
                    cmd = ";".join((cmd, tar_cmd))

            # Run jobs
            if self.parallel:
                while len(self._process_queue) == self.max_processes:
                    if self.verbose:
                        print("Number of processes currently:",
                              len(self._process_queue))
                    for process in self._process_queue:
                        if process.poll() is not None:
                            self._process_queue.remove(process)
                    time.sleep(self.poll_interval)
                self._process_queue.append(subprocess.Popen(cmd, shell=True,
                                                            stdout=log_file,
                                                            stderr=log_file))

            else:
                if self.terminal_output:
                    log_file.write("Outputting to terminal...")
                    subprocess.Popen(cmd, shell=True).wait()
                    log_file.write("Command completed.")
                else:
                    subprocess.Popen(cmd, shell=True, stdout=log_file,
                                     stderr=log_file).wait()

        # -- All jobs have been started --

        # Wait to exit while processes are still going
        if self.wait:
            while len(self._process_queue) > 0:
                time.sleep(self.poll_interval)
                for process in self._process_queue:
                    if process.poll() is not None:
                        self._process_queue.remove(process)
                print("Number of processes currently: %s" %
                      len(self._process_queue))

        return paths
