
from __future__ import print_function
from __future__ import absolute_import

import os
import numpy
import datetime

import storm

import batch.batch

days2seconds = lambda days: days * 60.0**2 * 24.0

class StormJob(batch.batch.job):
    r""""""

    def __init__(self, storm_num, base_path='./', storms_path='./'):

        super(StormJob, self).__init__()

        self.type = ""
        self.name = ""
        self.prefix = str(storm_num).zfill(5)
        self.storm_num = storm_num
        self.executable = "xgeoclaw"

        # Create base data object
        import setrun
        self.rundata = setrun.setrun()

        # Storm specific data
        self.storm_file_path = os.path.abspath(os.path.join(storms_path, 
                                                        "%s.storm" % storm_num))

        # Set storm file
        self.rundata.storm_data.storm_file = self.storm_file_path

        # Change time frame of simulation...
        # self.rundata.clawdata.t0 = days2seconds()
        # self.rundata.clawdata.tfinal = days2seconds()


    def __str__(self):
        output = super(StormJob, self).__str__()
        output += "\n  Storm Number: %s" % self.storm_num
        return output


    def write_data_objects(self):
        r""""""

        # Write out all data files
        super(StormJob, self).write_data_objects()

        # If any additional information per storm is needed do it here
        # ...
