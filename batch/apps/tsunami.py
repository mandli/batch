
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

    def __init__(self, ensemble, storm_num, base_path='./', storms_path='./'):

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
        self.ensemble = ensemble
        self.storm_file_path = os.path.join(storms_path
                               os.path.abspath("./%s.storm" % storm_num))

        # Set storm file
        self.rundata.storm_data.storm_type = 1
        self.rundata.storm_data.storm_file = self.storm_file_path


    def __str__(self):
        output = super(StormJob, self).__str__()
        output += "\n  Storm Number: %s" % self.storm_num
        return output


    def write_data_objects(self):
        r""""""

        # Modify output times to match storm track time interval
        # Assume that track data is grouped
        storm_indices = numpy.nonzero(self.storm_num == self.ensemble.storm_num)[0]
        if len(storm_indices) == 0:
            raise ValueError("Did not find %s storm in ensemble." % self.storm_num)
        new_year_date = datetime.datetime(
                               int(self.ensemble.time[storm_indices[-1]][0:4]),
                               1, 1, 0)
        final_date = datetime.datetime(
                               int(self.ensemble.time[storm_indices[-1]][0:4]), 
                               int(self.ensemble.time[storm_indices[-1]][4:6]), 
                               int(self.ensemble.time[storm_indices[-1]][6:8]), 
                               int(self.ensemble.time[storm_indices[-1]][8:10]))
        final_dt = final_date - new_year_date
        self.rundata.clawdata.tfinal = days2seconds(final_dt.days) \
                                                              + final_dt.seconds
        recurrence = 6
        self.rundata.clawdata.num_output_times = \
                   int((self.rundata.clawdata.tfinal - self.rundata.clawdata.t0) 
                                            * recurrence / (60**2 * 24))

        # Write out all data files
        super(StormJob, self).write_data_objects()

        # Create storm file
        self.ensemble.write(self.storm_file_path, storm_filter=self.storm_num)
