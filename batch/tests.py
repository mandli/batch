#!/usr/bin/env python

from __future__ import print_function
from __future__ import absolute_import

import unittest

import batch

class TestJob(batch.job):

    def __init__(self, test_param):

        super(TestJob, self).__init__()

        self.rundata = None
        self.type = None
        self.name = None
        self.prefix = None
        self.executable = None

    def write_data_objects(self):
        pass

def test_failed_job():
    job = TestJob(1)

    job_controller = batch.BatchController([job])

    job_controller.run()


if __name__ == "__main__":
    unittest.main()    
