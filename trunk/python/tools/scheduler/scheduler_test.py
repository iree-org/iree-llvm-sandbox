#!/usr/bin/env python3

from python.tools.scheduler.scheduler import JobScheduler, NoMoreJobsException

class ConcreteScheduler(JobScheduler):
  """A dummy scheduler for testing purposes."""

  def __init__(self, num_processes, process_job_fn, pinned_cpus=[]):
    super().__init__(num_processes, process_job_fn, pinned_cpus)
    self.state = 0
    self.results = []

  def get_next_job(self):
    if self.state > 10:
      raise NoMoreJobsException()
    self.state += 1
    return self.state

  def process_result(self, result):
    self.results.append(result)

def handler(job):
  """This function is run on multiple processes in parallel."""
  return job * job


s = ConcreteScheduler(5, handler)
s.start()
s.join()
s.results.sort()
assert s.results == [1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121], "wrong result"

