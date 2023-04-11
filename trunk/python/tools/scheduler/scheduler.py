#!/usr/bin/env python3

import abc
from cmath import pi
import multiprocessing as mp
import os
import queue
import threading


class NoMoreJobsException(Exception):
  """An exception indicating that no more jobs are available."""
  pass


def worker(job_queue, result_queue, shutdown_event, pinned_cpus_queue,
           pinned_cpus_semaphore, process_job_fn):
  """A worker function to be run in a separate process.

  This function is on every worker process. It iteratively retrieves jobs from
  job_queue until the job queue is empty and shutdown_event was set by the
  scheduler process. Results are enqueued in result_queue.

  If pinned_cpus_queue is set, this function pins the executing process to a
  CPU ID that is taken from the queue.
  """
  if pinned_cpus_queue:
    cpu_id = pinned_cpus_queue.get()
    os.sched_setaffinity(0, {cpu_id})
    pinned_cpus_semaphore.release()

  while not (job_queue.empty() and shutdown_event.is_set()):
    job = None
    try:
      # Block for up to 0.1s, then start a new iteration.
      job = job_queue.get(True, 0.1)
    except queue.Empty:
      continue
    result_queue.put(process_job_fn(job))


class JobScheduler(abc.ABC):
  """An abstract job scheduler.

  A job scheduler manages a pool of worker processes and dispatches jobs to
  them. The job scheduler cannot be accessed from worker processes.

  Subclasses must provide two methods: get_next_job generates a job to be
  dispatched to some available worker. When the job has been processed, the
  process_result callback is invoked.

  When creating a job scheduler, users must specify the number of processes
  and a function for processing a job (that is executed on worker processes).
  The scheduler keeps requesting and scheduling new jobs until get_next_job
  throws a NoMoreJobsException or the shutdown method is called.
  """

  def __init__(self, num_processes, process_job_fn, pinned_cpus=[]):
    """Initialize the job scheduler.

    num_processes is the desired number of worker processes. These can
    optionally be pinned to a CPU. In that case, the number of CPU IDs in
    pinned_cpus must match num_processes. process_job_fn is a callback for
    processing a job. It is executed in a worker process and has no access
    to the job scheduler.
    """
    assert len(pinned_cpus) == num_processes or len(
        pinned_cpus) == 0, "invalid number of pinned_cpus"
    self.job_queue = mp.Queue()
    self.pinned_cpus = pinned_cpus
    self.pinned_cpus_queue = mp.Queue() if len(pinned_cpus) > 0 else None
    self.pinned_cpus_semaphore = mp.Semaphore()
    self.result_processor_thread = threading.Thread(
        target=self._result_processor)
    self.result_queue = mp.Queue()
    self.shutdown_event = mp.Event()
    self.processes = [
        mp.Process(
            target=worker,
            args=(self.job_queue, self.result_queue, self.shutdown_event,
                  self.pinned_cpus_queue, self.pinned_cpus_semaphore,
                  process_job_fn)) for i in range(num_processes)
    ]
    self.num_jobs = 0
    self.was_started = False

  @abc.abstractmethod
  def get_next_job(self):
    """Generate a new job."""
    pass

  @abc.abstractmethod
  def process_result(self, result):
    """Process a job result."""
    pass

  def _enqueue_job(self):
    """Generate a new job and enqueue it."""
    if self.shutdown_event.is_set():
      raise NoMoreJobsException()
    job = self.get_next_job()
    self.num_jobs += 1
    self.job_queue.put(job)

  def _result_processor(self):
    """A function that is iteratively looking for results.
    
    This function is run in the job scheduler process, but in a separate
    thread. It takes results from the result queue and passes them to the
    abstract process_result method. For every result, it tries to enqueue
    a new job. When a result has been received for each job and no more jobs
    are available, the worker processes are shutdown.
    """
    while self.num_jobs > 0:
      result = self.result_queue.get()
      self.num_jobs -= 1
      self.process_result(result)
      try:
        self._enqueue_job()
      except NoMoreJobsException:
        pass
    self.shutdown_event.set()

  def start(self):
    """Start the job scheduler.
    
    This function starts the scheduler and returns immediately.
    """
    assert not self.was_started, "cannot start the scheduler multiple times"
    self.was_started = True
    self.shutdown_event.clear()

    # Enqueue pinned CPU IDs.
    for cpu_id in self.pinned_cpus:
      self.pinned_cpus_queue.put(cpu_id)

    # Start processes.
    for p in self.processes:
      p.start()

    # Wait until each process pinned itself to a CPU (if pinning is requested).
    if len(self.pinned_cpus) > 0:
      for i in range(len(self.processes)):
        self.pinned_cpus_semaphore.acquire()

    # Enqueue a few initial jobs. The number of jobs must be large enough to
    # keep the workers busy. For every received result, a new job will be
    # enqueued in _result_processor, so the overall number of jobs in the queue
    # will stay more or less the same throughout the execution.
    for i in range(2 * len(self.processes)):
      try:
        self._enqueue_job()
      except NoMoreJobsException:
        break

    # Start processing results in a new thread (but in the same process).
    self.result_processor_thread.start()

  def join(self):
    """Block until all jobs have been processed and the workers have been
    shut down.
    """
    self.result_processor_thread.join()
    for p in self.processes:
      p.join()

  def shutdown(self):
    """Stop enqueing new jobs."""
    self.shutdown_event.set()
