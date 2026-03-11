##########################################################################
# Copyright (c) 2026 Reinhard Caspary                                    #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import gc
import heapq
import logging
import logging.handlers
import multiprocessing
import multiprocessing.queues
import os
import queue

import psutil
import time

from ..matrix import Matrix
from .graph import Registry, MatrixNode
from .operators import matrix_args

vmem = psutil.virtual_memory()
MEM_CRITICAL = 0.05 * vmem.total
MEM_MAXIMAL = 0.10 * vmem.total


##########################################################################
# Memory statistics
##########################################################################

def memory():
    # Memory used by this process
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info().rss / 1024 ** 2

    # Total available memory
    vmem = psutil.virtual_memory()
    avail_mem = vmem.available / 1024 ** 2

    # Total amount of virtual memory on the system
    total_mem = vmem.total / 1024 ** 2

    return f"proc: {proc_mem:.0f} MB, avail: {avail_mem:.0f} MB, total: {total_mem:.0f} MB"


##########################################################################
# Synchronous direct build process (no graph)
##########################################################################

def build_matrix(config_name, name, state_space, reduced=False):
    logger = logging.getLogger("build")

    matrix_str = f"Matrix({config_name}, {name}, {state_space}, {reduced})"

    if Matrix.exists(config_name, name, state_space, reduced):
        logger.info(f"{matrix_str} exists.")
        return

    logger.info(f"Generate {matrix_str}.")
    t = time.time()
    Matrix(config_name, name, state_space, reduced)
    t = time.time() - t
    logger.info(f"Matrix generation took {t:.1f} seconds / {memory()}")


def run_sync(nums_electrons, handlers):
    t = time.time()

    # Register log handlers in root logger
    root = logging.getLogger()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    # Local logger
    logger = logging.getLogger("build")

    for num_electrons in nums_electrons:
        logger.info(f"Build matrices for lanthanide ion f{num_electrons}.")
        for kwargs in matrix_args(num_electrons):
            build_matrix(**kwargs)

    t = time.time() - t
    logger.info(f"Total execution time: {t:.1f} seconds")


##########################################################################
# TaskManager class
##########################################################################

class TaskManager:
    """ Manage tasks (node ids) in three states 'active' (ready to be processed), 'pending' (in process), and
    'finished' (finished successfully). """

    def __init__(self):
        # Priority queue for nodes ready for processing
        self._active_heap = []
        # Nodes currently in progress
        self._pending = set()
        # Successfully finished nodes
        self._finished = set()

    def add_active(self, node_id, weight):
        """ Add a node id to the active state if it hasn't been seen before. """

        # Sanity check, skip known node id
        if node_id in self._finished or node_id in self._pending:
            return
        if any(node_id == nid for _, nid in self._active_heap):
            return

        # Push node id on the active heap
        heapq.heappush(self._active_heap, (-weight, node_id))

    def get_next(self):
        """ Move next node id from active to pending state and return it. """

        # No active node id waiting
        if not self._active_heap:
            return None

        # Move node id from active heap to pending
        _, node_id = heapq.heappop(self._active_heap)
        self._pending.add(node_id)

        # Return node id
        return node_id

    def mark_finished(self, node_id):
        """ Move node id from pending to finished state. """

        self._pending.remove(node_id)
        self._finished.add(node_id)

    def revert_to_active(self, node_id, weight):
        """ Move node id from pending back to active state. """

        self._pending.remove(node_id)
        heapq.heappush(self._active_heap, (-weight, node_id))

    @property
    def len_active(self):
        """ Return number of active nodes. """

        return len(self._active_heap)

    @property
    def len_pending(self):
        """ Return number of pending nodes. """

        return len(self._pending)

    @property
    def len_finished(self):
        """ Return number of finished nodes. """

        return len(self._finished)

##########################################################################
# Worker process
##########################################################################

def pool_worker(task_queue, result_queue, log_queue, stop_event):
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)
    logger = logging.getLogger(f"[{os.getpid()}]")
    logger.info(f"Worker active / {memory()}")

    while True:
        node = task_queue.get()
        if node is None:
            break

        node_id = node.node_id
        logger.info(f"Worker generating {node_id}: {node}")
        try:
            node.generate()
            result_queue.put((node_id, os.getpid()))
        except Exception as e:
            logger.error(f"FATAL WORKER ERROR in node {node_id} / {memory()}: {e}", exc_info=True)
            stop_event.set()
            result_queue.put((node_id, os.getpid()))
            break
        gc.collect()
        logger.info(f"Worker done / {memory()}")


##########################################################################
# Pool executor
##########################################################################

class PoolScheduler:
    def __init__(self, logger, log_queue):
        self.logger = logger
        self.log_queue = log_queue

        self.max_workers = multiprocessing.cpu_count()
        self.workers = {}

        # Memory stats collected and updated for every processed node
        self.node_stats = {}

        # Initialise the task manager
        self.task_manager = TaskManager()

        # Initialise required queues and events
        self.manager = multiprocessing.Manager()
        self.result_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()

        self.registry = Registry()

    def max_rss(self, node_id):
        if node_id not in self.node_stats:
            return 0
        return self.node_stats[node_id]

    @staticmethod
    def get_rss(pid):
        try:
            rss = psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None
        return rss

    def update_rss(self):
        for proc, task_queue, node_id in self.workers.values():
            if node_id is None:
                continue
            rss = self.get_rss(proc.pid)
            if rss is None:
                continue
            if not node_id in self.node_stats or rss > self.node_stats[node_id]:
                self.node_stats[node_id] = rss

    @property
    def idle_workers(self):
        return [proc.pid for proc, task_queue, node_id in self.workers.values() if node_id is None]

    @property
    def busy_workers(self):
        return [proc.pid for proc, task_queue, node_id in self.workers.values() if node_id is not None]

    def terminate(self, pid):
        proc, task_queue, node_id = self.workers.pop(pid)
        task_queue.put(None)
        proc.terminate()
        proc.join()
        return node_id

    def memory_watchdog(self):

        # Skip if enough memory is still available
        vmem = psutil.virtual_memory()
        if len(self.workers) == 0 or vmem.available > MEM_MAXIMAL:
            return True

        # Terminate an idle worker
        if len(self.idle_workers) > 0:
            pid = self.idle_workers[0]
            rss = self.get_rss(pid) / 1024 ** 2
            self.logger.warning(f"LOW MEMORY ({vmem.percent:.1f} %): Terminating idle PID {pid} using {rss:.0f}MB")
            node_id = self.terminate(pid)
            assert node_id is None
            return False

        # Skip termination of busy workers if enough memory is still available or if just one busy worker is left
        if vmem.available > MEM_CRITICAL and len(self.busy_workers) > 1:
            return False

        # Claimed memory (RSS) and PID of the smallest worker
        worker_stats = [(self.get_rss(pid) / 1024 ** 2, pid) for pid in self.busy_workers]
        worker_stats.sort()
        rss, pid = worker_stats[0]

        # Terminate the smallest worker
        self.logger.warning(f"CRITICAL MEMORY ({vmem.percent:.1f} %): Terminating busy PID {pid} using {rss:.0f}MB")
        node_id = self.terminate(pid)

        # Put node on the heap again if the worker was active
        if node_id is not None:
            node = self.registry.nodes[node_id]
            self.task_manager.revert_to_active(node_id, -node.weight)
        return False

    def process_cleanup(self):
        for proc, task_queue, node_id in self.workers.values():
            if proc.is_alive():
                continue
            if node_id is None:
                continue
            node = self.registry.nodes[node_id]
            self.task_manager.revert_to_active(node_id, -node.weight)
            self.workers[proc.pid] = (proc, task_queue, None)

    def schedule_tasks(self):

        # Skip if maximum number of active workers reached
        if len(self.busy_workers) >= self.max_workers:
            return

        # Skip if no memory available
        vmem = psutil.virtual_memory()
        if vmem.available < MEM_MAXIMAL:
            return

        # Get next node
        node_id = self.task_manager.get_next()
        if node_id is None:
            return
        node = self.registry.nodes[node_id]

        # Skip if not enough memory available if at least one worker is busy
        rss = self.max_rss(node_id)
        if vmem.available - rss < MEM_CRITICAL:

            # Terminate an idle worker
            if len(self.idle_workers) > 0:
                pid = self.idle_workers[0]
                rss = self.get_rss(pid) / 1024 ** 2
                self.logger.warning(f"Terminating idle PID {pid} using {rss:.0f}MB")
                node_id = self.terminate(pid)
                assert node_id is None

            # Skip task only if at least one worker is busy
            if len(self.busy_workers) > 0:
                self.task_manager.revert_to_active(node_id, -node.weight)
                return

        # Get free worker
        free_workers = [(proc, queue) for proc, queue, node_id in self.workers.values() if node_id is None]
        if free_workers:
            proc, task_queue = free_workers[0]

        # Get new worker
        else:
            task_queue = self.manager.Queue()
            process_args = (task_queue, self.result_queue, self.log_queue, self.stop_event)
            proc = multiprocessing.Process(target=pool_worker, args=process_args)
            proc.start()

        # Deal task to worker
        self.workers[proc.pid] = (proc, task_queue, node_id)
        task_queue.put(node)

    def store_result(self):
        try:
            node_id, pid = self.result_queue.get(timeout=0.5)
        except queue.Empty:
            return

        # Mark node as finished
        self.task_manager.mark_finished(node_id)

        # Try to activate all child nodes
        node = self.registry.nodes[node_id]
        for child_id in node.children:
            if not node.exists and node.in_degree == 0:
                self.task_manager.add_active(node_id, -node.weight)

        # Log result
        rss = self.max_rss(node_id) / 1024 ** 2
        self.logger.info(f"Finished {node} [maximum RSS: {rss:.0f} MB].")

        # Release worker
        proc, task_queue, nid = self.workers[pid]
        assert proc.pid == pid
        assert node_id == nid
        self.workers[proc.pid] = (proc, task_queue, None)

    def register(self, num_electrons: int):
        self.logger.info(f"Register all matrices for lanthanide ion f{num_electrons}.")

        # Register all nodes
        for kwargs in matrix_args(num_electrons):
            self.registry.register(MatrixNode, **kwargs)

        # Activate all nodes with zero in-degree
        for node_id in self.registry.nodes.keys():
            node = self.registry.nodes[node_id]
            if not node.exists and node.in_degree == 0:
                self.task_manager.add_active(node_id, -node.weight)

        uf = self.unfinished
        ac = self.task_manager.len_active
        self.logger.info(f"Number of unfinished nodes: {uf} ({ac} active).")

    @property
    def unfinished(self):
        return self.registry.unfinished - self.task_manager.len_finished

    @property
    def status(self):
        avail = psutil.virtual_memory().available / 1024 ** 2
        limit = MEM_MAXIMAL / 1024 ** 2
        hard = MEM_CRITICAL / 1024 ** 2
        status = f"Mem avail: {avail:.0f} MB (limits: {limit:.0f}/{hard:.0f} MB)"

        w = len(self.workers)
        r = len(self.busy_workers)
        a = self.task_manager.len_active
        t = self.unfinished
        status += f" -- Workers: {w} ({r} busy) -- Tasks: {t} ({a} active)"
        return status

    def run(self, nums_electrons: list):
        self.logger.info(f"Build nodes in a pool of {self.max_workers} workers.")
        t = time.time()

        # Run until all nodes are completed
        last_status = ""
        while len(nums_electrons) or (self.unfinished and not self.stop_event.is_set()):
            if self.unfinished <= self.max_workers and not self.task_manager.len_active and nums_electrons:
                self.register(nums_electrons.pop(0))

            status = self.status
            if status != last_status:
                self.logger.info(f"Status: {status}")
                last_status = status

            self.update_rss()
            free_space = self.memory_watchdog()
            self.process_cleanup()
            if free_space:
                self.schedule_tasks()
            self.store_result()

        # Stop all workers
        self.logger.info(f"Stop all {len(self.workers)} workers.")
        for proc, task_queue, node_id in self.workers.values():
            task_queue.put(None)
            proc.join()

        t = time.time() - t
        self.logger.info(f"Total execution time: {t:.1f} seconds")


def run_pool(nums_electrons, handlers):
    # Start the worker log queue
    log_queue = multiprocessing.Queue()
    listener = logging.handlers.QueueListener(log_queue, *handlers)
    listener.start()

    # Register handler of the worker log queue in the root logger
    root = logging.getLogger()
    queue_handler = logging.handlers.QueueHandler(log_queue)
    root.addHandler(queue_handler)
    root.setLevel(logging.INFO)

    # Local logger
    logger = logging.getLogger("build")

    try:
        scheduler = PoolScheduler(logger, log_queue)
        scheduler.run(nums_electrons)
    finally:
        root.removeHandler(queue_handler)
        queue_handler.close()
        listener.stop()
