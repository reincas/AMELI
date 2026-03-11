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

    def add_active(self, node_id, priority):
        """ Add a node id to the active state if it hasn't been seen before. """

        # Sanity check, skip known node id
        if node_id in self._finished or node_id in self._pending:
            return
        if any(node_id == nid for _, nid in self._active_heap):
            return

        # Push node id on the active heap
        heapq.heappush(self._active_heap, (priority, node_id))

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

    def revert_to_active(self, node_id, priority):
        """ Move node id from pending back to active state. """

        self._pending.remove(node_id)
        heapq.heappush(self._active_heap, (priority, node_id))

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
# Worker process and ProcManager class
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


class ProcManager:
    """ Manager for all worker processes. """

    def __init__(self, max_workers, manager, result_queue, log_queue, stop_event):
        self.max_workers = max_workers
        self.manager = manager
        self.result_queue = result_queue
        self.log_queue = log_queue
        self.stop_event = stop_event

        # Workers: {pid: (process_obj, task_queue, node_id)}
        self.workers = {}

    @property
    def busy_count(self):
        return sum(1 for _, _, nid in self.workers.values() if nid is not None)

    @property
    def idle_count(self):
        return sum(1 for _, _, nid in self.workers.values() if nid is None)

    @property
    def idle_workers(self):
        return [pid for pid, (_, _, nid) in self.workers.items() if nid is None]

    @property
    def busy_workers(self):
        return [pid for pid, (_, _, nid) in self.workers.items() if nid is not None]

    @property
    def all_busy(self):
        return self.busy_count >= self.max_workers

    def get_busy_rss(self):
        """ Return PID, RSS and node_id for each busy worker. """

        result = []
        for pid, (_, _, node_id) in self.workers.items():
            if node_id is None:
                continue
            rss = self.get_rss(pid)
            if not rss:
                continue
            result.append((pid, rss, node_id))
        return result

    def get_worker(self):
        """ Return a (process, task_queue) tuple if a worker is available or can be created. """

        # Try to find an idle worker
        for proc, task_queue, nid in self.workers.values():
            if nid is None:
                return proc, task_queue

        # Try to spawn a new worker
        if len(self.workers) < self.max_workers:
            task_queue = self.manager.Queue()
            args = (task_queue, self.result_queue, self.log_queue, self.stop_event)
            proc = multiprocessing.Process(target=pool_worker, args=args)
            proc.start()
            self.workers[proc.pid] = (proc, task_queue, None)
            return proc, task_queue

        # No worker available
        return None, None

    def assign(self, pid, node_id):
        """ Assign node_id to the given worker. """

        proc, task_queue, _ = self.workers[pid]
        self.workers[pid] = (proc, task_queue, node_id)

    def release(self, pid):
        """ Mark given worker as idle. """

        if pid in self.workers:
            proc, task_queue, _ = self.workers[pid]
            self.workers[pid] = (proc, task_queue, None)

    def get_rss(self, pid):
        """ Return physical memory size (RSS) of given process, if it exists. """

        try:
            return psutil.Process(pid).memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def terminate_idle(self):
        """ Kill an idle worker and return its PID and RSS. """

        idle_workers = self.idle_workers
        if not idle_workers:
            return None, None

        pid = idle_workers[0]
        rss = self.get_rss(pid)
        node_id = self.terminate(pid)
        assert node_id is None
        return pid, rss

    def terminate_smallest(self):
        """ Kill smallest worker and return its PID, RSS, and node_id. """

        worker_stats = [(self.get_rss(pid), pid) for pid in self.busy_workers]
        worker_stats.sort()
        rss, pid = worker_stats[0]
        node_id = self.terminate(pid)
        return pid, rss, node_id

    def terminate(self, pid):
        """ Kill the given worker and return the node_id it was working on. """

        proc, task_queue, node_id = self.workers.pop(pid)
        task_queue.put(None)
        proc.terminate()
        proc.join()
        return node_id

    def cleanup_dead_workers(self):
        """ Return a list of node_ids that were lost to crashed worker processes. """

        lost_nodes = []
        for pid, (proc, task_queue, nid) in list(self.workers.items()):
            if not proc.is_alive():
                if nid is not None:
                    lost_nodes.append(nid)
                self.workers.pop(pid)
        return lost_nodes

    def stop(self):
        """ Stop all worker processes. """

        for proc, task_queue, _ in self.workers.values():
            task_queue.put(None)
            proc.join()


##########################################################################
# PoolScheduler class
##########################################################################

class PoolScheduler:
    def __init__(self, logger, log_queue):
        self.logger = logger
        self.log_queue = log_queue

        # Memory stats collected and updated for every processed node
        self.node_stats = {}

        # Initialise the task manager
        self.task_manager = TaskManager()

        # Initialise worker process manager
        max_workers = multiprocessing.cpu_count()
        self.manager = multiprocessing.Manager()
        self.result_queue = self.manager.Queue()
        self.stop_event = self.manager.Event()
        self.proc_manager = ProcManager(max_workers, self.manager, self.result_queue, self.log_queue, self.stop_event)

        # Initialize the node registry
        self.registry = Registry()

    def max_rss(self, node_id):
        if node_id not in self.node_stats:
            return 0
        return self.node_stats[node_id]

    def update_rss(self):
        for proc, rss, node_id in self.proc_manager.get_busy_rss():
            if not node_id in self.node_stats or rss > self.node_stats[node_id]:
                self.node_stats[node_id] = rss

    def memory_watchdog(self):

        # Skip if enough memory is still available
        vmem = psutil.virtual_memory()
        if len(self.proc_manager.workers) == 0 or vmem.available > MEM_MAXIMAL:
            return True

        # Try to terminate an idle worker
        pid, rss = self.proc_manager.terminate_idle()
        if pid:
            rss /= 1024 ** 2
            self.logger.warning(f"LOW MEMORY ({vmem.percent:.1f} %): Terminated idle PID {pid} using {rss:.0f}MB")
            return False

        # Skip termination of busy workers if enough memory is still available or if just one busy worker is left
        if vmem.available > MEM_CRITICAL and len(self.proc_manager.busy_workers) > 1:
            return False

        # Terminate the smallest worker
        pid, rss, node_id = self.proc_manager.terminate_smallest()
        rss /= 1024 ** 2
        self.logger.warning(f"CRITICAL MEMORY ({vmem.percent:.1f} %): Terminated busy PID {pid} using {rss:.0f}MB")

        # Put node on the heap again if the worker was active
        if node_id is not None:
            node = self.registry.nodes[node_id]
            self.task_manager.revert_to_active(node_id, node.priority)
        return False

    def process_cleanup(self):
        lost_node_ids = self.proc_manager.cleanup_dead_workers()
        for node_id in lost_node_ids:
            node = self.registry.nodes[node_id]
            self.task_manager.revert_to_active(node_id, node.priority)
            self.logger.warning(f"Rescheduled lost node {node_id}")

    def schedule_tasks(self):

        # Skip if maximum number of active workers reached
        if self.proc_manager.all_busy:
            return

        # Skip if no memory available
        vmem = psutil.virtual_memory()
        if vmem.available < MEM_MAXIMAL:
            return

        # Get next task
        node_id = self.task_manager.get_next()
        if node_id is None:
            return
        node = self.registry.nodes[node_id]

        # Kill idle workers until enough memory is available for this task
        rss = self.max_rss(node_id)
        while vmem.available - rss < MEM_CRITICAL and self.proc_manager.idle_count:
            pid, rss = self.proc_manager.terminate_idle()
            if pid:
                rss /= 1024 ** 2
                self.logger.warning(f"Terminated idle PID {pid} using {rss:.0f}MB")
            vmem = psutil.virtual_memory()

        # Skip task if still not enough memory available, but at least one busy worker
        if vmem.available - rss < MEM_CRITICAL and self.proc_manager.busy_count:
            self.task_manager.revert_to_active(node_id, node.priority)
            return

        # Get worker, if available
        proc, task_queue = self.proc_manager.get_worker()
        if not proc:
            self.task_manager.revert_to_active(node_id, node.priority)
            return

        # Schedule task to the worker
        self.proc_manager.assign(proc.pid, node_id)
        task_queue.put(node)

    def store_result(self):
        try:
            node_id, pid = self.result_queue.get(timeout=0.5)
        except queue.Empty:
            return

        # Mark node as finished
        self.task_manager.mark_finished(node_id)
        node = self.registry.nodes[node_id]
        node.exists = True

        # Try to activate all child nodes
        for child_id in node.children:
            self.add_node(child_id)

        # Log result
        rss = self.max_rss(node_id) / 1024 ** 2
        self.logger.info(f"Finished {node} [maximum RSS: {rss:.0f} MB].")

        # Release worker
        self.proc_manager.release(pid)

    def add_node(self, node_id):
        node = self.registry.nodes[node_id]
        if not node.exists and node.in_degree == 0:
            self.task_manager.add_active(node_id, node.priority)

    def register(self, num_electrons: int):
        self.logger.info(f"Register all matrices for lanthanide ion f{num_electrons}.")

        # Register all nodes
        for kwargs in matrix_args(num_electrons):
            self.registry.register(MatrixNode, **kwargs)

        # Activate all nodes with zero in-degree
        for node_id in self.registry.nodes.keys():
            self.add_node(node_id)

        uf = self.registry.unfinished
        ac = self.task_manager.len_active
        self.logger.info(f"Number of unfinished nodes: {uf} ({ac} active).")

    @property
    def status(self):
        avail = psutil.virtual_memory().available / 1024 ** 2
        limit = MEM_MAXIMAL / 1024 ** 2
        hard = MEM_CRITICAL / 1024 ** 2
        status = f"Mem avail: {avail:.0f} MB (limits: {limit:.0f}/{hard:.0f} MB)"

        w = len(self.proc_manager.workers)
        r = self.proc_manager.busy_count
        t = self.registry.unfinished
        a = self.task_manager.len_active
        status += f" -- Workers: {w} ({r} busy) -- Tasks: {t} ({a} active)"
        return status

    def run(self, nums_electrons: list):
        max_workers = self.proc_manager.max_workers
        self.logger.info(f"Build nodes in a pool of {max_workers} workers.")
        t = time.time()

        # Run until all nodes are completed
        last_status = time.time()
        while (self.registry.unfinished and not self.stop_event.is_set()) or nums_electrons:
            if nums_electrons:
                self.register(nums_electrons.pop(0))

            now = time.time()
            if now - last_status >= 5.0:
                self.logger.info(f"Status: {self.status}")
                last_status = now

            self.update_rss()
            free_space = self.memory_watchdog()
            self.process_cleanup()
            if free_space:
                self.schedule_tasks()
            self.store_result()

        # Stop all workers
        self.logger.info(f"Stop all {len(self.proc_manager.workers)} workers.")
        self.proc_manager.stop()

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
