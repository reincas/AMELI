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
import psutil
import time

from ..matrix import Matrix
from .graph import Registry, MatrixNode
from .operators import matrix_args

TIMEOUT = 1.0


##########################################################################
# Memory statistics
##########################################################################

def memory():
    # Memory used by this process
    process = psutil.Process(os.getpid())
    proc_mem = process.memory_info().rss / (1024 ** 3)

    # Total available memory
    vmem = psutil.virtual_memory()
    avail_mem = vmem.available / (1024 ** 3)

    # Total amount of virtual memory on the system
    total_mem = vmem.total / (1024 ** 3)

    return f"proc: {proc_mem:.3f} GB, avail: {avail_mem:.3f} GB, total: {total_mem:.3f} GB"


##########################################################################
# Synchronous build process
##########################################################################

def build_matrix(dtype, config_name, name, state_space, reduced=False):
    logger = logging.getLogger("build")

    matrix_str = f"Matrix({dtype}, {config_name}, {name}, {state_space}, reduced={reduced})"

    if Matrix.exists(dtype, config_name, name, state_space, reduced):
        logger.info(f"{matrix_str} exists.")
        return

    logger.info(f"Generate {matrix_str}.")
    t = time.time()
    Matrix(dtype, config_name, name, state_space, reduced)
    t = time.time() - t
    logger.info(f"Matrix generation took {t:.1f} seconds / {memory()}")


def run_sync(num_electrons, handlers):
    t = time.time()

    # Register log handlers in root logger
    root = logging.getLogger()
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(logging.DEBUG)

    # Local logger
    logger = logging.getLogger("build")
    logger.info(f"Build matrices for lanthanide ion f{num_electrons}.")

    for kwargs in matrix_args(num_electrons):
        build_matrix(**kwargs)

    t = time.time() - t
    logger.info(f"Total execution time: {t:.1f} seconds")


##########################################################################
# Worker process
##########################################################################

def pool_worker(task_queue, result_queue, log_queue, stop_event, registry):
    root = logging.getLogger()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.DEBUG)
    logger = logging.getLogger(f"[{os.getpid()}]")
    logger.info(f"Worker active / {memory()}")

    while True:
        node_id = task_queue.get()
        if node_id is None:
            break

        node = registry[node_id]
        logger.info(f"Worker generating {node}")
        try:
            node.generate()
            result_queue.put(node_id)
        except Exception as e:
            logger.error(f"FATAL WORKER ERROR in node {node_id} / {memory()}: {e}", exc_info=True)
            stop_event.set()
            result_queue.put(node_id)
            break
        gc.collect()
        logger.info(f"Worker done / {memory()}")


##########################################################################
# Registry housekeeping functions
##########################################################################

def activate(node_id, registry, in_degree, activated, completed, ready_heap):
    if node_id in activated:
        return
    if in_degree[node_id] != 0:
        return

    node = registry[node_id]
    if node.exists:
        mark_complete(node_id, registry, in_degree, activated, completed, ready_heap)
    else:
        heapq.heappush(ready_heap, (-node.weight, node_id))
    activated.add(node_id)


def mark_complete(node_id, registry, in_degree, activated, completed, ready_heap):
    if node_id in completed:
        return

    node = registry[node_id]
    for child_id in node.children:
        if in_degree[child_id] > 0:
            in_degree[child_id] -= 1
        activate(child_id, registry, in_degree, activated, completed, ready_heap)
    completed.add(node_id)


##########################################################################
# Pool executor
##########################################################################

def run_pool(num_electrons, handlers):
    t = time.time()

    manager = multiprocessing.Manager()
    task_queue = manager.Queue(maxsize=1)
    result_queue = manager.Queue()
    log_queue = manager.Queue()
    stop_event = manager.Event()

    # Start the listener
    listener = logging.handlers.QueueListener(log_queue, *handlers)
    listener.start()

    # Register queue handler in root logger
    root = logging.getLogger()
    root.addHandler(logging.handlers.QueueHandler(log_queue))
    root.setLevel(logging.INFO)

    # Local logger
    logger = logging.getLogger("build")

    # Register all available tensor operator matrices
    logger.info(f"Register all matrices for lanthanide ion f{num_electrons}.")
    registry = Registry()
    for kwargs in matrix_args(num_electrons):
        registry.register(MatrixNode, **kwargs)

    # Number of unfinished parents
    in_degree = {node_id: len(node.parents) for node_id, node in registry.items()}
    activated = set()
    total_nodes = len(registry)
    completed = set()
    ready_heap = []

    # Queue initial nodes without parents
    for node_id in in_degree:
        activate(node_id, registry, in_degree, activated, completed, ready_heap)

    # Run until all nodes are completed
    max_workers = multiprocessing.cpu_count()
    logger.info(f"Build matrices for lanthanide ion f{num_electrons} in a pool of {max_workers} workers.")
    slots = multiprocessing.Semaphore(max_workers)
    args = (task_queue, result_queue, log_queue, stop_event, registry.nodes)
    workers = []
    while len(completed) < total_nodes and not stop_event.is_set():

        if ready_heap and slots.acquire(block=False):
            priority, node_id = heapq.heappop(ready_heap)
            task_queue.put(node_id)

            if len(workers) < max_workers:
                proc = multiprocessing.Process(target=pool_worker, args=args)
                proc.start()
                workers.append(proc)

        try:
            finished_id = result_queue.get(timeout=TIMEOUT)
        except:
            continue

        mark_complete(finished_id, registry, in_degree, activated, completed, ready_heap)
        slots.release()
        logger.info(f"Finished {len(completed)}/{total_nodes} data container tasks.")

    # Kill all workers
    logger.info("Stop all workers.")
    for _ in range(len(workers)):
        task_queue.put(None)
    for worker in workers:
        worker.join()

    listener.stop()

    t = time.time() - t
    logger.info(f"Total execution time: {t:.1f} seconds")
