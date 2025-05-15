from multiprocessing import Process


def parallelize(worker_func, num_workers, task_list, args=()):
    n_tasks = len(task_list)

    assert num_workers >= 0
    assert n_tasks > 0

    if num_workers == 0:
        worker_func(task_list)
    else:

        num_workers = min(num_workers, n_tasks)
        process_list = []
        n_splits = n_tasks // num_workers

        for i in range(num_workers):
            i1 = i * n_splits
            i2 = i1 + n_splits
            sub_tasks = task_list[i1:i2]
            p = Process(target=worker_func, args=(sub_tasks, *args))
            process_list.append(p)

        # Start processes
        for i in range(num_workers):
            process_list[i].start()

        # Join processes
        for i in range(num_workers):
            process_list[i].join()


if __name__ == "__main__":

    def worker(tasks, custom_param):
        s = 0
        for i in range(len(tasks)):
            s += tasks[i]

        print(custom_param, s)

    task_list = list(range(100))
    parallelize(worker, 8, task_list, args=("Name",))
