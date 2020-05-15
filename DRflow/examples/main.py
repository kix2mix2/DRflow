

from DRflow.utils.evaluate import evaluate_files
import ray
import psutil
import time

if __name__ == "__main__":
    print("hello world")
    all_names = ['MNIST', 'coil-100', 'stanfordfaces', 'yalefaces', 'Caltech_instruments', 'Caltech_plants',
                 'Caltech_vehicles',
                 'fashionmnist', 'flowers',
                 'paintings', 'oxford_buildings', 'paris_buildings']

    ray.init(num_cpus = 20)
    time.sleep(2.0)

    res = []
    for name in all_names:
        # res.append(evaluate_files.remote(types = ['highlow'], high_dir = '../../data/{}/mini_batch/flatfiles/'.format(name),
        #                dr_dir = '../../data/{}/mini_batch/dr/'.format(name)))

        evaluate_files(types = ['supervised', 'highlow'], high_dir = '../../data/{}/mini_batch/flatfiles/'.format(name),
                       dr_dir = '../../data/{}/mini_batch/dr/'.format(name))


    time.sleep(80000)

    print('Job Done!')



