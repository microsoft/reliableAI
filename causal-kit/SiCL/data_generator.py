import numpy as np
from avici_Tools import simulate_data
from utils.tools import dag_to_dag, dag_to_vstrucs


def avici_continuous_generator(d=30, path="avici_Tools/myiddata.yaml", datasize=100,
                               n_interv=0, unbiased=False, forcetforks=False):
    while True:
        if n_interv:
            g, x, x_int = simulate_data(d=d, n=datasize - n_interv, path=path, seed=None, n_interv=n_interv)
            x = np.stack([x, x_int], axis=-1)


        else:
            g, x = simulate_data(d=d, n=datasize, path=path, seed=None)
            if unbiased:
                g = dag_to_dag(g)

        cube_tforks, cube_vstrucs = dag_to_vstrucs(g)
        if g.max() != 0:
            if forcetforks and cube_tforks.max() == 0:
                continue
            break
    return x, g


if __name__ == "__main__":
    x, g = avici_continuous_generator(d=5, unbiased=True)
    pass
