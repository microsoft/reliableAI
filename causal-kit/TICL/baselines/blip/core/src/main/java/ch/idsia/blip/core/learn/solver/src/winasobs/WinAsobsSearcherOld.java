package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.ParentSet;


/**
 * Windows iteration greedy hill exploration
 */
public class WinAsobsSearcherOld extends WinAsobsSearcher {

    private FastList<Integer> fastTodo;

    public WinAsobsSearcherOld(WinAsobsSolver winAsobsSolver) {
        super(winAsobsSolver);
    }

    @Override
    public ParentSet[] search() {

        // Find initial map!
        super.search();

        solver.checkTime();

        vars = new BayesianNetwork(str).getTopologicalOrder();
        ArrayUtils.reverse(vars);

        // list of variable for the optimization, one for each layer
        window = 1;

        while (solver.still_time) {

            if (window > max_windows) {
                break;
            }

            if (fastTodo == null) {
                initT(window);
            }

            // Choose a random variable
            int ix = fastTodo.rand();

            if (greedy(ix)) {
                window = 1;
                if (fastTodo.size() != n_var - 1) {
                    initT(window);
                }
            } else {
                fastTodo.delete(ix);
            }

            if (fastTodo.size() == 0) {
                window += 1;
                fastTodo = null;
            }

            solver.checkTime();
        }

        // solver.logf(2, "After greedy! %.5f - %.3f \n", solver.elapsed, sk);

        return str;
    }

    private void initT(int ly) {
        fastTodo = new FastList<Integer>(this.solver.rand);
        for (int j = 1; j < n_var - ly; j++) {
            fastTodo.add(j);
        }
    }
}
