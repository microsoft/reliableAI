package ch.idsia.blip.core.learn.solver.src.obs;


import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.BitSet;


public class ObsGreedySearcher extends ObsSearcher {

    public ObsGreedySearcher(BaseSolver solver) {
        super(solver);
    }

    /**
     * Try to improve map with a single switch in the order (second way, koller's)
     *
     * @param vars old variable order
     * @return if an improvement was possible
     */
    public boolean greedy(int[] vars) {

        // Index of the best switch
        int best_i = -1;
        int best_j = -1;
        // Gain from the best switch
        double best_gain = 0;
        // New best parent set
        ParentSet best_pset = null;

        forbidden = new boolean[n_var];

        for (int i = 0; i < (n_var - 1); i++) {

            int var = vars[i];

            BitSet domains = new BitSet();

            for (int p : str[var].parents) {
                domains.set(p);
            }

            for (int j = i + 1; j < n_var; j++) {
                int next = vars[j];

                // If this best parent set contains the next variable in the order
                if (domains.get(next)) {
                    // then it is not eligible for the switch
                    forbidden[var] = true;
                    break;
                }

                for (int p : str[next].parents) {
                    domains.set(p);
                }

                // System.out.printf("%d (%s) con %d, %d\n", var, Arrays.toString(new_str.get(var).parents), next,
                // Arrays.binarySearch(new_str.get(var).parents, next));

                ParentSet nextPSet = str[next];

                for (ParentSet pSet : m_scores[next]) {
                    if (!acceptable(pSet.parents, forbidden)) {
                        // If not acceptable, continue with the next one
                        continue;
                    }
                    double gain = pSet.sk - nextPSet.sk;

                    if (gain < 0) {
                        // If it doesn't beat the new_sk, continue
                        continue;
                    }

                    if (gain > best_gain) {
                        best_gain = gain;
                        best_i = i;
                        best_j = j;
                        best_pset = pSet;
                    }

                    break;
                }
            }

            forbidden[var] = true;
        }

        if (best_i == -1) {
            return false;
        } else {

            // System.out.printf("swapArray: %d and %d, gain: %.5f, new best: %s\n", vars.get(best_ix), vars.get(best_ix + 1), best_gain, best_pset);

            // Update the map for the following index
            str[vars[best_j]] = best_pset;
            // Do the switch
            ArrayUtils.swapArray(vars, best_i, best_j);
            // Update the new_sk
            sk += best_gain;

            return true;
        }

    }

    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        if (solver.verbose > 2) {
            solver.log("going! \n");
        }

        // Find initial map!
        super.search();

        if (solver.verbose > 2) {
            solver.logf("Initial: %.5f (check: %.5f) \n", sk, checkSk());
        }

        solver.checkTime();
        while (solver.still_time) {

            boolean improv = greedy(vars);

            if (solver.verbose > 1) {
                solver.logf("New Greedy! %.5f - %.3f \n", solver.elapsed,
                        sk);
            }

            if (!improv) {
                break;
            }

            solver.checkTime();
        }

        return str;
    }

}
