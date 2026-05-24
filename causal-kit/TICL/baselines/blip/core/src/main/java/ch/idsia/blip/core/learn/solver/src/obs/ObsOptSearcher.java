package ch.idsia.blip.core.learn.solver.src.obs;


import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.LinkedHashSet;


/**
 * (given an order, for each variable select the best parent set that doesn't contain any preceding variable)
 */
public class ObsOptSearcher extends ObsSearcher {

    boolean[] done;

    public ObsOptSearcher(BaseSolver solver) {
        super(solver);
    }

    /**
     * Find the best combination given the order (second way, koller's)
     */
    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        done = new boolean[n_var];

        sk = 0;

        forbidden = new boolean[n_var];

        LinkedHashSet todo = new LinkedHashSet();

        for (int v = 0; v < n_var; v++) {
            todo.add(v);
        }

        // index of the best parent set for that variable so far (at start, it is 0 for every variable)
        int[] best_ps = new int[n_var];

        // weight for the importance sampling
        double[] ws = new double[n_var];

        // indexes
        int[] ix = new int[n_var];

        int last_chosen = -1;

        ParentSet best;

        for (int i = 0; i < n_var; i++) {

            double tot = 0;
            int j = 0;

            // For each variable that has not been chosen yet (n_var - i), build the weight array
            for (int v = 0; v < n_var; v++) {
                if (done[v]) {
                    continue;
                }

                // Update the list of best parent sets; for each variable
                best = m_scores[v][best_ps[v]];
                // check if the last chosen blocks the current best, and find the new best
                if (last_chosen != -1 && find(last_chosen, best.parents)) {
                    best_ps[v] = new_best(v, forbidden, best_ps[v]);
                    best = m_scores[v][best_ps[v]];
                }

                // best_ps[v] has the best
                ws[j] = 1 / (-best.sk);
                tot += ws[j];
                ix[j] = v;
                j++;
            }

            // j now has the number of variables to check

            double r = solver.randDouble() - Math.pow(2, -10);
            int sel = -1;

            for (int v = 0; v < j && sel == -1; v++) {
                // Normalize weights
                double s = ws[v] /= tot;

                if (r < s) {
                    sel = v;
                }
                r -= s;
            }

            // "sel" is the selected index
            int var = ix[sel];

            forbidden[var] = true;
            last_chosen = var;
            done[var] = true;
            str[var] = m_scores[var][best_ps[var]];
            sk += str[var].sk;

            // pf("%d %s \n", var, str[var]);
        }

        return str;
    }

    // Finds the best
    private int new_best(int v, boolean[] forbidden, int start) {
        for (int i = start; i < m_scores[v].length; i++) {
            if (acceptable(m_scores[v][i].parents, forbidden)) {
                return i;
            }
        }

        return -1;
    }

}
