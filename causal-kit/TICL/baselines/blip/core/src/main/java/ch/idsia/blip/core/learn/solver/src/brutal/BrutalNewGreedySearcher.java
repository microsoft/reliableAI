package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceAndIncreaseArray;


/**
 * IMPROVES again?!
 * (may generate memory problem, in that case fall back to old searcher)
 */
public class BrutalNewGreedySearcher extends BrutalGreedySearcher {

    boolean[] already;

    public BrutalNewGreedySearcher(BaseSolver solver, int tw) {
        super(solver, tw);
    }

    @Override
    protected void clear() {
        super.clear();

        already = new boolean[n_var];
    }

    // Greedily winasobs a network!
    @Override
    public ParentSet[] search() {

        vars = smp.sample();

        // clear all
        clear();

        // Init the first maximal clique
        initClique();

        Pair<ParentSet, SIntSet> res;

        // For every new variable, add a new maximal clique
        for (int i = tw + 1; i < n_var; i++) {

            checkOptimals(i);

            if (already[i]) {
                continue;
            }

            int v = vars[i];

            // Search the best parent set given the handlers
            // (best parent set inside available quasi-maximal cliques)
            res = bestHandler(v);
            // update the chosen parent set
            update(v, res.getFirst());
            // add the new handlers
            // add new handler = new clique with size tw
            // created  just now
            SIntSet h = res.getSecond();

            for (int elim : h.set) {
                addHandler(new SIntSet(reduceAndIncreaseArray(h.set, v, elim)));
            }

            solver.checkTime();
            if (!solver.still_time) {
                return null;
            }
        }

        return new_str;
    }

    private void write(int i) {
        BayesianNetwork best = new BayesianNetwork(new_str);

        // best.writeGraph("/home/loskana/Desktop/what/" + i);
        solver.writeStructure("/home/loskana/Desktop/what/" + i, -10, new_str);

        try {
            best.checkTreeWidth(tw);
            best.checkAcyclic();
        } catch (Exception ex) {
            ex.printStackTrace();
            best.writeGraph("/home/loskana/Desktop/what/whuy");
        }
    }

    // From the position i in vars, check if there is any variable with already the optimal parent set in one handler
    private void checkOptimals(int start) {

        for (int j = start; j < n_var; j++) {

            if (already[j]) {
                continue;
            }

            int v = vars[j];

            // get best parent set for the variable
            ParentSet best = m_scores[v][0];

            // check if there are already some handlers
            TreeSet<SIntSet> good = evaluate(best);

            if (good.size() > 0) {
                SIntSet h = rand(good);

                update(v, best);
                for (int elim : h.set) {
                    addHandler(
                            new SIntSet(reduceAndIncreaseArray(h.set, v, elim)));
                }

                already[j] = true;
            }

        }
    }
}
