package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.learn.solver.WinAsobsLearningSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.ParentSet;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class WinAsobsLearningSearcher extends WinAsobsSearcherPerturbation {

    private final boolean inverse;

    // number of iterations we made
    private int iter = 0;

    // number of times we saw that variable in that position
    double[][] ranks;

    // list of variables, used only for computing ranks
    int[] todo;

    public WinAsobsLearningSearcher(WinAsobsLearningSolver solver) {
        super(solver);
        inverse = solver.inverse;
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        todo = new int[n_var];
        ranks = new double[n_var][];
        for (int i = 0; i < n_var; i++) {
            todo[i] = i;
            ranks[i] = new double[n_var];
        }
    }

    public ParentSet[] search() {

        if (iter < 100 || solver.randProb() > 0.5)
            vars = smp.sample();
        else
            vars = goNow();

        iter++;

        asobsOpt();

        winasobs();

        perturb();

        learnNow();

        return best_str;
    }

    private void learnNow() {
        for (int i = 0; i < n_var; i++) {
            int n = best_vars[i];
            ranks[n][i] += 1;
            // tot[n] += best_sk;
        }
    }

    private int[] goNow() {
        // it is the current position we are evaluating
        for (int it = 0; it < n_var; it++) {
            // compute sum of remaining variables
            double sum = 0;
            for (int jt = it; jt < n_var; jt++) {
                int v = todo[jt];
                sum += (ranks[v][it] +1);
            }

            double p = solver.randProb();
            int chosen = -1;
            for (int jt = it; jt < n_var && chosen < 0; jt++) {
                int v = todo[jt];
                double r = (ranks[v][it]+1) / sum;
                if (p < r)
                    chosen = jt;
                else
                    p -= r;
            }

            vars[it] = todo[chosen];
            ArrayUtils.swap(todo, it, chosen);
        }

        if (inverse)
            ArrayUtils.reverse(vars);
        return vars;
    }


}
