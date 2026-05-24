package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.learn.solver.WinAsobsPertSolver;
import ch.idsia.blip.core.utils.ParentSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.swap;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


/**
 * Hybrid greedy hill exploration
 */

public class WinAsobsSearcherPerturbation extends WinAsobsSearcher {
    protected int a = 3;
    protected int b = 10;
    protected int c = 5;
    private int[] impr_vars;
    private int cnt;
    private int cnt_win;

    public WinAsobsSearcherPerturbation(WinAsobsPertSolver solver) {
        super(solver);
        a = solver.a;
        b = solver.b;
        c = solver.c;
    }

    public ParentSet[] search() {
        vars = smp.sample();
        if (asobs > 0) {
            asobsOpt();
        }
        return perturb();
    }

    public ParentSet[] perturb() {
        double impr_sk = -1.7976931348623157E308D;

        cnt = 0;
        while (cnt < b) {
            solver.checkTime();
            if (!solver.still_time) {
                break;
            }
            initStr();

            winasobs();
            if (sk - 0.1D > impr_sk) {
                impr_sk = sk;
                cloneArray(vars, impr_vars);
                solver.newStructure(str);
                cnt = 0;
            } else {
                cloneArray(impr_vars, vars);
                cnt += 1;
            }
            makePerturbation(vars);
        }
        cnt_win += 1;
        if (cnt_win >= c) {
            if (max_windows < n_var) {
                max_windows += 1;
                if (solver.verbose > 1)
                pf("Improve max_win! %d %.2f \n", max_windows,
                        solver.elapsed);
            }
            solver.checkTime();

            cnt_win = 0;
        }
        return best_str;
    }

    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        impr_vars = new int[n_var];

        cnt_win = 0;
    }

    private void makePerturbation(int[] vars) {
        for (int i = 0; i < n_var * (a / 100.0D); i++) {
            swap(vars, randInt(n_var - 1),
                    randInt(n_var - 1));
        }
    }
}
