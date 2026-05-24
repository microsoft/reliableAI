package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.learn.solver.WinAsobsImprSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;


public class WinAsobsSearcherImprove extends WinAsobsSearcher {

    private int a;

    private int b;

    private int c;

    private int d;

    private int max_size;

    private Solution[] bests;

    private int size;

    private boolean fullSols;

    private int cnt_win;

    public WinAsobsSearcherImprove(WinAsobsImprSolver solver) {
        super(solver);
        this.a = solver.a;
        this.b = solver.b;
        this.c = solver.c;
        this.d = solver.d;
    }

    public ParentSet[] search() {
        int[] vars = this.smp.sample();

        if (this.asobs > 0) {
            asobsOpt();
        }

        return improve(vars);
    }

    public ParentSet[] improve(int[] in_vars) {
        this.vars = in_vars;
        initStr();
        winasobs();

        double opt_sk = this.sk;
        int[] opt_vars = ArrayUtils.cloneArray(this.vars);

        int cnt = 0;

        while (cnt < this.b) {
            this.solver.checkTime();
            if (!this.solver.still_time) {
                break;
            }
            makePerturbation();
            initStr();
            winasobs();
            if (this.sk - 0.1D > opt_sk) {
                opt_sk = this.sk;
                ArrayUtils.cloneArray(this.vars, opt_vars);
                cnt = 0;
            } else {
                cnt++;
                ArrayUtils.cloneArray(opt_vars, this.vars);
            }
        }
        addBestSolution(opt_sk, opt_vars);
        if (!this.fullSols) {
            return this.best_str;
        }
        for (int ix = 0; ix < this.size; ix++) {
            Solution sol = this.bests[ix];

            ArrayUtils.cloneArray(sol.vars, this.vars);

            makePerturbation();
            initStr();
            winasobs();
            if (this.sk > sol.sk) {
                this.bests[ix] = new Solution(this.vars, this.sk, 1);
            } else {
                sol.cnt += 1;
            }
        }
        this.cnt_win += 1;
        if (this.cnt_win >= this.d) {
            if (this.max_windows < this.n_var / 10) {
                this.max_windows += 1;
                RandomStuff.pf("Improve max_win! %d %.2f \n",
                        new Object[] {
                    Integer.valueOf(this.max_windows),
                    Double.valueOf(this.solver.elapsed)});
            }
            this.solver.checkTime();

            this.cnt_win = 0;
        }
        return this.best_str;
    }

    boolean addBestSolution(double sk, int[] var) {
        double worst_sk = 0.0D;
        int worst_ix = -1;

        for (int i = 0; i < this.max_size; i++) {
            if (this.bests[i] == null) {
                worst_ix = i;
                this.size += 1;
                break;
            }
            if (Math.abs(this.bests[i].sk - sk) < 0.01D) {
                return false;
            }
            if (this.bests[i].sk < worst_sk) {
                worst_sk = this.bests[i].sk;
                if (this.bests[i].sk < sk) {
                    worst_ix = i;
                }
            }
            if (i == this.max_size - 1) {
                this.fullSols = true;
            }
        }
        if (worst_ix != -1) {
            this.bests[worst_ix] = new Solution(var, sk, this.max_windows);
            return true;
        }
        return false;
    }

    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        this.max_size = this.c;

        this.bests = new Solution[this.max_size];
        this.size = 0;
    }

    private void makePerturbation() {
        for (int i = 0; i < this.n_var * this.a / 100.0D; i++) {
            ArrayUtils.swap(this.vars, randInt(this.n_var - 1),
                    randInt(this.n_var - 1));
        }
    }

    private class Solution
            implements Comparable<Solution> {
        int[] vars;
        double sk;
        int cnt;

        public Solution(int[] vars, double sk, int counter) {
            this.vars = ArrayUtils.cloneArray(vars);
            this.sk = sk;
            this.cnt = counter;
        }

        public int compareTo(Solution other) {
            if (this.sk < other.sk) {
                return 1;
            }
            return -1;
        }
    }
}
