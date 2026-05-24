package ch.idsia.blip.core.learn.solver.src.obs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.FastList;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.Arrays;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class InobsSearcher4 extends ObsSearcher {

    private int[][] cand;

    public InobsSearcher4(BaseSolver solver) {
        super(solver);
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        cand = new int[n_var][];

        for (int i = 0; i < n_var; i++) {
            TIntHashSet l = new TIntHashSet();

            for (ParentSet ps : scores[i]) {
                for (int p : ps.parents) {
                    l.add(p);
                }
            }

            cand[i] = l.toArray();
            Arrays.sort(cand[i]);
        }
    }

    /**
     * Select a variable, try to improve the score moving it on the list
     *
     * @param vars old variable order
     * @return if an improvement was possible
     */
    public boolean greedy(int[] vars, int ix) {

        int theChosen = vars[ix];

        // pf("theChosen: %d \n", theChosen);
        // p(Arrays.toString(vars));

        // Index of the best switch
        ParentSet[] best_str = null;
        int[] best_vars = new int[n_var];
        // Gain from the best switch
        double best_sk = sk;

        ParentSet[] new_str = cloneStr(str);

        int[] new_vars = new int[n_var];

        for (int i = 0; i < n_var; i++) {
            new_vars[i] = vars[i];
        }

        forbidden = new boolean[n_var];
        for (int i = 0; i < ix; i++) {
            forbidden[vars[i]] = true;
        }

        int lim;

        for (lim = 0; lim < ix; lim++) {
            if (find(vars[lim], cand[theChosen])) {
                break;
            }
        }

        for (int ix2 = ix; ix2 >= lim + 1; ix2--) {
            // Switch in the order ix2 and ix
            varSwitch(ix2, new_str, new_vars, forbidden);
            double sk = skore(new_str);

            if (sk > best_sk) {
                best_sk = sk;
                best_str = cloneStr(new_str);
                cloneArray(new_vars, best_vars);
            }

            // checkCorrect(new_str, new_vars);

        }

        // If no best gain, return false
        if (best_str == null) {
            return false;
        }

        // Save best
        str = best_str;
        sk = best_sk;
        cloneArray(best_vars, vars);
        return true;

    }

    private void checkCorrect(ParentSet[] new_str, int[] new_vars) {
        for (int i = 0; i < n_var; i++) {
            int jx = ArrayUtils.index(i, new_vars);

            for (int p : new_str[i].parents) {
                int jx2 = ArrayUtils.index(p, new_vars);

                if (jx2 < jx) {
                    p("Cdgfgfgdf");
                }
            }

        }

        BayesianNetwork bn = new BayesianNetwork(new_str);

        try {
            bn.checkAcyclic();
        } catch (CyclicGraphException e) {
            e.printStackTrace();
        }
    }

    private ParentSet[] cloneStr(ParentSet[] s) {
        ParentSet[] new_s = new ParentSet[s.length];

        for (int i = 0; i < s.length; i++) {
            new_s[i] = s[i];
        }
        return new_s;
    }

    private void varSwitch(int ix, ParentSet[] str, int[] vars, boolean[] forbidden) {
        int a = vars[ix];
        int b = vars[ix - 1];

        forbidden[a] = true;

        // pf("Old parent set for %d: %s \n", b, str[b]);

        // If in the parent set of a there is b, we need to assign a new parent set
        if (find(a, str[b].parents)) {
            bests(b, str, forbidden);
        }

        // pf("New parent set for %d: %s \n", b, str[b]);

        // pf("Old parent set for %d: %s \n", a, str[a]);

        // Find new best parent set for a, now that b is available
        forbidden[b] = false;
        if (find(b, cand[a])) {
            bests(a, str, forbidden);
        }

        // pf("New parent set for %d: %s \n", a, str[a]);
        // p("");

        ArrayUtils.swapArray(vars, ix, ix - 1);
    }

    private void bests(int a, ParentSet[] str, boolean[] forbidden) {
        for (ParentSet pSet : m_scores[a]) {
            if (acceptable(pSet.parents, forbidden)) {
                str[a] = pSet;
                return;
            }
        }
    }

    private double skore(ParentSet[] s) {
        double check = 0.0;

        for (ParentSet p : s) {
            check += p.sk;
        }
        return check;

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

        FastList<Integer> todo = new FastList<Integer>(solver.rand);

        initT(todo);

        while (solver.still_time) {

            // Choose a random variable
            int ix = todo.rand();

            if (greedy(vars, ix)) {
                if (todo.size() != n_var - 1) {
                    initT(todo);
                }
            } else {
                todo.delete(ix);
            }

            if (todo.size() == 0) {
                break;
            }

            solver.checkTime();
        }

        if (solver.verbose > 2) {
            solver.logf("After greedy! %.5f - %.3f \n", solver.elapsed, sk);
        }

        return str;
    }

    private void initT(FastList todo) {
        todo.reset();
        for (int i = 1; i < n_var; i++) {
            todo.add(i);
        }
    }

}
