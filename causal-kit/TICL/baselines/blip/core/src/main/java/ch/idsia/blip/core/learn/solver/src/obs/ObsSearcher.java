package ch.idsia.blip.core.learn.solver.src.obs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.learn.solver.src.ScoreSearcher;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.exp.CyclicGraphException;
import ch.idsia.blip.core.utils.ParentSet;


/**
 * (given an order, for each variable select the best parent set that doesn't contain any preceding variable)
 */
public class ObsSearcher extends ScoreSearcher {

    protected boolean[] voidB;

    protected boolean[] fullB;

    protected boolean[] forbidden;

    protected double eps = -Math.pow(2.0D, -18.0D);

    protected boolean[][] cand;

    protected double old_sk;

    private ParentSet pSet;

    public ObsSearcher(BaseSolver solver) {
        super(solver);
    }

    // Find the best combination given the order (second way, koller's)
    @Override
    public ParentSet[] search() {
        vars = smp.sample();

        obs();

        boolean gain = true;

        while (gain) {
            gain = greedy();
        }
        return str;
    }

    public boolean greedy() {
        int best_ix = -1;

        double best_gain = -eps;

        ArrayUtils.cloneArray(voidB, forbidden);
        for (int ix = 0; ix < n_var - 1; ix++) {
            int v = vars[ix];
            int next = vars[(ix + 1)];

            forbidden[v] = false;
            double gain = -str[next].sk + best(next).sk;

            forbidden[next] = true;
            gain += -str[v].sk + best(v).sk;
            if (gain > best_gain) {
                best_ix = ix;
                best_gain = gain;
            }
            forbidden[v] = true;
        }
        if (best_ix == -1) {
            return false;
        }
        ArrayUtils.cloneArray(voidB, forbidden);
        for (int ix = 0; ix < best_ix; ix++) {
            forbidden[vars[ix]] = true;
        }
        int v = vars[best_ix];
        int next = vars[(best_ix + 1)];

        forbidden[v] = false;
        str[next] = best(next);

        forbidden[next] = true;
        str[v] = best(v);

        ArrayUtils.swapArray(vars, best_ix, best_ix + 1);

        sk += best_gain;
        try {
            new BayesianNetwork(str).checkAcyclic();
        } catch (CyclicGraphException e) {
            e.printStackTrace();
        }
        solver.newStructure(str);

        return true;
    }

    public ParentSet[] obs() {
        sk = 0.0D;

        ArrayUtils.cloneArray(voidB, forbidden);
        for (int v : vars) {
            pSet = best(v);
            str[v] = pSet;
            sk += pSet.sk;

            forbidden[v] = true;
        }
        
        return str;
    }

    protected ParentSet best(int v) {
        for (ParentSet pSet : m_scores[v]) {
            if (acceptable(pSet.parents, forbidden)) {
                return pSet;
            }
        }
        return null;
    }

    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        str = new ParentSet[n_var];

        forbidden = new boolean[n_var];

        voidB = new boolean[n_var];
        fullB = new boolean[n_var];
        for (int i = 0; i < n_var; i++) {
            voidB[i] = false;
            fullB[i] = true;
        }
        cand = new boolean[n_var][];
        for (int i = 0; i < n_var; i++) {
            cand[i] = new boolean[n_var];
            for (ParentSet ps : m_scores[i]) {
                for (int p : ps.parents) {
                    cand[i][p] = true;
                }
            }
        }
    }

    protected boolean acceptable(int[] parents, boolean[] forbidden) {
        for (int p : parents) {
            if (forbidden[p]) {
                return false;
            }
        }
        return true;
    }

}
