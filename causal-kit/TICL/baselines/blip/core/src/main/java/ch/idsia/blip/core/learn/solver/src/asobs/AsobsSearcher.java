package ch.idsia.blip.core.learn.solver.src.asobs;


import ch.idsia.blip.core.utils.other.TopologicalOrder;
import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.learn.solver.src.obs.ObsSearcher;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.BitSet;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class AsobsSearcher extends ObsSearcher {

    protected BitSet[] descendant;

    protected BitSet[] ancestor;

    private int[][] l_parent_var;

    public AsobsSearcher(BaseSolver solver) {
        super(solver);
    }

    /**
     * Find the best combination given the order (first way, cassio's)
     */
    public ParentSet[] search() {
        vars = smp.sample();

        asobsGain();

        return str;
    }

    public void asobsGain() {
        asobs();

        for (int i = 0; i < n_var; i++) {
            l_parent_var[i] = str[i].parents;
        }
        vars = TopologicalOrder.find(n_var, l_parent_var);
        ArrayUtils.reverse(vars);
        obs();

        boolean gain = true;

        while (gain) {
            gain = greedy();
        }
    }

    public void asobs() {

        sk = 0;

        for (int i = 0; i < n_var; i++) {
            ancestor[i].clear();
            descendant[i].clear();
        }

        for (int v : vars) {
            // System.err.println("var: " + v);
            findBest(v);
        }

        sk = checkSk();
    }

    @Override
    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        ancestor = new BitSet[n_var];
        descendant = new BitSet[n_var];

        for (int i = 0; i < n_var; i++) {
            ancestor[i] = new BitSet(n_var);
            descendant[i] = new BitSet(n_var);
        }

        l_parent_var = new int[n_var][];
    }

    public void findBest(int v) {
        // Search first acceptable parent set (none of the proposed parents is a descendant of the variable)
        // System.err.println("Trying: " + m_scores[v].length);
        int j = 0;

        for (ParentSet pSet : m_scores[v]) {

            // System.err.print (j++ + " ");
            if (acceptable(pSet.parents, descendant[v])) {

                updateStr(v, pSet);

                updateAncDesc(ancestor, descendant, v, pSet);

                return;
            }
        }

        System.out.println(v);
        System.out.println("WAHHHHHHHH");
    }

    protected void updateStr(int v, ParentSet pSet) {
        str[v] = pSet;
        sk += pSet.sk;
    }

    protected void updateAncDesc(int v, ParentSet best) {
        updateAncDesc(ancestor, descendant, v, best);
    }

    protected void updateAncDesc(BitSet[] ancestor, BitSet[] descendant, int v, ParentSet pSet) {
        // Update descendant
        for (int p : pSet.parents) {
            // Add, as descendant of the parent, the variable and all the dand
            // descendant of the variable
            updateAux(descendant, v, p);
            // Do the same for every ascendant of the parent
            for (int i2 : variables) {
                if (ancestor[p].get(i2)) {
                    updateAux(descendant, v, i2);
                }
            }

            // Add, as ancestor of the variable, the parent and all the dand
            // ancestor of the variable
            updateAux(ancestor, p, v);
            // Do the same for every ascendant of the parent
            for (int i2 : variables) {
                if (descendant[v].get(i2)) {
                    updateAux(ancestor, p, i2);
                }
            }

        }
    }

    /**
     * Update the Bitset
     */
    protected void updateAux(BitSet[] t, int v, int p) {
        t[p].set(v);
        t[p].or(t[v]);
    }

    protected boolean acceptable(int[] parents, BitSet forbidden) {
        for (int p : parents) {
            if (forbidden.get(p)) {
                return false;
            }
        }
        return true;
    }
}
