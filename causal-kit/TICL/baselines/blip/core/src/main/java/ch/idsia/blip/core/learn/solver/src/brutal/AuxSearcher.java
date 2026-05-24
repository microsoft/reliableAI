package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.learn.solver.samp.SimpleSampler;
import ch.idsia.blip.core.learn.solver.src.WinObsSearcher;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class AuxSearcher extends WinObsSearcher {

    private int[] clique;

    public AuxSearcher(BaseSolver solver, int max_windows) {
        super(solver);
        this.max_windows = max_windows;
    }

    public void setClique(ParentSet[][] scores, int[] initCl) {
        this.clique = initCl;

        Arrays.sort(this.clique);

        ParentSet[][] newPSet = new ParentSet[this.clique.length][];

        for (int i = 0; i < this.clique.length; i++) {
            List<ParentSet> l = new ArrayList();

            for (ParentSet orig : scores[this.clique[i]]) {
                boolean keep = true;
                int[] new_parents = new int[orig.parents.length];

                for (int j = 0; j < orig.parents.length; j++) {
                    int pos = pos(orig.parents[j], this.clique);

                    if (pos >= 0) {
                        new_parents[j] = pos;
                    } else {
                        keep = false;
                    }
                }
                if (keep) {
                    l.add(new ParentSet(orig.sk, new_parents));
                }
            }
            newPSet[i] = new ParentSet[l.size()];
            for (int j = 0; j < l.size(); j++) {
                newPSet[i][j] = ((ParentSet) l.get(j));
            }
        }
        init(newPSet);

        this.smp = new SimpleSampler(this.clique.length, this.solver.rand);
    }

    public ParentSet[] getComplete(ParentSet[][] scores, ParentSet[] best_str) {
        ParentSet[] comp = new ParentSet[scores.length];

        for (int i = 0; i < this.clique.length; i++) {
            int[] new_parents = new int[best_str[i].parents.length];

            for (int k = 0; k < best_str[i].parents.length; k++) {
                new_parents[k] = this.clique[best_str[i].parents[k]];
            }
            comp[this.clique[i]] = new ParentSet(best_str[i].sk, new_parents);
        }
        return comp;
    }
}
