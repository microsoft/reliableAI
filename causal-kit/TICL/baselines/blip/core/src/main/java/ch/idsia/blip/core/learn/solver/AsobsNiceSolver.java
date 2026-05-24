package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.asobs.AsobsSearcher;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.BitSet;


/**
 * Asobs nice solver - for nice graphs
 */
public class AsobsNiceSolver extends AsobsSolver {

    @Override
    protected String name() {
        return "AsobsNice";
    }

    @Override
    protected Searcher getSearcher() {
        return new AsobsNiceSearcher(this);
    }

    private static class AsobsNiceSearcher extends AsobsSearcher {
        TIntArrayList[] childrens;

        public AsobsNiceSearcher(BaseSolver solver) {
            super(solver);
        }

        @Override
        public void init(ParentSet[][] scores, int thread) {
            super.init(scores, thread);

            childrens = new TIntArrayList[n_var];
            for (int i = 0; i < n_var; i++) {
                childrens[i] = new TIntArrayList();
            }
        }

        @Override
        public ParentSet[] search() {

            vars = smp.sample();

            // solver.log(3, "Trying new one! \n");

            for (int i = 0; i < n_var; i++) {
                childrens[i].clear();
            }

            asobs();

            return str;
        }

        @Override
        protected void updateStr(int v, ParentSet pSet) {
            super.updateStr(v, pSet);

            for (int p : pSet.parents) {
                childrens[p].add(v);
            }
        }

        @Override
        protected boolean acceptable(int[] parents, BitSet forbidden) {

            // Check parents for variable
            int max_degree = 4;

            if (parents.length > max_degree) {
                return false;
            }

            // Check usual stuff
            if (!super.acceptable(parents, forbidden)) {
                return false;
            }

            // Check this new parents for maximum degree
            for (int p : parents) {
                if (childrens[p].size() + 1 > max_degree) {
                    return false;
                }
            }

            return true;
        }
    }

}
