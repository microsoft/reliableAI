package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalMaxDirectedSearcher;


/**
 * BRTL approach, MAX
 */
public class BrutalMaxSolver extends BrutalSolver {

    @Override
    protected String name() {
        return "BRUTAL MAX";
    }

    @Override
    protected Searcher getSearcher() {

        return new BrutalMaxDirectedSearcher(this, tw);

        // return new BrutalMaxDirectedSearcherWeight(this, tw);
    }
}
