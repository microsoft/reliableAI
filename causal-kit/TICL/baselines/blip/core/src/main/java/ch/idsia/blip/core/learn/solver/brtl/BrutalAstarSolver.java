package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalAstarSearcher;


/**
 * BRTL approach, A*
 */
public class BrutalAstarSolver extends BrutalSolver {

    public int exp_limit = 0;

    @Override
    protected String name() {
        return "BRUTAL A* Score";
    }

    @Override
    protected Searcher getSearcher() {
        return new BrutalAstarSearcher(this, tw, exp_limit);
    }

}
