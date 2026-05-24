package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalAstarSearcher;


/**
 * BRTL approach, A*
 */
public class BrutalPcAstarSolver extends BrutalPcGreedySolver {

    @Override
    protected String name() {
        return "Brutal A* on a skeleton";
    }

    @Override
    protected Searcher getSearcher() {
        return new BrutalAstarSearcher(this, tw, 0);
    }
}
