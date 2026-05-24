package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.SkelSolver;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalOldSearcher;


/**
 * BRTL approach, A*
 */
public class BrutalPcGreedySolver extends SkelSolver {

    protected int tw = 3;

    @Override
    protected String name() {
        return "Greedy on a skeleton";
    }

    @Override
    protected Searcher getSearcher() {
        return new BrutalOldSearcher(this, tw);
    }

    public void init(int time, String s, int tw) {
        this.tw = tw;
        init(time, s);
    }
}
