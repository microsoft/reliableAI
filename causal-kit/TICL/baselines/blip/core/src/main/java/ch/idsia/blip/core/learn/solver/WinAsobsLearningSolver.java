package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.winasobs.WinAsobsLearningSearcher;
import ch.idsia.blip.core.learn.solver.src.winasobs.WinAsobsSearcher;

import java.util.HashMap;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class WinAsobsLearningSolver extends WinAsobsPertSolver {

    public boolean inverse;

    @Override
    public void prepare() {
        super.prepare();

        if (verbose > 0) {
            logf("inverse: %b \n", inverse);
        }
    }

    @Override
    protected Searcher getSearcher() {
        return new WinAsobsLearningSearcher(this);
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        this.inverse = gBool("inverse");
    }

}
