package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.winasobs.WinAsobsSearcher;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class WinAsobsSolver extends WinObsSolver {

    @Override
    protected Searcher getSearcher() {
        return new WinAsobsSearcher(this);
    }

}
