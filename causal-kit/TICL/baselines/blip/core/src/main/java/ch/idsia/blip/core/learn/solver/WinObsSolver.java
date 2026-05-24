package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.WinObsSearcher;

import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class WinObsSolver extends ObsSolver {

    public int max_windows;

    @Override
    public void prepare() {
        super.prepare();

        if (verbose > 0) {
            logf("max window: %d \n", max_windows);
        }
    }

    @Override
    protected Searcher getSearcher() {
        return new WinObsSearcher(this);
    }

    @Override
    protected String name() {
        return f("WinAsobs %d", max_windows);
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        max_windows = gInt("win", 4);
    }
}

