package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.winasobs.WinAsobsSearcherPerturbation;

import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class WinAsobsPertSolver extends WinAsobsSolver {

    public int a;

    public int b;

    public int c;

    @Override
    public void prepare() {
        super.prepare();

        if (verbose > 0) {
            logf("a: %d \n", a);
            logf("b: %d \n", b);
        }

    }

    @Override
    protected Searcher getSearcher() {
        return new WinAsobsSearcherPerturbation(this);
    }

    @Override
    protected String name() {
        return f("WinAsobs Pert %d", max_windows);
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        this.a = gInt("pa", 1);
        this.b = gInt("pb", 1);
        this.c = gInt("pc", 1);
    }

}
