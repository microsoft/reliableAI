package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.winasobs.WinAsobsSearcherImprove;

import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class WinAsobsImprSolver extends WinAsobsSolver {

    public int a;

    public int b;

    public int c;

    public int d;

    @Override
    public void prepare() {
        super.prepare();

        if (verbose > 0) {
            logf("a: %d \n", a);
            logf("b: %d \n", b);
            logf("c: %d \n", c);
            logf("c: %d \n", d);
        }
    }

    @Override
    protected Searcher getSearcher() {
        return new WinAsobsSearcherImprove(this);
    }

    @Override
    protected String name() {
        return f("WinAsobs Impr");
    }

    public void init(HashMap<String, String> options) {

        super.init(options);
        this.a = gInt("pa", 3);
        this.b = gInt("pb", 10);
        this.c = gInt("pc", 10);
        this.d = gInt("pd", 5);
    }
}

