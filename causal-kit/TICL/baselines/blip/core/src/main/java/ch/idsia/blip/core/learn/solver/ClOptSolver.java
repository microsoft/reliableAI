package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.ClOptSearcher;
import ch.idsia.blip.core.learn.solver.src.Searcher;

import static ch.idsia.blip.core.utils.RandomStuff.f;


public class ClOptSolver extends ObsSolver {

    public String ph_gobnilp = "experiments/gobnilp";
    public String ph_work = "temp";

    // optimization parameter
    public int w = 5;

    public ClOptSolver(String search) {
        this.searcher = search;
    }

    public ClOptSolver() {}

    @Override
    protected Searcher getSearcher() {
        return new ClOptSearcher(this);
    }

    @Override
    protected String name() {
        return f("ClOptSolver %s", searcher);
    }

    public void setup(String work, String gob) {
        this.ph_work = work;
        this.ph_gobnilp = gob;
    }
}
