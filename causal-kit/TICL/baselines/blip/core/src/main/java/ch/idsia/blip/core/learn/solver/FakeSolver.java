package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.samp.SimpleSampler;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.utils.ParentSet;


public class FakeSolver extends BaseSolver {
    public FakeSolver(BaseSolver solver) {
        this.rand = solver.rand;
    }

    protected String name() {
        return null;
    }

    public Sampler getSampler() {
        return new SimpleSampler(this.n_var, this.rand);
    }

    protected Searcher getSearcher() {
        return null;
    }

    protected Provider getProvider() {
        return null;
    }

    public void newStructure(ParentSet[] new_str) {}
}
