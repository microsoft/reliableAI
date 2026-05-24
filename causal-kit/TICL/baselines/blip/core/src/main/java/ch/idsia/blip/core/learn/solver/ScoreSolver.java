package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.ps.MaxScoreProvider;
import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.ps.SimpleProvider;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.samp.SimpleSampler;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.HashMap;


public abstract class ScoreSolver extends BaseSolver {

    public String dat_path;

    public int max_parents;

    @Override
    public Sampler getSampler() {
        return new SimpleSampler(sc.length, this.rand);
    }

    @Override
    protected Provider getProvider() {
        if (max_parents == 0) {
            return new SimpleProvider(sc);
        } else {
            return new MaxScoreProvider(sc, max_parents);
        }
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        max_parents = gInt("max_parents", 0);
        dat_path = gStr("dat_path", null);
    }

    public void init(ParentSet[][] sc) {
        this.sc = sc;
        this.n_var = sc.length;
    }

    public void init(ParentSet[][] sc, int time) {
        init(sc);
        this.max_exec_time = time;
    }
}
