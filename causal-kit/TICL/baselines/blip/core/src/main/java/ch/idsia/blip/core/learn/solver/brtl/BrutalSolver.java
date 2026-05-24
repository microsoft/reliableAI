package ch.idsia.blip.core.learn.solver.brtl;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.ps.MaxScoreProvider;
import ch.idsia.blip.core.learn.solver.ps.Provider;
import ch.idsia.blip.core.learn.solver.samp.Sampler;
import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.brutal.*;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.HashMap;

import static ch.idsia.blip.core.learn.solver.samp.SamplerUtils.getAdvSampler;


/**
 * BRTL approach, Greedy
 */
public class BrutalSolver extends ScoreSolver {

    public int tw;

    public String sampler;

    public String searcher;

    // public List<Clique> bestJuncTree;

    @Override
    protected String name() {
        return "BRUTAL Greedy";
    }

    @Override
    public void prepare() {
        super.prepare();

        if (tw == 0) {
            tw = 3;
        }

        if (verbose > 0) {
            log("tw: " + tw + "\n");
            log("sampler: " + sampler + "\n");
            log("sercher: " + searcher + "\n");
        }
    }

    @Override
    public void init(HashMap<String, String> options) {
        super.init(options);
        sampler = gStr("sampler", null);
        searcher = gStr("sercher", null);

        tw = gInt("tw", 5);
    }

    @Override
    protected Provider getProvider() {
        return new MaxScoreProvider(sc, tw - 1);
    }

    @Override
    public Sampler getSampler() {
        return getAdvSampler(sampler, dat_path, sc.length, this.rand);
    }

    @Override
    protected Searcher getSearcher() {
        if ("old".equals(searcher)) {
            return new BrutalOldSearcher(this, tw);
        } else if ("new".equals(searcher)) {
            return new BrutalNewGreedySearcher(this, tw);
        } else if ("max".equals(searcher)) {
            return new BrutalMaxDirectedSearcher(this, tw);
        } else if ("weight".equals(searcher)) {
            return new BrutalMaxDirectedSearcherWeight(this, tw);
        } else {
            return new BrutalGreedySearcher(this, tw);
        }
    }

    public void init(ParentSet[][] sc, int time, int tw) {
        this.tw = tw;
        super.init(sc, time);
    }

    /*
     public void propose(double new_sk, ParentSet[] new_str, List<Clique> junctTree) {
     synchronized (lock) {
     if (new_sk > best_sk) {
     bestJuncTree = junctTree;
     }
     }
     }*/
}
