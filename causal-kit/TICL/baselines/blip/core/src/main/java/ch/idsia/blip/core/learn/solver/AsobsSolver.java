package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.src.Searcher;
import ch.idsia.blip.core.learn.solver.src.asobs.AsobsSearcher;
import ch.idsia.blip.core.learn.solver.src.asobs.InAsobsSearcher;
import ch.idsia.blip.core.learn.solver.src.obs.InobsSearcher4;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * (given an order, for each variable select the best parent set compatible with the previous assignment).
 */
public class AsobsSolver extends ObsSolver {

    @Override
    protected Searcher getSearcher() {

        if ("inasobs".equals(searcher)) {
            return new InAsobsSearcher(this);
        } else if ("inasobs".equals(searcher)) {
            return new InAsobsSearcher(this);
        } else if ("inasobsnew".equals(searcher)) {
            return new InAsobsSearcher(this);
        } else if ("inobs4".equals(searcher)) {
            return new InobsSearcher4(this);
        }

        return new AsobsSearcher(this);
    }

    @Override
    protected String name() {
        return f("Asobs %s", searcher);
    }

}
