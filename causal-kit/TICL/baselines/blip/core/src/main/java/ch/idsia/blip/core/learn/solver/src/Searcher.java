package ch.idsia.blip.core.learn.solver.src;


import ch.idsia.blip.core.utils.ParentSet;


/**
 * New search following the given order
 */
public interface Searcher {

    ParentSet[] search();

    void init(ParentSet[][] scores, int thread);
}
