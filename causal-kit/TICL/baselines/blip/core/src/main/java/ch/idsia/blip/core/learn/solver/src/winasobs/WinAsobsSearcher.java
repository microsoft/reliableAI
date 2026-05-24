package ch.idsia.blip.core.learn.solver.src.winasobs;


import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.src.WinObsSearcher;
import ch.idsia.blip.core.learn.solver.src.asobs.AsobsSearcher;
import ch.idsia.blip.core.utils.ParentSet;


/**
 * Hybrid greedy hill exploration
 */
public class WinAsobsSearcher extends WinObsSearcher {

    private int[][] l_parent_var;

    private AsobsSearcher s;

    public WinAsobsSearcher(WinAsobsSolver solver) {
        super(solver);
        max_windows = solver.max_windows;
    }

    public void init(ParentSet[][] scores, int thread) {
        super.init(scores, thread);

        l_parent_var = new int[n_var][];

        s = new AsobsSearcher(solver);
        s.init(m_scores, thread);
    }

    public ParentSet[] search() {
        vars = smp.sample();

        asobsOpt();

        winasobs();

        return str;
    }

    public void asobsOpt() {
        s.vars = vars;
        s.asobsGain();
        this.vars = s.vars;
    }
}
