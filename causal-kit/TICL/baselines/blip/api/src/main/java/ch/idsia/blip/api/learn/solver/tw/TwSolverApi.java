package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.api.learn.solver.ScoreSolverApi;
import org.kohsuke.args4j.Option;


public abstract class TwSolverApi extends ScoreSolverApi {

    @Option(name = "-w", required = true, usage = "maximum treewidth")
    protected int tw;

}
