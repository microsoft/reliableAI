package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class BrutalGreedyAdvSolverApi extends TwSolverApi {

    private static final Logger log = Logger.getLogger(
            BrutalGreedyAdvSolverApi.class.getName());

    @Option(name = "-d", usage = "Datafile path (.dat format)")
    private String dat_path;

    @Option(name = "-smp", usage = "Advanced sampler")
    private String sampler;

    @Option(name = "-src", usage = "Advanced searcher")
    private String searcher;

    public static void main(String[] args) {
        defaultMain(args, new BrutalGreedyAdvSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new BrutalSolver();
    }
}
