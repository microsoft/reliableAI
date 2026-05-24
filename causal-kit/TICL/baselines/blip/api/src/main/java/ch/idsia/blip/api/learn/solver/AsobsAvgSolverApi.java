package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.core.learn.solver.AsobsAvgSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class AsobsAvgSolverApi extends ScoreSolverApi {

    private static final Logger log = Logger.getLogger(
            AsobsAvgSolverApi.class.getName());

    @Option(name = "-d", required = true, usage = "Datafile path (.dat format)")
    String ph_dat;

    public static void main(String[] args) {
        defaultMain(args, new AsobsAvgSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new AsobsAvgSolver();
    }

    @Override
    public void exec() throws Exception {
        ((AsobsAvgSolver) solver).ph_dat = ph_dat;
        super.exec();
    }

}
