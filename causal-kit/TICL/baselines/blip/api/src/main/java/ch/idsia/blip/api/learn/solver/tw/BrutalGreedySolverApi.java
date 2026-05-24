package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;

import java.util.logging.Logger;


public class BrutalGreedySolverApi extends TwSolverApi {

    private static final Logger log = Logger.getLogger(
            BrutalGreedySolverApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new BrutalGreedySolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new BrutalSolver();
    }

}
