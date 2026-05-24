package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalMaxSolver;

import java.util.logging.Logger;


public class BrutalMaxSolverApi extends TwSolverApi {

    private static final Logger log = Logger.getLogger(
            BrutalMaxSolverApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new BrutalMaxSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new BrutalMaxSolver();
    }

}
