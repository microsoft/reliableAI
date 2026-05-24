package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalAstarSolver;

import java.util.logging.Logger;


public class BrutalAstarSolverApi extends TwSolverApi {

    private static final Logger log = Logger.getLogger(
            BrutalAstarSolverApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new BrutalAstarSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new BrutalAstarSolver();
    }
}
