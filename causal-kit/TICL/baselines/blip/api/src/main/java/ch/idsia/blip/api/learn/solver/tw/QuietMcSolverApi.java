package ch.idsia.blip.api.learn.solver.tw;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.brtl.QuietGreedySolver;

import java.util.logging.Logger;


public class QuietMcSolverApi extends TwSolverApi {

    private static final Logger log = Logger.getLogger(
            QuietMcSolverApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new QuietMcSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new QuietGreedySolver();
    }
}
