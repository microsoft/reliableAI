package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.core.learn.solver.ObsSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;

import java.util.logging.Logger;


public class ObsSolverApi extends ScoreSolverApi {

    private static final Logger log = Logger.getLogger(
            ObsSolverApi.class.getName());

    public static void main(String[] args) {
        defaultMain(args, new ObsSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new ObsSolver();
    }
}
