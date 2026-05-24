package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;

import java.util.logging.Logger;


public class AsobsSolverApi extends ScoreSolverApi {

    public static void main(String[] args) {
        defaultMain(args, new AsobsSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new AsobsSolver();
    }

}
