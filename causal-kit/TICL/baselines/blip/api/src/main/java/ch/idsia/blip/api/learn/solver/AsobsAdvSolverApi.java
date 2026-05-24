package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.ObsSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class AsobsAdvSolverApi extends ObsAdvSolverApi {

    public static void main(String[] args) {
        defaultMain(args, new AsobsAdvSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new AsobsSolver();
    }

}
