package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.learn.solver.ScoreSolverApi;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinObsSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class WinObsSolverApi extends ScoreSolverApi {

    @Option(name = "-win", usage = "Maximum window size")
    protected int win = 5;

    public static void main(String[] args) {
        defaultMain(args, new WinObsSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinObsSolver();
    }

}
