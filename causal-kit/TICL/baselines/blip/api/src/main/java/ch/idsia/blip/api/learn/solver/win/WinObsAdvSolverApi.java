package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.learn.solver.ObsAdvSolverApi;
import ch.idsia.blip.api.learn.solver.ObsSolverApi;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.ObsSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinObsSolver;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.kohsuke.args4j.Option;

import java.io.File;
import java.util.logging.Logger;


public class WinObsAdvSolverApi extends ObsAdvSolverApi {

    @Option(name = "-win", usage = "Maximum window size")
    protected int win = 5;

    public static void main(String[] args) {
        defaultMain(args, new WinObsAdvSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinObsSolver();
    }
}
