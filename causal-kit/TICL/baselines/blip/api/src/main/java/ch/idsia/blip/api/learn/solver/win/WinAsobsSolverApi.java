package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.learn.solver.AsobsAdvSolverApi;
import ch.idsia.blip.api.learn.solver.AsobsSolverApi;
import ch.idsia.blip.api.learn.solver.ScoreSolverApi;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;

import static ch.idsia.blip.api.Api.defaultMain;


public class WinAsobsSolverApi extends AsobsSolverApi {

    @Option(name = "-win", usage = "Maximum window size")
    protected int win = 5;

    public static void main(String[] args) {
        defaultMain(args, new WinAsobsSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinAsobsSolver();
    }

}
