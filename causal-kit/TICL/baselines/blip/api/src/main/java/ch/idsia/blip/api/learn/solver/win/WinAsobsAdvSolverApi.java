package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.learn.solver.ObsAdvSolverApi;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import ch.idsia.blip.core.learn.solver.WinObsSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class WinAsobsAdvSolverApi extends WinObsAdvSolverApi {

    public static void main(String[] args) {
        defaultMain(args, new WinAsobsAdvSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinAsobsSolver();
    }

}
