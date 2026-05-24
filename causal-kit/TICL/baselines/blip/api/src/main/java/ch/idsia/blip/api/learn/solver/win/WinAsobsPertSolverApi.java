package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.Api;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsPertSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class WinAsobsPertSolverApi extends WinAsobsSolverApi {

    @Option(name = "-pa", usage = "pa")
    protected int pa = 10;

    @Option(name = "-pb", usage = "pb")
    protected int pb = 10;

    @Option(name = "-pc", usage = "pb")
    protected int pc = 5;

    public static void main(String[] args) {
        Api.defaultMain(args, new WinAsobsPertSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinAsobsPertSolver();
    }

}