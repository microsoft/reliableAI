package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsImprSolver;
import org.kohsuke.args4j.Option;

import java.util.logging.Logger;


public class WinAsobsImprSolverApi extends WinAsobsSolverApi {

    @Option(name = "-pa", usage = "A")
    protected int pa = 5;

    @Option(name = "-pb", usage = "B")
    protected int pb = 15;

    @Option(name = "-pc", usage = "C")
    protected int pc = 3;

    @Option(name = "-pd", usage = "D")
    protected int pd = 5;

    public static void main(String[] args) {
        defaultMain(args, new WinAsobsImprSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinAsobsImprSolver();
    }

}
