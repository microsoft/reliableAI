package ch.idsia.blip.api.learn.solver.win;


import ch.idsia.blip.api.learn.solver.AsobsSolverApi;
import ch.idsia.blip.core.learn.solver.ScoreSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsLearningSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsPertSolver;
import ch.idsia.blip.core.learn.solver.WinAsobsSolver;
import org.kohsuke.args4j.Option;


public class WinAsobsLearningSolverApi extends WinAsobsPertSolverApi {

    @Option(name = "-inv", usage = "inverse")
    protected boolean inverse = false;

    public static void main(String[] args) {
        defaultMain(args, new WinAsobsLearningSolverApi());
    }

    @Override
    protected ScoreSolver getSolver() {
        return new WinAsobsLearningSolver();
    }

}
