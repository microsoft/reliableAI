package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import org.junit.Test;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class BrutalApiTest extends TheTest {

    @Test
    public void tryit() throws Exception {
        String net = "random90-1";
        String base = "/home/loskana/Desktop/tw/";

        BayesianNetwork bn = BnNetReader.ex(base + net + ".net");

        bn.writeGraph(base + net);

        if (!new File(base + net + "-5000.dat").exists()) {
            SamGe s = new SamGe();

            s.go(bn, base + net, 5000);
            p("... created dat");
        }

        if (!new File(base + net + ".jkl").exists()) {
            IndependenceScorer is = new IndependenceScorer();

            is.ph_scores = base + net + ".jkl";
            is.max_exec_time = 1;
            is.go(base + net + "-5000.dat");
            p("... created jkl");
        }

        // String[] args = {
        // "",
        // "-set", base + "child-5000.jkl",
        // "-d", base + "child-5000.dat",
        // "-r", base + "child-5000.res",
        // "-p", base + "gob/",
        // "-g", base + "gobnilp",
        // "-pa", "core/src/main/resources/astar",
        // "-v", "1",
        // "-pb", "1",
        // "-t", "10",
        // "-w", "4"};
        //
        // GobnilpS2SolverApi.main(args);


    }
}
