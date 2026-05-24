package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.ktree.S2Solver;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


public class ExpS2 extends TheTest {

    @Test
    public void test() throws IOException, IncorrectCallException {

        // String set = "dna.test";
        String s = "bbc.test";

        String path = basePath + "/tw/" + s;

        S2Solver solver = new S2Solver();

        solver.ph_gobnilp = basePath + "/gobnilp";
        solver.ph_dat = path + ".dat";
        solver.thread_pool_size = 1;
        solver.ph_work = path;
        solver.init(getScoreReader(path + ".jkl"), 3600);
        solver.verbose = 3;
        solver.go();
    }
}
