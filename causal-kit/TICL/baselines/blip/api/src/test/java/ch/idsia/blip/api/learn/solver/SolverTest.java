package ch.idsia.blip.api.learn.solver;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.IOException;


public class SolverTest extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        // greedyComplete("child-5000");
        asobs("net");
    }

    private void asobs(String s) {
        String scores = basePath + "solver/" + s + ".jkl";

        int verbose = 2;
        int max_exec_time = 60;

        String[] args = {
            "", "-set", scores, "-t", String.valueOf(max_exec_time), "-v",
            String.valueOf(verbose)};

        AsobsSolverApi.main(args);
    }

    @Test
    public void whatever2() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        oneAndOnlyTrueTest("child-5000");
    }

    private void oneAndOnlyTrueTest(String s) throws IOException, IncorrectCallException {

        String path = basePath + "scorer/" + s;

        String scores = path + ".scores";

        int verbose = 2;
        int max_exec_time = 2;

        String[] args = {
            "", "-set", scores, "-t", String.valueOf(max_exec_time), "-v",
            String.valueOf(verbose)};

        AsobsSolverApi.main(args);
        // ForwardSolverApi.main(args);

    }

    public void greedyComplete(String s) throws IOException, IncorrectCallException {

        String path = basePath + "scorer/" + s;

        String scores = path + ".scores";

        AsobsSolver so = new AsobsSolver();

        so.verbose = 2;
        so.max_exec_time = 2;
        ParentSet[][] sc = RandomStuff.getScoreReader(scores, so.verbose);

        int[] vars = new int[] {
            0, 10, 9, 18, 11, 12, 1, 3, 6, 13, 5, 7, 17, 2, 4, 8, 19, 14, 16, 15};

        /*
         AsobsSolver.AsobsSearcher r = so.getNewSearcher();

         // Find initial map!
         r.findNew(vars, so.m_scores);

         System.out.println(so.printQuick(r.str));

         boolean pb = r.greedy(vars, so.m_scores);

         System.out.println(pb);
         Assert.assertFalse(pb);

         so.printQuick(r.str);

         pb = r.greedy(vars, so.m_scores);
         System.out.println(pb);
         */
    }
}
