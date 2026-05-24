package ch.idsia.blip.core.learn.solver;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.utils.ParentSet;
import org.junit.Test;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BrutalGreedyTest extends TheTest {

    public BrutalGreedyTest() {
        basePath += "asobs/ktree/mc3/";
    }

    @Test
    public void sampleKTreeTest() throws Exception {

        String s = "simple-5000";

        check(5, s);
    }

    private void check(int maxTw, String s) throws Exception {

        clean(basePath);

        ParentSet[][] sc = getScoreReader(basePath + s + ".jkl");
        BrutalSolver a = new BrutalSolver();

        a.verbose = 2;

        /*
         a.prepare();
         // a.verbose = 2;
         BrutalGreedySolver.BrutalGreedySearcher b = a.getNewSearcher();

         b.vars = new int[l_sc.n_var];
         for (int thread = 0; thread < l_sc.n_var; thread++) {
         b.vars[thread] = thread;
         }

         long start = System.currentTimeMillis();
         double elapsed = 0;
         int thread = 0;

         TreeWidth t = new TreeWidth();

         while (elapsed < 10) {

         try {
         b.searchNew();

         b.new_bn.checkTreeWidth(maxTw);

         pf("%.3f \n", b.new_sk);

         } catch (Exception e) {
         e.printStackTrace();

         String f = basePath + s + "-" + thread;
         Directed d = b.new_bn.directed();

         d.graph(f + "-dir");

         Undirected m = b.new_bn.moralize();

         m.graph(f + "-mor");

         pf("%d \n", t.exec(m));
         t.ar.graph(f + "-tri");

         whichOneIs(t.ar, maxTw);
         thread++;
         }

         elapsed = (System.currentTimeMillis() - start) / 1000.0;
         }

         */
    }

    private void whichOneIs(Undirected ar, int tw) {
        for (int i = 0; i < ar.n; i++) {
            if (ar.biggerClique(i, tw + 1)) {
                pf("v: %d ", i);
            }
        }
        p("");
    }

    private void clean(String basePath) {
        File folder = new File(basePath);

        for (File f : folder.listFiles()) {
            if (f.getName().endsWith(".dot") || f.getName().endsWith(".png")) {
                f.delete(); // may fail mysteriously - returns boolean you may want to check
            }
        }

    }

}
