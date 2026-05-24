package ch.idsia.blip.core.learn.constraint;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.learn.constraints.PcAlgo;
import ch.idsia.blip.core.learn.solver.brtl.BrutalPcAstarSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalPcGreedySolver;
import org.junit.Test;

import java.io.File;
import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class PcAlgoTest extends TheTest {

    public PcAlgoTest() {
        basePath += "pc/";
    }

    // String r = "child";
    // String r = "random2000";
    private final String r = "child";
    // String r = "random50";
    /*
     @Test
     public void testBlah() throws Exception {

     int n = 5000;

     BayesianNetwork b = getBnFromFile(f("%s.net", r));

     String g = f("%s%s.png", basePath, r);

     if (!(new File(g).exists()))
     b.writeGraph(basePath + r);

     String d = f("%s%s-%d.dat", basePath, r, n);

     if (!(new File(d).exists()))
     SamGe.ex(b, basePath + r, n);

     DataSet dat = getDataSet(d);

     String s = f("%s%s.jkl", basePath, r);
     if (!(new File(s).exists())) {
     IndependenceScorer is = new IndependenceScorer();
     is.ph_scores = s;
     is.max_exec_time = 2;
     is.ex(dat);
     }

     String u = f("%s%s.und", basePath, r);
     skeleton(basePath + r, dat);

     Undirected skeleton = Undirected.read(u);

     BaseSolver as = new BrutalPcGreedySolver(100, 3, skeleton);
     as.thread_pool_size = 1;
     as.ex(new ScoreReader(s));
     as.res_path = basePath + r + "-solv";
     as.writeGraph(basePath + r + "-out", dat.l_nm_var);

     }
     */

    @Test
    public void testBlah3() throws Exception {

        String o = f("%s%s-out", basePath, r);
        BrutalPcAstarSolver as = new BrutalPcAstarSolver();

        as.init(100, o, 3);
        as.thread_pool_size = 1;
        as.verbose = 1;
        as.go(f("%s%s-astar-nets", basePath, r));
    }

    @Test
    public void testBlah4() throws Exception {

        String o = f("%s%s-out", basePath, r);
        BrutalPcGreedySolver as = new BrutalPcGreedySolver();

        as.init(100, o, 3);
        as.thread_pool_size = 1;
        as.verbose = 1;
        as.go(f("%s%s-greedy-nets", basePath, r));
    }

    @Test
    public void testBlah2() throws Exception {

        String r = "500";

        String d = f("%s%s.dat", basePath, r);
        DataSet dat = getDataSet(d);

        skeleton(basePath + r, dat);
    }

    private void skeleton(String s, DataSet dat) throws IOException {
        String u = f("%s.und", s);

        if (!(new File(u).exists())) {
            PcAlgo p = new PcAlgo();

            p.verbose = 2;
            p.oracle = new MutualInformation(dat, 0.1, 1);
            Undirected skeleton = p.skeleton(dat);

            // p.printSkeleton(skeleton);
            skeleton.names = dat.l_nm_var;
            skeleton.graph(basePath + r + "-skel");
            skeleton.write(u);
        }
    }
}
