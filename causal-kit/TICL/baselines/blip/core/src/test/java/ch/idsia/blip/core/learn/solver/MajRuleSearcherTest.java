package ch.idsia.blip.core.learn.solver;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class MajRuleSearcherTest extends TheTest {

    public MajRuleSearcherTest() {
        basePath += "majRule/";
    }

    @Test
    public void testOne() throws Exception {
        // testTw("trainfile-0");

        // testTw("child-5000", 3);
        // testNormal("child-5000");
        // testTw("child-5000", 2);

        testOneTw("random30");

    }

    private void testOneTw(String r) throws Exception {

        int n = 5000;

        BayesianNetwork b = getBnFromFile(f("%s.net", r));

        String g = f("%s%s.png", basePath, r);

        if (!(new File(g).exists())) {
            b.writeGraph(basePath + r);
        }

        String d = f("%s%s-%d.dat", basePath, r, n);

        if (!(new File(d).exists())) {
            SamGe.ex(b, basePath + r, n);
        }

        DataSet dat = getDataSet(d);

        String s = f("%s%s.jkl", basePath, r);

        if (!(new File(s).exists())) {
            IndependenceScorer is = new IndependenceScorer();

            is.ph_scores = s;
            is.max_exec_time = 2;
            is.go(dat);
        }

        ParentSet[][] sc = RandomStuff.getScoreReader(s, 0);
        AsobsAvgSolver solver = new AsobsAvgSolver();
        String m = f("%s%s.maj", basePath, r);

        if (!(new File(m).exists())) {
            solver.init(sc, 20);
            solver.go(m);
        }

        // solver.mtx = UpperMatrix.read(m);

        // for (double eps: new double[]{0.95, 0.9, 0.8, 0.7}) {
        // solver.eps = eps;
        // solver.graph(f("%s%s-out-%.2f", basePath, r, eps), dat.l_nm_var);
        // }

    }

}
