package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnErgWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import org.junit.Test;

import java.io.File;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSampler extends TheTest {

    String path = basePath + "/sampler";
    int learning = 600;

    protected void parle(DataSet dat, String s, BayesianNetwork bn) {
        ParLeBayes p = new ParLeBayes(1);

        bn = p.go(bn, dat);
        BnErgWriter.ex(s, bn);
        BnUaiWriter.ex(s, bn);
        // bn.writeGraph(set);
    }

    class Learner extends Thread {

        String net;
        int tw;

        public Learner(String net, int tw) {
            this.net = net;
            this.tw = tw;
        }

        @Override
        public void run() {

            pf("Start %s %d", net, tw);

            try {
                learn();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void learn() throws Exception {
            String r = f("%s/%s/", path, net);

            String d = r + "dat";
            DataSet dat = getDataSet(d);

            String s = r + "jkl";

            // Brutal
            String bru = r + "brut-" + tw;

            if (!(new File(bru).exists())) {
                BrutalSolver sv = new BrutalSolver();

                sv.init(getScoreReader(s), learning);
                sv.tw = tw;
                sv.out_solutions = 10;
                sv.go(bru);
                // sv.writeGraph(bru);
            }

            for (int i = 0; i < 10; i++) {
                String br = f("%s-%d", bru, i);
                BayesianNetwork BnBru = BnResReader.ex(br);

                parle(dat, br, BnBru);

            }
        }
    }


    class Prep extends Thread {

        String net;

        public Prep(String net) {
            this.net = net;
        }

        @Override
        public void run() {

            pf("Start Prep %s", net);

            try {
                learn();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void learn() throws Exception {
            int n = 100000;

            String r = f("%s/%s/", path, net);
            File dir = new File(r);

            if (!new File(r).exists()) {
                dir.mkdir();
            }

            BayesianNetwork orig = BnUaiReader.ex(r + net + ".uai");

            BnErgWriter.ex(r + net, orig);

            // BnUaiWriter.ex(r+net, orig);
            String d = r + "dat";

            SamGe.ex(orig, d, n);
            SamGe.ex(orig, d + "2", n);
            parle(getDataSet(d), r + "new", orig);

            DataSet dat = getDataSet(d);

            String s = r + "jkl";

            if (!(new File(s).exists())) {
                IndependenceScorer is = new IndependenceScorer();

                is.ph_scores = s;
                is.max_exec_time = 5;
                is.scoreNm = "bdeu";
                is.alpha = 1;
                is.max_pset_size = 5;
                is.go(dat);
            }
        }
    }

    @Test
    public void gh() {

        for (String s : new String[] { "BN_34"}) { // , "BN_N_1", "BN_N_2"}) {
            go(new Prep(s));

            for (int tw : new int[] { 2, 4, 6}) { // , 4 , 6, 8, 10
                go(new Learner(s, tw));
            }
        }
    }

    private void go(Thread t) {
        t.start();
        try {
            t.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
