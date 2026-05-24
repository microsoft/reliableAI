package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnErgReader;
import ch.idsia.blip.core.io.bn.BnErgWriter;
import ch.idsia.blip.core.io.bn.BnResReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLeBayes;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.learn.solver.ktree.S2PlusSolver;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpUai06Small {

    String path = "/home/loskana/Desktop/uai06/data-small";
    int learning = 120;

    protected void parle(DataSet dat, String s, BayesianNetwork bn) {
        ParLeBayes p = new ParLeBayes(1);

        bn = p.go(bn, dat);
        BnErgWriter.ex(s, bn);
        BnUaiWriter.ex(s, bn);
        bn.writeGraph(s);
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

            // S2+
            String s2 = r + "s2-" + tw;

            if (!(new File(s2).exists())) {
                S2PlusSolver sv = new S2PlusSolver();

                sv.init(getScoreReader(s), learning);
                sv.tw = tw;
                sv.ph_dat = d;
                sv.ph_astar = path + "/astar/";
                sv.ph_work = f("%s/%s/gob%d", path, net, tw);
                sv.ph_gobnilp = path + "/gobnilp";
                sv.thread_pool_size = 1;
                sv.go(s2);
            }

            File f = new File(s2);

            if (f.exists() && f.length() > 100) {
                BayesianNetwork BnS2 = BnResReader.ex(s2);

                parle(dat, s2, BnS2);
            }

            // Brutal
            String bru = r + "brut-" + tw;

            if (!(new File(bru).exists())) {
                BrutalSolver sv = new BrutalSolver();

                sv.init(getScoreReader(s), learning);
                sv.tw = tw;
                sv.thread_pool_size = 1;
                sv.go(bru);
                sv.writeGraph(bru);
            }

            BayesianNetwork BnBru = BnResReader.ex(bru);

            parle(dat, bru, BnBru);
        }
    }


    class Prep extends Thread {

        String net;

        public Prep(String net) {
            this.net = net;
        }

        @Override
        public void run() {

            pf("Start Prep %s %d", net);

            try {
                learn();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void learn() throws Exception {
            int n = 1000000;

            String r = f("%s/%s/", path, net);
            File dir = new File(r);

            if (!new File(r).exists()) {
                dir.mkdir();
            }

            BayesianNetwork orig = BnErgReader.ex(r + net + ".erg");
            // BnUaiWriter.ex(r+net, orig);
            String d = r + "dat";

            if (!(new File(d).exists())) {
                SamGe.ex(orig, d, n);
            }

            DataSet dat = getDataSet(d);

            if (!(new File(r + "new.net").exists())) {
                parle(dat, r + "new", orig);
            }

            String s = r + "jkl";

            if (!(new File(s).exists())) {
                IndependenceScorer is = new IndependenceScorer();

                is.ph_scores = s;
                is.max_exec_time = 5;
                is.scoreNm = "bdeu";
                is.alpha = 1;
                is.max_pset_size = 5;
                is.thread_pool_size = 1;
                is.go(dat);
            }

            for (int l : new int[] { 0, 1, 2, 3, 4, 5}) {
                String best = r + "best" + l;

                if (!(new File(best).exists())) {
                    AsobsSolver aso = new AsobsSolver();

                    aso.max_exec_time = learning / 10;
                    aso.thread_pool_size = 1;
                    aso.init(getScoreReader(s, l));
                    aso.go(best);
                }
                BayesianNetwork BnBest = BnResReader.ex(best);

                parle(dat, best, BnBest);
            }

        }
    }

    @Test
    public void gh() {

        try {
            ExecutorService es = Executors.newFixedThreadPool(6);

            for (String s : new String[] {
                "BN_28", "BN_100", "BN_102", "BN_104",
                "BN_106", "BN_108", "BN_112"}) {
                // es.go(new Prep(set));
                for (int tw : new int[] { 2, 4}) { // , 4 , 6, 8, 10
                    es.execute(new Learner(s, tw));
                }
            }

            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
