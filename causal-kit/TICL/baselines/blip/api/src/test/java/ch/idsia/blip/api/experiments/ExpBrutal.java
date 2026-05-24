package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnErgReader;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.solver.brtl.BrutalAstarSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpBrutal extends TheTest {

    String path = basePath + "/brutal";
    int learning = 600;

    class Learner extends Thread {

        String net;
        int exp_limit;

        public Learner(String net, int tw) {
            this.net = net;
            this.exp_limit = tw;
        }

        @Override
        public void run() {

            // pf("Start %s %d \n", net, exp_limit);

            try {
                learn();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void learn() throws Exception {
            String r = f("%s/%s/", path, net);

            String s = r + "jkl";

            // Brutal
            String bru = r + "brut-" + exp_limit;

            if (true || !(new File(bru).exists())) {
                pf("Doing %s %d \n", net, exp_limit);
                BrutalAstarSolver sv = new BrutalAstarSolver();

                sv.init(getScoreReader(s), learning);
                sv.tw = 4;
                sv.exp_limit = exp_limit;
                sv.thread_pool_size = 1;
                sv.go(bru);
                sv.writeGraph(bru);
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

            // pf("Prep %s \n", net);

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
                pf("dat %s \n", net);
                SamGe.ex(orig, d, n);
            }

            DataSet dat = getDataSet(d);

            String s = r + "jkl";

            if (!(new File(s).exists())) {
                pf("jkl %s \n", net);
                IndependenceScorer is = new IndependenceScorer();

                is.ph_scores = s;
                is.max_exec_time = 5;
                is.scoreNm = "bdeu";
                is.alpha = 1;
                is.max_pset_size = 5;
                is.thread_pool_size = 1;
                is.go(dat);
            }

            String bru = r + "brut-0";

            if (!(new File(bru).exists())) {
                pf("Doing %s greedy \n", net);
                BrutalSolver sv = new BrutalSolver();

                sv.init(getScoreReader(s), learning);
                sv.tw = 4;
                sv.thread_pool_size = 1;
                sv.go(bru);
                sv.writeGraph(bru);
            }
        }
    }

    @Test
    public void gh() {

        try {

            String[] nets = new String[] { "BN_4", "BN_28", "BN_100", "BN_102"}; // , };
            ExecutorService es = Executors.newFixedThreadPool(6);

            for (String s : nets) {
                es.execute(new Prep(s));
            }
            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

            es = Executors.newFixedThreadPool(6);
            for (String s : nets) {
                for (int tw : new int[] { 1, 2, 3, 4, 5, 6, 7}) { // , 4 , 6, 8, 10
                    es.execute(new Learner(s, tw));
                }
            }

            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void see() throws Exception {
        BrutalAstarSolver sv = new BrutalAstarSolver();

        sv.init(getScoreReader(path + "/BN_4/jkl"), learning);
        sv.tw = 4;
        sv.verbose = 1;
        sv.exp_limit = 2;
        sv.thread_pool_size = 1;
        sv.go(path + "/BN_4/test");
        sv.max_exec_time = 100;
    }
}
