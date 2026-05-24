package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.learn.solver.samp.EntropySampler;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpTreewidth extends TheTest {

    String path = basePath + "tw/samp";

    class Learner extends Thread {

        // iteration
        int it;

        // learning method
        String lear;

        // network to work on
        String net;

        // treewidth
        int tw;

        int learning = 600;

        public Learner(int it, String lear, String net, int tw) {
            this.it = it;
            this.lear = lear;
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
            String r = f("%s/../data/%s", path, net);
            String r2 = f("%s/%s", path, net);

            new File(r2).mkdir();
            String output = f("%s/%s-%d-%d", r2, lear, tw, it);

            // if (new File(output).exists())
            // return;


            String d = r + ".dat";

            String s = r + ".jkl";

            BrutalSolver sv = new BrutalSolver();

            if (sv == null) {
                p("NOOOOO WHHAAAATTT");
                return;
            }

            sv.sampler = lear;
            sv.dat_path = d;
            sv.init(getScoreReader(s), learning);
            sv.tw = tw;
            sv.thread_pool_size = 1;
            sv.logWr = getWriter(output + ".log");
            sv.verbose = 2;
            sv.go(output);
            sv.writeGraph(output);

        }
    }

    @Test
    public void gh() {

        try {
            ExecutorService es = Executors.newFixedThreadPool(6);

            for (int it = 0; it < 3; it++) {
                for (String l : new String[] {
                    "d", "ent", "r_ent", "mi", "r_mi"}) {
                    for (File file : new File(f("%s/../data/", path)).listFiles()) {
                        p(file.getName());
                        if (file.getName().endsWith((".data"))) {
                            // es.go(new Prep(set));
                            for (int tw : new int[] { 4}) { // , 4 , 6, 8, 10
                                es.execute(
                                        new Learner(it, l,
                                        file.getName().replace(".dat", ""), tw));
                            }
                        }
                    }
                }
            }
            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testGen() {
        int n = 10;

        Random rnd = getRandom();

        double[] w = new double[n];
        int t = 0;

        for (int i = 0; i < n; i++) {
            w[i] = rnd.nextInt(10) + 1;
            t += w[i];
        }

        int tot = 1000;

        double[][] freq = new double[n][];

        for (int j = 0; j < n; j++) {
            freq[j] = new double[n];
        }

        EntropySampler s = new EntropySampler(null, 0, rnd);

        for (int i = 0; i < tot; i++) {
            int[] ord = s.sampleWeighted(n, w);

            p(Arrays.toString(ord));

            for (int j = 0; j < n; j++) {
                freq[j][ord[j]] += 1;
            }
        }

        p("freqs");

        for (int j = 0; j < n; j++) {
            pf("\nposition %d", j);

            for (int i = 0; i < n; i++) {
                pf("%5d ", i);
            }
            p("");

            for (int i = 0; i < n; i++) {
                pf("%5.2f ", w[i] / t);
            }
            p("");

            for (int i = 0; i < n; i++) {
                pf("%5.2f ", freq[j][i] / tot);
            }
            p("");
        }
    }
}
