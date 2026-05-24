package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpVarsGreedy extends TheTest {

    String path = basePath + "tw/new/";

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

            // pf("Start %s %d \n", net, tw);

            try {
                learn();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        private void learn() throws Exception {
            String r = f("%s/%s", path, net);

            new File(r).mkdir();
            String output = f("%s/%s-%d-%d", r, lear, tw, it);

            // if (new File(output).exists())
            // return;

            String s = r + ".jkl";

            p("Start: " + output);

            BrutalSolver sv = new BrutalSolver();

            sv.searcher = lear;

            if (sv == null) {
                p("NOOOOO WHHAAAATTT");
                return;
            }

            sv.sampler = lear;
            sv.init(getScoreReader(s), learning);
            sv.tw = tw;
            sv.thread_pool_size = 1;
            sv.logWr = getWriter(output + ".log");
            sv.verbose = 2;
            sv.go(output);

        }
    }

    @Test
    public void gh() {

        try {
            ExecutorService es = Executors.newFixedThreadPool(6);

            for (File file : new File(path).listFiles()) {

                if (file.getName().endsWith((".jkl"))) {

                    for (int it = 0; it < 3; it++) {
                        for (String l : new String[] { "new2"}) { // "old", "new2",

                            // es.go(new Prep(set));
                            for (int tw : new int[] { 4}) { // , 4 , 6, 8, 10
                                es.execute(
                                        new Learner(it, l,
                                        file.getName().replace(".jkl", ""), tw));
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

}
