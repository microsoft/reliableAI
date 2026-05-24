package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSearcherAsobs extends TheTest {

    String path = basePath + "data/data-sets/";

    String results = basePath + "asobs/searcher/";

    class Learner extends Thread {

        // iteration
        int it;

        // learning method
        String lear;

        // network to work on
        String net;

        int learning = 60;

        public Learner(int it, String lear, String net) {
            this.it = it;
            this.lear = lear;
            this.net = net;
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
            String s = r + ".jkl";

            String r1 = f("%s/%s", results, net);

            new File(r1).mkdir();
            String result = f("%s/%s/%s-%d", results, net, lear, it);

            // if (new File(output).exists())
            // return;

            p("Start: " + result);

            AsobsSolver sv = new AsobsSolver();

            sv.searcher = lear;

            if (sv == null) {
                p("NOOOOO WHHAAAATTT");
                return;
            }

            sv.init(getScoreReader(s), learning);
            sv.thread_pool_size = 1;
            sv.logWr = getWriter(result + ".log");
            sv.verbose = 2;
            sv.go(result);

        }
    }

    @Test
    public void gh() {

        try {
            ExecutorService es = Executors.newFixedThreadPool(6);

            // ExecutorService es = Executors.newFixedThreadPool(1);

            for (File file : new File(path).listFiles()) {

                if (file.getName().endsWith((".jkl"))) {

                    for (int it = 0; it < 3; it++) {
                        for (String l : new String[] { "old", "new"}) { // "old", "new2",

                            // es.go(new Prep(set));

                            es.execute(
                                    new Learner(it, l,
                                    file.getName().replace(".jkl", "")));
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
    public void whaaaaaaaaat() throws Exception {

        for (String l : new String[] { "new2", "new", "new3", "greedy", "std"}) {
            whaaaatGO(l);
        }

    }

    private void whaaaatGO(String lear) throws Exception {

        AsobsSolver sv = new AsobsSolver();

        sv.searcher = lear;

        String s = f("%s/asobs/net550.pc.jkl", basePath);
        String o = f("%s/asobs/net550-%s.res", basePath, lear);

        sv.init(getScoreReader(s), 5);
        sv.thread_pool_size = 1;
        sv.logWr = getWriter(o + ".log");
        sv.verbose = 2;
        sv.testAcycility = true;
        sv.max_exec_time = 10;
        sv.go(o);

    }
}
