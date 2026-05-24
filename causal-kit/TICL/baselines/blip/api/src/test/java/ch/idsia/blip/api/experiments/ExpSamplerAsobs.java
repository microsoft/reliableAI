package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import org.junit.Test;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpSamplerAsobs extends TheTest {

    String path = basePath + "data/data-sets/";

    String results = basePath + "asobs/sampler/";

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
            String d = r + ".dat";
            String r1 = f("%s/%s", results, net);

            new File(r1).mkdir();
            String result = f("%s/%s/%s-%d", results, net, lear, it);

            // if (new File(output).exists())
            // return;

            p("Start: " + result);

            AsobsSolver sv = new AsobsSolver();

            sv.sampler = lear;
            sv.dat_path = d;
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
            ExecutorService es = Executors.newFixedThreadPool(5);

            // ExecutorService es = Executors.newFixedThreadPool(1);

            for (File file : new File(path).listFiles()) {

                if (file.getName().endsWith((".jkl"))) {

                    for (int it = 0; it < 3; it++) {
                        for (String l : new String[] {
                            "std", "ent", "r_ent",
                            "ent_b", "mi", "r_mi", "mi_b"}) { // "old", "new2",

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

        for (String l : new String[] { "old", "current", "new"}) {
            whaaaatGO(l);
        }

    }

    private void whaaaatGO(String lear) throws Exception {

        BrutalSolver sv = new BrutalSolver();

        sv.searcher = lear;

        String s = f("%s/tw/child-5000.jkl", basePath);
        String o = f("%s/tw/child-%s.res", basePath, lear);

        sv.sampler = lear;
        sv.init(getScoreReader(s), 5);
        sv.tw = 4;
        sv.thread_pool_size = 1;
        sv.logWr = getWriter(o + ".log");
        sv.verbose = 2;
        sv.go(o);
    }

    @Test
    public void whaaatest() throws Exception {

        AsobsSolver sv = new AsobsSolver();

        sv.sampler = "ent_b";

        String s = f("%s/tw/child-5000.jkl", basePath);
        String d = f("%s/tw/child-5000.dat", basePath);
        String o = f("%s/tw/child-ent_b.res", basePath);

        sv.init(getScoreReader(s), 60);
        sv.dat_path = d;
        sv.thread_pool_size = 1;
        sv.logWr = getWriter(o + ".log");
        sv.verbose = 2;
        sv.go(o);
    }
}
