package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnErgWriter;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLe;
import org.junit.Test;

import java.io.IOException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class ExpUai06Big {

    String path = "/home/loskana/Desktop/uai06/data-big";

    @Test
    public void gen() throws IOException {

        try {
            ExecutorService es = Executors.newFixedThreadPool(2);

            for (int tw : new int[] { 2, 4, 6, 8, 10}) {
                for (String s : new String[] { "BN_4"}) { // , "BN_30", "BN_32", "BN_34", "BN_40", "BN_55", "BN_57", "BN_63"}) { //
                    es.execute(new Learner(s, tw));
                }
            }

            es.shutdown();
            es.awaitTermination(Integer.MAX_VALUE, TimeUnit.MINUTES);

        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }

    private class Learner implements Runnable {
        private final String s;
        private final int tw;

        public Learner(String s, int tw) {
            this.s = s;
            this.tw = tw;
        }

        @Override
        public void run() {
            String f = f("%s/%s/net-%d", path, s, tw);
            BayesianNetwork bn = BnNetReader.ex(f);
            BayesianNetwork bnNew = ParLe.ex(bn,
                    getDataSet(f("%s/%s-50000.dat", path, s)));

            BnErgWriter.ex(f, bnNew);
            BnUaiWriter.ex(f, bnNew);
            bn.writeGraph(f);
        }
    }
}
