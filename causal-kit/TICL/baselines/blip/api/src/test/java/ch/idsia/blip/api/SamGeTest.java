package ch.idsia.blip.api;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetReader;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class SamGeTest extends TheTest {

    @Test
    public void testSimple() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        int n_sample = 100;

        File f_bn = new File(basePath + "old/random10-1.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn = BnNetReader.ex(rd);

        int[] ord = bn.getTopologicalOrder();

        System.out.println(Arrays.toString(ord));

        SamGe samGe = new SamGe();

        samGe.go(bn, String.format("%srandom10-new", basePath), n_sample);

    }

    /*
     R code
     library("bnlearn")
     fn <- read.table("random10-new.tab")
     bn <- read.net("random10-1.net")
     descr <- modelstring(bn)
     n_bn <- model2network(descr)
     f_bn <- bn.fit(n_bn, fn, method="bayes", iss=1)
     graph.net("random10-new.net", f_bn)
     */

    @Test

    /**
     * Let'set see if we can re-learn pa network from pa sample. Given the R code, we compare the two networks.
     */
    public void testSameNetwork() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        File f_bn = new File(basePath + "old/random10-1.net");
        BufferedReader rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn_orig = BnNetReader.ex(rd);

        f_bn = new File(basePath + "old/random10-1.new.net");
        rd = new BufferedReader(new FileReader(f_bn));
        BayesianNetwork bn_learned = BnNetReader.ex(rd);

        assertEquals(bn_orig.n_var, bn_learned.n_var);
        assertEquals(bn_orig.l_ar_var, bn_learned.l_ar_var);
        assertEquals(bn_orig.l_nm_var, bn_learned.l_nm_var);
        assertEquals(bn_orig.l_values_var, bn_learned.l_values_var);

        for (int i = 0; i < bn_orig.n_var; i++) {
            int[] orig = bn_orig.parents(i);
            int[] cp_orig = Arrays.copyOf(orig, orig.length);

            Arrays.sort(cp_orig);
            int[] learned = bn_learned.parents(i);
            int[] cp_learned = Arrays.copyOf(learned, learned.length);

            Arrays.sort(cp_learned);
            assertEquals(Arrays.toString(cp_orig), Arrays.toString(cp_learned));
        }

        SamGe samGe = new SamGe();

        samGe.go(bn_orig, null, 0);
        int[] ord = bn_orig.getTopologicalOrder();
        int cntr = 0;
        int tot = 10000;

        for (int i = 0; i < tot; i++) {
            short[] sample = samGe.getSample(ord);
            double orig_ll = bn_orig.getLogLik(sample);
            double learned_ll = bn_learned.getLogLik(sample);

            // System.out.println(orig_ll);
            // System.out.println(learned_ll);
            double diff = Math.abs((orig_ll - learned_ll) / orig_ll);

            // System.out.println(diff);

            if (diff > 0.2) {
                cntr += 1;
            }
        }

        double p = (cntr * 100.0) / tot;

        System.out.println(p);
        System.out.println(cntr);
        assertTrue(p < 0.1);
    }

}
