package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class BnNetReaderTest extends TheTest {

    @Test
    public void easyNow() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn = getBnFromFile("train-file0.net");

        System.out.println(bn.l_nm_var);
    }

    @Test
    public void testInsurance() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn = getBnFromFile("old/andes.net");

        printBnToFile(bn, "old/andes_new.net");

        System.out.println(bn.l_nm_var);
    }

    @Test
    public void getScoreOriginal() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        /*
         for (String s: new String[]{"random2000", "random4000", "random10000"}) {
         for (int thread = 0; thread < 5; thread++) {
         computeOrigScore(s+"-" + thread);
         }
         }
         */
        computeOrigScore("random2000-0");
    }

    private void computeOrigScore(String s) throws IOException, IncorrectCallException {
        BayesianNetwork bn = getBnFromFile("scorer/orig/" + s + ".net");

        System.out.println(bn.l_nm_var);

        DataSet dat = getDataSet("../experiments/scorer/orig/" + s + "-5000.dat");

        double t_sk = 0;

        for (int n = 0; n < bn.n_var; n++) {
            // BIC score = new BIC(dat.n_datapoints, dat.l_n_arity);
            BDeu score = new BDeu(1, dat);

            int[] pars = bn.parents(n);
            // pars = new int[0];
            double sk;

            if (pars.length > 0) {
                int[][] p_values = score.computeParentSetValues(pars);

                sk = score.computeScore(n, pars);
            } else {
                sk = score.computeScore(n);
            }

            String o = "";

            for (int p : pars) {
                o += " " + bn.name(p);
            }
            System.out.println(sk + " [" + o + " ]");
            System.out.println(sk + Arrays.toString(pars));
            t_sk += sk;

            break;
        }

        System.out.printf("%s %.2f \n", s, t_sk);
    }

    @Test
    public void testReaderSimple() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn_random = getBnFromFile("old/random10-1.net");

        printBnToFile(bn_random, "old/random10-1-new.net");

        /*
         short sample[] = { 1, 0, 0, 1, 0, 1, 0, 1, 1, 0};
         System.out.println(bn_random.getPotential(0, sample));
         checkPot(0.57388418, bn_random.getPotential(0, sample)));
         checkPot(0.485495885, bn_random.getPotential(1, sample)));
         checkPot(0.9784004, bn_random.getPotential(2, sample)));
         checkPot(0.90815791, bn_random.getPotential(3, sample)));
         checkPot(0.2641813, bn_random.getPotential(4, sample)));
         checkPot(0.821967200, bn_random.getPotential(5, sample)));
         checkPot(0.5469369, bn_random.getPotential(6, sample)));
         checkPot(0.26401624, bn_random.getPotential(7, sample)));
         checkPot(0.90596272, bn_random.getPotential(8, sample)));
         checkPot(0.93771801, bn_random.getPotential(9, sample)));

         checkPot(-5.02149, bn_random.getLogLik(sample), 5));
         */

        BayesianNetwork bn_asia = getBnFromFile("old/asia.net");

        short[] sample2 = { 1, 0, 0, 1, 0, 1, 0, 1};

        checkPot(0.99, bn_asia.getPotential(0, sample2));
        System.out.println(bn_asia.getPotential(1, sample2));
        checkPot(0.3, bn_asia.getPotential(1, sample2));
        checkPot(0.8, bn_asia.getPotential(2, sample2));
        checkPot(0.0, bn_asia.getPotential(3, sample2));
        checkPot(0.01, bn_asia.getPotential(4, sample2));
        checkPot(0.5, bn_asia.getPotential(5, sample2));
        checkPot(0.01, bn_asia.getPotential(6, sample2));
        checkPot(0.95, bn_asia.getPotential(7, sample2));

        checkPot(-25.2549, bn_asia.getLogLik(sample2));
    }

    private boolean doubleEqual(double x, double y) {
        return doubleEqual(x, y, 20);
    }

    private boolean doubleEqual(double x, double y, int p) {
        return Math.abs(x - y) < Math.pow(2, -p);
    }

    @Test
    public void testLogLikTimes() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn_random = getBnFromFile("old/random10-1.net");

        long start = System.currentTimeMillis();

        short[] sample = { 1, 0, 0, 1, 0, 1, 0, 1, 1, 0};

        for (int i = 0; i < 10000; i++) {
            bn_random.getLogLik(sample);
        }

        System.out.println(
                String.format("Required: %.5f",
                (System.currentTimeMillis() - start) / 1000.0));
    }

    @Test
    public void testReaderMonster() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn = getBnFromFile("old/monster.net");

        int tot = 0;

        for (int n = 0; n < bn.n_var; n++) {
            tot += bn.parents(n).length;
        }
        System.out.println(tot);
    }

    private static void checkPot(double v, double p) {

        // System.out.println(String.format("Different values: %.6f - %.6f", v, v2));
        Assert.assertTrue(String.format("Different values: %.4f - %.4f", v, p),
                Math.abs(v - p) < 0.0001);
    }
}
