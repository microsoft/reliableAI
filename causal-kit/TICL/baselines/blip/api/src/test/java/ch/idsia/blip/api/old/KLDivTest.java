package ch.idsia.blip.api.old;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.other.KLDiv;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;


public class KLDivTest extends TheTest {

    @Test
    public void simpleTest() throws Exception {
        BayesianNetwork trueNet = getBnFromFile("old/random10-true.net");
        BayesianNetwork learnedNet1 = getBnFromFile("old/random10-learned1.net");
        BayesianNetwork learnedNet2 = getBnFromFile("old/random10-learned2.net");

        KLDiv klaus = new KLDiv();

        klaus.verbose = false;

        double kl1 = klaus.getKLDivergence(learnedNet1, trueNet);

        System.out.println(kl1);
        double kl2 = klaus.getKLDivergence(learnedNet2, trueNet);

        System.out.println(kl2);
    }

    @Test
    public void alarmTest() throws Exception {
        BayesianNetwork trueNet = getBnFromFile("old/asia-true.net");
        BayesianNetwork learnedNet1 = getBnFromFile("old/asia-learned-1.net");

        KLDiv klaus = new KLDiv();

        klaus.verbose = true;

        double kl1 = klaus.getKLDivergence(trueNet, trueNet);

        System.out.println(kl1);

        double kl2 = klaus.getKLDivergence(trueNet, learnedNet1);

        System.out.println(kl2);
    }

    @Test
    public void distr() throws IOException {

        KLDiv klaus = new KLDiv();

        klaus.verbose = true;

        BayesianNetwork bn = getBnFromFile("old/asia-learned-1.net");

        System.out.println("### Joint");
        for (int i = 0; i < bn.n_var; i++) {
            double[] d = klaus.getJointDistribution(bn, i);
            double[] o = bn.potentials(i);

            System.out.println(
                    String.format("Var %d:\nd: %s\no: %s", i, Arrays.toString(o),
                    Arrays.toString(d)));
        }

        System.out.println("\n### Cond");
        for (int i = 0; i < bn.n_var; i++) {
            double[] c = klaus.getCondDistribution(bn, i);
            double[] o = bn.potentials(i);

            System.out.println(
                    String.format("Var %d:\no: %s\npc: %s", i,
                    Arrays.toString(o), Arrays.toString(c)));

            for (int j = 0; j < c.length; j++) {
                checkPot(c[j], o[j]);
            }
        }
    }

    private static void checkPot(double v, double p) {

        // System.out.println(String.format("Different values: %.6f - %.6f", v, v2));
        Assert.assertTrue(String.format("Different values: %.4f - %.4f", v, p),
                Math.abs(v - p) < 0.0001);
    }

}
