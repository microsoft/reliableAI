package ch.idsia.blip.core.sampler;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnErgWriter;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.inference.sample.BayesianSampler;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import org.junit.Test;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.*;
import static org.junit.Assert.assertTrue;


public class BayesianSamplerTest extends TheTest {

    private BayesianSampler samp;

    @Test
    public void test2() throws Exception {

        for (double t : new double[] { 0.5, 1.0, 2.0, 10.0}) {
            go2("BN_34", t);
        }
    }

    @Test
    public void testPRmultiple() {
        go3("BN_N_20_0", new int[] { 0, 2}, 0.0656058);
        go3("BN_N_20_0", new int[] { 0, 2, 5}, 0.0347697);
        go3("BN_N_20_0", new int[] { 0, 2, 5, 10, 12}, 0.0017015);
    }

    @Test
    public void testPRsingle() {

        go("BN_test2",
                new double[] {
            0.255235, 0.17429, 0.394189, 0.865862, 0.531247, 0.062774, 0.729822,
            0.08741, 0.684355, 0.498944
        });

        go("BN_test",
                new double[] {
            0.967262, 0.31825, 0.347895, 0.728076, 0.852163, 0.956497, 0.62552,
            0.603067, 0.596948, 0.29477
        });

        go("BN_N_20_0",
                new double[] {
            0.282108, 0.438524, 0.232565, 0.858942, 0.417696, 0.421095, 0.277517,
            0.369018, 0.509523, 0.430687, 0.771311, 0.741216, 0.0784126,
            0.772598, 0.462437, 0.318727, 0.374542, 0.704721, 0.309898, 0.601215
        });

    }

    @Test
    public void testMMAP() throws Exception {
        goMMAP("cancer", "3 0 2 4", "[1, 1, 0]");
    }

    private void goMMAP(String s, String q, String sol) {
        String h = f("%s/sampler/%s.uai", basePath, s);
        BayesianNetwork bn = getBayesianNetwork(h);

        samp = new BayesianSampler(bn);
        TIntIntHashMap evid = new TIntIntHashMap();
        int[] query = samp.getQuery(q);
        short[] map;

        for (int i = 0; i < 10; i++) {
            map = samp.MMAP(evid, query, 1);
            assertTrue(sol.equals(Arrays.toString(map)));
        }
    }

    @Test
    public void testMAP() throws Exception {
        goMAP("cancer", "[1, 1, 1, 1, 0]");
        goMAP("BN_simple", "[0, 1, 0, 0, 0]");
    }

    private void goMAP(String s, String sol) {
        String h = f("%s/sampler/%s.uai", basePath, s);
        BayesianNetwork bn = getBayesianNetwork(h);

        BnErgWriter.ex(f("%s/sampler/%s.erg", basePath, s), bn);
        BnNetWriter.ex(bn, f("%s/sampler/%s.net", basePath, s));
        samp = new BayesianSampler(bn);
        TIntIntHashMap evid = new TIntIntHashMap();
        short[] map;

        for (int i = 0; i < 10; i++) {
            map = samp.MAP(evid, 1);
            assertTrue(sol.equals(Arrays.toString(map)));
        }
    }

    @Test
    public void gh() throws Exception {
        BayesianNetwork bn = BnNetReader.ex(f("%s/sampler/cancer.net", basePath));

        BnUaiWriter.ex(f("%s/sampler/cancer.uai", basePath), bn);
    }

    @Test
    public void testMAR() throws Exception {

        goMAR("BN_simple", "1");
        goMAR("BN_N_20_0", "1");
        goMAR("BN_N_20_0", "2");
    }

    private void goMAR(String s, String res) throws IOException, InterruptedException {
        String h = f("%s/sampler/MAR/%s.%s", basePath, s, res);
        BayesianNetwork bn = getBayesianNetwork(
                f("%s/sampler/%s.uai", basePath, s));

        samp = new BayesianSampler(bn);
        TIntIntHashMap evid = samp.getEvidence(h + ".evid");

        bn.writeGraph(h + ".graph", evid.keys());

        double[][] P_x_w = samp.MAR(evid, 1.0);

        samp.writeMARoutput(h, P_x_w);

        Runtime rt = Runtime.getRuntime();
        Process pr = rt.exec(f("meld %s %s", h + ".MAR2", h + ".MAR"));

        pr.waitFor();
    }

    private void go(String s, double[] est) {

        BayesianNetwork bn = getBayesianNetwork(
                f("%s/sampler/%s.uai", basePath, s));

        samp = new BayesianSampler(bn);
        for (int i = 0; i < est.length; i++) {

            double tru = est[i];
            double res = samp.PR(i, 0, 1.0);

            pf("Estimate for %d: %.5f, real: %.5f, error: %.5f \n", i, res, tru,
                    Math.abs(res - tru));
            assertTrue(Math.abs(tru - res) < 0.1);
        }
    }

    private void go3(String s, int[] query, double tru) {

        BayesianNetwork bn = getBayesianNetwork(
                f("%s/sampler/%s.uai", basePath, s));

        samp = new BayesianSampler(bn);
        Arrays.sort(query);

        TIntIntHashMap values = new TIntIntHashMap();

        for (int q : query) {
            values.put(q, 0);
        }

        double res = samp.PR(values, 1.0);

        pf("Estimate for %s: %.5f, real: %.5f, error: %.5f \n",
                Arrays.toString(query), res, tru, Math.abs(res - tru));
        assertTrue(Math.abs(tru - res) < 0.1);

    }

    private void go2(String s, double t) throws Exception {

        BayesianNetwork bn = getBayesianNetwork(
                f("%s/sampler/%s/%s.uai", basePath, s, s));

        samp = new BayesianSampler(bn);
        Writer res = getWriter(
                f("%s/sampler/%s/%s-%.2f.samp", basePath, s, s, t));

        for (int i = 0; i < bn.n_var; i++) {
            double v = samp.PR(i, 0, 0.1);

            wf(res, "%.5f \n", v);
            res.flush();
        }
    }
}
