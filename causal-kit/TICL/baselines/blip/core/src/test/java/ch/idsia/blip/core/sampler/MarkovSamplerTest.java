package ch.idsia.blip.core.sampler;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.MarkovNetwork;
import ch.idsia.blip.core.io.MarkovUaiReader;
import ch.idsia.blip.core.inference.sample.MarkovSampler;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.pf;
import static org.junit.Assert.assertTrue;


public class MarkovSamplerTest extends TheTest {

    private MarkovSampler samp;

    @Test
    public void testMAR() throws Exception {
        goMAR("simplest");
        goMAR("Grids_17");
        goMAR("378-Promedus_33");

    }

    private void goMAR(String s) throws Exception {

        String h = f("%s/sampler/markov/%s", basePath, s);

        MarkovNetwork mn = MarkovUaiReader.ex(h + ".uai");

        samp = new MarkovSampler(mn);
        TIntIntHashMap evid = samp.getEvidence(h + ".evid");

        double[][] cnt = samp.MAR(evid, 5.0);

        samp.writeMARoutput(h, cnt);
    }

    @Test
    public void testMAP() throws Exception {
        goMAP("simplest", "[1, 0, 1]");
        goMAP("Grids_17", "[1, 0, 0, 1, 1, 0, 1, 0, 0, 1]");
    }

    private void goMAP(String s, String sol) throws Exception {

        String h = f("%s/sampler/markov/%s", basePath, s);

        MarkovNetwork mn = MarkovUaiReader.ex(h + ".uai");

        samp = new MarkovSampler(mn);
        TIntIntHashMap evid = samp.getEvidence(h + ".evid");
        short[] map;

        for (int i = 0; i < 10; i++) {
            map = samp.MAP(evid, 1);
            assertTrue(sol.equals(Arrays.toString(map)));
        }
    }

    @Test
    public void testPRsingle() throws FileNotFoundException {

        goPR("undirected_test", new double[] {
            1.6274, 1.5798, 1.3222
        });
    }

    private void goPR(String s, double[] est) throws FileNotFoundException {

        MarkovNetwork bn = MarkovUaiReader.ex(
                f("%s/sampler/markov/%s.uai", basePath, s));

        samp = new MarkovSampler(bn);
        for (int i = 0; i < est.length; i++) {

            double tru = est[i];
            double res = samp.PR(i, 0, 1.0);

            pf("Estimate for %d: %.5f, real: %.5f, error: %.5f \n", i, res, tru,
                    Math.abs(res - tru));
            assertTrue(Math.abs(tru - res) < 0.1);
        }
    }

}
