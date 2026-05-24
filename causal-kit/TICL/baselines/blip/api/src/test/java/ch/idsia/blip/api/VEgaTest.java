package ch.idsia.blip.api;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import org.junit.Test;

import java.util.Arrays;

import static org.junit.Assert.assertTrue;


public class VEgaTest extends TheTest {

    @Test
    public void completeRound() throws Exception {
        BayesianNetwork bn = getBnFromFile("old/random10-1.net");

        boolean verbose = true;

        VariableElimination inf_bn = new VariableElimination(bn, verbose);

        int[] query = new int[] { 3};
        TIntIntHashMap evidence = new TIntIntHashMap();
        BayesianFactor infere = inf_bn.query(query, evidence);
        double[] inf = infere.getPotent();

        System.out.println(Arrays.toString(inf));

        checkPot(0.390435, inf[0]);
        checkPot(0.609565, inf[1]);

        query = new int[] { 4};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.004671, inf[0]);
        checkPot(0.995329, inf[1]);

        query = new int[] { 8};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.330736, inf[0]);
        checkPot(0.123515, inf[1]);
        checkPot(0.079099, inf[2]);
        checkPot(0.466650, inf[3]);

        query = new int[] { 5};
        evidence = new TIntIntHashMap() {
            {
                put(4, 0);
            }
        };
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.552975, inf[0]);
        checkPot(0.447025, inf[1]);

        query = new int[] { 3};
        evidence = new TIntIntHashMap() {
            {
                put(4, 0);
            }
        };
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.237045, inf[0]);
        checkPot(0.762955, inf[1]);

        // ive.verbose = true;

        query = new int[] { 3, 5, 6};
        evidence = new TIntIntHashMap() {
            {
                put(4, 0);
            }
        };
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        System.out.println(Arrays.toString(inf));

    }

    @Test
    public void completeRoundSmall() throws Exception {
        BayesianNetwork bn = getBnFromFile("old/random4.net");

        System.out.println(bn.getTopologicalOrder());

        VariableElimination inf_bn = new VariableElimination(bn, false);

        boolean verbose = false;

        int[] query = new int[] { 3};
        TIntIntHashMap evidence = new TIntIntHashMap();
        BayesianFactor infere = inf_bn.query(query, evidence);
        double[] inf = infere.getPotent();

        checkPot(0.1125842, inf[0]);
        checkPot(0.8874158, inf[1]);

        query = new int[] { 2};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.1887048, inf[0]);
        checkPot(0.8112952, inf[1]);

        query = new int[] { 1};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.3621626, inf[0]);
        checkPot(0.0867952, inf[1]);
        checkPot(0.5510422, inf[2]);

        query = new int[] { 3};
        evidence = new TIntIntHashMap() {
            {
                put(1, 0);
            }
        };
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.021892473, inf[0]);
        checkPot(0.9781075, inf[1]);

        query = new int[] { 2};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.1107765, inf[0]);
        checkPot(0.8892234, inf[1]);
    }

    @Test
    public void completeRoundAlarm() throws Exception {
        BayesianNetwork bn = getBnFromFile("old/alarm.net");

        System.out.println(bn.getTopologicalOrder());

        VariableElimination inf_bn = new VariableElimination(bn, false);

        boolean verbose = false;

        int[] query = new int[] { 1};
        TIntIntHashMap evidence = new TIntIntHashMap() {
            {
                put(3, 0);
                put(4, 0);
            }
        };
        BayesianFactor infere = inf_bn.query(query, evidence);
        double[] inf = infere.getPotent();

        System.out.println(Arrays.toString(inf));
        System.out.print(inf[0]);
        checkPot(0.28417184, inf[0]);
        checkPot(0.71582816, inf[1]);

        query = new int[] { 2};
        infere = inf_bn.query(query, evidence);
        inf = infere.getPotent();
        checkPot(0.1761, inf[0]);
        checkPot(0.8239, inf[1]);
    }

    private void checkPot(double v, double p) {

        // System.out.println(String.format("Different values: %.6f - %.6f", v, v2));
        assertTrue(String.format("Different values: %.4f - %.4f", v, p),
                Math.abs(v - p) < 0.0001);
    }

    /* TEST AS:
     library("bnlearn")
     bn <- read.net("random10-1.net")

     cpquery(bn, (node3 == "state0"), TRUE)
     cpquery(bn, (node3 == "state1"), TRUE)

     */
}

