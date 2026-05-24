package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.Simulation;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;
import org.junit.Test;

import java.io.IOException;
import java.util.*;

import static ch.idsia.blip.core.utils.data.ArrayUtils.shuffleArray;
import static ch.idsia.blip.core.utils.RandomStuff.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


public class InferenceTest extends TheTest {

    @Test
    public void tryNets() {}

    @Test
    public void trytry() throws IOException {
        BayesianNetwork bn = getBnFromFile("inference/test.net");

        BnUaiWriter.ex(bn, basePath + "inference/test.uai");
    }

    @Test
    public void checkIJGP() throws Exception {
        String base_path = System.getProperty("user.dir") + "/" + basePath
                + "inference/";

        for (String s : new String[] { "test.uai", "cancer.uai", "BN_test.uai"}) { // ,

            p(
                    s);

            BayesianNetwork bn = getBayesianNetwork(base_path + s);

            // BayesianFactor.logComp = false;

            VariableElimination inf = new VariableElimination(bn, true);

            for (int i = 0; i < bn.n_var; i++) {
                // check marginals
                double[] mg = new double[bn.arity(i)];

                for (int j = 0; j < bn.arity(i); j++) {
                    TIntIntHashMap a = new TIntIntHashMap();

                    a.put(i, j);

                    double v = ijgp(base_path + s, a);

                    mg[j] = v;
                }

                BayesianFactor f = inf.query(i);

                check(s, i, f, mg);
            }

            // check evidence
            int[] vars = new int[bn.n_var];

            for (int i = 0; i < bn.n_var; i++) {
                vars[i] = i;
            }
            Random r = new Random(System.currentTimeMillis());

            for (int i = 0; i < 100; i++) {
                shuffleArray(vars, r);

                int evids = r.nextInt(2) + 1;
                int q = vars[0];
                TIntIntHashMap a = new TIntIntHashMap();

                for (int e = 0; e < evids; e++) {
                    int var = vars[e + 1];
                    int val = r.nextInt(bn.arity(var));

                    a.put(var, val);
                }

                BayesianFactor f = inf.query(q, a);

                double cng = ijgp(base_path + s, a);
                double[] mg = new double[bn.arity(q)];

                for (int j = 0; j < bn.arity(q); j++) {
                    a.put(q, j);
                    double v = ijgp(base_path + s, a);

                    mg[j] = v - cng;
                }

                check(s, i, f, mg);
            }
        }

    }

    private void check(String s, int i, BayesianFactor f, double[] mg) throws Exception {

        pf("Variable %d \nInf:  %s \nIjgp: %s \n\n", i, printPotent(f.potent),
                printPotent(mg));

        if (f.potent.length != mg.length) {
            throw new Exception(f("%s - Different size!", s));
        }

        for (int j = 0; j < f.potent.length; j++) {
            if (Math.abs(f.potent[j] - mg[j]) > 0.001) {
                throw new Exception(f("%s - Different value at %d!!", s, j));
            }
        }
    }

    public String printPotent(double[] potent) {
        StringBuilder str = new StringBuilder();

        str.append("[ ");
        for (int j = 0; j < potent.length; j++) {
            str.append(String.format("%.6f ", Math.pow(10, potent[j])));
        }
        str.append("]");

        return str.toString();
    }

    @Test
    public void tryAnotherInference() throws IOException {
        BayesianNetwork bn = getBnFromFile("inference/test.net");
        // http://www.bnlearn.com/documentation/man/cpquery.html

        VariableElimination inf = new VariableElimination(bn, false);

        TIntArrayList q = new TIntArrayList();
        TIntIntHashMap e = new TIntIntHashMap();

        BayesianFactor p_inf = inf.query(0);

        assertEquals("[ 0.333600 0.334000 0.332400 ]", p_inf.printPotent());

        p_inf = inf.query(1);
        assertEquals("[ 0.472400 0.113600 0.414000 ]", p_inf.printPotent());

        p_inf = inf.query(2);
        assertEquals("[ 0.743400 0.204800 0.051800 ]", p_inf.printPotent());

        p_inf = inf.query(3);
        assertEquals("[ 0.352764 0.314507 0.332728 ]", p_inf.printPotent());

        p_inf = inf.query(4);
        assertEquals("[ 0.388614 0.297737 0.313649 ]", p_inf.printPotent());

        e.put(1, 0);

        p_inf = inf.query(0, e);
        assertEquals("[ 0.604572 0.314564 0.080864 ]", p_inf.printPotent());

        p_inf = inf.query(2, e);
        assertEquals("[ 0.743400 0.204800 0.051800 ]", p_inf.printPotent());

        p_inf = inf.query(3, e);
        assertEquals("[ 0.490784 0.278508 0.230709 ]", p_inf.printPotent());

        p_inf = inf.query(4, e);
        assertEquals("[ 0.603607 0.293112 0.103281 ]", p_inf.printPotent());

        p_inf = inf.query(5, e);
        assertEquals("[ 0.501800 0.498200 ]", p_inf.printPotent());

        e.put(0, 1);

        p_inf = inf.query(2, e);
        assertEquals("[ 0.743400 0.204800 0.051800 ]", p_inf.printPotent());

        p_inf = inf.query(3, e);
        assertEquals("[ 0.267298 0.630757 0.101945 ]", p_inf.printPotent());

        p_inf = inf.query(4, e);
        assertEquals("[ 0.603607 0.293112 0.103281 ]", p_inf.printPotent());

        p_inf = inf.query(5, e);
        assertEquals("[ 0.501800 0.498200 ]", p_inf.printPotent());

    }

    @Test
    public void tryInference() throws IOException {
        BayesianNetwork bn = getBnFromFile("random10-1.net");

        VariableElimination inf = new VariableElimination(bn, false);

        int[] q;

        q = new int[] { 4};
        BayesianFactor p_inf = inf.query(q);

        assertEquals("[ 0.00467061 0.99532939 ]", p_inf.printPotent());

        q = new int[] { 1};
        p_inf = inf.query(q);
        assertEquals("[ 0.09590780 0.90409220 ]", p_inf.printPotent());

        q = new int[] { 2};
        p_inf = inf.query(q);
        assertEquals("[ 0.59774033 0.16603578 0.06889111 0.16733278 ]",
                p_inf.printPotent());

        q = new int[] { 9};
        p_inf = inf.query(q);
        assertEquals("[ 0.83035827 0.16964173 ]", p_inf.printPotent());

        q = new int[] { 7};
        p_inf = inf.query(q);
        assertEquals("[ 0.59906015 0.40093985 ]", p_inf.printPotent());

        q = new int[] { 0};
        p_inf = inf.query(q);
        assertEquals("[ 0.21580753 0.32466630 0.45952617 ]", p_inf.printPotent());

        q = new int[] { 1};
        TIntIntHashMap evidence = new TIntIntHashMap();

        evidence.put(4, 0);
        p_inf = inf.query(q, evidence);
        assertEquals("[ 0.86837436 0.13162564 ]", p_inf.printPotent());

        evidence.put(6, 1);
        p_inf = inf.query(q, evidence);
        assertEquals("[ 0.86013743 0.13986257 ]", p_inf.printPotent());

        q = new int[] { 7};
        p_inf = inf.query(q, evidence);
        assertEquals("[ 0.60801598 0.39198402 ]", p_inf.printPotent());

        q = new int[] { 8};
        p_inf = inf.query(q, evidence);
        assertEquals("[ 0.28249117 0.13885711 0.05324394 0.52540778 ]",
                p_inf.printPotent());

        q = new int[] { 3, 5};
        p_inf = inf.query(q, evidence);
        assertEquals("[ 0.28249117 0.13885711 0.05324394 0.52540778 ]",
                p_inf.printPotent());

    }

    @Test
    public void testArcs() {
        playWithArcs(3);
        playWithArcs(4);
        playWithArcs(5);
    }

    private void playWithArcs(int n) {
        Undirected arcs = new Undirected(n);

        System.out.println("length: " + arcs.arcs.length);
        System.out.println("index: ");
        for (int n1 = 0; n1 < n; n1++) {
            for (int n2 = n1 + 1; n2 < n; n2++) {
                System.out.println(arcs.index(n1, n2));
            }
        }
        System.out.println("r_index: ");
        for (int i = 0; i < arcs.arcs.length; i++) {
            System.out.println(Arrays.toString(arcs.r_index(i)));
        }
    }

    private int size(int n) {
        return (n * (n - 1)) / 2;
    }

    @Test
    public void playWithEliminationOrder() throws IOException {

        BayesianNetwork bn_random = getBnFromFile("old/random10-1.net");
        int[] vars = new int[bn_random.n_var];

        for (int v = 0; v < bn_random.n_var; v++) {
            vars[v] = v;
        }
        VariableElimination inf = new VariableElimination(bn_random, false);
        int[] order = inf.findEliminationOrder(vars);

        System.out.println("ord: " + Arrays.toString(order));

        assertTrue(
                "[8, 2, 9, 6, 3, 4, 5, 0, 1, 7]".equals(Arrays.toString(order)));
    }

    @Test
    public void simulateEliminationOrder() throws IOException {

        BayesianNetwork bn_random = getBnFromFile("old/random30.net");

        int[] vars = new int[bn_random.n_var];

        for (int v = 0; v < bn_random.n_var; v++) {
            vars[v] = v;
        }
        VariableElimination inf = new VariableElimination(bn_random, false);
        Simulation sim = new Simulation(bn_random, false);

        int[] order = inf.tryEliminationOrder(vars,
                VariableElimination.EliminMethod.MinFill);
        double cost = sim.simulateInference(order);

        System.out.println("MinFill: " + cost + " - " + Arrays.toString(order));

        int[] best = inf.findEliminationOrder(vars);

        cost = sim.simulateInference(best);

        System.out.println("Best: " + cost + " - " + Arrays.toString(best));
    }

    @Test
    public void playWithConstruction() {
        // first variable
        int v1 = 5;
        int d1 = 2;
        // second variable
        int v2 = 4;
        int d2 = 3;
        // third variable
        int v3 = 7;
        int d3 = 4;

        BayesianFactor psi1 = getFirstFactor(v1, d1, v2, d2);

        assertEquals("{4, 5}", psi1.dom.toString());
        assertEquals("{3, 2}", psi1.card.toString());

        BayesianFactor psi2 = new BayesianFactor(psi1, 4);

        assertEquals("{5}", psi2.dom.toString());
        assertEquals("{2}", psi2.card.toString());

        BayesianFactor psi3 = new BayesianFactor(psi1, 5);

        assertEquals("{4}", psi3.dom.toString());
        assertEquals("{3}", psi3.card.toString());

        BayesianFactor psi4 = getSecondFactor(v1, d1, v3, d3);

        BayesianFactor psi5 = new BayesianFactor(psi1, psi4);

        assertEquals("{4, 5, 7}", psi5.dom.toString());
        assertEquals("{3, 2, 4}", psi5.card.toString());

        BayesianFactor psi6 = new BayesianFactor(psi5, psi4);

        assertEquals("{4, 5, 7}", psi5.dom.toString());
        assertEquals("{3, 2, 4}", psi5.card.toString());

    }

    @Test
    public void playWithFactors() {

        // first variable
        int v1 = 5;
        int d1 = 2;
        // second variable
        int v2 = 4;
        int d2 = 3;
        // third variable
        int v3 = 7;
        int d3 = 4;

        BayesianFactor psi1 = getFirstFactor(v1, d1, v2, d2);
        String s_psi1 = psi1.toString();

        System.out.println("\n&&& psi1 &&& \n" + s_psi1);

        BayesianFactor psi2 = getSecondFactor(v1, d1, v3, d3);
        String s_psi2 = psi2.toString();

        System.out.println("\n&&& psi2 &&& \n" + s_psi2);

        BayesianFactor psi3 = psi1.product(psi2);
        String s_psi3 = psi3.toString();

        System.out.println("\n&&& psi3 &&& \n" + s_psi3);

        BayesianFactor psi4 = psi3.marginalization(v3);
        String s_psi4 = psi4.toString();

        System.out.println("\n&&& psi4 &&& \n" + s_psi4);

        BayesianFactor psi5 = psi3.marginalization(v2);
        String s_psi5 = psi5.toString();

        System.out.println("\n&&& psi5 &&& \n" + s_psi5);

        BayesianFactor psi6 = psi3.marginalization(v1);
        String s_psi6 = psi6.toString();

        System.out.println("\n&&& psi6 &&& \n" + s_psi6);

        BayesianFactor psi7 = psi1.reduction(5, 1);
        String s_psi7 = psi7.toString();
        BayesianFactor psi8 = psi1.reduction(4, 1);
        String s_psi8 = psi8.toString();

        System.out.println("\n&&& psi1 &&& \n" + s_psi1);
        System.out.println("\n&&& psi7 &&& \n" + s_psi7);
        System.out.println("\n&&& psi8 &&& \n" + s_psi8);

        BayesianFactor psi9 = psi3.reduction(5, 1);
        String s_psi9 = psi9.toString();
        BayesianFactor psi10 = psi9.reduction(4, 1);
        String s_psi10 = psi10.toString();

        System.out.println("\n&&& psi3 &&& \n" + s_psi3);
        System.out.println("\n&&& psi9 &&& \n" + s_psi9);
        System.out.println("\n&&& psi10 &&& \n" + s_psi10);
    }

    private BayesianFactor getSecondFactor(int v1, int d1, int v3, int d3) {
        TreeSet<Integer> dom2 = new TreeSet<Integer>();

        dom2.add(v1);
        dom2.add(v3);
        Map<Integer, Integer> card2 = new HashMap<Integer, Integer>();

        card2.put(v1, d1);
        card2.put(v3, d3);
        BayesianFactor psi2 = new BayesianFactor(dom2, card2);
        double[] potent2 = {
            0.30036414, 0.06502641, 0.08531442, 0.00524338, 0.12548187,
            0.24284191, 0.005041721, 0.17068614};

        psi2.updatePotent(potent2);
        return psi2;
    }

    private BayesianFactor getFirstFactor(int v1, int d1, int v2, int d2) {
        TreeSet<Integer> dom1 = new TreeSet<Integer>();

        dom1.add(v1);
        dom1.add(v2);
        Map<Integer, Integer> card1 = new HashMap<Integer, Integer>();

        card1.put(v1, d1);
        card1.put(v2, d2);
        double[] potent1 = {
            0.065785445, 0.032717004, 0.14413065, 0.17580023, 0.29846358,
            0.28310302};
        BayesianFactor psi1 = new BayesianFactor(dom1, card1);

        psi1.updatePotent(potent1);
        return psi1;
    }

    @Test
    public void playWithAncestorsDescendant() throws IOException {
        BayesianNetwork bn_random = getBnFromFile("old/random10-1.net");

        System.out.println(Arrays.toString(bn_random.getTopologicalOrder()));
        for (int v = 0; v < bn_random.n_var; v++) {
            System.out.printf("%d - anc: %s, desc: %s \n", v,
                    bn_random.getAncestors(v), bn_random.getDescendents(v));
        }
    }

}
