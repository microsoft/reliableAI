package ch.idsia.blip.core.learn.constraint;


import ch.idsia.blip.core.learn.solver.samp.SkelSampler;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class ConstraintOrderSamplerTest {

    @Test
    public void testSamples() {
        int n = 5;

        int[][] parents = new int[n][];

        for (int i = 0; i < n; i++) {
            parents[i] = new int[0];
        }

        parents[0] = new int[] { 1};
        parents[2] = new int[] { 3};

        test(n, parents);

    }

    @Test
    public void testSamples2() {
        int n = 5;

        int[][] parents = new int[n][];

        for (int i = 0; i < n; i++) {
            parents[i] = new int[0];
        }

        parents[0] = new int[] { 1, 4};

        parents[2] = new int[] { 3};
        parents[3] = new int[] { 4};

        test(n, parents);

    }

    private void test(int n, int[][] parents) {
        SkelSampler cos = new SkelSampler(parents, getRandom());

        cos.init();

        HashMap<String, Integer> cac = new HashMap<String, Integer>();

        for (int i = 0; i < 100000; i++) {
            int[] ord = cos.sample();
            String s = Arrays.toString(ord);

            if (!cac.containsKey(s)) {
                cac.put(s, 0);
            }
            cac.put(s, cac.get(s) + 1);
        }

        for (String s : cac.keySet()) {
            pf("%s -> %d \n", s, cac.get(s));
        }
    }

}
