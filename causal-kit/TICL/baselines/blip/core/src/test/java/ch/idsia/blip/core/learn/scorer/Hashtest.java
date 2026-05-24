package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.data.SIntSet;
import org.junit.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.data.ArrayUtils.hashToParentSet;
import static ch.idsia.blip.core.utils.RandomStuff.getRandom;


public class Hashtest {

    @Test
    public void hashtest() {

        IndependenceScorer heu = new IndependenceScorer();

        heu.n_var = 20;

        HashSet<Long> closed = new HashSet<Long>();

        long start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            int[] p = hashToParentSet(i, heu.n_var);

            closed.add((long) i);
        }

        System.out.println(
                "Long, adding:  "
                        + (System.currentTimeMillis() - start) / 1000.0);

        start = System.currentTimeMillis();

        Random r = getRandom();

        for (int i = 0; i < 100000; i++) {
            int j = r.nextInt(100000);
            int[] p = hashToParentSet(j, heu.n_var);
            boolean b = closed.contains(j);
        }

        System.out.println(
                "Long, searching:  "
                        + (System.currentTimeMillis() - start) / 1000.0);

        HashSet<int[]> closed2 = new HashSet<int[]>();

        start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            int[] p = hashToParentSet(i, heu.n_var);

            closed2.add(p);
        }

        System.out.println(
                "int, adding:  " + (System.currentTimeMillis() - start) / 1000.0);

        start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            int j = r.nextInt(100000);
            int[] p = hashToParentSet(j, heu.n_var);
            boolean b = closed2.contains(p);
        }

        System.out.println(
                "int, searching:  "
                        + (System.currentTimeMillis() - start) / 1000.0);

        TreeSet<SIntSet> closed3 = new TreeSet<SIntSet>();

        start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            int[] p = hashToParentSet(i, heu.n_var);

            closed3.add(new SIntSet(p));
        }

        System.out.println(
                "int2, adding:  "
                        + (System.currentTimeMillis() - start) / 1000.0);

        start = System.currentTimeMillis();

        for (int i = 0; i < 100000; i++) {
            int j = r.nextInt(100000);
            int[] p = hashToParentSet(j, heu.n_var);
            boolean b = closed3.contains(new SIntSet(p));
        }

        System.out.println(
                "int2, searching:  "
                        + (System.currentTimeMillis() - start) / 1000.0);

        closed3 = new TreeSet<SIntSet>();

        closed3.add(new SIntSet(new int[] { 2, 3}));

        System.out.println(closed3.size());

        for (SIntSet aClosed3 : closed3) {
            System.out.println(aClosed3);
        }

        System.out.println(closed3.contains(new SIntSet(new int[] { 2, 3})));

    }

}
