package ch.idsia.blip.core.learn.solver;


import ch.idsia.blip.core.learn.solver.brtl.BrutalSolver;
import ch.idsia.blip.core.learn.solver.src.brutal.BrutalGreedySearcher;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import org.junit.Test;

import java.util.Arrays;
import java.util.TreeSet;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class BrutalGreedyNewTest {

    @Test
    public void testIntersectN() {
        int[] a1 = new int[] { 1};
        int[] a2 = new int[] { 0, 1, 2, 3};

        p(Arrays.toString(ArrayUtils.intersect(a1, a2)));
    }

    @Test
    public void testIntersect() {
        BrutalGreedySearcher searcher = new BrutalGreedySearcher(
                new BrutalSolver(), 10);

        TreeSet<SIntSet> a1 = new TreeSet<SIntSet>();

        a1.add(new SIntSet(new int[] { 1, 3, 7, 9}));

        TreeSet<SIntSet> a2 = new TreeSet<SIntSet>();

        a2.add(new SIntSet(new int[] { 2, 4, 7, 9}));
        a2.add(new SIntSet(new int[] { 1, 3, 7, 9}));
        a2.add(new SIntSet(new int[] { 0, 3, 7, 9}));

        TreeSet<SIntSet> aux = searcher.intersect(a1, a2);

        for (SIntSet s : aux) {
            p(Arrays.toString(s.set));
        }

        a1.add(new SIntSet(new int[] { 0, 1, 7, 9}));

        aux = searcher.intersect(a1, a2);
        for (SIntSet s : aux) {
            p(Arrays.toString(s.set));
        }

        a1 = new TreeSet<SIntSet>();
        a1.add(new SIntSet(new int[] { 2, 3, 5, 7}));
        a1.add(new SIntSet(new int[] { 1, 3, 5, 7}));
        a1.add(new SIntSet(new int[] { 1, 2, 5, 7}));

        a2 = new TreeSet<SIntSet>();
        a2.add(new SIntSet(new int[] { 2, 3, 7, 9}));
        a2.add(new SIntSet(new int[] { 2, 3, 5, 7}));
        a2.add(new SIntSet(new int[] { 1, 3, 7, 9}));
        a2.add(new SIntSet(new int[] { 1, 3, 5, 7}));
        a2.add(new SIntSet(new int[] { 1, 2, 3, 7}));
        a2.add(new SIntSet(new int[] { 1, 2, 3, 5}));

        aux = searcher.intersect(a1, a2);
        for (SIntSet s : aux) {
            p(Arrays.toString(s.set));
        }
    }
}
