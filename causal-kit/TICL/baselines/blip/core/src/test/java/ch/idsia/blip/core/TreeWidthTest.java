package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.tw.TreeWidth;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Assert;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class TreeWidthTest extends TheTest {

    @Test
    public void testTW() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        Assert.assertEquals(findTW("tw/tw-0.net"), 2);
        Assert.assertEquals(findTW("tw/alarm.net"), 4);
        Assert.assertEquals(findTW("tw/barley.net"), 7);
        Assert.assertEquals(findTW("tw/mildew.net"), 4);
        Assert.assertEquals(findTW("tw/pigs.net"), 11);
        Assert.assertEquals(findTW("tw/water.net"), 10);

    }

    private int findTW(String f) throws IOException {

        BayesianNetwork bn = getBnFromFile(f);
        // printBn(f, bn);

        TreeWidth t = new TreeWidth();
        int tw = TreeWidth.go(bn);

        return tw;
    }

    private void printBn(String f, BayesianNetwork bn) {
        System.out.printf("\n### %s\n", f);
        for (int n = 0; n < bn.n_var; n++) {
            System.out.println(n + " - " + Arrays.toString(bn.parents(n)));
        }
        System.out.println();
    }

    @Test
    public void testMorale() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        List<TIntArrayList> paths = findCordless("tw/tw-3.net");

        paths = findCordless("tw/tw-2.net");
        Assert.assertTrue(paths.size() == 1);
        Assert.assertEquals(paths.get(0).toString(), "{0, 4, 3, 5}");

        paths = findCordless("tw/tw-0.net");
        Assert.assertTrue(paths.size() == 1);
        Assert.assertEquals(paths.get(0).toString(), "{0, 1, 4, 3}");

        // findCordless("random10-1.net");

    }

    private List<TIntArrayList> findCordless(String f) throws IOException {
        BayesianNetwork bn = getBnFromFile(f);

        System.out.printf("\n### %s\n\n", f);
        System.out.println(bn);

        Undirected ar = bn.moralize();

        printMoralization(bn, ar);

        List<TIntArrayList> paths = bn.findPaths(ar);

        System.out.println("\nCordless cycles: ");
        for (TIntArrayList path : paths) {
            System.out.println(path);
        }

        System.out.println();

        return paths;
    }

    private void printMoralization(BayesianNetwork bn, Undirected ar) {
        System.out.printf("%6s", "");
        for (int n = 0; n < bn.n_var; n++) {
            System.out.printf("%5d ", n);
        }
        System.out.println();
        for (int n = 0; n < bn.n_var; n++) {
            System.out.printf("%5d ", n);
            for (int n2 = 0; n2 < bn.n_var; n2++) {
                String s;

                if (n == n2) {
                    s = "-";
                } else {
                    s = ar.check(n, n2) ? "*" : " ";
                }
                System.out.printf("%5s ", s);
            }
            System.out.println();
        }
    }
}
