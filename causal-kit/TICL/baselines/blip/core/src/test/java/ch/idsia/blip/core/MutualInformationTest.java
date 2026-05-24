package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Entropy;
import ch.idsia.blip.core.utils.analyze.LogLikelihood;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Assert;
import org.junit.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;

import static ch.idsia.blip.core.utils.data.ArrayUtils.expandArray;
import static ch.idsia.blip.core.utils.RandomStuff.doubleEquals;
import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class MutualInformationTest extends TheTest {

    @Test
    public void someTimes() throws Exception {

        whatYouThink("child-5000.dat");
        // letsTryIt("child-10000.scores");

        /*
         greedyTest("child-10000.scores");

         for (int p = 0; p < 100; p++) {
         greedyTest("child-10000.scores");
         }
         */

    }

    private void whatYouThink(String s) throws IOException, IncorrectCallException {

        s = basePath + "/scorer/" + s;

        DataSet dat = getDataSet(s);
        BufferedWriter writer = new BufferedWriter(
                new OutputStreamWriter(System.out));

        MutualInformation mi = new MutualInformation(dat);
        LogLikelihood ll = new LogLikelihood(dat);
        Entropy h = new Entropy(dat);

        for (int i = 0; i < dat.n_var; i++) {
            for (int j = 0; j < dat.n_var; j++) {
                if (i == j) {
                    continue;
                }

                double m = mi.computeMi(i, j);

                double ll_i = ll.computeLL(i);
                double ll_ij_c = ll.computeLL(i, j);

                double ll_j = ll.computeLL(j);
                double ll_ji_c = ll.computeLL(j, i);

                double h_i = h.computeH(i);
                double h_j = h.computeH(j);
                double h_ij_c = h.computeHCond(i, j);
                double h_ij_c2 = h.computeHCond(i, new int[] { j});

                Assert.assertTrue(doubleEquals(h_ij_c, h_ij_c2));
                double h_ji_c = h.computeHCond(j, i);
                double h_ji_c2 = h.computeHCond(j, new int[] { i});

                Assert.assertTrue(doubleEquals(h_ji_c, h_ji_c2));

                double h_ij = h.computeH(i, j);

                // System.out.printf("mi: %.3f - h(x) - h(x|y): %.3f \n", m, h_i - h_ij );
                // System.out.printf("h(x): %.3f, ll(x): %.3f, h(x|y): %.3f, ll(x|y): %.3f \n", h_i * dat.n_datapoints, ll_i, h_ij_c * dat.n_datapoints, ll_ij );
                // System.out.printf("mi: %.3f - ll(x) - ll(x|y): %.3f ### \tll thread: %.3f, ll thread - j: %.3f \n", m, ll_i - ll_ij, ll_i, ll_ij );

                double a = h_i + h_j - h_ij;

                System.out.printf("mi(x|y): %.5f, h(x) + h(y) - h(x,y): %.5f \n",
                        m, a);
                Assert.assertTrue(doubleEquals(m, a));

                a = h_i - h_ij_c;
                System.out.printf("mi(x|y): %.5f, h(x) - h(x|y): %.5f \n", m, a);
                Assert.assertTrue(doubleEquals(m, a));

                a = h_j - h_ji_c;
                System.out.printf("mi(x|y): %.5f, h(y) - h(y|x): %.5f \n", m, a);
                Assert.assertTrue(doubleEquals(m, a));

                a = h_ij - h_ji_c - h_ij_c;
                System.out.printf(
                        "mi(x|y): %.5f, h(x,y) - h(y|x) - h(x|y): %.5f \n", m, a);
                Assert.assertTrue(doubleEquals(m, a));

                a = -(ll_i - ll_ij_c) / dat.n_datapoints;
                System.out.printf("mi(x|y): %.5f, ll(x) - ll(x|y): %.5f \n", m,
                        a);
                Assert.assertTrue(doubleEquals(m, a));

                a = -(ll_j - ll_ji_c) / dat.n_datapoints;
                System.out.printf("mi(x|y): %.5f, ll(y) - ll(y|x): %.5f \n", m,
                        a);
                Assert.assertTrue(doubleEquals(m, a));

                System.out.println();
            }
        }

        // for every x
        for (int x = 0; x < dat.n_var; x++) {
            for (int y = 0; y < dat.n_var; y++) {
                if (x == y) {
                    continue;
                }

                double h1 = h.computeHCond(x, y);
                double h2 = h.computeHCond(x, new int[] { y});

                System.out.printf("h(x|y): %.5f, h(x|{y}): %.5f \n", h1, h2);
                Assert.assertTrue(doubleEquals(h1, h2));
            }
        }

        // for every x
        for (int x = 0; x < dat.n_var; x++) {
            for (int y = 0; y < dat.n_var; y++) {
                if (x == y) {
                    continue;
                }

                for (int z = 0; z < dat.n_var; z++) {
                    if (z == x || z == y) {
                        continue;
                    }

                    double m1 = mi.computeCMI(x, y, z);
                    // double m2 = mi.computeMiCond2(x, y, z);
                    double m3 = mi.computeCMI(x, y, new int[] { z});

                    // double m4 = mi.computeMiCond3(x, y, new int[] { z});

                    System.out.printf("m(x;y|z): %.5f, m(x;y|{z}): %.5f \n", m1,
                            m3);
                    Assert.assertTrue(doubleEquals(m1, m3));
                }
            }
        }

        // for every x
        for (int x = 0; x < dat.n_var; x++) {

            // for every y
            for (int y = 0; y < dat.n_var; y++) {
                if (x == y) {
                    continue;
                }

                // for every subset
                int allMasks = (1 << dat.n_var);

                for (int i = 1; i < allMasks; i++) {

                    // get new subset
                    TIntArrayList z = new TIntArrayList();

                    for (int j = 0; j < dat.n_var; j++) {
                        if ((i & (1 << j)) > 0) {
                            z.add(j);
                            // System.out.println(s);
                        }
                    }

                    // simple check
                    if (z.size() > 3 || z.contains(x) || z.contains(y)) {
                        continue;
                    }

                    double h_x_z = h.computeHCond(x, z.toArray());
                    double h_x_yz = h.computeHCond(x,
                            expandArray(z.toArray(), y));
                    double m = mi.computeCMI(x, y, z.toArray());

                    double a = h_x_z - h_x_yz;

                    System.out.printf(
                            "mi(x;y|z): %.5f, h(x|z) - h(x|y,z): %.5f \n", m, a);
                    Assert.assertTrue(doubleEquals(m, a));

                    double h_y_z = h.computeHCond(y, z.toArray());
                    double h_y_xz = h.computeHCond(y,
                            expandArray(z.toArray(), x));

                    a = h_y_z - h_y_xz;
                    System.out.printf(
                            "mi(x;y|z): %.5f, h(y|z) - h(y|x,z): %.5f \n", m, a);
                    Assert.assertTrue(doubleEquals(m, a));
                }
            }
        }
    }
}

