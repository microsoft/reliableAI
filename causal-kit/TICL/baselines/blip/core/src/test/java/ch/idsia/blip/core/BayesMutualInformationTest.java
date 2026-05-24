package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.utils.analyze.BayesMutualInformation;
import ch.idsia.blip.core.utils.analyze.BayesMutualInformationMatlab;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.data.ArrayUtils.expandArray;
import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BayesMutualInformationTest extends TheTest {

    public BayesMutualInformationTest() {
        basePath += "mi/";
    }

    @Test
    public void testMi() throws IncorrectCallException, IOException {

        basePath += "size/";

        BayesianNetwork bn = getBnFromFile("simple.net");

        for (int n : new int[] { 10, 50, 100, 500, 1000, 5000}) {
            SamGe.ex(bn, basePath + "simple", n);
            DataSet d = getDataSet(f("%s%s-%d.dat", basePath, "simple", n));

            BayesMutualInformation bmi = new BayesMutualInformation(d);
            MutualInformation mi = new MutualInformation(d);

            int x = 0;
            int y = 1;
            int z = 2;

            /*
             int[][] x_r = d.row_values[x];
             int[][] y_r = d.row_values[y];

             p(ArrayUtils.intersectN(x_r[0], y_r[0]));
             p(ArrayUtils.intersectN(x_r[0], y_r[1]));
             */

            pf("%.7f \n", mi.computeCMI(x, y, z));
            pf("%.7f \n", bmi.computeCMI(x, y, z));
            bmi.out(x, y, z, basePath, f("%s-%d-p", "simple", n));

        }

    }

    @Test
    public void testMiCond() throws IncorrectCallException, IOException {

        basePath += "cond/";

        BayesianNetwork bn = getBnFromFile("simple.net");

        int n = 1000;

        SamGe.ex(bn, basePath + "simple", n);
        DataSet d = getDataSet(f("%s%s-%d.dat", basePath, "simple", n));

        for (int z : new int[] { 3, 4, 5, 6, 7, 8}) {

            BayesMutualInformation bmi = new BayesMutualInformation(d);
            MutualInformation mi = new MutualInformation(d);

            int x = 0;
            int y = 1;

            p(bn.name(x));
            p(bn.name(y));

            /*
             int[][] x_r = d.row_values[x];
             int[][] y_r = d.row_values[y];

             p(ArrayUtils.intersectN(x_r[0], y_r[0]));
             p(ArrayUtils.intersectN(x_r[0], y_r[1]));
             */

            pf("%.7f \n", mi.computeCMI(x, y, z));
            pf("%.7f \n", bmi.computeCMI(x, y, z));
            bmi.out(x, y, z, basePath, f("%s-%d-p", "simple", z));

        }

    }

    @Test
    public void testMiIncre() throws IncorrectCallException, IOException {

        basePath += "incre/";

        String net = "child";

        BayesianNetwork bn = getBnFromFile(net + ".net");

        int n = 5000;

        SamGe.ex(bn, basePath + net, n);
        DataSet d = getDataSet(f("%s%s-%d.dat", basePath, net, n));
        BayesMutualInformationMatlab bm = new BayesMutualInformationMatlab(d);
        BayesMutualInformation bmi = new BayesMutualInformation(d);
        MutualInformation mi = new MutualInformation(d);

        int x = 0;
        int y = 1;

        p(bn.name(x));
        p(bn.name(y));

        int[] z = new int[] {};

        for (int c : new int[] { 3, 4, 5, 6, 7, 8}) {

            z = expandArray(z, c);

            /*
             int[][] x_r = d.row_values[x];
             int[][] y_r = d.row_values[y];

             p(ArrayUtils.intersectN(x_r[0], y_r[0]));
             p(ArrayUtils.intersectN(x_r[0], y_r[1]));
             */

            p(z.length);
            pf("matlab: %.7f \n", bm.computeCMI(x, y, z));
            pf("mi: %.7f \n", mi.computeCMI(x, y, z));
            pf("%b \n", mi.condInd(x, y, z));
            pf("bmi: %.7f \n", bmi.computeCMI(x, y, z));

            p("\n");
            bm.out(f("%s%s-%d-p2", basePath, net, z.length));

            bmi.out(x, y, z, basePath, f("%s-%d-p", net, z.length));

        }

    }

    @Test
    public void testTest() throws IncorrectCallException, IOException {

        basePath += "test/";

        int n = 1000;

        double[] thetaz = new double[] { 0.25, 0.75};
        double[][] thetaw = new double[2][];

        thetaw[0] = new double[] { 0.25, 0.3, 0.4, 0.05};
        thetaw[1] = new double[] { 0.4, 0.15, 0.15, 0.3};

        for (int ix = 0; ix < 10; ix++) {

            if (ix != 0) {
                writeDataFile(f("%s%s-%d.dat", basePath, "simple", ix), n,
                        thetaz, thetaw);
            }

            DataSet d = getDataSet(f("%s%s-%d.dat", basePath, "simple", ix));

            BayesMutualInformation bmi = new BayesMutualInformation(d);
            MutualInformation mi = new MutualInformation(d);

            int x = 0;
            int y = 1;
            int z = 2;

            /*
             int[][] x_r = d.row_values[x];
             int[][] y_r = d.row_values[y];

             p(ArrayUtils.intersectN(x_r[0], y_r[0]));
             p(ArrayUtils.intersectN(x_r[0], y_r[1]));
             */

            double m = mi.computeCMI(x, y, z);

            pf("%.7f \n", m);
            double b = bmi.computeCMI(x, y, z);

            pf("%.7f \n", b);
            bmi.out(x, y, z, basePath, f("%s-%d-p", "simple", ix));

            if (ix == 0) {
                assertDoubleEquals(0.1034, m);
                assertDoubleEquals(0.1039, b);
            }

        }

    }

    @Test
    public void testTest2() throws IncorrectCallException, IOException {

        basePath += "test2/";

        int n = 1000;

        double[] thetaz = new double[] { 0.25, 0.75};
        double[][] thetaw = new double[2][];

        thetaw[0] = new double[] { 0.1474, 0.8383, 0.0021, 0.012};
        thetaw[1] = new double[] { 0.2262, 0.0989, 0.4695, 0.2053};

        for (int ix = 0; ix < 10; ix++) {

            writeDataFile(f("%s%s-%d.dat", basePath, "simple", ix), n, thetaz,
                    thetaw);

            DataSet d = getDataSet(f("%s%s-%d.dat", basePath, "simple", ix));

            BayesMutualInformation bmi = new BayesMutualInformation(d);
            MutualInformation mi = new MutualInformation(d);

            int x = 0;
            int y = 1;
            int z = 2;

            /*
             int[][] x_r = d.row_values[x];
             int[][] y_r = d.row_values[y];

             p(ArrayUtils.intersectN(x_r[0], y_r[0]));
             p(ArrayUtils.intersectN(x_r[0], y_r[1]));
             */

            double m = mi.computeCMI(x, y, z);

            pf("%.7f \n", m);
            double b = bmi.computeCMI(x, y, z);

            pf("%.7f \n", b);
            bmi.out(x, y, z, basePath, f("%s-%d-p", "simple", ix));

        }

    }

    private void writeDataFile(String s, int n, double[] thetaz, double[][] thetaw) throws IOException {
        Writer w = getWriter(s);

        wf(w, "%d \n", 3);
        wf(w, "X Y Z\n");
        wf(w, "2 2 2\n");
        wf(w, "%d\n", n);
        for (int i = 0; i < n; i++) {
            // get a value for z
            double r = Math.random() + Math.pow(2, -20); // Get a random number in (0, 1)

            short z = 0;

            while (z < 2 && (thetaz[z] < r)) { // Select a value from the probabilities given the random
                r -= thetaz[z];
                z++;
            }

            double[] tw = thetaw[z];

            r = Math.random() + Math.pow(2, -20);
            int iw = 0;

            while ((iw < 4) && (tw[iw] < r)) { // Select a value from the probabilities given the random
                r -= tw[iw];
                iw++;
            }

            int x = iw / 2;
            int y = iw % 2;

            wf(w, "%d %d %d\n", x, y, z);
            w.flush();
        }

        w.close();
    }

    @Test
    public void testDrch() throws IncorrectCallException, FileNotFoundException {
        DataSet d = getDataSet(basePath + "child-5000.dat");
        BayesMutualInformation bmi = new BayesMutualInformation(d);

        double[][] r = bmi.drchrnd(new double[] { 3.5, 7.5}, 1000);
        double s1 = 0;
        double s2 = 0;

        for (double[] rw : r) {
            p(Arrays.toString(rw));
            s1 += rw[0];
            s2 += rw[1];
        }
        p(s1);
        p(s2);
    }

    @Test
    public void testNoMissing() throws IncorrectCallException, FileNotFoundException {
        DataSet d = getDataSet(basePath + "child-5000.dat");
        BayesMutualInformation bmi = new BayesMutualInformation(d);

        bmi.getZRowsNoMissing(new int[] { 1, 2});
    }

}
