package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.io.dat.DatFileReader;
import ch.idsia.blip.core.learn.feature.Iamb;
import ch.idsia.blip.core.learn.feature.IambBMi;
import ch.idsia.blip.core.learn.feature.IambBMi3;
import ch.idsia.blip.core.learn.feature.IambMi;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Test;

import java.io.*;
import java.util.*;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class FeatureSelectionTest extends TheTest {

    public FeatureSelectionTest() {
        basePath += "feature/";
    }

    @Test
    public void testSimple() throws Exception {

        System.out.println(
                "Working Directory = " + System.getProperty("user.dir"));

        String[] rest = new String[] { "alarm"};
        int n = 10000;

        for (String r : rest) {
            String g1 = basePath + r + "/";

            new File(basePath + r + "/").mkdir();
            g1 += "net";
            for (int i = 0; i < 5; i++) {

                // Open pa simple network, make some data
                BayesianNetwork bn = BnNetReader.ex(f("%s%s.net", basePath, r));
                SamGe sm = new SamGe();
                String h = f("%s-%d", g1, i);

                sm.go(bn, h, n);
                DataSet dat = getDataSet(f("%s-%d.dat", h, n));

                // Test pa new blanket searcher
                List<String> lst = new ArrayList<String>();

                // testBlanket(new IambBdeu(dat), h, bn, lst);
                // testBlanket(new IambDummy(dat), h, bn, lst);
                // testBlanket(new IambMi(dat), h, bn, lst);
                // testBlanket(new IambDyBIC(dat), h, bn, lst);

                // testBlanket(new IAMB_ent(dat), h, bn);

                plot(h, lst);
            }
        }
    }

    private void testEVERYTHING(Iamb iamb, String r, BayesianNetwork bn) throws Exception {
        String h = f("%s-%s", r, iamb.getName());

        p(h);
        Writer w = getWriter(basePath + h);

        iamb.verb = true;

        for (double t : iamb.getTresholds()) {
            double[] res = testThreshold(iamb, bn, t, false);

            if (cr(res[0]) && cr(res[1])) {
                wf(w, "%.3f %.3f \n", res[0], res[1]);
            }
            w.flush();
        }
        w.close();
    }

    private boolean cr(double re) {
        return !(Double.isNaN(re) || Double.isInfinite(re));
    }

    private void plot(String t, List<String> lst) throws IOException {
        String h = f("%s.plt", t);
        Writer w = getWriter(h);

        wf(w, "set terminal png size 800,600 \n");
        wf(w, "set output '%s.png' \n", t);
        wf(w, "set xlabel 'recall' \n");
        wf(w, "set xrange [0:1] \n");
        wf(w, "set ylabel 'precision' \n");
        wf(w, "set yrange [0:1] \n");
        wf(w, "set key top right \n");

        wf(w, "plot");
        boolean first = true;

        for (String s : lst) {
            if (first) {
                first = false;
            } else {
                wf(w, ", \\\n");
            }

            wf(w, " '%s' using 1:2", s);
        }
        w.close();

        h = f("gnuplot %s.plt", t);
        cmd(h);
    }

    private double[] testThreshold(Iamb iamb, BayesianNetwork bn, double t, boolean verb) throws Exception {

        // true positive
        int tp = 0;
        // real positive
        int r_p = 0;
        // classified positive
        int i_p = 0;

        double precis, recall;

        for (int v = 0; v < bn.n_var; v++) {
            int[] r_blanket = bn.blanket(v);
            // int[] i_blanket = iamb.ex(thread);
            int[] i_blanket = iamb.go(v, t);

            pf("%d - var:%s, real: %s, iamb: %s \n", v, bn.name(v),
                    iamb.fgh(r_blanket), iamb.fgh(i_blanket));

            tp += ArrayUtils.intersectN(r_blanket, i_blanket);
            r_p += r_blanket.length;
            i_p += i_blanket.length;

            if (verb) {
                pf("%d - var:%s \nreal: %s\niamb: %s \n", v, bn.name(v),
                        iamb.fgh(r_blanket), iamb.fgh(i_blanket));
                precis = tp * 1.0 / i_p;
                recall = tp * 1.0 / r_p;
                pf("t: %.5f, pre: %.3f, rec: %.3f \n", t, precis, recall);
            }

        }

        precis = tp * 1.0 / i_p;
        recall = tp * 1.0 / r_p;

        pf("t: %.5f, pre: %.3f, rec: %.3f \n", t, precis, recall);
        pf("tp: %d, r_p: %d, i_p: %d \n", tp, r_p, i_p);

        double[] res = new double[2];

        res[1] = precis;
        res[0] = recall;
        return res;
    }

    @Test
    public void testSimpler() throws Exception {
        String r = "alarm";
        int n = 10000;
        String h = basePath + r + "/";

        new File(h).mkdir();
        h += "net";

        // Open pa simple network, make some data
        BayesianNetwork bn = BnNetReader.ex(basePath + r + ".net");

        bn.writeGraph(h + "345");
        SamGe sm = new SamGe();

        // sm.ex(bn, h, n);

        DatFileReader dr = new DatFileReader();

        dr.init(h + "-" + n + ".dat");
        MutualInformation mi = new MutualInformation(dr.read());

        // IambDyBIC iamb = new IambDyBIC(dat);

        p(bn.l_nm_var.toString());

        /*
         IambBdeu iamb = new IambBdeu(dat);
         testThreshold(iamb, bn, 0.000001, false);
         testThreshold(iamb, bn, 0.00001, false);
         testThreshold(iamb, bn, 0.0001, false);
         testThreshold(iamb, bn, 0.001, false);
         testThreshold(iamb, bn, 0.01, false);
         */


        IambMi iamb = new IambMi(dr);

        iamb.verb = true;
        iamb.go(34, 0.2);

        iamb.verb = false;
        testThreshold(iamb, bn, 0.2, false);
        // testThreshold(iamb, bn, 0.05, false);


        /*
         IambBdeu iamb = new IambBdeu(dat);
         // testThreshold(iamb, bn, 100, true);

         int v = 0;        double t = 6;

         int[] r_blanket = bn.blanket(v);
         // int[] i_blanket = iamb.ex(thread);
         int[] i_blanket = iamb.ex(v, t);

         pf("%d - var: %s, threshold: %.2f \nreal: %s\niamb: %s \n", v, bn.name( v), t,
         iamb.fgh(r_blanket), iamb.fgh(i_blanket));
         */
    }

    private int[] bests(int i, DataSet dat, MutualInformation mi) {
        HashMap<Integer, Double> g = new HashMap<Integer, Double>();

        for (int j = 0; j < dat.n_var; j++) {
            if (i == j) {
                continue;
            }
            double f = mi.computeMi(i, j);

            pf("%s %.5f \n", dat.l_nm_var[j], f);
            g.put(j, f);

        }
        int k = 0;
        SortedSet<Map.Entry<Integer, Double>> e = RandomStuff.entriesSortedByValues(
                g, true);
        Iterator<Map.Entry<Integer, Double>> e1 = e.iterator();
        TIntArrayList f = new TIntArrayList();

        while (k < 10) {
            Map.Entry<Integer, Double> h = e1.next();

            f.add(h.getKey());
            k++;
        }
        int[] res = f.toArray();

        Arrays.sort(res);
        return res;
    }

    @Test
    public void testMutualInfo() throws FileNotFoundException, UnsupportedEncodingException {
        String r = "test";
        BayesianNetwork bn = BnNetReader.ex(basePath + r + ".net");
        SamGe sm = new SamGe();

        sm.go(bn, basePath + r, 1000);

        MutualInformation mi = new MutualInformation(
                getDataSet(basePath + r + "-1000.dat"));

        double m = mi.computeCMI(2, 1, new int[] { 0});

        pf("%.3f \n", m);
    }

    @Test
    public void testEasy() throws Exception {
        String r = "hepar2";
        int i = 0;
        int n = 1000;

        BayesianNetwork bn = BnNetReader.ex(f("%s%s.net", basePath, r));

        BnNetWriter.ex(bn, f("%s%s-new.net", basePath, r));
        String h = f("%s%s-%d", basePath, r, i);

        SamGe.ex(bn, h, n);

        DatFileReader dat = new DatFileReader();

        dat.init(f("%s-%d.dat", h, n));

        // Test pa new blanket searcher
        List<String> lst = new ArrayList<String>();

        int v = 1;

        /*
         int[] r_blanket = bn.blanket(v);
         int[] i_blanket = iamb.ex(v, 0.05);

         pf("%d - var:%s, real: %s, iamb: %s \n", v, bn.name( v),
         iamb.fgh(r_blanket), iamb.fgh(i_blanket));

         iamb.verb = false;

         // testThreshold(iamb, bn, 0.1, false);

         // testThreshold(iamb, bn, 0.05, false);
         */


        Iamb[] iambs = new Iamb[] {
            // new IambBMi3(dat),
            // new IambBMi1(dat),
            new IambBMi(dat)
        };

        Writer w = getWriter(basePath + r + ".plt");

        wf(w, "set terminal png\n");
        wf(w, "set output '%s.png'\n", r);
        wf(w, "set xrange [0:1]\n");
        wf(w, "set yrange [0:1]\n");

        wf(w, "plot ");
        boolean s = true;

        for (Iamb ia : iambs) {
            if (s) {
                s = false;
            } else {
                wf(w, ", \\\n");
            }
            wf(w, " '%s-%s' using 1:2", r, ia.getName());
        }

        w.close();

        for (Iamb ia : iambs) {
            testEVERYTHING(ia, r, bn);
        }

        String fs = f("gnuplot %s.plt", r);
        Process proc = Runtime.getRuntime().exec(fs, new String[0],
                new File(basePath));
        int exitVal = waitForProc(proc, 10000);

    }

    @Test
    public void testEasy2() throws Exception {
        String r = "child";
        int i = 0;
        int n = 1000;

        BayesianNetwork bn = BnNetReader.ex(f("%s%s.net", basePath, r));
        String h = f("%s%s-%d", basePath, r, i);

        SamGe.ex(bn, h, n);

        DatFileReader dr = new DatFileReader();

        dr.init(f("%s-%d.dat", h, n));
        IambBMi3 iamb3 = new IambBMi3(dr);

        iamb3.verb = true;
        iamb3.go(2, 0.05);

    }

}
