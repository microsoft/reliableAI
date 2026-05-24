package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.other.BnBuilder;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.common.SamGe;
import ch.idsia.blip.core.io.bn.BnNetWriter;
import ch.idsia.blip.core.utils.score.BIC;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Test;

import java.io.*;
import java.util.ArrayList;


/**
 * Failing case of the heuristic
 */
public class FailComplete extends TheTest {

    private BnBuilder bn;
    private String p;

    private PrintWriter wr_1_ok;
    private PrintWriter wr_1_ko;
    private PrintWriter wr_2_ok;
    private PrintWriter wr_2_ko;
    private int n_datapoints;
    private BIC sc;

    @Test
    public void failTest() throws IOException, IncorrectCallException {

        prepare();
        prepareBN();

        // BC > BD, and I guess
        wr_1_ok = new PrintWriter(String.format("%s-1-ok.res", p), "UTF-8");
        // BC > BD, and I miss
        wr_1_ko = new PrintWriter(String.format("%s-1-ko.res", p), "UTF-8");
        // BD > BC, and I guess
        wr_2_ok = new PrintWriter(String.format("%s-2-ok.res", p), "UTF-8");
        // BD > BC, and I guess
        wr_2_ko = new PrintWriter(String.format("%s-2-ko.res", p), "UTF-8");

        int lim = 100;

        for (int i1 = 1; i1 < lim; i1++) {
            for (int i2 = 1; i2 < lim; i2++) {
                testComplete((i1 * 1.0) / lim, (i2 * 1.0) / lim);
            }

            wr_1_ok.flush();
            wr_1_ko.flush();
            wr_2_ok.flush();
            wr_2_ko.flush();
        }

        wr_1_ok.close();
        wr_1_ko.close();
        wr_2_ok.close();
        wr_2_ko.close();
    }

    private void prepare() {

        n_datapoints = 10000;
        p = basePath + "exp/failcase";
    }

    private void prepareBN() {
        bn = new BnBuilder(4);
        bn.n_var = 4;
        bn.l_ar_var = new TIntArrayList();
        bn.l_nm_var = new ArrayList<String>();
        bn.l_values_var = new ArrayList<String[]>();
        for (int i = 0; i < 4; i++) {
            bn.l_ar_var.add(2);
            bn.l_nm_var.add(String.valueOf(i));
            bn.l_values_var.add(new String[] { "0", "1"});
        }

        bn.l_parent_var = new ArrayList<int[]>();
        bn.l_potential_var = new ArrayList<double[]>();
        // 1 -> 0, 2 -> 0
        bn.l_parent_var.add(new int[] { 1, 2});
        bn.l_potential_var.add(new double[] {});
        // 1
        bn.l_parent_var.add(new int[] {});
        bn.l_potential_var.add(new double[] { 0.5, 0.5});
        // 3 -> 2
        bn.l_parent_var.add(new int[] { 3});
        bn.l_potential_var.add(new double[] { 0.8, 0.2, 0.2, 0.8});
        // 3
        bn.l_parent_var.add(new int[] {});
        bn.l_potential_var.add(new double[] { 0.5, 0.5});
    }

    @Test
    public void testSingle() throws IOException, IncorrectCallException {

        prepare();
        prepareBN();

        // Write bn to file
        OutputStreamWriter bn_stream = new OutputStreamWriter(
                new FileOutputStream(p + "-bn.net"), "utf-8");
        BufferedWriter bn_wr = new BufferedWriter(bn_stream);

        BnNetWriter.ex(bn.toBn(), bn_wr);

        DataSet dat = getDat(0.9, 0.8);

        // BIC void score of 0
        double sk_0 = sc.computeScore(0);

        // BIC score of 1
        double sk_0_1 = sc.computeScore(0, 1);

        System.out.println("sk_0_1:" + sk_0_1);

        // BIC score of 2
        double sk_0_2 = sc.computeScore(0, 2);

        System.out.println("sk_0_2:" + sk_0_2);

        // BIC score of 3
        double sk_0_3 = sc.computeScore(0, 3);

        System.out.println("sk_0_3:" + sk_0_3);

        // BIC score of 1-2
        int[] pars = { 1, 2};
        int[][] p_values = sc.computeParentSetValues(pars);
        double sk_0_12 = sc.computeScore(0, pars);

        System.out.println("sk_0_12:" + sk_0_12);

        // BIC score of 2-3
        pars = new int[] { 2, 3};
        p_values = sc.computeParentSetValues(pars);
        double sk_0_23 = sc.computeScore(0, pars);

        System.out.println("sk_0_23:" + sk_0_23);
    }

    private DataSet getDat(double k1, double k2) throws IOException, IncorrectCallException {
        // bn.l_potential_var.set(0, new double[]{k1, 1 - k1, 1 - k2, k2, k2, 1 - k2, 1 - k1, k1});
        bn.l_potential_var.set(0,
                new double[] { k1, 1 - k1, 1 - k2, k2, k2, 1 - k2, 1 - k1, k1});

        // Generate sample
        SamGe samGe = new SamGe();

        samGe.go(bn.toBn(), p, n_datapoints);

        // Read data
        DataSet dat = RandomStuff.getDataSet(
                String.format("%s-%d.dat", p, n_datapoints));

        return dat;
    }

    private void testComplete(double k1, double k2) throws IOException, IncorrectCallException {

        DataSet dat = getDat(k1, k2);

        // BIC void score of 0
        double sk_0 = sc.computeScore(0);

        // BIC score of 1
        double sk_0_1 = sc.computeScore(0, 1);

        // BIC score of 2
        double sk_0_2 = sc.computeScore(0, 2);

        // BIC score of 3
        double sk_0_3 = sc.computeScore(0, 3);

        // Heu score of 1-2
        double heu_0_12 = sk_0_1 + sk_0_2;

        // Heu score of 1-3
        double heu_0_23 = sk_0_1 + sk_0_3;

        // BIC score of 1-2
        int[] pars = { 1, 2};
        int[][] p_values = sc.computeParentSetValues(pars);
        double sk_0_12 = sc.computeScore(0, pars);

        // BIC score of 2-3
        pars = new int[] { 2, 3};
        p_values = sc.computeParentSetValues(pars);
        double sk_0_23 = sc.computeScore(0, pars);

        String msg = String.format("%.2f %.2f", k1, k2);

        if (((heu_0_12 > heu_0_23) && (sk_0_12 > sk_0_23))) {
            // BC > BD, I guess
            wr_1_ok.println(msg);
        } else if (((heu_0_12 > heu_0_23) && (sk_0_23 > sk_0_12))) {
            // BC > BD, I miss
            wr_1_ko.println(msg);
        } else if (((heu_0_23 > heu_0_12) && (sk_0_23 > sk_0_12))) {
            // BD > BC, I guess
            wr_2_ok.println(msg);
        } else if (((heu_0_23 > heu_0_12) && (sk_0_12 > sk_0_23))) {
            // BD > BC, I miss
            wr_2_ko.println(msg);
        }

        double sk_2_3 = sc.computeScore(2, 3);

        System.out.printf(
                "sk_0_12: %.3f, sk_0_23:%.3f, sk_0_1:%.3f, sk_0_3:%.3f  \n",
                sk_0_12, sk_0_23, sk_0_1, sk_0_3);

    }

}
