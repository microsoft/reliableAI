package ch.idsia.blip.api.experiments;


import ch.idsia.blip.api.TheTest;
import org.junit.Test;

import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;


public class FailPowah extends TheTest {

    private int n_datapoints;
    private String p;

    private void prepare() {

        n_datapoints = 10000;

        p = basePath + "exp/powah/failcase";
    }

    @Test
    public void testPowah() throws FileNotFoundException, UnsupportedEncodingException {

        testPowahSingle(0.99);
        testPowahSingle(0.9);
        testPowahSingle(0.8);
        testPowahSingle(0.7);
        testPowahSingle(0.6);
        testPowahSingle(0.51);
    }

    private void testPowahSingle(double k3) throws FileNotFoundException, UnsupportedEncodingException {
        PrintWriter wr_sc = new PrintWriter(
                String.format("%s-l_sc-%.2f.res", p, k3), "UTF-8");

        int lim = 100;

        for (int i1 = 1; i1 < lim; i1++) {
            for (int i2 = 1; i2 < lim; i2++) {
                testPower((i1 * 1.0) / lim, (i2 * 1.0) / lim, k3, wr_sc, false);
            }

            wr_sc.flush();

        }

        wr_sc.close();
    }

    private double simpleBic(double v1, double v2) {
        System.out.printf("%.2f %.2f, ", v1 / 2, v2 / 2);
        return n_datapoints
                * (((v1 / 2) * Math.log(v1 / 2)) + ((v2 / 2) * Math.log(v2 / 2)));
    }

    private void testPower(double k1, double k2, double k3, PrintWriter wr_sc, boolean v) {

        double b1 = k1 + k2;
        double b2 = 2 - k1 - k2;
        double c1 = (1 + k1) - k2;
        double c2 = (1 + k2) - k1;

        double bic_b = simpleBic(b1, b2);

        if (v) {
            System.out.printf("bic_b: %s%n", bic_b);
        }
        double bic_c = simpleBic(c1, c2);

        if (v) {
            System.out.printf("bic_c: %s%n", bic_c);
        }

        double d1 = (k3 * c1) + ((1 - k3) * c2);
        double d2 = (k3 * c2) + ((1 - k3) * c1);

        double bic_d = simpleBic(d1, d2);

        if (v) {
            System.out.printf("bic_d: %s%n", bic_d);
        }

        d1 = Math.abs((k2 - k1) * (1 - (2 * k3)));
        d2 = (k3 * c2) + ((1 - k3) * c1);

        bic_d = simpleBic(d1, d2);
        if (v) {
            System.out.printf("bic_d: %s%n", bic_d);
        }

        double bic_bc = advBic(k1, 1 - k1, k2, 1 - k2);

        if (v) {
            System.out.printf("bic_bc: %s%n", bic_bc);
        }

        double bic_cd = advBic(c1 / 2, (1 - (c1 / 2)), c2 / 2, (1 - (c2 / 2)));

        if (v) {
            System.out.printf("bic_cd: %s%n", bic_cd);
            return;
        }

        String msg = String.format("%.2f %.2f", k1, k2);

        if (bic_d > bic_b) {
            wr_sc.println(msg);
        }

    }

    private double advBic(double k1, double ik1, double k2, double ik2) {
        System.out.printf("%.2f %.2f %.2f %.2f, ", k1, ik1, k2, ik2);
        return (n_datapoints
                * ((k1 * Math.log(k1)) + (ik1 * Math.log(ik1))
                + (k2 * Math.log(k2)) + ((ik2) * Math.log(ik2))))
                        / 2;
    }

    @Test
    public void testLook() {

        prepare();

        testPower(0.9, 0.8, 0.3, null, true);
    }
}
