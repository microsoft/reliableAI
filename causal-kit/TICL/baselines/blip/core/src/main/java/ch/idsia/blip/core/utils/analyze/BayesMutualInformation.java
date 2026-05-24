package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.math.FastMath;
import ch.idsia.blip.core.utils.other.Gamma;

import java.io.File;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class BayesMutualInformation extends MutualInformation {

    private final Gamma r;

    private double[] samps;

    public double ess = 1.0;

    public BayesMutualInformation(DataSet dat) {
        super(dat);
        r = new Gamma();
    }

    @Override
    public double computeCMI(int x, int y, int z) {
        return this.computeCMI(x, y, new int[] { z});
    }

    @Override
    public double computeCMI(int x, int y, int[] z) {
        return computeCMI(x, y, z, 0.5);
    }

    /**
     * Bayesian PR of CMI: MI(X, Y | Z)
     * percentile to compute
     */
    public double computeCMI(int x, int y, int[] z, double percentile) {
        if (z.length == 0) {
            return computeMi(x, y);
        }

        int[][] x_r = dat.row_values[x];
        int x_ar = dat.l_n_arity[x];

        int[][] y_r = dat.row_values[y];
        int y_ar = dat.l_n_arity[y];

        int w_ar = x_ar * y_ar;

        int[][] z_r = getZRowsNoMissing(z);
        int z_ar = z_r.length;

        double[] n_z = getCounts(z_r, z_ar, ess);

        double[][] n_x = getCondCounts(x_r, x_ar, z_r, z_ar, ess);
        double[][] n_y = getCondCounts(y_r, y_ar, z_r, z_ar, ess);

        double[][] n_w = getCondJointCounts(x_r, x_ar, y_r, y_ar, w_ar, z_r,
                z_ar, ess);

        int mcsamples = 1000;

        samps = new double[mcsamples];

        // start sampling          
        for (int s_i = 0; s_i < mcsamples; s_i++) {

            double mi = 0;

            // Get dirichlet samples for Z counts
            double[] th_Z = drch(n_z);

            for (int z_i = 0; z_i < z_ar; z_i++) {

                // get dirichlet samples for X-Y
                double[] th_w = drch(n_w[z_i]);

                // get sum dirichlet samples for X and Y
                double[] th_x = new double[x_ar];
                double[] th_y = new double[y_ar];

                for (int i = 0; i < th_w.length; i++) {
                    th_x[i / y_ar] += th_w[i];
                    th_y[i % y_ar] += th_w[i];
                }

                // p(z)
                double p_z = th_Z[z_i];

                // double p_z_c = getFreq(z_r[z_i].length, z_ar);

                for (int x_i = 0; x_i < x_ar; x_i++) {

                    // p(x | z)
                    double p_xz = th_x[x_i];

                    // int xc = ArrayUtils.intersectN(x_r[x_i], z_r[z_i]);
                    // double p_xz_c = getFreq(xc, z_ar * x_ar);

                    for (int y_i = 0; y_i < y_ar; y_i++) {

                        // p(x, y | z)
                        double p_xyz = th_w[x_i * y_ar + y_i];
                        // int xyc = ArrayUtils.intersectN(
                        // ArrayUtils.intersect(x_r[x_i], y_r[y_i]), z_r[z_i]);
                        // double p_xyz_c = getFreq(xyc, z_ar * w_ar);

                        // p(y | z)
                        double p_yz = th_y[y_i];

                        // int yc = ArrayUtils.intersectN(y_r[y_i], z_r[z_i]);
                        // double p_yz_c = getFreq(yc, z_ar * y_ar);

                        mi += p_z * p_xyz
                                * (FastMath.log(p_xyz)
                                - (FastMath.log(p_xz) + FastMath.log(p_yz)));

                        /*
                         pf(" %.5f * log ( (%.5f * %.5f) / (%.5f * %.5f) ) -> %.5f \n",
                         p_xyz, p_z, p_xyz, p_xz, p_yz, mi);
                         */
                    }
                }
            }

            // pf(" %.5f \n", mi);

            samps[s_i] = mi;
        }

        Arrays.sort(samps);

        return samps[(int) (mcsamples * percentile)];
    }

    double[][] getCondJointCounts(int[][] x_r, int x_ar, int[][] y_r, int y_ar, int w_ar, int[][] z_r, int z_ar, double ess) {
        // Uniform prior for joint X-Y
        double[][] tx_y = new double[x_ar][];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            tx_y[x_i] = new double[y_ar];
            for (int y_i = 0; y_i < y_ar; y_i++) {
                tx_y[x_i][y_i] = 1.0 / w_ar;
            }
        }

        // Compute counts for joint X-Y
        double[][] n_w = new double[z_ar][];

        for (int z_i = 0; z_i < z_ar; z_i++) {
            n_w[z_i] = new double[w_ar];
            for (int x_i = 0; x_i < x_ar; x_i++) {
                int[] x_z_r = ArrayUtils.intersect(x_r[x_i], z_r[z_i]);

                for (int y_i = 0; y_i < y_ar; y_i++) {
                    // int[] xy = ArrayUtils.intersect(x_r[x_i], y_r[y_i]);
                    n_w[z_i][x_i * y_ar + y_i] = ArrayUtils.intersectN(y_r[y_i],
                            x_z_r)
                            + ess * tx_y[x_i][y_i];
                }
            }
        }
        return n_w;
    }

    double[] getCounts(int[][] x_r, int x_ar, double ess) {
        // Uniform prior for x (used for all precise classificators)
        double[] tx = new double[x_ar];

        for (int i = 0; i < x_ar; i++) {
            tx[i] = 1.0 / x_ar;
        }

        // Compute joints for x
        double[] n_x = new double[x_ar];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            n_x[x_i] = x_r[x_i].length + ess * tx[x_i];
        }
        return n_x;
    }

    private double[][] getCondCounts(int[][] r, int ar, int[][] z_r, int z_ar, double ess) {
        // Uniform prior for x (used for all precise classificators)
        double[] tx = new double[ar];

        for (int i = 0; i < ar; i++) {
            tx[i] = 1.0 / ar;
        }

        // Compute joints for Z
        double[][] n_xz = new double[z_ar][];

        for (int z_i = 0; z_i < z_ar; z_i++) {
            n_xz[z_i] = new double[ar];
            for (int i = 0; i < ar; i++) {
                n_xz[z_i][i] = ArrayUtils.intersectN(r[i], z_r[z_i])
                        + ess * tx[i];
            }
        }

        return n_xz;
    }

    private int getZAr(int[] z) {
        int z_ar = 1;

        for (int e_z : z) {
            z_ar *= dat.l_n_arity[e_z];
        }
        return z_ar;
    }

    public int[][] getZRowsNoMissing(int[] z) {

        int z_ar = getZAr(z);

        int[][] z_rows = getZRows(z);

        int[][] new_z_rows = new int[z_ar][];
        int j = 0;

        for (int z_i = 0; z_i < z_rows.length; z_i++) {
            // if (containsMissing(z_i, z)) {
            // continue;
            // }
            new_z_rows[j] = z_rows[z_i];
            j++;
        }

        return new_z_rows;
    }

    /**
     * Sample "a" times from a Dirichlet with parameters "p"
     */
    public double[][] drchrnd(double[] p, int a) {
        double[][] s = new double[a][];

        for (int i = 0; i < a; i++) {
            s[i] = drch(p);
        }
        return s;
    }

    /**
     * Sample from a Dirichlet with parameters "p"
     */
    private double[] drch(double[] p) {
        double sum = 0;
        double[] pr = new double[p.length];

        for (int i = 0; i < p.length; i++) {
            // pr[i] = r.random(p[i], 1);
            // TOCHECK
            sum += pr[i];
        }
        for (int i = 0; i < p.length; i++) {
            pr[i] /= sum;
        }
        return pr;
    }

    private void write(String basePath, String s, double precise) throws IOException {
        Writer w = getWriter(basePath + s + ".dat");
        double[][] hist = getHist(80);
        double max = 0;

        for (double[] h : hist) {
            wf(w, "%.7f %.7f\n", Math.exp(h[0]), h[1]);
            if (h[1] > max) {
                max = h[1];
            }
        }
        w.close();

        w = getWriter(basePath + s + ".plt");
        wf(w, "reset \n");
        wf(w, "set terminal png \n");
        wf(w, "set output '%s.png' \n", s);
        // wf(w, "set logscale x \n");
        wf(w, "set boxwidth 0.95 relative \n");
        wf(w, "set style fill transparent solid 0.5 noborder  \n");
        wf(w, "set arrow from %.5f,0 to %.5f,%.1f nohead lw 3 front \n", precise,
                precise, max);
        wf(w, "plot '%s.dat' using 1:2 w boxes \n", s);
        w.close();
    }

    // Create histogram data for the samples
    private double[][] getHist(int t) {
        Arrays.sort(samps);
        double s = Math.log(samps[0]);
        double e = Math.log(samps[samps.length - 1]);

        double step = (e - s) / t;
        double[][] hist = new double[t + 1][];
        double r = s - step / 2;
        int t_i = -1;

        for (double samp : samps) {
            double c = Math.log(samp);

            while (c > r) {
                r += step;
                t_i++;
                hist[t_i] = new double[] { r, 0};
            }

            hist[t_i][1]++;
        }

        return hist;
    }

    public void out(int x, int y, int z, String basePath, String s) throws IOException {
        out(x, y, new int[] { z}, basePath, s);
    }

    public void out(int x, int y, int[] z, String basePath, String s) throws IOException {
        computeCMI(x, y, z);

        double precise = super.computeCMI(x, y, z);

        write(basePath, s, precise);

        String d = f("gnuplot %s.plt", s);
        Process proc = Runtime.getRuntime().exec(d, new String[0],
                new File(basePath));
        int exitVal = waitForProc(proc, 100);

    }
}
