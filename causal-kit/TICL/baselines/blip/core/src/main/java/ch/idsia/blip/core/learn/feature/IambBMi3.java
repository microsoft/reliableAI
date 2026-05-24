package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.io.dat.DatFileReader;
import ch.idsia.blip.core.utils.score.BDeu;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.other.Gamma;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.pf;


/**
 * BMI - alpha optimized trough BDeu
 */
public class IambBMi3 extends IambBMi {

    private BDeu b;
    private final double[] alphas = new double[] {
        0.01, 0.1, 1, 10, 50, 100,
        1000, 1500, 2000};

    public IambBMi3(DatFileReader dr) throws IOException {
        super(dr);
        b = new BDeu(1, dat);
    }

    @Override
    public String getName() {
        return "iambBMi3";
    }

    @Override
    protected boolean condInd(int x, int y, int[] z, double alpha) {
        fixEss(x, y, z);

        double est_I = computeCMI(x, y, z);

        return est_I < alpha;

    }

    @Override
    protected double computeCMI(int x, int y, int[] z) {

        fixEss(x, y, z);

        return bmi.computeCMI(x, y, z);

    }

    private void fixEss(int x, int y, int[] z) {

        if (z.length == 0) {
            return;
        }

        int[][] x_r = dat.row_values[x];
        int x_ar = dat.l_n_arity[x];

        int[][] y_r = dat.row_values[y];
        int y_ar = dat.l_n_arity[y];

        int w_ar = x_ar * y_ar;

        int[][] z_r = bmi.getZRowsNoMissing(z);
        int z_ar = z_r.length;

        // Compute joints for x
        int[] n_z = new int[z_ar];

        for (int i = 0; i < z_ar; i++) {
            n_z[i] = z_r[i].length;
        }

        // Compute counts for joint X-Y
        int[][] n_w = new int[z_ar][];

        for (int z_i = 0; z_i < z_ar; z_i++) {
            n_w[z_i] = new int[w_ar];
            for (int x_i = 0; x_i < x_ar; x_i++) {
                int[] x_z_r = ArrayUtils.intersect(x_r[x_i], z_r[z_i]);

                for (int y_i = 0; y_i < y_ar; y_i++) {
                    // int[] xy = ArrayUtils.intersect(x_r[x_i], y_r[y_i]);
                    n_w[z_i][x_i * y_ar + y_i] = ArrayUtils.intersectN(y_r[y_i],
                            x_z_r);
                }
            }
        }

        double best_sk = -Double.MAX_VALUE;
        double best_a = -1;

        for (double a : alphas) {
            double sk = computeScore(n_z, z_ar, n_w, w_ar, a);

            if (sk > best_sk) {
                best_sk = sk;
                best_a = a;

                if (verb) {
                    pf("sk: %.4f, alpha %.3f \n", sk, a);
                }
            }
        }

        if (verb) {
            pf("chosen alpha %.3f \n", best_a);
        }
        bmi.ess = best_a;

    }

    private double computeScore(int[] x_v, int x_ar, int[][] y_v, int y_ar, double alpha) {

        double a_ij = alpha / y_ar;
        double a_ijk = a_ij / x_ar;

        double skore = 0;

        skore += (Gamma.lgamma(a_ij) * y_ar)
                - (Gamma.lgamma(a_ijk) * y_ar * x_ar);

        for (int i = 0; i < x_ar; i++) {

            skore -= Gamma.lgamma(a_ij + x_v[i]);

            for (int j = 0; j < y_ar; j++) {
                skore += Gamma.lgamma(a_ijk + y_v[i][j]);
            }

        }

        return skore;
    }
}
