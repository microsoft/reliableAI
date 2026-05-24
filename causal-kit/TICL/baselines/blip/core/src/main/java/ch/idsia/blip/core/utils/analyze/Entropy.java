package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class Entropy extends Analyzer {

    /**
     * Alpha for counts
     */
    double alpha = 1.0;

    /**
     * Num of variables
     */
    private int r;

    public Entropy(DataSet dat) {
        super(dat);
    }

    public double computeH(int x) {
        double h = 0;

        double p;
        int k;

        for (int v = 0; v < dat.l_n_arity[x]; v++) {
            k = dat.row_values[x][v].length;
            p = getFreq(k, dat.l_n_arity[x]);
            if (p == 0) {
                continue;
            }

            h += p * log(p);
        }

        if (h == 0)
            h = -Math.pow(2, -10);

        return -h;
    }

    public double computeHCond(int x, int y) {

        double h = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            // P(x)
            int[] r_x = dat.row_values[x][x_i];
            double p_x = getFreq(r_x.length, x_ar);

            for (int y_i = 0; y_i < y_ar; y_i++) {
                // P(y)
                int[] r_y = dat.row_values[y][y_i];
                double p_y = getFreq(r_y.length, y_ar);

                // P(x, y)
                double p_xy = getFreq(ArrayUtils.intersectN(r_x, r_y),
                        x_ar * y_ar);

                if (p_xy > 0) {
                    h += p_xy * log(p_xy / p_y);
                }

                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, Fastlog(p_xy / (p_x * p_y)));
            }
        }

        // System.out.printf("mi: x %d, y %d -> %.5f\n", x, y, mi);

        return -h;
    }

    public double computeHCond(int x, int[] y) {

        int[][] y_rows;

        if (y.length > 1) {
            y_rows = computeParentSetValues(y);
        } else if (y.length == 1) {
            y_rows = dat.row_values[y[0]];
        } else {
            return computeH(x);
        }

        int y_ar = 1;

        for (int e_y : y) {
            y_ar *= dat.l_n_arity[e_y];
        }

        double h = 0;

        int x_ar = dat.l_n_arity[x];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            // P(x)
            int[] r_x = dat.row_values[x][x_i];
            double p_x = getFreq(r_x.length, x_ar);

            for (int y_i = 0; y_i < y_rows.length; y_i++) {
                // if (containsMissing(z_i, z)) {
                // continue;
                // }

                // P(y)
                int[] r_y = y_rows[y_i];
                double p_y = getFreq(r_y.length, y_ar);

                int r_xy = ArrayUtils.intersectN(r_x, r_y);

                if (r_xy == 0) {
                    continue;
                }

                // P(x, y)
                double p_xy = getFreq(r_xy, x_ar * y_ar);

                h += p_xy * log(p_xy / p_y);

                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, Fastlog(p_xy / (p_x * p_y)));
            }
        }

        // System.out.printf("mi: x %d, y %d -> %.5f\n", x, y, mi);

        return -h;
    }

    public double computeH(int x, int y) {

        double h = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            int[] r_x = dat.row_values[x][x_i];

            for (int y_i = 0; y_i < y_ar; y_i++) {

                int[] r_y = dat.row_values[y][y_i];

                double p_xy = getFreq(ArrayUtils.intersectN(r_x, r_y),
                        x_ar * y_ar);

                h += p_xy * log(p_xy);

                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, Fastlog(p_xy / (p_x * p_y)));
            }
        }

        // System.out.printf("mi: x %d, y %d -> %.5f\n", x, y, mi);

        return -h;
    }

    public double computeH(int[] x) {
        double h = 0;

        int[][] x_rows;

        if (x.length > 1) {
            x_rows = computeParentSetValues(x);
        } else {
            x_rows = dat.row_values[x[0]];
        }
        int x_ar = 1;

        for (int e_x : x) {
            x_ar *= dat.l_n_arity[e_x];
        }

        for (int x_i = 0; x_i < x_rows.length; x_i++) {
            // if (containsMissing(z_i, z)) {
            // continue;
            // }
            // P(x)
            int[] r_x = x_rows[x_i];
            double p_x = getFreq(r_x.length, x_ar);

            h += p_x * log(p_x);
        }

        return -h;
    }

    public double computeH(int x, int[] y) {

        double h = 0;

        int x_ar = dat.l_n_arity[x];

        int[][] y_rows;

        if (y.length > 1) {
            y_rows = computeParentSetValues(y);
        } else {
            y_rows = dat.row_values[y[0]];
        }
        int y_ar = 1;

        for (int e_y : y) {
            y_ar *= dat.l_n_arity[e_y];
        }

        for (int y_i = 0; y_i < y_rows.length; y_i++) {
            // if (containsMissing(z_i, z)) {
            // continue;
            // }
            int[] r_y = y_rows[y_i];

            for (int x_i = 0; x_i < x_ar; x_i++) {
                int[] r_x = dat.row_values[x][x_i];

                int r_xy = ArrayUtils.intersectN(r_x, r_y);

                if (r_xy == 0) {
                    continue;
                }

                double p_xy = getFreq(r_xy, x_ar * y_ar);

                h += p_xy * log(p_xy);

                if (Double.isNaN(h)) {
                    p("ciao");
                }

                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, Fastlog(p_xy / (p_x * p_y)));
            }
        }

        // System.out.printf("mi: x %d, y %d -> %.5f\n", x, y, mi);
        return -h;
    }
}
