package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.math.ChiSquare;
import ch.idsia.blip.core.learn.constraints.oracle.Oracle;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.math.FastMath;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class MutualInformation extends Oracle {

    /**
     * Computed distance
     */
    private double[] mis;

    /**
     * Num of variables
     */
    private int r;

    private final int maxDF;

    public MutualInformation(DataSet dat) {
        this(dat, 0.95, 1);
    }

    public MutualInformation(DataSet dat, double alpha, int maxDF) {
        super(dat);
        this.alpha = alpha;

        this.maxDF = maxDF;

    }

    /*
     private int computeMaxDf(int N, double a, double w) throws OutOfRangeException {
     // FN bound
     double b = 0.05;
     maxDF = 0;
     double power = 1;
     while( power >= (1 -b) && maxDF <= 20) {
     maxDF ++;
     ChiSqrDistribution d = new ChiSqrDistribution(maxDF);
     double c = d.inverse(1 - a);
     double delta = 2 * N * Math.log(2)  * Math.pow(w, 2);
     NonCentralChiSquare s = new NonCentralChiSquare(maxDF, delta);
     power = s.cumulative(c, true, true);
     }
     return 0;
     }
     */

    /**
     * Compute the MI between all the variables
     *
     * @param v
     * @throws java.io.IOException if there is an error with the reader
     */
    private void compute(int v) throws IOException {

        r = dat.n_var;
        int size = (r * (r - 1)) / 2;

        if (v > 0) {
            pf("size: %d\n", size);
        }
        mis = new double[size];

        for (int p1 = 0; p1 < dat.n_var; p1++) {
            if (v > 0) {
                pf("%d\n", p1);
            }

            for (int p2 = p1 + 1; p2 < dat.n_var; p2++) {
                double mi = computeMi(p1, p2);

                mis[index(p1, p2)] = mi;
                // System.out.println(String.format("%d : %d -> %.8f (%d) ", p1, p2, mi * 100, index(p1, p2)));
            }
        }
    }

    public void compute() throws IOException {
        compute(0);
    }

    /**
     * I(X,Y)
     */
    public double computeMi(int x, int y) {

        double mi = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];

        for (int x_i = 0; x_i < x_ar; x_i++) {
            // P(x)
            int[] r_x = dat.row_values[x][x_i];

            if (r_x.length == 0) {
                continue;
            }

            double p_x = getFreq(r_x.length, x_ar);

            for (int y_i = 0; y_i < y_ar; y_i++) {
                // P(y)
                int[] r_y = dat.row_values[y][y_i];

                if (r_y.length == 0) {
                    continue;
                }
                double p_y = getFreq(r_y.length, y_ar);

                int n_xy = ArrayUtils.intersectN(r_x, r_y);

                if (n_xy == 0) {
                    continue;
                }
                // P(x, y)
                double p_xy = getFreq(n_xy, x_ar * y_ar);

                double t1 = FastMath.log(p_x) + FastMath.log(p_y);
                double t2 = FastMath.log(p_xy);
                double t = t2 - t1;

                t = p_xy * (t);
                mi += t;
                // pf("%.5f \n", mi);
                // System.out.printf(" %.5f * log ( %.5f / %.5f) - %.5f * %.5f \n", p_xy, p_xy, p_x * p_y, p_xy, FastMath.log(p_xy / (p_x * p_y)));
            }
        }

        return mi;
    }

    /**
     * I(X,Y|Z)
     */
    public double computeCMI(int x, int y, int z) {

        double mi = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];
        int z_ar = dat.l_n_arity[z];

        for (int z_i = 0; z_i < z_ar; z_i++) {
            // P(z)
            int[] r_z = dat.row_values[z][z_i];
            double p_z = getFreq(r_z.length, z_ar);

            for (int x_i = 0; x_i < x_ar; x_i++) {
                // P(x)
                int[] r_x = dat.row_values[x][x_i];
                double p_x = getFreq(r_x.length, x_ar);

                for (int y_i = 0; y_i < y_ar; y_i++) {
                    // P(y)
                    int[] r_y = dat.row_values[y][y_i];
                    double p_y = getFreq(r_y.length, y_ar);

                    int r_xy[] = ArrayUtils.intersect(r_x, r_y);

                    // P(x, y)
                    double p_xyz = getFreq(ArrayUtils.intersectN(r_xy, r_z),
                            x_ar * y_ar * z_ar);

                    double p_xz = getFreq(ArrayUtils.intersectN(r_x, r_z),
                            x_ar * z_ar);

                    double p_yz = getFreq(ArrayUtils.intersectN(r_y, r_z),
                            y_ar * z_ar);

                    mi += p_xyz
                            * ((FastMath.log(p_z) + FastMath.log(p_xyz))
                                    - (FastMath.log(p_xz) + FastMath.log(p_yz)));

                    pf(
                            " %.5f * log ( (%.5f * %.5f) / (%.5f * %.5f) ) -> %.5f \n",
                            p_xyz, p_z, p_xyz, p_xz, p_yz, mi);

                }
            }
        }

        return mi;
    }

    /**
     * I(X,Y|Z)
     */
    public double computeCMI(int x, int y, int[] z) {

        if (z.length == 0) {
            return computeMi(x, y);
        }

        int[][] z_rows = getZRows(z);

        double mi = 0;

        int x_ar = dat.l_n_arity[x];
        int y_ar = dat.l_n_arity[y];

        int z_ar = 1;

        for (int e_z : z) {
            z_ar *= dat.l_n_arity[e_z];
        }

        for (int x_i = 0; x_i < x_ar; x_i++) {
            // P(x)
            int[] r_x = dat.row_values[x][x_i];
            double p_x = getFreq(r_x.length, x_ar);

            for (int y_i = 0; y_i < y_ar; y_i++) {
                // P(y)
                int[] r_y = dat.row_values[y][y_i];
                double p_y = getFreq(r_y.length, y_ar);

                int r_xy[] = ArrayUtils.intersect(r_x, r_y);

                for (int z_i = 0; z_i < z_rows.length; z_i++) {

                    // if (containsMissing(z_i, z)) {
                    // continue;
                    // }

                    // P(z)
                    int[] r_z = z_rows[z_i];
                    double p_z = getFreq(r_z.length, z_ar);
                    // P(x, y)
                    double p_xyz = getFreq(ArrayUtils.intersectN(r_xy, r_z),
                            x_ar * y_ar * z_ar);

                    double p_xz = getFreq(ArrayUtils.intersectN(r_x, r_z),
                            x_ar * z_ar);

                    double p_yz = getFreq(ArrayUtils.intersectN(r_y, r_z),
                            y_ar * z_ar);

                    mi += p_xyz * FastMath.log(p_z * p_xyz / (p_xz * p_yz));

                    // System.out.printf(
                    // " %.5f * log ( %.5f / (%.5f * %.5f * %.5f)) - %.5f * %.5f \n",
                    // p_xyz, p_xyz, p_x, p_y, p_z, p_xyz,
                    // FastMath.log(p_xyz / (p_x * p_y * p_z)));
                }
            }
        }

        // System.out.printf("mi: x %d, y %d, z %d -> %.5f\n", x, y, z, mi);

        return mi;
    }

    int[][] getZRows(int[] z) {
        if (z.length == 1) {
            return dat.row_values[z[0]];
        }

        return computeParentSetValues(z);
    }

    /**
     * I(X,Y)
     */
    public double computeMi(int[] x, int[] y) {

        double mi = 0;

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

        for (int x_i = 0; x_i < x_rows.length; x_i++) {

            // P(x)
            int[] r_x = x_rows[x_i];
            double p_x = getFreq(r_x.length, x_ar);

            for (int y_i = 0; y_i < y_rows.length; y_i++) {

                // if (containsMissing(z_i, z)) {
                // continue;
                // }

                // P(y)
                int[] r_y = y_rows[y_i];
                double p_y = getFreq(r_y.length, y_ar);

                // P(x, y)
                double p_xy = getFreq(ArrayUtils.intersectN(r_x, r_y),
                        x_ar * y_ar);

                mi += p_xy * FastMath.log(p_xy / (p_x * p_y));

            }
        }

        return mi;
    }

    /*
     public double computeMiCond2(int x, int y, int z) {

     int z_ar = dat.l_n_arity[z];
     int x_ar = dat.l_n_arity[x];
     int y_ar = dat.l_n_arity[x];

     double mi = 0;
     double d;
     double n;

     for (int x_i = 0; x_i < dat.l_n_arity[x]; x_i++) {
     int[] r_x = dat.row_values[x][x_i];

     for (int y_i = 0; y_i < y_ar; y_i++) {
     int[] r_y = dat.row_values[y][y_i];
     int[] r_xy = ArrayUtils.intersect(r_x, r_y);

     for (int z_i = 0; z_i < z_ar; z_i++) {
     int[] r_z = dat.row_values[z][z_i];

     double p_z = getFreq(r_z.length, z_ar);

     double p_xyz = getFreq(ArrayUtils.intersectN(r_xy, r_z),
     x_ar * y_ar * z_ar);

     double p_yz = getFreq(ArrayUtils.intersectN(r_y, r_z),
     y_ar * z_ar);

     double p_xz = getFreq(ArrayUtils.intersectN(r_x, r_z),
     x_ar * z_ar);

     d = FastMath.log(p_z) + FastMath.log(p_xyz);
     n = FastMath.log(p_xz) + FastMath.log(p_yz);
     double f1 = p_xyz;
     double f2 = d - n;

     mi += f1 * f2;

     System.out.printf(
     " %.5f * log ( %.5f * %.5f / %.5f * %.5f) - %.5f * %.5f \n",
     p_xyz[x_i][y_i][z_i], p_z[z_i], p_xyz[x_i][y_i][z_i],
     p_xz[x_i][z_i], p_yz[y_i][z_i], p_xyz[x_i][y_i][z_i],
     FastMath.log(d / n));

     }
     }
     }

     System.out.printf("mi: x %d, y %d, z %s -> %.5f\n", x, y, z, mi);

     return mi;
     }
     */

    /*
     public double computeCMI(int x, int y, int[] z) {

     if (z.length == 0) {
     return computeCMI(x, y);
     }

     int[][] z_rows;

     if (z.length > 1) {
     z_rows = computeParentSetValues(z);
     } else {
     z_rows = dat.row_values[z[0]];
     }

     int z_ar = 1;

     for (int e_z : z) {
     z_ar *= dat.l_n_arity[e_z];
     System.out.println(e_z + " - " + dat.l_n_arity[e_z]);
     }

     double[] p_z = new double[z_ar];

     for (int z_i = 0; z_i < z_ar; z_i++) {
     p_z[z_i] = getFreq(z_rows[z_i].length, z_ar);
     }

     int x_ar = dat.l_n_arity[x];
     double[][] p_xz = new double[x_ar][];

     for (int x_i = 0; x_i < x_ar; x_i++) {
     p_xz[x_i] = new double[z_ar];
     for (int z_i = 0; z_i < z_ar; z_i++) {
     p_xz[x_i][z_i] = getFreq(
     ArrayUtils.intersectN(dat.row_values[x][x_i],
     z_rows[z_i]),
     x_ar * z_ar);
     }
     }

     int y_ar = dat.l_n_arity[y];
     double[][] p_yz = new double[y_ar][];

     for (int y_i = 0; y_i < y_ar; y_i++) {
     p_yz[y_i] = new double[z_ar];
     for (int z_i = 0; z_i < z_ar; z_i++) {
     p_yz[y_i][z_i] = getFreq(
     ArrayUtils.intersectN(dat.row_values[y][y_i],
     z_rows[z_i]),
     y_ar * z_ar);
     }
     }

     double[][][] p_xyz = new double[x_ar][][];

     for (int x_i = 0; x_i < x_ar; x_i++) {
     p_xyz[x_i] = new double[y_ar][];
     for (int y_i = 0; y_i < y_ar; y_i++) {
     int[] xy = ArrayUtils.intersect(dat.row_values[y][y_i],
     dat.row_values[y][y_i]);

     p_xyz[x_i][y_i] = new double[z_ar];
     for (int z_i = 0; z_i < z_ar; z_i++) {
     p_xyz[x_i][y_i][z_i] = getFreq(
     ArrayUtils.intersectN(xy, z_rows[z_i]),
     x_ar * y_ar * z_ar);
     }
     }

     }

     double mi = 0;
     double d;
     double n;

     for (int x_i = 0; x_i < dat.l_n_arity[x]; x_i++) {
     for (int y_i = 0; y_i < y_ar; y_i++) {
     for (int z_i = 0; z_i < z_ar; z_i++) {
     if (containsMissing(z_i, z)) {
     continue;
     }
     d = FastMath.log(p_z[z_i])
     + FastMath.log(p_xyz[x_i][y_i][z_i]);
     n = FastMath.log(p_xz[x_i][z_i])
     + FastMath.log(p_yz[y_i][z_i]);
     double f1 = p_xyz[x_i][y_i][z_i];
     double f2 = d - n;

     mi += f1 * f2;

     /* System.out.printf(
     " %.5f * log ( %.5f * %.5f / %.5f * %.5f) - %.5f * %.5f \n",
     p_xyz[x_i][y_i][z_i], p_z[z_i], p_xyz[x_i][y_i][z_i],
     p_xz[x_i][z_i], p_yz[y_i][z_i], p_xyz[x_i][y_i][z_i],
     FastMath.log(d / n));

     }
     }
     }

     System.out.printf("mi: x %d, y %d, z %s -> %.5f\n", x, y,
     Arrays.toString(z), mi);

     return mi;
     }
     */
    public double getMI(int p1, int p2) {
        return mis[index(p1, p2)];
    }

    /**
     * Index in matrix from row / column
     *
     * @param n1 row
     * @param n2 column
     * @return index
     */
    private int index(int n1, int n2) {
        if (n1 > n2) {
            int t = n1;

            n1 = n2;
            n2 = t;
        }

        int n = 0;

        for (int i = 0; i < n1; i++) {
            n += (r - i - 2);
        }
        n += (n2 - 1);
        return n;
    }

    public double distance(int j, int v) {
        if (j == v) {
            return 0;
        }

        return mis[index(j, v)];
    }

    // Hypothesis test: X is conditionally independent of Y given CMB, under the significance
    // level alpha.
    public boolean condInd(int x, int y, int[] z) {

        // Computes estimator for conditional mutual information
        double est_I = computeCMI(x, y, z);

        // 2*N*log(2)*I^(x, y|z) ~ X^2_{df}
        est_I *= 2 * dat.n_var * FastMath.log(2);

        // Computes degree of freedom
        int df = (dat.l_n_arity[x] - 1) * (dat.l_n_arity[y] - 1);

        for (int c : z) {
            df *= dat.l_n_arity[c];
        }

        df = Math.min(df, maxDF);

        double p_value = ChiSquare.pochisq(est_I, df);

        // My test is wheter they are INDEPENDENT, means CMI == 0
        return p_value > alpha;
    }

    public boolean condInd(int x, int y) {

        // estimator for mutual information
        double est_I = computeMi(x, y);

        // 2*N*log(2)*I^(x, y|z) ~ X^2_{df}
        est_I *= 2 * dat.n_var * FastMath.log(2);

        // Computes degree of freedom
        int df = (dat.l_n_arity[x] - 1) * (dat.l_n_arity[y] - 1);

        df = Math.min(df, maxDF);

        double p_value = ChiSquare.pochisq(est_I, df);

        // pf("%d - %.2f \n", y, p_value);

        // My test is wheter they are INDEPENDENT, means MI == 0
        return p_value > alpha;
    }

    @Override
    public String toString() {
        return f("Mutual Information (alpha: %.3f, maxDF: %d", alpha, maxDF);
    }
}
