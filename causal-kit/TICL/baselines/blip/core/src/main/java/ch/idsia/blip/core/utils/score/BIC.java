package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;

import java.util.Map;


/**
 * Computes the BIC
 */
public class BIC extends Score {

    public BIC(DataSet dat) {
        super(dat);
    }

    @Override
    public double computeScore(int n) {

        numEvaluated++;

        int arity = dat.l_n_arity[n];
        int[][] values = dat.row_values[n];

        double skore = 0;

        // for(int[] v: values)
        // System.out.println(Arrays.toString(v));

        double alpha = 1;
        double alpha_i = (alpha / arity);

        for (int v = 0; v < arity; v++) {
            double p = (values[v].length + alpha_i) / (dat.n_datapoints + alpha);

            skore += values[v].length * log(p);
            // System.out.printf("%d - %.2f, ", values[v].length, p);
            // System.out.println((values[v].length * 1.0) + (alpha / arity));
            // System.out.println((n_datapoints + alpha));

            // System.out.println(values[v].length + " & " + log(p) + " & " + p + " % " + skore);
        }

        double pen = getPenalization(arity);

        skore -= pen;

        return skore;
    }

    double getPenalization(int arity) {
        return log(dat.n_datapoints) * (arity - 1) / 2;
    }

    @Override
    public double computeScore(int n, int[] set_p, int[][] p_values) {

        numEvaluated++;

        double skore = 0;

        int arity = dat.l_n_arity[n];

        int p_arity = 1;

        for (int p : set_p) {
            p_arity *= dat.l_n_arity[p];
        }

        double alpha = 1;
        double alpha_i = alpha / arity;
        double alpha_ij = alpha / (arity * p_arity);

        for (int p_v = 0; p_v < p_values.length; p_v++) {

            // Check if it contains a missing value; in case, don't consider it
            // if (containsMissing(z_i, z)) {
            // continue;
            // }
            int valcount;

            for (int v = 0; v < arity; v++) {
                valcount = ArrayUtils.intersectN(dat.row_values[n][v],
                        p_values[p_v]);

                if (valcount == 0) {
                    continue;
                }

                // System.out.printf("%.4f, %d - %d, %.3f \n", skore, valcount[v], p_values[p_v].length,  log((valcount[v] * 1.0) / p_values[p_v].length));

                skore += valcount
                        * (log(valcount + alpha_ij)
                                - log(p_values[p_v].length + alpha_i));

                // System.out.printf("%d- %.2f, ", valcount[v], p);

                // System.out.println(valcount[v] + "   " + log(p) + "   " + p + "   " + skore);
            }

        }
        // Penalization term
        skore -= getPenalization(arity, p_arity);

        return skore;
    }

    double getPenalization(int arity, int p_arity) {
        double pen = log(dat.n_datapoints);

        pen *= (arity - 1);
        pen *= p_arity / 2;
        return pen;
    }

    public double getPenalization(int n, int[] pset) {
        int p_arity = 1;

        for (int p : pset) {
            p_arity *= dat.l_n_arity[p];
        }
        return getPenalization(dat.l_n_arity[n], p_arity);
    }

    public double inter(int n, int[] set, int p2) {

        // Compute interaction
        int p1_ar = 0;

        for (int v1 : set) {
            p1_ar = dat.l_n_arity[v1];
        }

        int p2_ar = dat.l_n_arity[p2];

        return getPenalization(dat.l_n_arity[n], (p1_ar + p2_ar - p1_ar * p2_ar));
    }

    @Override
    public String descr() {
        return "BIC";
    }

    @Override
    public double computePrediction(int n, int[] p1, int p2, Map<int[], Double> scores) {
        return scores.get(p1) + scores.get(new int[] { p2}) + inter(n, p1, p2);
    }

}
