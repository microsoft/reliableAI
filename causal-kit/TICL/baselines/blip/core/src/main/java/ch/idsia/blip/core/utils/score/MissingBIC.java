package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;


/**
 * Computes the BIC
 */
public class MissingBIC extends BIC {

    private final int n_var;

    private int[][][] completion;

    private double[] weights;

    public MissingBIC(DataSet dat, TIntArrayList[][] completion, TDoubleArrayList weights) {
        super(dat);

        this.n_var = completion.length;

        this.completion = new int[n_var][][];
        for (int n = 0; n < n_var; n++) {
            this.completion[n] = new int[dat.l_n_arity[n]][];
            for (int v = 0; v < dat.l_n_arity[n]; v++) {
                this.completion[n][v] = completion[n][v].toArray();
            }
        }

        this.weights = weights.toArray();
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

            double weight = values[v].length;

            for (int r : completion[n][v]) {
                weight += weights[r];
            }

            double p = (weight + alpha_i) / (dat.n_datapoints + alpha);

            skore += weight * log(p);
            // System.out.printf("%d - %.2f, ", values[v].length, p);
            // System.out.println((values[v].length * 1.0) + (alpha / arity));
            // System.out.println((n_datapoints + alpha));

            // System.out.println(values[v].length + " & " + log(p) + " & " + p + " % " + skore);
        }

        double pen = getPenalization(arity);

        skore -= pen;

        return skore;
    }

    @Override
    public double computeScore(int n, int[] set_p) {

        int[][] p_values = computeParentSetValues(set_p);

        int[][] comp_values = computeParentSetValues(set_p, completion);

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

        for (int p_v = 0; p_v < comp_values.length; p_v++) {

            // Check if it contains a missing value; in case, don't consider it
            // if (containsMissing(z_i, z)) {
            // continue;
            // }

            // App weight for the parent configuration
            double weight_p = p_values[p_v].length;

            for (int r : comp_values[p_v]) {
                weight_p += weights[r];
            }
            if (weight_p == 0) {
                continue;
            }

            for (int v = 0; v < arity; v++) {

                // Weight of missing rows
                double weight_n = ArrayUtils.intersectN(dat.row_values[n][v],
                        p_values[p_v]);

                for (int r : ArrayUtils.intersect(completion[n][v],
                        comp_values[p_v])) {
                    weight_n += weights[r];
                }
                if (weight_n == 0) {
                    continue;
                }

                // System.out.printf("%.4f, %d - %d, %.3f \n", skore, valcount[v], p_values[p_v].length,  log((valcount[v] * 1.0) / p_values[p_v].length));

                skore += weight_n
                        * (log(weight_n + alpha_ij) - log(weight_p + alpha_i));

                // System.out.printf("%d- %.2f, ", valcount[v], p);

                // System.out.println(valcount[v] + "   " + log(p) + "   " + p + "   " + skore);
            }

        }
        // Penalization term
        skore -= getPenalization(arity, p_arity);

        return skore;
    }

    @Override
    public String descr() {
        return "BIC with missing values";
    }

}
