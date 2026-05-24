package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;

import java.util.Arrays;
import java.util.Map;


/**
 * Computes the BDeu.
 * <p/>
 * Remenber: last index in values is for missing data (let's simply ignore them!)
 */
public class MIT extends Score {

    private double alpha = 0.999;

    public MIT(double alpha, DataSet dat) {
        super(dat);
        if (alpha != 0) {
            this.alpha = alpha;
        }
    }

    @Override
    public double computeScore(int n) {

        return -dat.n_datapoints;
    }

    @Override
    public double computeScore(int n, int[] set_p, int[][] p_values) {

        /*
         numEvaluated++;

         Arrays.sort(set_p);

         double skore = 0;

         int arity = dat.l_n_arity[n];

         int[] totcount = new int[arity + 1];

         for (int v = 0; v < arity; v++) {
         totcount[v] = 0;
         }

         for (int r : values) {
         totcount[r] += 1;
         }

         for (int p_v = 0; p_v < p_values.length; p_v++) {

         // Check if it contains a missing value; in case, don't consider it
         if (containsMissing(p_v, set_p)) {
         continue;
         }

         int[] valcount = new int[arity + 1];

         for (int v = 0; v < arity; v++) {
         valcount[v] = 0;
         }

         for (int r : p_values[p_v]) {
         valcount[values[r]] += 1;
         }

         for (int v = 0; v < arity; v++) {

         if (valcount[v] == 0) {
         continue;
         }

         double p_x_y = valcount[v] * 1.0 / dat.n_datapoints;

         double p_y = p_values[p_v].length * 1.0 / dat.n_datapoints;

         double p_x = totcount[v] * 1.0 / dat.n_datapoints;

         // System.out.printf("%.4f, %d - %d, %.3f \n", skore, valcount[v], p_values[p_v].length,  log((valcount[v] * 1.0) / p_values[p_v].length));

         skore += p_x_y
         * (log(p_x_y) - log(p_x) - log(p_y));
         // System.out.println(skore);

         // System.out.printf("%d- %.2f, ", valcount[v], p);

         // System.out.println(valcount[v] + "   " + log(p) + "   " + p + "   " + skore);
         }

         }

         double pen = penalizationTerm(n, set_p);
         double sk = 2 * dat.n_datapoints * skore - pen;

         if (debug) {
         System.out.printf("%d, %10s -> %10.2f, %10.2f   --- %10.2f \n", n,
         Arrays.toString(set_p), 2 * dat.n_datapoints * skore, pen,
         sk);
         }

         return sk;
         */

        return -Double.MAX_VALUE;
    }

    private double penalizationTerm(int n, int[] set_p) {

        int arity = dat.l_n_arity[n];

        int[] p_arity = new int[set_p.length];

        for (int i = 0; i < set_p.length; i++) {
            p_arity[i] = dat.l_n_arity[set_p[i]];
        }

        // Penalization term
        // skore -= log(n_datapoints) * (arity - 1) * p_arity / 2.0;
        Arrays.sort(p_arity);
        // System.out.println(Arrays.toString(p_arity));

        double pen = 0;

        for (int j = 0; j < set_p.length; j++) {
            int df = (arity - 1) * (p_arity[j] - 1);

            for (int j1 = j + 1; j1 < set_p.length; j1++) {
                df *= p_arity[j1];
            }

            double p = AChiSq();

            // System.out.println(df + " " + p);
            pen += p;
        }
        return pen;
    }

    @Override
    public String descr() {
        return "MIT";
    }

    private double AChiSq() {
        double v = 0.5;
        double dv = 0.5;
        double x = 0;

        /*
         ChiSquaredDistribution d = new ChiSquaredDistribution(df);

         while (dv > 1e-10) {
         x = 1 / v - 1;
         dv = dv / 2;
         // System.out.print (x + "-"+ d.cumulativeProbability(x) + " ... ");
         if (d.cumulativeProbability(x) < p) {
         v = v - dv;
         } else {
         v = v + dv;
         }
         }
         return x;
         */

        return 0;
    }

    @Override
    public double computePrediction(int n, int[] p1, int p2, Map<int[], Double> scores) {
        return scores.get(p1) + scores.get(new int[] { p2});
    }

}
