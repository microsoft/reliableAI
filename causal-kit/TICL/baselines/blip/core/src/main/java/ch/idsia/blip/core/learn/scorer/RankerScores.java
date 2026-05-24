package ch.idsia.blip.core.learn.scorer;


import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.logging.Logger;


public class RankerScores {

    /**
     * Logger
     */
    private static final Logger log = Logger.getLogger(
            RankerScores.class.getName());

    public double alpha = 0.05;

    public int debug = 0;

    public int execute(ParentSet[][] f_psets, ParentSet[][] s_psets) throws Exception {

        if (f_psets.length != s_psets.length) {
            log.info("Different number of variables!");
            return 0;
        }

        int n = f_psets.length;

        double f = 0;
        double s = 0;

        for (int v = 0; v < n; v++) {

            int len = Math.min(f_psets[v].length, s_psets[v].length);

            ParentSet[] f1 = new ParentSet[len];
            ParentSet[] s1 = new ParentSet[len];

            System.arraycopy(f_psets[v], 0, f1, 0, len);
            System.arraycopy(s_psets[v], 0, s1, 0, len);

            Pair<Double, Double> res = compare(f1, s1);

            if (res.getFirst() > res.getSecond()) {
                f += 1;
            } else if (res.getSecond() > res.getFirst()) {
                s += 1;
            } else {
                f += 0.5;
                s += 0.5;
            }

            if (debug > 1) {
                System.out.printf("%d - f: %.1f, s: %.1f\n", v, res.getFirst(),
                        res.getSecond());
            }
        }

        if (debug > 0) {// System.out.printf("%.1f - %.1f\n", f, s);
        }

        int f1 = (int) Math.floor(f);
        int s1 = (int) Math.floor(s);

        /*
         BinomialDistribution binomial = new BinomialDistribution(f1 + s1, 0.5);

         if (f1 > s1) {
         double p_value = 1 - binomial.cumulativeProbability(f1);

         if (debug > 0) {
         System.out.printf("f > s: %.3f\n", p_value);
         }
         if (p_value < alpha) {
         return 1;
         }
         } else {
         double p_value = binomial.cumulativeProbability(f1);

         if (debug > 0) {
         System.out.printf("f < s: %.3f\n", p_value);
         }
         if (p_value < alpha) {
         return -1;
         }
         }*/

        return 0;
    }

    /**
     * Wilcoxon rank-sum test of the two parent sets lists
     *
     * @param f_pset first parent sets lists
     * @param s_pset second  parent sets lists
     * @return whether the first lists wins the statistic test
     */
    public Pair<Double, Double> compare(ParentSet[] f_pset, ParentSet[] s_pset) {

        double f_skore = 0, s_skore = 0;
        int f_ix = 0, s_ix = 0, f_l = f_pset.length, s_l = s_pset.length;
        boolean f, s;

        // For each existing parent set
        while (f_ix < f_l || s_ix < s_l) {

            int r;

            if (f_ix == f_l) {
                r = -1;
            } else if (s_ix == s_l) {
                r = 1;
            } else {
                r = Double.compare(f_pset[f_ix].sk, s_pset[s_ix].sk);
            }

            switch (r) {
            case 1:
                f_ix++;
                f_skore += s_l - s_ix;
                break;

            case 0:
                f_ix++;
                s_ix++;
                f_skore += 0.5 + s_l - s_ix;
                s_skore += 0.5 + f_l - f_ix;
                break;

            case -1:
                s_ix++;
                s_skore += f_l - f_ix;
                break;
            }
        }

        // f_skore = upd(f_skore, f_pset.length);
        // s_skore = upd(s_skore, s_pset.length);

        return new Pair<Double, Double>(f_skore, s_skore);
    }

    private double upd(double s, int l) {
        return s - l * (l + 1) / 2.0;
    }
}
