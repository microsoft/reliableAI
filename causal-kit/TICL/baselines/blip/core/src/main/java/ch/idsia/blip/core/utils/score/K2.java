package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.other.Gamma;
import ch.idsia.blip.core.utils.RandomStuff;

import java.util.Map;


public class K2 extends Score {
    protected double alpha = 10.0D;

    public K2(DataSet dat) {
        super(dat);
    }

    public double computeScore(int n) {
        this.numEvaluated += 1;

        int arity = this.dat.l_n_arity[n];

        double skore = Gamma.lgamma(arity)
                - Gamma.lgamma(arity + this.dat.n_datapoints);

        for (int v = 0; v < arity; v++) {
            int weight = this.dat.row_values[n][v].length;

            skore += Gamma.lgamma(1 + weight);
        }
        return skore;
    }

    public double computeScore(int n, int[] set_p, int[][] p_values) {

        this.numEvaluated += 1;

        int arity = this.dat.l_n_arity[n];

        double skore = 0.0D;

        for (int[] p_value : p_values) {

            skore += Gamma.lgamma(arity) - Gamma.lgamma(arity + p_value.length);
            for (int v = 0; v < arity; v++) {
                int valcount = ArrayUtils.intersectN(this.dat.row_values[n][v],
                        p_value);

                skore += Gamma.lgamma(valcount + 1);
            }
        }
        return skore;
    }

    @Override
    public double computePrediction(int n, int[] p1, int p2, Map<int[], Double> scores) {
        return scores.get(p1) + scores.get(new int[] { p2});
    }

    public String descr() {
        return RandomStuff.f("K2");
    }
}
