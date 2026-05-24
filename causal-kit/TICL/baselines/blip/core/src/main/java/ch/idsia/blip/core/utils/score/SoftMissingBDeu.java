package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.Gamma;


public class SoftMissingBDeu extends BDeu {
    private final int n_var;
    private TIntArrayList[][] completion;
    private TDoubleArrayList weights;

    public SoftMissingBDeu(DataSet dat, TIntArrayList[][] completion, TDoubleArrayList weights, double alpha) {
        super(alpha, dat);

        this.n_var = completion.length;

        this.completion = completion;

        this.weights = weights;
    }

    public double computeScore(int n) {
        this.numEvaluated += 1;

        int arity = this.dat.l_n_arity[n];

        double a_ij = this.alpha;
        double a_ijk = this.alpha / arity;

        int[][] values = this.dat.row_values[n];

        double tot = 0.0D;
        double skore = 0.0D;

        for (int v = 0; v < arity; v++) {
            double weight = values[v].length;

            for (int j = 0; j < this.completion[n][v].size(); j++) {
                int r = this.completion[n][v].get(j);

                weight += this.weights.get(r);
            }
            skore = skore + (Gamma.lgamma(a_ijk + weight) - Gamma.lgamma(a_ijk));

            tot += weight;
        }
        skore = skore + Gamma.lgamma(a_ij) - Gamma.lgamma(a_ij + tot);

        return skore;
    }

    public double computeScore(int n, int[] set_p) {

        int[][] p_values = computeParentSetValues(set_p);

        TIntArrayList[] comp_values = computeParentSetValues(set_p,
                this.completion);

        this.numEvaluated += 1;

        int arity = this.dat.l_n_arity[n];

        int p_arity = 1;

        for (int p : set_p) {
            p_arity *= this.dat.l_n_arity[p];
        }
        double a_ij = this.alpha / p_arity;
        double a_ijk = a_ij / arity;

        double skore = 0.0D;

        skore = skore
                + (Gamma.lgamma(a_ij) * p_arity
                        - Gamma.lgamma(a_ijk) * p_arity * arity);
        for (int p_v = 0; p_v < p_values.length; p_v++) {
            double weight_p = p_values[p_v].length;

            for (int j = 0; j < comp_values[p_v].size(); j++) {
                int r = comp_values[p_v].get(j);

                weight_p += this.weights.get(r);
            }
            skore -= Gamma.lgamma(a_ij + weight_p);
            for (int v = 0; v < arity; v++) {
                double weight_n = ArrayUtils.intersectN(
                        this.dat.row_values[n][v], p_values[p_v]);
                TIntArrayList inter = ArrayUtils.intersect(this.completion[n][v],
                        comp_values[p_v]);

                for (int j = 0; j < inter.size(); j++) {
                    int r = inter.get(j);

                    weight_n += this.weights.get(r);
                }
                skore += Gamma.lgamma(a_ijk + weight_n);
            }
        }
        return skore;
    }

    public String descr() {
        return "BDeu with missing values";
    }

    public TIntArrayList[] computeParentSetValues(int[] set_p, TIntArrayList[][] rows) {
        int p_arity = 1;

        for (int p : set_p) {
            p_arity *= this.dat.l_n_arity[p];
        }
        TIntArrayList[] p_values = new TIntArrayList[p_arity];

        for (int i = 0; i < p_arity; i++) {
            TIntArrayList values = null;

            int n = i;

            for (int p : set_p) {
                int ar = this.dat.l_n_arity[p];
                short val = (short) (n % ar);

                n /= ar;

                TIntArrayList par_var = rows[p][val];

                if (values == null) {
                    values = par_var;
                } else {
                    values = ArrayUtils.intersect(values, par_var);
                }
            }
            p_values[i] = values;
        }
        return p_values;
    }
}
