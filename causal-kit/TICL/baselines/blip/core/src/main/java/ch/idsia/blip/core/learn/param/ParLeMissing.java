package ch.idsia.blip.core.learn.param;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


public class ParLeMissing extends ParLe {
    private int n_var;
    private double[] weights;
    private int[][][] completion;
    protected double alpha;

    public ParLeMissing(double in_alpha) {
        this.alpha = in_alpha;
    }

    public double[] computePotentials(int n) {
        int[] parents = this.bn.parents(n);
        int ar = this.bn.arity(n);

        int n_par_conf = 1;

        for (int parent : parents) {
            n_par_conf *= this.bn.arity(parent);
        }
        int n_potent = n_par_conf * ar;

        double[] potent = new double[n_potent];

        double alpha_j = this.alpha / n_par_conf;
        double alpha_ij = this.alpha / n_potent;

        TIntIntHashMap conf = new TIntIntHashMap();

        conf.put(n, 0);
        for (int p : parents) {
            conf.put(p, 0);
        }
        for (int j = 0; j < n_par_conf; j++) {
            int[] parents_var = getParentsConf(parents, j, this.completion);

            int n_j = 0;

            double weight_p = 0.0D;

            for (int r : parents_var) {
                weight_p += this.weights[r];
            }
            if (weight_p != 0.0D) {
                int ix = j * ar;

                for (int v = 0; v < ar; v++) {
                    double weight_n = 0.0D;

                    for (int r : ArrayUtils.intersect(this.completion[n][v],
                            parents_var)) {
                        weight_n += this.weights[r];
                    }
                    if (weight_n != 0.0D) {
                        potent[(ix++)] = ((weight_n + alpha_ij)
                                / (weight_p + alpha_j));
                    }
                }
            }
        }
        return potent;
    }

    public double[] computePotentialsSimple(int var) {
        int ar = this.bn.arity(var);
        double[] potent = new double[ar];

        int[][] vl_var = this.dat.row_values[var];

        double alpha_j = this.alpha;
        double alpha_ij = this.alpha / ar;

        for (int v = 0; v < ar; v++) {
            double weight = 0.0D;

            for (int r : this.completion[var][v]) {
                weight += this.weights[r];
            }
            double p = (weight * 1.0D + alpha_ij)
                    / (this.n_datapoints + alpha_j);

            potent[v] = p;
        }
        return potent;
    }

    public static BayesianNetwork ex(BayesianNetwork bn, DataSet dat, TIntArrayList[][] completion, TDoubleArrayList weights) {
        ParLeMissing pm = new ParLeMissing(10.0D);

        return pm.go(bn, dat, completion, weights);
    }

    private BayesianNetwork go(BayesianNetwork bn, DataSet dat, TIntArrayList[][] completion, TDoubleArrayList weights) {
        this.n_var = completion.length;

        this.completion = new int[this.n_var][][];
        for (int n = 0; n < this.n_var; n++) {
            this.completion[n] = new int[dat.l_n_arity[n]][];
            for (int v = 0; v < dat.l_n_arity[n]; v++) {
                this.completion[n][v] = completion[n][v].toArray();
            }
        }
        this.weights = weights.toArray();

        return super.go(bn, dat);
    }
}
