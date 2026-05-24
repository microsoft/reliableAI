package ch.idsia.blip.core.learn.param;


import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


/**
 * Implementation for Bayes
 */
public class ParLeBayes extends ParLe {

    protected double alpha;

    public ParLeBayes(double in_alpha) {
        super();
        alpha = in_alpha;
    }

    // Compute potentials for given variable
    public double[] computePotentials(int var) {

        // System.out.println("Var: " + var);

        int[] parents = bn.parents(var);
        int ar = bn.arity(var);

        // System.out.println("Parents: " + Arrays.toString(bn.parents(var)));

        int n_par_conf = 1;

        for (int parent : parents) {
            n_par_conf *= bn.arity(parent);
        }
        int n_potent = n_par_conf * ar;

        double[] potent = new double[n_potent];

        double alpha_j = alpha / n_par_conf;
        double alpha_ij = alpha / n_potent;

        TIntIntHashMap conf = new TIntIntHashMap();

        conf.put(var, 0);
        for (int p : parents) {
            conf.put(p, 0);
        }

        for (int j = 0; j < n_par_conf; j++) {

            int[] n_ij = computeCardinalities(var, parents, j);

            // Compute sums
            int n_j = 0;

            for (int v = 0; v < ar; v++) {
                n_j += n_ij[v];
            }

            int ix = j * ar;

            for (int v = 0; v < ar; v++) {
                potent[ix++] = (n_ij[v] + alpha_ij) / (n_j + alpha_j);
            }
        }

        return potent;
    }

    public double[] computePotentialsSimple(int var) {
        int ar = bn.arity(var);
        double[] potent = new double[ar];

        int[][] vl_var = this.dat.row_values[var];

        double alpha_j = alpha;
        double alpha_ij = alpha / ar;

        for (int v = 0; v < ar; v++) {
            double p = ((vl_var[v].length * 1.0) + alpha_ij)
                    / (n_datapoints + alpha_j);

            potent[v] = p;
        }
        return potent;
    }
}
