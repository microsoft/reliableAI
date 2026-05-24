package ch.idsia.blip.core.learn.param;


import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


public class ParLeWeight extends ParLe {

    private final double alpha;

    private final double[] weight;

    public ParLeWeight(double in_alpha, double[] weight) {
        super();
        alpha = in_alpha;
        this.weight = weight;
    }

    double[] computeWeightedCardinalities(int var, int[] parents, int j) {

        int n = j;

        int[] parents_var = getParentsConf(parents, n);

        int ar = bn.arity(var);
        double[] n_ij = new double[ar];

        int[][] vl_var = this.dat.row_values[var];

        // System.out.println(var + " ... " + ar + " .... " + vl_var.length);
        // if (verbose > 1 && parents_var.length < 50 )
        // pf("WARNING! Variable %s, less than 50 datapoints in parent configuration! There are: %d \n", bn.name(var), parents_var.length);

        // For every variable configuration, compute the n's
        for (int v = 0; v < ar; v++) {
            int[] lg = ArrayUtils.intersect(parents_var, vl_var[v]);

            for (int g : lg) {
                n_ij[v] += weight[g];
            }
        }
        return n_ij;
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

            double[] n_ij = computeWeightedCardinalities(var, parents, j);

            // Compute sums
            double n_j = 0;

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

        double sum = 0;

        double[] n_i = new double[ar];

        for (int v = 0; v < ar; v++) {
            for (int g : vl_var[v]) {
                n_i[v] += weight[g];
                sum += weight[g];
            }
        }

        for (int v = 0; v < ar; v++) {

            double p = (n_i[v] + alpha_ij) / (sum + alpha_j);

            potent[v] = p;
        }
        return potent;
    }
}

