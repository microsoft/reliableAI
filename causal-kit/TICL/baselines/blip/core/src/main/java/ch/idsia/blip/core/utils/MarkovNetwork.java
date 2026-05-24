package ch.idsia.blip.core.utils;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.util.Random;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;


/**
 * A Markov network, and the main operations to do on them.
 */
public class MarkovNetwork {

    private static final Logger log = Logger.getLogger(
            MarkovNetwork.class.getName());

    /**
     * Number of variable in the network.
     */
    public int n_var;

    /**
     * Name of each variable.
     */
    public String[] l_nm_var;

    /**
     * Arity of each variable.
     */
    public int[] l_ar_var;

    /**
     * List of values of each variable.
     */
    public String[][] l_values_var;

    /**
     * List of variables in each clique
     */
    public int[][] l_vars_cliques;

    /**
     * Potentials of each clique
     */
    public double[][] l_pot_cliques;

    public int n_cliques;

    private int[][] l_vars_assign;

    /**
     * Construct void network
     *
     * @param n size of network
     */
    public MarkovNetwork(int n) {

        l_nm_var = new String[n];
        l_ar_var = new int[n]; // List of variables arity
        l_values_var = new String[n][]; // List of variables row_values
        l_vars_cliques = new int[0][]; // List of parent set for variable
        l_pot_cliques = new double[0][]; // List of probabilities

        for (int i = 0; i < n; i++) {
            l_nm_var[i] = "N" + String.valueOf(i);
            l_ar_var[i] = 0;
            l_values_var[i] = new String[0];
        }
        n_var = n;
    }

    public void sample(int n_sample) {

        short[] sample = new short[n_var];
        Random r = getRandom();

        updateCliqueAssignments();

        // Initial random assignment of values
        for (int i = 0; i < n_var; i++) {
            sample[i] = (short) r.nextInt(arity(i));
        }

        for (int i = 0; i < n_sample; i++) {
            for (int v = 0; v < n_var; v++) {
                resample(sample, v, r);
            }
        }
    }

    public void resample(short[] sample, int v, Random rand) {
        // Initial potential
        double[] pt = new double[arity(v)];

        for (int j = 0; j < arity(v); j++) {
            pt[j] = 1.0;
        }
        // Multiply with all the cliques where it appears
        for (int clique : l_vars_assign[v]) {
            multiply(pt, v, clique, sample);
        }
        double tot = 0;

        for (double p : pt) {
            tot += p;
        }
        double r = rand.nextDouble() * tot;
        int i = 0;

        while ((i < arity(v)) && (pt[i] < r)) { // Select a value from the probabilities given the random
            r -= pt[i];
            i++;
        }
        sample[v] = (short) i;
    }

    /**
     * @param pt     potential to be multiplied
     * @param v      variable
     * @param clique clique
     * @param sample
     */
    private void multiply(double[] pt, int v, int clique, short[] sample) {
        int[] vars = l_vars_cliques[clique];

        // For each value of the variable
        for (int a = 0; a < pt.length; a++) {
            int ix = 0;
            int ix_ml = 1;

            for (int i = vars.length - 1; i >= 0; i--) {
                int p = vars[i];
                int val;

                if (p == v) {
                    val = a;
                } else {
                    val = sample[p];
                }

                ix += val * ix_ml; // Shift index
                ix_ml *= arity(p); // Compute cumulative shifter

            }
            double p = l_pot_cliques[clique][ix];

            pt[a] *= p;
        }
    }

    public void updateCliqueAssignments() {
        if (l_vars_assign != null) {
            return;
        }

        TIntArrayList[] aux = new TIntArrayList[n_var];

        for (int i = 0; i < n_var; i++) {
            aux[i] = new TIntArrayList();
        }

        for (int i = 0; i < n_cliques; i++) {
            for (int v : l_vars_cliques[i]) {
                aux[v].add(i);
            }
        }

        l_vars_assign = new int[n_var][];
        for (int i = 0; i < n_var; i++) {
            l_vars_assign[i] = aux[i].toArray();
        }
    }

    private int arity(int i) {
        return l_ar_var[i];
    }

}
