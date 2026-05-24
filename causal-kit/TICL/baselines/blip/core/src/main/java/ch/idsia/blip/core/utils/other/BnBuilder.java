package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;


/**
 * A Bayesian network builder
 */
public class BnBuilder implements Serializable {

    public int n_var;

    public List<String> l_nm_var;

    public TIntArrayList l_ar_var;

    public ArrayList<String[]> l_values_var;

    public List<int[]> l_parent_var;

    public List<double[]> l_potential_var;

    /**
     * Default constructor
     */
    public BnBuilder() {
        l_nm_var = new ArrayList<String>(); // List of variables names
        l_ar_var = new TIntArrayList(); // List of variables arity
        l_values_var = new ArrayList<String[]>(); // List of variables row_values

        l_parent_var = new ArrayList<int[]>(); // List of parent set for variable

        l_potential_var = new ArrayList<double[]>(); // List of probabilities
    }

    /**
     * Construct void network
     *
     * @param n size of network
     */
    public BnBuilder(int n) {
        this();

        for (int i = 0; i < n; i++) {
            l_nm_var.add("N" + String.valueOf(i));
            l_ar_var.add(0);
            l_values_var.add(new String[0]);
            l_parent_var.add(new int[0]);
            l_potential_var.add(new double[0]);
        }
        n_var = n;
    }

    public BayesianNetwork toBn() {
        BayesianNetwork bn = new BayesianNetwork(n_var);

        for (int i = 0; i < n_var; i++) {
            bn.l_nm_var[i] = l_nm_var.get(i);
            bn.l_ar_var[i] = l_ar_var.get(i);
            bn.l_values_var[i] = l_values_var.get(i);
            bn.l_parent_var[i] = l_parent_var.get(i);
            bn.l_potential_var[i] = l_potential_var.get(i);
        }

        return bn;
    }
}

