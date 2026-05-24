package ch.idsia.blip.core.utils;


public class DataSet {

    // Number of variables
    public int n_var;

    // List of names of variables
    public String[] l_nm_var;

    // List of arities of variables
    public int[] l_n_arity;

    // Number of datapoints in sample
    public int n_datapoints;

    // For each variable, and for each value, an array of all the row where that variable appears with that value.
    public int[][][] row_values;

    // For each variable with missing data, array of row with missing value
    public int[][] missing_l;

    // Names of the states for each variable
    public String[][] l_nm_states;

    public DataSet(DataSet dat) {
        this.n_var = dat.n_var;
        this.l_nm_var = dat.l_nm_var;
        this.l_n_arity = dat.l_n_arity;
        this.n_datapoints = dat.n_datapoints;
        this.l_nm_states = dat.l_nm_states;
    }

    public DataSet() {}
}
