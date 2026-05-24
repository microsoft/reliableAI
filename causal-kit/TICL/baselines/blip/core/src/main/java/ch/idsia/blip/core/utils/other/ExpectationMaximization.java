package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;
import ch.idsia.blip.core.utils.data.ArrayUtils;


/**
 * Perform EM on given row values
 */
public class ExpectationMaximization extends Analyzer {

    private short[] new_values;

    private int[][] p_values;

    private int[] set_p;

    private int[] arities;

    private int ar;

    private int[] missing_row;

    private int[] missing_pv;

    public ExpectationMaximization(DataSet dat) {
        super(dat);
    }

    /*
     private int n_var;

     private int[] missing;

     private int[] vars;

     private int[][][] filledValues;

     private int[][][] originalValues;
     */

    public short[] performEM(DataSet dat, int missingVar, int[] set_p) {

        this.ar = dat.l_n_arity[missingVar];

        // new_values = new short[dat.sample[missingVar].length];
        // ArrayUtils.cloneArray(dat.sample[missingVar], new_values);

        this.missing_row = dat.row_values[missingVar][ar];
        this.p_values = computeParentSetValues(set_p);

        this.set_p = set_p;
        this.arities = dat.l_n_arity;

        // Parent configuration for each missing row of the variable
        this.missing_pv = new int[missing_row.length];
        for (int i = 0; i < missing_row.length; i++) {
            boolean found = false;

            for (int p_v = 0; p_v < p_values.length && !found; p_v++) {
                if (find(missing_row[i], p_values[p_v])) {
                    missing_pv[i] = p_v;
                    found = true;
                }
            }

        }

        return performEM();
    }

    /**
     * Perform EM on the values of a variable
     */
    private short[] performEM() {

        int it = 0;
        boolean change = true;

        // System.out.println(Arrays.toString(new_values));

        while (change) {
            // System.out.println("\n# Iter: " + it);

            // Expectation phase
            short[] exp = expectation();

            // Maximization phase
            change = maximization(exp);
            it++;

            // System.out.println(Arrays.toString(new_values));
        }

        return new_values;
    }

    private boolean maximization(short[] exp) {

        boolean change = false;

        // For each row
        for (int i = 0; i < missing_row.length; i++) {
            int p_v = missing_pv[i];

            if (new_values[i] != exp[p_v]) {
                change = true;
            }
            new_values[i] = exp[p_v];
        }

        return change;
    }

    private short[] expectation() {

        // Expectation: most probable value for given parent set configuration
        short[] exp = new short[p_values.length];

        for (int p_v = 0; p_v < p_values.length; p_v++) {

            // Check if it contains a missing value; in case, don't consider it
            if (containsMissing(p_v, set_p)) {
                continue;
            }

            int[] valcount = new int[ar];

            for (int v = 0; v < ar; v++) {
                valcount[v] = 0;
            }

            for (int r : p_values[p_v]) {
                if (new_values[r] < ar) {
                    valcount[new_values[r]] += 1;
                }
            }

            exp[p_v] = (short) ArrayUtils.max_index(valcount);

            // System.out.printf("p_v: %d, valcount: %s\n", p_v, Arrays.toString(valcount));
        }

        // System.out.printf("Expectations: %s\n", Arrays.toString(exp));

        return exp;
    }

    /*
     public int[][][] performEM(int[][][] in_originalValues, int[] in_vars) {

     Arrays.sort(in_vars);
     vars = in_vars;
     originalValues = in_originalValues;

     // Values filled for missing variables
     filledValues = new int[n_var][][];
     for (int thread : vars) {
     filledValues[thread] = new int[arity[thread]][];
     for (int m = 0; m < arity[thread]; m++) {
     filledValues[thread][m] = Arrays.copyOf(originalValues[thread][m],
     originalValues[thread][m].length);
     }
     }

     // Find all the missing values
     getMissingVar();

     System.out.println("Missing vars: " + Arrays.toString(missing));

     int it = 0;
     boolean change = true;

     while (change) {
     System.out.println("\n# Iter: " + it);
     // Expectation phase
     int[] cnts = expectation();

     System.out.printf("Counts: %s \n", Arrays.toString(cnts));
     // Maximization phase
     change = maximization(cnts);
     System.out.printf("Counts: %s \n", Arrays.toString(cnts));
     it++;
     }

     return filledValues;
     }

     private void getMissingVar() {

     TIntArrayList missingVar = new TIntArrayList();

     for (int thread : vars) {
     if (originalValues[thread][arity[thread]].length > 0) {
     missingVar.add(thread);
     }
     }
     missingVar.sort();
     missing = missingVar.toArray();
     }*/

    /**
     * Get counts of the different configurations from filledValues
     *
     * @return counts for every variable configuration
     */
    
    /*
     private int[] expectation() {

     // Get possible configurations
     int n_conf = 1;

     for (int v : vars) {
     n_conf *= arity[v];
     }

     int[] cnts = new int[n_conf];

     for (int v = 0; v < n_conf; v++) {
     cnts[v] = 0;
     }

     short[] sample = new short[n_var];

     // Get counts
     for (int j = 0; j < n_conf; j++) {
     cnts[j] = computeCounts(vars, filledValues, j);
     }

     return cnts;
     }

     private int computeCounts(int[] vars, int[][][] filledValues, int j) {
     int[] values = null;

     int n = j;

     for (int v : vars) {

     // Get value for the parent in this configuration
     short val = (short) (n % arity[v]);

     n /= arity[v];

     // System.out.printf("Var: %d, val: %d \n", v, val);

     // Update set containing sample rows for the chosen configuration
     int[] par_var = filledValues[v][val];

     if (values == null) {
     values = par_var;
     } else {
     values = RandomStuff.intersect(values, par_var);
     }
     }

     return values.length;
     }

     private boolean maximization(int[] cnts) {

     boolean change = false;

     // For every variables
     for (int thread : missing) {

     // Get line with missing values
     List<TIntArrayList> maximized = new ArrayList<TIntArrayList>();

     for (int m = 0; m < arity[thread]; m++) {
     maximized.add(new TIntArrayList(originalValues[thread][m]));
     }

     // For every line where it has a missing value
     for (int l : originalValues[thread][arity[thread]]) {
     System.out.printf("Choosing value for var: %d, line: %d - ", thread,
     l);
     // Predict the most probable value
     int v = mostProbable(thread, l, cnts);

     System.out.printf(" final: %d\n", v);
     // Add line to value
     maximized.get(v).add(l);
     }

     for (int m = 0; m < arity[thread]; m++) {
     maximized.get(m).sort();
     int[] newValues = maximized.get(m).toArray();

     // Check if there was some changes
     int s = RandomStuff.intersectN(newValues, filledValues[thread][m]);

     if (s != filledValues[thread][m].length) {
     change = true;
     }

     filledValues[thread][m] = newValues;
     }
     }

     return change;
     } */

    /*
     private int mostProbable(int thread, int l, int[] cnts) {

     int best_val = -1;
     int max_cnts = -1;

     // Compile the values for that line
     short[] sample = new short[n_var];

     for (int v : vars) {
     int val = -1;

     for (int vl = 0; vl < arity[v]; vl++) {
     if (Arrays.binarySearch(originalValues[v][vl], l) >= 0) {
     val = vl;
     }
     }
     sample[v] = (short) val;
     }

     // Choose the value with the most counts for that sample
     for (int m = 0; m < arity[thread]; m++) {
     sample[thread] = (short) m;

     // Get the index count
     int mx = 1;
     int ix = 0;

     for (int tw = 0; tw < vars.length; tw++) {
     ix += sample[vars[tw]] * mx;
     mx *= arity[vars[tw]];
     }

     // Check if this value has the best counts
     if (cnts[ix] > max_cnts) {
     max_cnts = cnts[ix];
     best_val = m;
     }

     System.out.printf("samp: %s, best: %d - ", Arrays.toString(sample),
     cnts[ix]);

     }

     return best_val;
     } */
    protected boolean containsMissing(int n, int[] set_p) {
        for (int p : set_p) {
            int ar = arities[p] + 1;
            short val = (short) (n % ar);

            n /= ar;

            if (val == arities[p]) {
                return true;
            }
        }
        return false;
    }
}
