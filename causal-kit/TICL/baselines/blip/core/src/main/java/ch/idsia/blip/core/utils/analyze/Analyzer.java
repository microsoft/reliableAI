package ch.idsia.blip.core.utils.analyze;


import ch.idsia.blip.core.Base;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.cache.LRUCache;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.math.FastMath;

import java.util.logging.Logger;


public class Analyzer extends Base {

    private static final Logger log = Logger.getLogger(Analyzer.class.getName());

    // Data file
    public DataSet dat;

    // Alpha for counts
    public double alpha = 1.0;

    public boolean base2 = false;

    protected LRUCache<SIntSet, int[][]> cache;

    public int max_cache = 0;

    public Analyzer(DataSet dat) {
        this.dat = dat;

        if (max_cache > 0) {
            cache = new LRUCache<SIntSet, int[][]>(max_cache);
        }
    }

    protected double getFreq(int v, int i) {

        // return ((v + (alpha / thread)) / (dat.n_datapoints + alpha));
        return (v * 1.0 / dat.n_datapoints);
        // double d = ((v + (alpha / i)) / (dat.n_datapoints + alpha));
        // double d1 = d+ d * (new Random().nextDouble() - 0.5) / 10.0;
        // return d;
    }

    public int[][] computeParentSetValues(int p) {
        return computeParentSetValues(new int[] { p});
    }

    public int[][] computeParentSetValues(int[] set_p) {

        // if (cache == null)
        return computeParentSetValues(set_p, dat.row_values);

        /*
         int[][] values = cache.get(set_p);
         if (values == null) {
         values = computeParentSetValues(set_p, dat.row_values);
         cache.put(set_p, values);
         }

         return values;*/
    }

    public int[][] expandParentSetValues(int[] s, int[][] p_values, int new_p) {
        return expandParentSetValues(s, p_values, new_p, dat.row_values);
    }

    /**
     * Compute the set of datapoints indexes for each parent configuration.
     * <p/>
     * (Please remember that for each variable, the position equal to (arity of that variable + 1)
     * is for missing data
     *
     * @param set_p parents to consider
     * @return for each configurations of the parents, array of datapoints indexes where that configuration appears
     */
    public int[][] computeParentSetValues(int[] set_p, int[][][] rows) {

        if (set_p.length == 1) {
            return dat.row_values[set_p[0]];
        }

        int p_arity = 1;

        for (int p : set_p) {
            p_arity *= dat.l_n_arity[p];
        }

        int[][] p_values = new int[p_arity][];

        for (int i = 0; i < p_arity; i++) {
            int[] values = null;

            int n = i;

            // Get value for the parent in this configuration
            for (int p : set_p) {
                int ar = dat.l_n_arity[p];
                short val = (short) (n % ar);

                n /= ar;

                // System.out.print("p: " + p + " v: " + val + ", ");

                // Update set containing sample rows for the chosen configuration
                // System.out.printf("%d (%d) - %d - %d\n", p,  dat.l_n_arity[p], val, dat.row_values[p].length);
                int[] par_var = rows[p][val];

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

    public int[][] expandParentSetValues(int[] set_p, int[][] old_p_values, int new_p, int[][][] rows) {

        int p_arity = 1;

        for (int p : set_p) {
            p_arity *= dat.l_n_arity[p];
        }

        int[][] p_values = new int[p_arity][];

        int p;

        for (int i = 0; i < p_arity; i++) {

            int n = i;

            int new_ix = 0;

            int old_ix = 0;
            int ix_ml = 1;

            // Get value for the parent in this configuration
            for (int j = 0; j < set_p.length; j++) {

                p = set_p[j];
                int ar = dat.l_n_arity[p];
                short val = (short) (n % ar);

                n /= ar;

                if (p == new_p) {
                    new_ix = val;
                } else {
                    old_ix += val * ix_ml; // Shift index
                    ix_ml *= ar; // Compute cumulative shifter
                }
            }

            p_values[i] = ArrayUtils.intersect(old_p_values[old_ix],
                    dat.row_values[new_p][new_ix]);
        }

        return p_values;
    }

    /*
     protected boolean containsMissing(int n, int[] set_p) {
     for (int p : set_p) {
     int ar = dat.l_n_arity[p] + 1;
     short val = (short) (n % ar);

     n /= ar;

     if (val == dat.l_n_arity[p]) {
     return true;
     }
     }
     return false;
     }*/

    protected double log(double p) {
        if (base2) {
            return FastMath.log(p) / FastMath.log(2);
        }
        return FastMath.log(p);
    }
}
