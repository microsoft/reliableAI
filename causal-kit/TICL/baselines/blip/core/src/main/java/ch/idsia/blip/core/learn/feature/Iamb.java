package ch.idsia.blip.core.learn.feature;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.io.dat.DatFileReader;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static ch.idsia.blip.core.utils.data.ArrayUtils.expandArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.find;
import static ch.idsia.blip.core.utils.data.ArrayUtils.reduceArray;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


/**
 * General IAMB analyzer
 */
public abstract class Iamb {

    final DataSet dat;

    public boolean verb = false;

    protected boolean forward;

    Iamb(DatFileReader dr) throws IOException {
        this.dat = dr.read();
        int[] ar = dat.l_n_arity;
    }

    /**
     * @param x variable to search
     * @param t treeshold
     */
    
    /**
     * @param x     variable to search
     * @param alpha signficance
     */
    public int[] go(int x, double alpha) throws Exception {

        prepare();

        // markov blanket
        int[] CMB = new int[0];

        // forward phase
        forward = true;
        boolean change = true;

        while (change) {
            int best_y = -1;
            double best_f = 0;

            for (int y = 0; y < dat.n_var; y++) {
                if (y == x || find(y, CMB)) {
                    continue;
                }

                double f = computeCMI(x, y, CMB);

                // pf("m(%s;%s|%s): %.5f \n", dat.l_nm_var[x], dat.l_nm_var[y], fgh(CMB.toArray()), f);
                /*
                 pf("%d - %d - %s -> mi: %.5f \n", x, y, CMB, f);
                 pf("%.5f, %.5f, %.5f \n", weight.computeHCond(y, CMB.toArray()),
                 weight.computeHCond(y,
                 expandArray(CMB.toArray(), x)),

                 weight.computeHCond(y, CMB.toArray())-
                 weight.computeHCond(y,
                 expandArray(CMB.toArray(), x)));*/

                if (f > best_f) {
                    best_f = f;
                    best_y = y;
                }
            }

            if (!(condInd(x, best_y, CMB, alpha))) {
                CMB = expandArray(CMB, best_y);

                if (verb) {
                    pf("Adding %s (t: %.5f, mi: %.5f) - %s \n",
                            dat.l_nm_var[best_y], alpha, best_f,
                            Arrays.toString(CMB));
                }

            } else {
                change = false;
            }
        }

        forward = false;
        // backward phase
        change = true;
        while (change && CMB.length > 0) {
            int best_y = -1;
            double best_f = Double.MAX_VALUE;

            for (int y : CMB) {
                int[] n_CMB = reduceArray(CMB, y);
                double f = computeCMI(x, y, n_CMB);

                if (f < best_f) {
                    best_f = f;
                    best_y = y;
                }
            }

            int[] n_CMB = reduceArray(CMB, best_y);

            if (condInd(x, best_y, n_CMB, alpha)) {
                CMB = n_CMB;

                if (verb) {
                    pf("Removing %d (mi: %.5f) - %s \n", best_y, best_f, CMB);
                }

            } else {
                change = false;
            }
        }

        // Arrays.sort(CMB);
        return CMB;
    }

    protected abstract void prepare();

    protected abstract double computeCMI(int x, int y, int[] z);

    protected abstract boolean condInd(int x, int y, int[] z, double alpha);

    public List<String> fgh(int[] s) {
        return fgh(dat, s);
    }

    List<String> fgh(DataSet dat, int[] s) {
        List<String> g = new ArrayList<String>();

        for (int e : s) {
            g.add(dat.l_nm_var[e]);
        }
        return g;
    }

    public abstract String getName();

    public abstract double[] getTresholds();

}
