package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;

import static ch.idsia.blip.core.utils.RandomStuff.wf;
import static ch.idsia.blip.core.utils.other.StringUtils.join;


public class BnErgWriter extends BnWriter {

    @Override
    public void go(Writer wr, BayesianNetwork bn) throws IOException {

        wf(wr, "%d\n", bn.n_var);

        // cardinalities
        for (int i = 0; i < bn.n_var; i++) {
            if (i != 0) {
                wf(wr, " ");
            }
            wf(wr, "%d", bn.l_ar_var[i]);
        }
        wf(wr, "\n", bn.n_var);

        // write parents
        for (int i = 0; i < bn.n_var; i++) {
            int[] g = bn.parents(i);

            Arrays.sort(g);
            wf(wr, "%d\t%s\n", g.length, join(g, "\t"));
        }

        // write cardinalities
        wf(wr, "\n/* Probabilities */\n");
        for (int i = 0; i < bn.n_var; i++) {
            writePotential(wr, bn, i);
        }
    }

    /*
     protected void writePotential(Writer wr, BayesianNetwork bn, int i) throws IOException {
     wf(wr, "\n");
     double[] pt = bn.potentials(i);
     int t = 0;
     wf(wr, "%d\n", pt.length);
     int ar = bn.l_ar_var[i];
     int n_par = pt.length / ar;
     for (int j = 0; j < n_par; j++) {
     wf(wr, " ");
     for (int z = 0; z < ar;z++) {
     writeP(wr, pt[t++]);
     }
     wf(wr, "\n");
     }
     }
     */

    public static void ex(String s, BayesianNetwork bn) {
        new BnErgWriter().go(s, bn);
    }

}
