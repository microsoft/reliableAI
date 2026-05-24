package ch.idsia.blip.core.io.bn;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.IOException;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.wf;
import static ch.idsia.blip.core.utils.other.StringUtils.join;


public class BnUaiWriter extends BnWriter {

    @Override
    public void go(Writer wr, BayesianNetwork bn) throws IOException {

        wf(wr, "BAYES\n");
        wf(wr, "%d\n", bn.n_var);

        // cardinalities
        for (int i = 0; i < bn.n_var; i++) {
            if (i != 0) {
                wf(wr, " ");
            }
            wf(wr, "%d", bn.l_ar_var[i]);
        }
        wf(wr, "\n%d\n", bn.n_var);

        // write parents
        for (int i = 0; i < bn.n_var; i++) {
            TIntArrayList g = new TIntArrayList();

            g.addAll(bn.parents(i));
            g.sort();
            g.add(i);
            wf(wr, "%d\t%s\n", g.size(), join(g, "\t"));
        }

        // write cardinalities
        for (int i = 0; i < bn.n_var; i++) {
            writePotential(wr, bn, i);
        }
    }

    public static void ex(String s, BayesianNetwork bn) {
        new BnUaiWriter().go(s, bn);
    }

    public static void ex(BayesianNetwork bn, String s) {
        ex(s, bn);
    }
}
