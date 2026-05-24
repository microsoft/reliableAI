package ch.idsia.blip.core.common;


import ch.idsia.blip.core.utils.DataSet;

import java.io.IOException;
import java.io.Writer;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.logExp;
import static ch.idsia.blip.core.utils.RandomStuff.wf;


public class Marginals {

    private static final Logger log = Logger.getLogger(Marginals.class.getName());

    public static void ex(DataSet dat_rd, Writer wr) {
        try {
            new Marginals().go(dat_rd, wr);
        } catch (IOException e) {
            logExp(log, e);
        }
    }

    private void go(DataSet dat, Writer wr) throws IOException {

        int n = dat.n_var;
        double d = dat.n_datapoints * 1.0;

        wf(wr, "%d\n", n);
        for (int i = 0; i < n; i++) {
            int a = dat.l_n_arity[i];

            wf(wr, "%d %d", i, a);
            for (int v = 0; v < a; v++) {
                wf(wr, " %5.4f", dat.row_values[i][v].length / d);
            }
            wf(wr, "\n");
        }
    }
}
