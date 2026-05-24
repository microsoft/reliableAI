package ch.idsia.blip.core.common;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import java.io.IOException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.intersect;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


public class Query {

    private static final Logger log = Logger.getLogger(Query.class.getName());

    public static double ex(DataSet dat_rd, TIntIntHashMap q) {
        try {
            return new Query().go(dat_rd, q);
        } catch (IOException e) {
            logExp(log, e);
        }

        return 0;
    }

    private double go(DataSet dat, TIntIntHashMap q) throws IOException {
        int[] rows = null;

        for (int var : q.keys()) {
            int val = q.get(var);
            int[] aux = dat.row_values[var][val];

            if (rows == null) {
                rows = aux;
            } else {
                rows = intersect(rows, aux);
            }

            pf("%d %d %d \n", var, val, rows.length);
        }

        if (rows == null || rows.length == 0) {
            return 0;
        }

        return Math.log10(rows.length * 1.0) - Math.log10(dat.n_datapoints);
    }

}
