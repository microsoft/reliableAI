package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.data.ArrayUtils.index;
import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * Reads the content of a datapoints file
 */
public class DataFileReader extends DatFileReader {

    private static final Logger log = Logger.getLogger(
            DatFileLineReader.class.getName());

    @Override
    protected String[] getSplit(String ln) {
        return ln.split(",");
    }

    @Override
    protected void readMetaData() throws IOException {

        nextLine = rd_dat.readLine();
        dSet.n_var = getSplit(nextLine).length;
        dSet.l_nm_var = new String[dSet.n_var];
        for (int i = 0; i < dSet.n_var; i++) {
            dSet.l_nm_var[i] = f("N%d", i);
        }

        dSet.n_datapoints = 0;
        dSet.l_n_arity = new int[dSet.n_var];
    }

    @Override
    public void readValuesCache() throws IOException {

        // Collect row values
        List<List<TIntArrayList>> v_aux = new ArrayList<List<TIntArrayList>>(
                dSet.n_var);

        for (int n = 0; n < dSet.n_var; n++) {
            v_aux.add(new ArrayList<TIntArrayList>());
        }

        List<TIntArrayList> missing_aux_v = new ArrayList<TIntArrayList>();
        TIntArrayList missing_aux_l = new TIntArrayList();

        String[] sp;
        short v;
        List<TIntArrayList> lu;

        dSet.n_datapoints = 0;
        while (nextLine != null) {
            String line = nextLine.trim();

            if ("".equals(line)) {
                continue;
            }

            if (readMissing && line.contains("?")) {
                continue;
            }

            sp = getSplit(line);
            if (sp.length != dSet.n_var) {
                notifyError(dSet.n_datapoints + 1, sp.length);
                return;
            }

            // For each variable
            for (int i = 0; i < dSet.n_var; i++) {
                if ("?".equals(sp[i])) {
                    int pos = index(i, missing_aux_l);

                    if (pos < 0) {
                        pos = missing_aux_l.size();
                        missing_aux_v.add(new TIntArrayList());
                        missing_aux_l.add(i);
                    }

                    missing_aux_v.get(pos).add(dSet.n_datapoints);
                } else {
                    lu = v_aux.get(i);
                    v = Short.valueOf(sp[i]);
                    for (int j = dSet.l_n_arity[i]; j <= v; j++) {
                        dSet.l_n_arity[i]++;
                        lu.add(new TIntArrayList());
                    }
                    lu.get(v).add(dSet.n_datapoints);
                }
            }

            dSet.n_datapoints++;

            nextLine = rd_dat.readLine();
        }

        dSet.row_values = compact(v_aux);

        dSet.missing_l = missing(missing_aux_v, missing_aux_l);

        rd_dat.close();

    }

}
