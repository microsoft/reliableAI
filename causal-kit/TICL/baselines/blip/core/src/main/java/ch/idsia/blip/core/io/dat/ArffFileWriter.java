package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.other.StringUtils;

import java.io.IOException;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.wf;


public class ArffFileWriter extends DatFileWriter {

    public ArffFileWriter() {
        separator = ",";
    }

    public static void ex(DataSet dat, String s) throws IOException {
        new ArffFileWriter().go(dat, s);
    }

    @Override
    protected void writeMetaData(DataSet dat, Writer wr) throws IOException {

        wf(wr, "@relation '%s'\n\n", "rel");

        for (int i = 0; i < dat.n_var; i++) {
            wf(wr, "@attribute '%s' {", dat.l_nm_var[i]);

            wf(wr, "%s}\n", StringUtils.join(dat.l_nm_states[i], ","));
            wr.flush();
        }

        wf(wr, "\n@data\n");
    }

    @Override
    public String value(DataSet dat, int n, int v) {
        return dat.l_nm_states[n][v];
    }

}
