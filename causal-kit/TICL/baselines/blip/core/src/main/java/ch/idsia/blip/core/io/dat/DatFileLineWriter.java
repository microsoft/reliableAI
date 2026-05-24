package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.other.StringUtils;

import java.io.IOException;
import java.io.Writer;

import static ch.idsia.blip.core.utils.RandomStuff.wf;


/**
 * Write the content of a datapoints file in a custom format.
 */
public class DatFileLineWriter extends BaseFileLineWriter {

    public DatFileLineWriter(BayesianNetwork bn, Writer writer) {
        super(bn, writer);
    }

    public DatFileLineWriter(DataSet dat, Writer writer) {
        super(dat, writer);
    }

    /**
     * Write metadata in the file.
     *
     * @throws IOException if there is a problem in writing.
     */
    @Override
    public void writeMetaData() throws IOException {
        wf(wr, "%s\n", StringUtils.join(l_nm_var, " "));
        wf(wr, "%s\n", StringUtils.join(l_ar_var, " "));
    }

    /**
     * Write the next line of sample
     *
     * @param sample sample to graph (value for each variable)
     * @throws IOException if there is a problem writing
     */
    @Override
    public void next(short[] sample) throws IOException {

        String l = "";

        l += sample[0];
        for (int i = 1; i < n_var; i++) {
            l += " " + sample[i];
        }
        l += "\n";

        // Replace missing values
        wr.write(l.replace("-1", "?"));

        wr.flush();
    }

    /**
     * Close the writer map
     *
     * @throws IOException if there is a problem
     */
    public void close() throws IOException {
        if (wr != null) {
            wr.close();
            wr = null;
        }
    }
}
