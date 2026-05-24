package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.RandomStuff;
import ch.idsia.blip.core.utils.other.StringUtils;

import java.io.IOException;
import java.io.Writer;


public class ArffFileLineWriter extends BaseFileLineWriter {
    private String name;

    public ArffFileLineWriter(BayesianNetwork bn, Writer wr, String name) {
        super(bn, wr);
        this.name = name;
    }

    public ArffFileLineWriter(DataSet dat, String s) {
        super(dat, RandomStuff.getWriter(s));
    }

    public ArffFileLineWriter(DataSet dat, Writer writer) {
        super(dat, writer);
    }

    public void writeMetaData()
        throws IOException {
        RandomStuff.wf(this.wr, "@relation '%s'\n\n", new Object[] { this.name});
        for (int i = 0; i < this.n_var; i++) {
            RandomStuff.wf(this.wr, "@attribute '%s' {",
                    new Object[] { this.l_nm_var[i]});
            RandomStuff.wf(this.wr, "%s}\n",
                    new Object[] { StringUtils.join(this.l_values_var[i], ",")});
        }
        RandomStuff.wf(this.wr, "\n@data\n", new Object[0]);
    }

    public void next(short[] sample)
        throws IOException {
        String l = null;

        for (int i = 0; i < this.n_var; i++) {
            if (l == null) {
                l = "";
            } else {
                l = l + ",";
            }
            if (sample[i] >= 0) {
                l = l + this.l_values_var[i][sample[i]];
            } else {
                l = l + "?";
            }
        }
        l = l + "\n";

        RandomStuff.wf(this.wr, l, new Object[0]);
    }

    public void close()
        throws IOException {
        if (this.wr != null) {
            this.wr.close();
            this.wr = null;
        }
    }
}
