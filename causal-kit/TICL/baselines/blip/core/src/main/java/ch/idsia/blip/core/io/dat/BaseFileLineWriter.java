package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.DataSet;

import java.io.IOException;
import java.io.Writer;


/**
 * Write a datapoints file
 */
public abstract class BaseFileLineWriter {

    // Writer for data
    protected Writer wr;

    protected final String[][] l_values_var;

    protected final int n_var;

    protected final String[] l_nm_var;

    protected final int[] l_ar_var;

    public BaseFileLineWriter(BayesianNetwork bn, Writer wr) {
        this.wr = wr;
        this.n_var = bn.n_var;
        this.l_values_var = bn.l_values_var;
        this.l_nm_var = bn.l_nm_var;
        this.l_ar_var = bn.l_ar_var;
    }

    public BaseFileLineWriter(DataSet d, Writer wr) {
        this.wr = wr;
        this.n_var = d.n_var;
        this.l_values_var = d.l_nm_states;
        this.l_nm_var = d.l_nm_var;
        this.l_ar_var = d.l_n_arity;
    }

    public abstract void writeMetaData() throws IOException;

    /**
     * Write the next line of sample
     *
     * @param sample sample to graph (value for each variable)
     * @throws IOException if there is a problem writing
     */
    public abstract void next(short[] sample) throws IOException;

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
