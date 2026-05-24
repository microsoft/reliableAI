package ch.idsia.blip.core.io.dat;


import java.io.*;
import java.util.logging.Logger;


/**
 * Write a datapoints file
 */
public abstract class BaseFileLineReader implements Closeable {

    private static final Logger log = Logger.getLogger(
            BaseFileLineReader.class.getName());

    public short[] samp;

    public String path;

    // Concluded sample reading
    public boolean concluded;

    // Reader map.
    protected BufferedReader rd_dat;

    public int n_var;

    public String[] l_s_names;

    public int n_datapoints;

    protected String nextLine;



    public BaseFileLineReader(String s) throws FileNotFoundException {
        this.rd_dat = new BufferedReader(new FileReader(s));
        this.path = s;
    }

    public abstract short[] next();

    public abstract boolean readMetaData() throws IOException;

    public void close() throws IOException {
        if (rd_dat != null) {
            rd_dat.close();
        }
    }

    protected String[] splitSpace(String ln) {
        return ln.split("\\s+");
    }

    protected String readLn() throws IOException {
        String s = readL();

        if (s == null) {
            return null;
        }
        while (s.equals("")) {
            s = readL();
            if (s == null) {
                return null;
            }
        }
        return s;
    }

    private String readL() throws IOException {
        String s = rd_dat.readLine();

        if (s == null) {
            return null;
        }
        s = s.trim();
        return s;
    }
}
