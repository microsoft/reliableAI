package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Logger;


/**
 * Reads the content of a datapoints file, line by line
 */
public class DatFileLineReader extends BaseFileLineReader {

    private static final Logger log = Logger.getLogger(
            DatFileLineReader.class.getName());

    private String sep;

    public DatFileLineReader(String s) throws FileNotFoundException {
        super(s);
    }

    protected String[] getSplit(String ln) {
        return ln.split(sep);
    }

    @Override
    public short[] next() {

        try {

            int i = 0; // variable index

            String[] aux_l = getSplit(nextLine);

            if (samp == null)
                samp = new short[aux_l.length];

            for (String aux: aux_l) {
                if ("?".equals(aux)) {
                    samp[i] = -1;
                } else {
                    samp[i] = Short.valueOf(aux);
                }
                i++;
            }

            n_datapoints++;

            nextLine = rd_dat.readLine();
            if (nextLine == null) {
                concluded = true;
            } else {

                nextLine = nextLine.trim();

                if (nextLine.length() == 0) {
                    concluded = true;
                }
            }

        } catch (IOException e) {
            RandomStuff.logExp(log, e);
        }

        return samp;
    }

    public boolean readMetaData() {
        try {
            String ln = rd_dat.readLine();

            if (ln.contains(","))
                sep = ",";
            else
                sep = "\\s+";

            l_s_names = getSplit(ln);
            n_var = l_s_names.length;

            nextLine = rd_dat.readLine();

            nextLine = rd_dat.readLine();
        } catch (IOException e) {
            RandomStuff.logExp(log, e);
            return true;
        }

        return false;
    }
}
