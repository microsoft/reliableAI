package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * Reads the content of a datapoints file, line by line
 */
public class DataFileLineReader extends DatFileLineReader {

    private static final Logger log = Logger.getLogger(
            DatFileLineReader.class.getName());

    public DataFileLineReader(String s) throws FileNotFoundException {
        super(s);
    }

    @Override
    protected String[] getSplit(String ln) {
        return ln.split(",");
    }

    @Override
    public boolean readMetaData() {
        try {
            nextLine = rd_dat.readLine();
            n_var = getSplit(nextLine).length;
            l_s_names = new String[n_var];
            for (int i = 0; i < n_var; i++) {
                l_s_names[i] = f("N%d", i);
            }
        } catch (IOException e) {
            RandomStuff.logExp(log, e);
            return true;
        }

        return false;
    }
}

