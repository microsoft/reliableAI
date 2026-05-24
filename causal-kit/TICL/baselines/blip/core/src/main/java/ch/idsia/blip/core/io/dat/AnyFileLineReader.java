package ch.idsia.blip.core.io.dat;


import ch.idsia.blip.core.utils.RandomStuff;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;


/**
 * Reads the content of a datapoints file, line by line
 */
public class AnyFileLineReader extends DatFileLineReader {

    private static final Logger log = Logger.getLogger(
            AnyFileLineReader.class.getName());

    private List<List<String>> values;

    public AnyFileLineReader(String s) throws FileNotFoundException {
        super(s);
    }

    @Override
    public short[] next() {

        short[] samp = new short[n_var];

        short v;
        List<String> lv;

        try {

            int i = 0; // variable index

            for (String aux : getSplit(nextLine)) {
                lv = values.get(i);
                v = (short) lv.indexOf(aux);
                if (v < 0) {
                    v = (short) lv.size();
                    lv.add(aux);
                }
                samp[i] = v;
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

    @Override
    public boolean readMetaData() {
        if (!super.readMetaData()) {
            return false;
        }

        values = new ArrayList<List<String>>();
        for (int i = 0; i < n_var; i++) {
            values.add(new ArrayList<String>());
        }

        return true;
    }
}
