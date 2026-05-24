package ch.idsia.blip.core.io;


import ch.idsia.blip.core.utils.ParentSet;

import java.io.Closeable;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.Writer;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.getWriter;
import static ch.idsia.blip.core.utils.RandomStuff.logExp;


/**
 * Print scores to file
 */
public class ScoreWriter implements Closeable {

    private static final Logger log = Logger.getLogger(
            ScoreWriter.class.getName());

    public String path;

    private Writer wr = null;

    public ScoreWriter(String s) throws FileNotFoundException {
        try {
            this.path = s;
            wr = getWriter(s);
        } catch (Exception e) {
            log.severe(e.getMessage());
        }
    }

    public void go(ParentSet[][] pSets) {

        try {

            wr.write(String.format("%d\n", pSets.length));
            for (int i = 0; i < pSets.length; i++) {
                wr.write(String.format("%d %d\n", i, pSets[i].length));
                for (ParentSet p : pSets[i]) {
                    wr.write(p.prettyPrint() + "\n");
                }
                wr.flush();
            }

        } catch (Exception e) {
            log.severe(e.getMessage());
        }
    }

    @Override
    public void close() throws IOException {
        wr.close();
    }

    public static void go(ParentSet[][] scores, String s) {
        try {
            ScoreWriter sc = new ScoreWriter(s);

            sc.go(scores);
        } catch (Exception e) {
            logExp(log, e);
        }
    }
}
