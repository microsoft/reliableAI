package ch.idsia.blip.core.io;


import ch.idsia.blip.core.utils.ParentSet;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * Reads a pre-computed file with local scores.
 */
public class ScoreReader implements Closeable {

    /**
     * Logger
     */
    private static final Logger log = Logger.getLogger(
            ScoreReader.class.getName());

    public String filename;

    /**
     * Number of variables
     */
    public int n_var;

    /**
     * Scores for each variable
     */
    public ParentSet[][] m_scores;
    public boolean completed = false;

    /**
     * Reader from the score file
     */
    private BufferedReader rd_scores;

    public int max_size = 10;

    public ScoreReader(String s, int max_size) throws FileNotFoundException {
        try {
            rd_scores = new BufferedReader(new FileReader(s));
            this.filename = s;
            this.max_size = max_size;

            String ln = rd_scores.readLine();

            while (ln.startsWith("#")) {
                ln = rd_scores.readLine();
            }

            n_var = Integer.parseInt(ln);
        } catch (FileNotFoundException e) {
            log.info("File not found: " + s);
        } catch (IOException e) {
            log.severe(f("Problem while reading file %s", filename));
            logExp(log, e);
        }
    }

    public ScoreReader(String s) throws FileNotFoundException {
        this(s, 10);
    }

    /**
     * Close the reader to the file
     *
     * @throws IOException problem with the file?
     */
    public void close() throws IOException {
        if (rd_scores != null) {
            rd_scores.close();
        }
    }

    /**
     * Read scores from the given file.
     */

    public ParentSet[][] readScores() throws IOException {

        if (completed) {
            return new ParentSet[0][];
        }

        int i, j = 0, k = 0;

        String[] ln = null, aux = null;

        m_scores = new ParentSet[n_var][];

        String l = rl();

        while (l != null && (l.startsWith("#") || l.equals(""))) {
            l = rl();
        }

        for (i = 0; i < n_var; i++) {

            try {
                ln = l.split(" ");
                k++;
            } catch (Exception e) {
                e.printStackTrace();
                log.severe(
                        String.format(
                                "Problem while reading variable: %s. Var %d, ln: %s (line number: %d)",
                                filename, i, Arrays.toString(ln), k));
                logExp(log, e);
                return new ParentSet[0][];
            }

            int v = Integer.parseInt(ln[0]);

            if (v != i) {
                pf("WHAAAAAAAT?! v: %d, thread: %d, line number: %d \n", v, i, k);
            }

            int n = Integer.parseInt(ln[1]);

            List<ParentSet> t = new ArrayList<ParentSet>();

            try {

                for (j = 0; j < n; j++) {

                    aux = rd_scores.readLine().split(" ");
                    k++;
                    if (Integer.valueOf(aux[1]) <= max_size) {
                        t.add(new ParentSet(aux));
                    }
                }

            } catch (Exception e) {
                log.severe(
                        String.format(
                                "Problem while reading parent set: %s. Var %d, parent set %d. aux: %s",
                                filename, i, j, Arrays.toString(aux)));
                logExp(log, e);
                return new ParentSet[0][];
            }

            Collections.sort(t);
            m_scores[i] = t.toArray(new ParentSet[t.size()]);

            l = rl();
        }

        completed = true;

        return m_scores;

    }

    private String rl() throws IOException {
        if (rd_scores == null) {
            throw new IOException("No reader defined!");
        }

        String s = rd_scores.readLine();

        if (s != null) {
            s = s.trim();
        }
        return s;
    }

}
