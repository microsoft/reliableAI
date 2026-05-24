package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.learn.scorer.SeqScorer;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;


public class SeqScorerTest extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        oneAndOnlyTrueTest("child-5000", "bic");

    }

    private void oneAndOnlyTrueTest(String s, String m) throws IOException, IncorrectCallException {
        String path = basePath + "scorer/" + s;

        String file = path + ".dat";
        String scores = String.format("%s-seqnew-%s.scores", path, m);

        int verbose = 1;
        double max_exec_time = 3;
        int max_pset_size = 3;

        String[] args = {
            "", "-f", file, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose)};

        SeqScorerApi.main(args);
    }

    @Test
    public void trynew() throws IOException, IncorrectCallException {
        String path = basePath + "scorer/dat";

        String[] args = {
            "", "-d", path, "-set", path + ".jkl", "-t", "10",
            "-pc", "bic", "-u", "13", "-v", "2"
        };

        SeqScorerApi.main(args);
    }

    @Test
    public void incrementTest() {

        int pset_size = 2;

        int[] pset = new int[pset_size];

        for (int i = 0; i < pset_size; i++) {
            pset[i] = i;
        }

        boolean cnt = true;

        while (cnt) {

            cnt = SeqScorer.incrementPset(pset, pset.length - 1, 10);

            System.out.println(Arrays.toString(pset));
        }

    }
}
