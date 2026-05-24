package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;


public class SeqAdvScorerTest extends TheTest {

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
            "", "-d", file, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose)};

        SeqAdvScorerApi.main(args);
    }
}
