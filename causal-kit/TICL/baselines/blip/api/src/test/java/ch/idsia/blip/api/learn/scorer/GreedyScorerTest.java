package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;


public class GreedyScorerTest extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        oneAndOnlyTrueTest("child-50000", "bic");

    }

    private void oneAndOnlyTrueTest(String s, String m) throws IOException, IncorrectCallException {

        String path = basePath + "scorer/" + s;

        String file = path + ".dat";
        String scores = String.format("%s-greedy-%s.scores", path, m);

        int verbose = 2;
        double max_exec_time = 1;
        int max_pset_size = 3;

        String[] args = {
            "", "-f", file, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose)};

        GreedyScorerApi.main(args);
    }
}
