package ch.idsia.blip.api.learn.scorer;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;


public class AdvK2Test extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {
        System.out.println(System.getProperty("user.dir"));

        // oneAndOnlyTrueTest("simple", "10000");

        // oneAndOnlyTrueTest("random20-5-3200");

        oneAndOnlyTrueTest("child-50000", "bic");

        // oneAndOnlyTrueTest("random10-1", "10240");

        // oneAndOnlyTrueTest("ovarian_61902");


        /*


         oneAndOnlyTrueTest("alarm", "10000");

         oneAndOnlyTrueTest("insurance", "10000");

         oneAndOnlyTrueTest("diabetes", "1000");

         oneAndOnlyTrueTest("diabetes", "10000");

         */

        // oneAndOnlyTrueTest("link", "50000");
    }

    private void oneAndOnlyTrueTest(String s, String m) throws IOException, IncorrectCallException {

        String path = basePath + "scorer/" + s;

        String file = path + ".dat";
        String scores = String.format("%s-advk2-%s.scores", path, m);

        int verbose = 2;
        double max_exec_time = 1;
        int max_pset_size = 3;

        String[] args = {
            "", "-f", file, "-set", scores, "-n", String.valueOf(max_pset_size),
            "-t", String.valueOf(max_exec_time), "-pc", m, "-v",
            String.valueOf(verbose)};

        AdvK2ScorerApi.main(args);

    }
}
