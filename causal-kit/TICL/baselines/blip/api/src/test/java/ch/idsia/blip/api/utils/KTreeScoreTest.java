package ch.idsia.blip.api.utils;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.api.old.KTreeScoreApi;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import org.junit.Test;

import java.io.IOException;


public class KTreeScoreTest extends TheTest {

    @Test
    public void whatever() throws IOException, IncorrectCallException {

        // oneAndOnlyTrueTest("simple-5000", "bic");
        oneAndOnlyTrueTest("random4000-0", 10);

    }

    private void oneAndOnlyTrueTest(String s, int treewidth) throws IOException, IncorrectCallException {
        String path = basePath + "ktree/" + s;

        String scores = path + ".jkl";

        String[] args = {
            "", "-set", scores, "-tw", String.valueOf(treewidth), "-w", s
        };

        KTreeScoreApi.main(args);
    }
}
