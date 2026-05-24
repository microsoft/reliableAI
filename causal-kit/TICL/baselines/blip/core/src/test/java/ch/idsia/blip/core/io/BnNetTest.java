package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import org.junit.Test;

import java.io.IOException;


public class BnNetTest extends TheTest {

    @Test
    public void testReaderSimple() throws IOException {
        System.out.println(System.getProperty("user.dir"));

        BayesianNetwork bn_random = getBnFromFile("old/random10-1.net");

        printBnToFile(bn_random, "old/random10-1.new.net");
    }

}
