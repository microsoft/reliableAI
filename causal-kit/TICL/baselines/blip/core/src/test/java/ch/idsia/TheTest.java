package ch.idsia;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnNetWriter;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.doubleEquals;
import static org.junit.Assert.assertTrue;


public class TheTest {

    protected String basePath = "../experiments/";

    protected BayesianNetwork getBnFromFile(String s) throws IOException {
        return BnNetReader.ex(basePath + s);
    }

    protected void printBnToFile(BayesianNetwork bn, String fileName) {
        BnNetWriter.ex(bn, basePath + fileName);
    }

    protected static void assertDoubleEquals(double a, double b) {
        assertTrue(doubleEquals(a, b));
    }

}
