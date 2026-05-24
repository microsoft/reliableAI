package ch.idsia.blip.core.utils.score;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertTrue;


public class BICTest extends TheTest {

    @Test
    public void testBIC() throws IncorrectCallException, IOException {

        String s = "water-50000";

        String path = basePath + "/scorer/" + s;

        String file = path + ".dat";

        DataSet dat = RandomStuff.getDataSet(file);

        BIC score = new BIC(dat);

        // First void
        double sk = score.computeScore(0);

        System.out.println("void: " + sk);
        assertTrue(isScore(sk));

        // Then one more complex
        int[] pars = { 8};

        sk = score.computeScore(0, pars);

        System.out.println("one: " + sk);
        assertTrue(isScore(sk));

    }

    private boolean isScore(double sk) {
        return !Double.isNaN(sk);
    }
}
