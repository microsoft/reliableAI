package ch.idsia.blip.core.utils.score;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.IOException;


public class MITTest extends TheTest {

    private DataSet dat;
    private MIT score;

    @Test
    public void testMIT() throws IncorrectCallException, IOException {

        String s = "child-50000";

        String path = basePath + "/scorer/" + s;

        String file = path + ".dat";

        dat = RandomStuff.getDataSet(file);

        int n = 0;

        score = new MIT(0.999, dat);
        score.debug = true;

        // First void
        double sk = score.computeScore(n);

        System.out.println("void: " + sk);

        // Then one more complex
        scoreit(new int[] { 2}, n);
        scoreit(new int[] { 5}, n);
        scoreit(new int[] { 6}, n);

        scoreit(new int[] { 2, 6}, n);
        scoreit(new int[] { 5, 6}, n);

    }

    private void scoreit(int[] pars, int n) {

        int[][] p_values = score.computeParentSetValues(pars);
        // double sk = score.computeScore(n, p_values, pars);
        // System.out.printf("%d, %s -> %.2f\n", n, Arrays.toString(pars), sk);

    }

    private boolean isScore(double sk) {
        return !Double.isNaN(sk);
    }
}
