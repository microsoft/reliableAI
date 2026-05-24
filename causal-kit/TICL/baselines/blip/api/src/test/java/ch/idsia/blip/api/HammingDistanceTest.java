package ch.idsia.blip.api;


import ch.idsia.blip.api.common.HammingDist;
import ch.idsia.blip.core.utils.BayesianNetwork;
import org.junit.Test;

import java.io.IOException;

import static org.junit.Assert.assertEquals;


public class HammingDistanceTest extends TheTest {

    @Test
    public void testSimple() throws IOException {
        System.out.println(System.getProperty("user.dir"));
        assertEquals(distance("tw/tw-1.net", "tw/tw-1.net"), 0, 0.0001);
        assertEquals(distance("tw/tw-1.net", "tw/tw-2.net"), 8, 0.0001);
        assertEquals(distance("tw/tw-2.net", "tw/tw-1.net"), 8, 0.0001);
    }

    private double distance(String s1, String s2) throws IOException {
        BayesianNetwork f = getBnFromFile(s1);
        BayesianNetwork s = getBnFromFile(s2);

        HammingDist h = new HammingDist();
        double dist = h.computeDist(f, s);

        System.out.println(f);
        System.out.println(s);
        System.out.printf("%s , %s -> %d \n", s1, s2, dist);
        return dist;
    }
}
