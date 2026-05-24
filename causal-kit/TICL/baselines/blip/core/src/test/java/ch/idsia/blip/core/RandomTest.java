package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.other.BetterNets;
import ch.idsia.blip.core.common.NetToGraph;
import ch.idsia.blip.core.io.bn.BnNetReader;
import org.junit.Test;

import java.io.FileNotFoundException;

import static ch.idsia.blip.core.utils.data.ArrayUtils.removeElementAt;


public class RandomTest extends TheTest {

    @Test
    public void test() throws FileNotFoundException {
        String s = "/home/loskana/Desktop/10000/10000";
        BayesianNetwork bn = BnNetReader.ex(s + ".net");
        NetToGraph n = new NetToGraph();

        n.go(bn, s);
    }

    @Test
    public void fhdf() {
        int[] g = new int[] { 0, 1, 2};

        g = removeElementAt(g, 0);
        g = removeElementAt(g, 1);
        g = removeElementAt(g, 0);

        g = new int[] { 0, 1, 2};
        g = removeElementAt(g, 1);

    }

    @Test
    public void testBetter() throws FileNotFoundException {
        String s = "/home/loskana/Desktop/test";
        BayesianNetwork bn = BnNetReader.ex(s + ".net");
        BetterNets n = new BetterNets();

        n.go(bn, s);
    }

}
