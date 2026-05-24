package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnMdlReader;
import org.junit.Test;

import java.io.IOException;


public class BnMdlReaderTest extends TheTest {

    @Test
    public void testReadSimple() throws IOException {
        String s = "[node0|node1][node1|node4:node5:node7][node2][node3|node4:node9]"
                + "[node4|node8][node5|node2:node9][node6][node7|node6][node8|node7][node9|node2:node6]";

        BayesianNetwork bn = BnMdlReader.go(s);

        System.out.println(bn);
    }
}
