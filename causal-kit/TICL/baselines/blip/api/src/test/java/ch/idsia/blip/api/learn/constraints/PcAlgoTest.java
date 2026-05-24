package ch.idsia.blip.api.learn.constraints;


import ch.idsia.blip.api.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.MutualInformation;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.other.SubsetIterator;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class PcAlgoTest extends TheTest {

    @Test
    public void powerSetTest() {
        int[] o = new int[] { 0, 1, 2, 4};

        List<int[]> result = ArrayUtils.powerSet(o);

        for (int[] r : result) {
            System.out.println(Arrays.toString(r));
        }
    }

    @Test
    public void subSetIteratorTest() {

        SubsetIterator s = new SubsetIterator(new int[] { 0, 1, 2, 4});

        while (s.hasNext()) {
            System.out.println(Arrays.toString(s.next()));
        }

    }

    @Test
    public void subsetsTest() {

        int[] o = new int[] { 0, 1, 2, 4};

        for (int i = 0; i < 4; i++) {
            System.out.println("#### " + i);
            List<int[]> result = ArrayUtils.getSubsets(o, i);

            for (int[] r : result) {
                System.out.println(Arrays.toString(r));
            }
        }

    }

    @Test
    public void testComputeMI() throws IncorrectCallException, IOException {
        String s = "child-5000.dat";
        DataSet dat = getDataSet(basePath + "constraints/" + s);
        MutualInformation mutualInf = new MutualInformation(dat);

        int x = 0;
        int y = 1;
        int z = 2;

        mutualInf.computeMi(x, y);
        mutualInf.computeMi(y, x);
        mutualInf.computeCMI(x, y, z);
        mutualInf.computeCMI(y, x, z);
        mutualInf.computeCMI(z, x, y);
        mutualInf.computeMi(new int[] { z}, new int[] { x, y});
        mutualInf.computeMi(new int[] { y}, new int[] { x, z});
        mutualInf.computeMi(new int[] { x}, new int[] { z, y});

        mutualInf.computeCMI(x, y, new int[] { z});
    }

}
