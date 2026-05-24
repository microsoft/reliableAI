package ch.idsia.blip.core.utils.score;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.RandomStuff;
import org.junit.Test;

import java.io.FileNotFoundException;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;


public class BDeuTest extends TheTest {

    @Test
    public void checkScore() throws FileNotFoundException {
        DataSet dat = getDataSet(basePath + "scorer/child-5000.dat");

        BDeu bDeu = new BDeu(1, dat);

        double sk = bDeu.computeScore(0, new int[] { 6, 18});

        RandomStuff.doubleEquals(-3627.0266, sk);
    }
}
