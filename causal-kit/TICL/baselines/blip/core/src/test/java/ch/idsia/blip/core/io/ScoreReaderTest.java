package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import org.junit.Test;

import java.io.IOException;


public class ScoreReaderTest extends TheTest {

    @Test
    public void testRead() throws IOException {

        ScoreReader sc = new ScoreReader(basePath + "old/alarm-10000.scores", 1);

        sc.readScores();

        for (int i = 0; i < sc.n_var; i++) {
            System.out.println(i);
            System.out.println(sc.m_scores[i]);
        }
    }
}
