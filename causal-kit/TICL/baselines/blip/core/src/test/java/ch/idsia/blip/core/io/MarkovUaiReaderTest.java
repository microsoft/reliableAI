package ch.idsia.blip.core.io;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.MarkovNetwork;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

import static ch.idsia.blip.core.utils.RandomStuff.getRandom;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class MarkovUaiReaderTest extends TheTest {

    @Test
    public void testReadSimple() throws IOException {
        MarkovNetwork mk = MarkovUaiReader.ex(
                basePath + "sampler/markov/378-Promedus_33.uai");

        p(mk.n_var);
        mk.sample(10);
    }

    @Test
    public void testSamplest() throws IOException {
        MarkovNetwork mk = MarkovUaiReader.ex(
                basePath + "sampler/markov/undirected_test.uai");

        p(mk.n_var);
        short[] sample = new short[] { 1, 1, 1};

        mk.updateCliqueAssignments();

        Random rand = getRandom();

        mk.resample(sample, 0, rand);

        for (int i = 0; i < 100; i++) {
            // Chose a random variable to resample
            int v = rand.nextInt(mk.n_var);

            mk.resample(sample, v, rand);

            p(Arrays.toString(sample));
        }
    }

}
