package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.learn.solver.AsobsSolver;
import ch.idsia.blip.core.learn.solver.src.asobs.AsobsSearcher;
import ch.idsia.blip.core.utils.other.IncorrectCallException;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import org.junit.Test;

import java.io.IOException;

import static ch.idsia.blip.core.utils.RandomStuff.getScoreReader;


/**
 * Created by loskana on 11/12/16.
 */
public class ColinTest {

    @Test
    public void test() throws IOException, IncorrectCallException {
        int[] order = new int[] {
            19, 20, 47, 22, 51, 25, 40, 46, 63, 39, 9, 28,
            53, 42, 54, 61, 31, 58, 35, 15, 11, 50, 64, 44, 34, 43, 6, 49, 13, 5,
            24, 67, 52, 59, 29, 37, 17, 38, 14, 1, 56, 57, 8, 41, 7, 36, 26, 30,
            4, 66, 23, 65, 60, 27, 10, 12, 0, 21, 48, 45, 2, 68, 55, 16, 62, 3,
            32, 33, 18};

        ArrayUtils.reverse(order);
        AsobsSolver s = new AsobsSolver();
        ParentSet[][] sc = getScoreReader(
                "/home/loskana/Desktop/colin/plants.blip.jkl");

        s.init(sc, 10);

        AsobsSearcher g = new AsobsSearcher(s);

        g.init(sc);
        g.search();
    }

    @Test
    public void test2() throws IOException, IncorrectCallException {
        int[] order = new int[] {
            50, 14, 31, 17, 29, 7, 21, 44, 48, 66, 23, 24,
            36, 53, 73, 28, 59, 64, 63, 100, 101, 45, 108, 13, 75, 52, 56, 84,
            91, 42, 30, 89, 47, 71, 9, 85, 40, 90, 1, 0, 93, 92, 49, 32, 79, 86,
            94, 83, 51, 38, 60, 16, 102, 98, 77, 33, 110, 69, 72, 76, 70, 99, 78,
            97, 15, 107, 58, 46, 19, 62, 109, 106, 74, 81, 68, 3, 2, 10, 54, 65,
            55, 61, 43, 5, 95, 22, 37, 25, 96, 26, 20, 67, 8, 27, 6, 11, 35, 41,
            4, 34, 87, 57, 88, 103, 12, 105, 18, 82, 80, 104, 39};

        ArrayUtils.reverse(order);
        AsobsSolver s = new AsobsSolver();
        ParentSet[][] sc = getScoreReader(
                "/home/loskana/Desktop/colin/accidents.blip.jkl");

        s.init(sc, 10);

        AsobsSearcher g = new AsobsSearcher(s);

        g.init(sc);
        g.search();
    }
}
