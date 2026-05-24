package ch.idsia.blip.core.utils;


import ch.idsia.blip.core.utils.data.ArrayUtils;
import org.junit.Test;

import java.util.Arrays;


public class RandomStuffTest {

    @Test
    public void testIntersect() {
        int[] t1 = { 1, 3, 4, 5};
        int[] t2 = { 1, 5, 9, 12};

        System.out.println(Arrays.toString(ArrayUtils.intersect(t1, t2)));
        System.out.println(ArrayUtils.intersectN(t1, t2));
    }
}
