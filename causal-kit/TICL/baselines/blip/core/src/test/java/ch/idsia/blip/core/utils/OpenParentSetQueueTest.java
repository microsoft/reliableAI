package ch.idsia.blip.core.utils;


import ch.idsia.blip.core.utils.other.ParentSetQueue2;
import org.junit.Test;


public class OpenParentSetQueueTest {

    @Test
    public void testQueue() {
        ParentSetQueue2 q = new ParentSetQueue2(5);

        System.out.println(q);
        q.add(10, 1, 13);
        System.out.println(q);
        q.add(5, 8, 7);
        System.out.println(q);
        q.add(1, 3, 2);
        System.out.println(q);
        q.add(6, 7, 9);
        System.out.println(q);

        q.popBest();
        System.out.println(q);
        q.popWorst();
        System.out.println(q);
        q.popBest();
        System.out.println(q);
        q.popBest();
        System.out.println(q);
    }
}
