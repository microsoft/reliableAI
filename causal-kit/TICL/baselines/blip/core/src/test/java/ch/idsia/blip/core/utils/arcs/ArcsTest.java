package ch.idsia.blip.core.utils.arcs;


import org.junit.Test;

import java.util.Arrays;


public class ArcsTest {

    @Test
    public void testDirected() {
        int n = 10;

        Directed dir = new Directed(n);

        dir.mark(1, 2);
        dir.mark(1, 3);
        dir.mark(2, 5);
        dir.mark(5, 7);
        dir.mark(2, 7);

        dir.mark(9, 3);
        dir.mark(4, 3);
        dir.mark(9, 4);

        System.out.println(dir);

        System.out.println("Childrens:");
        for (int i = 0; i < n; i++) {
            System.out.printf("%d - %s    ", i,
                    Arrays.toString(dir.childrens(i)));
        }

        System.out.println("\n\nDescendants:");
        for (int i = 0; i < n; i++) {
            System.out.printf("%d - %s    ", i,
                    Arrays.toString(dir.descendants(i)));
        }

        System.out.println("\n\nParents:");
        for (int i = 0; i < n; i++) {
            System.out.printf("%d - %s    ", i, Arrays.toString(dir.parents(i)));
        }

        System.out.println("\n\nAncestors:");
        for (int i = 0; i < n; i++) {
            System.out.printf("%d - %s    ", i,
                    Arrays.toString(dir.ancestors(i)));
        }

        dir.ancestors(1);
        dir.descendants(1);

        dir.ancestors(2);

    }

    @Test
    public void testCliqueSize() {

        int n = 10;

        Undirected tri = new Undirected(n);

        /*
         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         Assert.assertEquals(mor.sizeClique(thread, j), 2);
         }
         }

         mor.mark(1, 2);

         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         Assert.assertEquals(mor.sizeClique(thread, j), 2);
         }
         }

         mor.mark(2, 3);

         Assert.assertEquals(mor.sizeClique(1, 3), 3);

         int v, h;

         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         v = (thread == 1 && j == 3) ? 3 : 2;
         h = mor.sizeClique(thread, j);
         Assert.assertEquals(h, v);
         }
         }

         mor.mark(1, 4);

         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         v = (thread == 1 && j == 3) || (thread == 2 && j == 4) ? 3 : 2;
         h = mor.sizeClique(thread, j);
         Assert.assertEquals(h, v);
         }
         }

         mor.mark(2, 4);

         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         if ((thread == 1 && j == 3) || (thread == 2 && j == 4)
         || (thread == 1 && j == 2) || (thread == 1 && j == 4)
         || (thread == 3 && j == 4)) {
         v = 3;
         } else {
         v = 2;
         }
         h = mor.sizeClique(thread, j);
         System.out.printf("%d %d %d \n", thread, j, h);
         Assert.assertEquals(h, v);
         }
         }

         mor.mark(1, 3);

         for (int thread = 0; thread < n; thread++) {
         for (int j = thread + 1; j < n; j++) {
         if ((thread == 1 && j == 3) || (thread == 2 && j == 4)
         || (thread == 2 && j == 3) || (thread == 1 && j == 2)
         || (thread == 1 && j == 4)) {
         v = 3;
         } else if (thread == 3 && j == 4) {
         v = 4;
         } else {
         v = 2;
         }
         h = mor.sizeClique(thread, j);
         Assert.assertEquals(h, v);
         }
         }

         System.out.println(mor.sizeClique(3, 4));
         }

         @Test
         public void r_index() {
         int n = 10;
         Directed A = new Directed(n);

         for (int thread = 0; thread < n; thread++) {
         for (int j = 0; j < n; j++) {
         int v = A.index(thread, j);

         System.out.printf("%d %d -> %d -> %s\n", thread, j, v,
         Arrays.toString(A.r_index(v)));
         }
         }*/
    }
}
