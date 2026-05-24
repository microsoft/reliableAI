package ch.idsia.blip.core.utils.tw;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;


public class TreeWidth {

    public Undirected ar;
    private TIntArrayList[] neighboor;
    private int[] degree;
    private int[] fill_in;
    private int[] fill_in_exclude_one;

    public static int go(BayesianNetwork bn) {
        return new TreeWidth().exec(bn.moralize());
    }

    public static int go(Undirected und) {
        return new TreeWidth().exec(und);
    }

    public int exec(Undirected und) {
        ar = und;
        int n_var = und.n;

        // Compute neighboordhod for each node
        neighboor = new TIntArrayList[n_var];

        degree = new int[n_var];
        fill_in = new int[n_var];
        fill_in_exclude_one = new int[n_var];

        for (int n = 0; n < n_var; n++) {
            neighboor[n] = new TIntArrayList();
            for (int n2 = 0; n2 < n_var; n2++) {
                if (n == n2) {
                    continue;
                }
                if (ar.check(n, n2)) {
                    neighboor[n].add(n2);
                }
            }

            updateNode(n);
        }

        /*
         System.out.println("Neighboor");
         for (TIntArrayList p : neighboor) {
         System.out.println(p);
         }
         */

        int tree_width = 0;

        int elim = 0;
        boolean[] eliminated = new boolean[n_var];
        int[] order = new int[n_var];

        while (elim < n_var) {

            /*
             System.out.println("\nOrder");
             System.out.println(Arrays.toString(order));

             System.out.println("Degree");
             System.out.println(Arrays.toString(degree));

             System.out.println("Fill-in");
             System.out.println(Arrays.toString(fill_in));

             System.out.println("Fill-in-exclude");
             System.out.println(Arrays.toString(fill_in_exclude_one));
             */

            int chos = -1;

            // Pick one simplicial (fill_in 0)
            for (int n = 0; n < n_var && chos < 0; n++) {
                if (eliminated[n]) {
                    continue;
                }
                if (fill_in[n] == 0) {
                    chos = n;
                }
            }

            // If there were none, pick one almost simplicial (fill_in_exclude_one 0)
            for (int n = 0; n < n_var && chos < 0; n++) {
                if (eliminated[n]) {
                    continue;
                }
                if (fill_in_exclude_one[n] == 0 && degree[n] <= tree_width) {
                    chos = n;
                }
            }

            // If there was nothing
            if (chos < 0) {
                int min_chos = -1;
                int min_fill = Integer.MAX_VALUE;

                for (int n = 0; n < n_var && chos < 0; n++) {
                    if (eliminated[n]) {
                        continue;
                    }
                    if (fill_in[n] < min_fill) {
                        min_chos = n;
                        min_fill = fill_in[n];
                    }

                }

                chos = min_chos;
            }

            order[elim] = chos;
            elim++;
            eliminated[chos] = true;

            // Update the tree-width
            tree_width = Math.max(tree_width, degree[chos]);

            // Update the neighboorhood: remove the chosen node, add arcs to not-already joining neighbors
            TIntArrayList p = neighboor[chos];

            for (int j1 = 0; j1 < p.size(); j1++) {
                int p1 = p.get(j1);

                // Remove this node as neighbour
                neighboor[p1].remove(chos);

                for (int j2 = j1 + 1; j2 < p.size(); j2++) {
                    int p2 = p.get(j2);

                    if (!ar.check(p1, p2)) {
                        ar.mark(p1, p2);
                        neighboor[p1].add(p2);
                        neighboor[p2].add(p1);
                    }
                }

                updateNode(p1);
            }

        }

        // System.out.println(Arrays.toString(order));

        return tree_width;
    }

    private void updateNode(int n) {

        TIntArrayList p = neighboor[n];

        degree[n] = p.size();

        int fill = 0;

        // System.out.println("fill: " + n);
        for (int j = 0; j < p.size(); j++) {
            for (int j1 = j + 1; j1 < p.size(); j1++) {
                // System.out.printf("%d, %d - %b \n",p[j], p[j1], ar.check(p[j], p[j1]) );
                if (!ar.check(p.get(j), p.get(j1))) {
                    fill += 1;
                }
            }
        }
        fill_in[n] = fill;

        // System.out.println("fill_exclude: " + n);
        int min_fill_exclude = Integer.MAX_VALUE;

        // For every element of the neighbor, compute the fill_in_exclude without it
        for (int c = 0; c < p.size(); c++) {
            int fill_exclude = 0;

            for (int j = 0; j < p.size(); j++) {
                if (j == c) {
                    continue;
                }
                for (int j1 = j + 1; j1 < p.size(); j1++) {
                    if (j1 == c) {
                        continue;
                    }
                    // System.out.printf("%d, %d - %b \n",p[j], p[j1], ar.check(p[j], p[j1]) );
                    if (!ar.check(p.get(j), p.get(j1))) {
                        fill_exclude += 1;
                    }
                }
            }

            if (fill_exclude < min_fill_exclude) {
                min_fill_exclude = fill_exclude;
                // System.out.println(p[c] + " - " + min_fill_exclude);
            }
        }
        fill_in_exclude_one[n] = min_fill_exclude;
    }
}
