package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class TopologicalOrder {

    public static int[] find(int n_var, int[][] l_parent_var) {

        TIntArrayList[] E = new TIntArrayList[n_var]; // Container for the edges

        for (int i = 0; i < n_var; i++) {
            E[i] = new TIntArrayList();
        }

        TIntArrayList P = new TIntArrayList(n_var); // Parent set dom_size residual for each node

        TIntArrayList L = new TIntArrayList(); // Empty list that will contain the sorted elements

        TIntArrayList S = new TIntArrayList(); // Set of all nodes with no incoming edges

        for (int i = 0; i < n_var; i++) {
            int[] parents = l_parent_var[i];

            if (parents.length == 0) {
                S.add(i);
            }

            P.add(parents.length);

            for (int parent : parents) {
                E[parent].add(i);
            }
        }

        while (!S.isEmpty()) { // while S is non-empty do
            int n = S.iterator().next(); // remove a node n from S

            S.remove(n);

            L.add(n);
            for (int i = 0; i < E[n].size(); i++) { // for each child of node
                int c = E[n].get(i);
                int s = P.get(c) - 1; // reduce size of parent set

                if (s == 0) {
                    S.add(c);
                } // if m has no incoming edges then
                if (s < 0) {
                    p("WHAT?");
                }
                P.set(c, s);
            }
        }

        return L.toArray();
    }

    public static boolean isAcyclic(int n_var, int[][] l_parent_var) {
        int[] res = find(n_var, l_parent_var);

        return ((res != null) && (res.length == n_var));
    }
}
