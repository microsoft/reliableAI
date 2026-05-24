package ch.idsia.blip.core.utils.arcs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.data.set.TIntSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.Formatter;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;
import static ch.idsia.blip.core.utils.data.ArrayUtils.zeros;
import static ch.idsia.blip.core.utils.RandomStuff.*;


/**
 * Matrix of undirected arcs of a bayesian network
 */
public class Undirected extends Arcs {

    public TIntHashSet[] neigh;

    public Undirected(BayesianNetwork bn) {
        super(bn);
    }

    @Override
    void prepare() {
        neigh = new TIntHashSet[n];
        for (int i = 0; i < n; i++) {
            neigh[i] = new TIntHashSet();
        }
    }

    public Undirected(int n_var) {
        this(n_var, false);

    }

    public Undirected(int n_var, boolean b) {
        super(n_var, b);

        prepare();
    }

    @Override
    protected int getSize() {
        return (n * (n - 1)) / 2;
    }

    /**
     * Index in matrix from row / column
     *
     * @param v1 row
     * @param v2 column
     * @return index
     */
    @Override
    public int index(int v1, int v2) {
        return RandomStuff.index(n, v1, v2);
    }

    /**
     * Reverse index (row / column from index)
     *
     * @param i index
     * @return row / column
     */
    @Override
    public int[] r_index(int i) {
        int[] arr = zeros(2);

        int n = this.n - 1;

        while (i >= n) {
            i -= n;
            n--;
            arr[0]++;
        }
        arr[1] = arr[0] + i + 1;

        return arr;
    }

    @Override
    public int[] neighbours(int v) {
        return neigh[v].toArray();
    }

    @Override
    public void mark(int v1, int v2) {
        neigh[v1].add(v2);
        neigh[v2].add(v1);
        super.mark(index(v1, v2));
    }

    public void mark(int i) {
        int[] r = r_index(i);

        neigh[r[0]].add(r[1]);
        neigh[r[1]].add(r[0]);
        super.mark(i);
    }

    public void empty(int v1, int v2) {
        neigh[v1].remove(v2);
        neigh[v2].remove(v1);
        super.empty(index(v1, v2));
    }

    public void empty(int i) {
        int[] r = r_index(i);

        neigh[r[0]].remove(r[1]);
        neigh[r[1]].remove(r[0]);
        super.empty(i);
    }

    /**
     * Find arcs that will be added by removing v
     *
     * @param v    variable of interest
     * @param vars list of variables
     * @return list of arcs
     */
    public int[] findFillArcs(int v, TIntArrayList vars) {
        // Variables related to the given in the arcs
        // (parents or children of v, and contained in vars)
        TIntArrayList related = new TIntArrayList();

        for (int o = 0; o < n; o++) {
            if (o == v) {
                continue;
            }

            if (check(o, v) || check(v, o)) {
                if (vars.contains(o)) {
                    related.add(o);
                }
            }
        }
        related.sort();
        // List of arcs between the related to add
        TIntArrayList new_arcs = new TIntArrayList();

        for (int n1 = 0; n1 < related.size(); n1++) {
            for (int n2 = n1 + 1; n2 < related.size(); n2++) {
                int i = index(related.get(n1), related.get(n2));

                if (!check(i)) {
                    new_arcs.add(i);
                }
            }
        }
        return new_arcs.toArray();
    }

    /**
     * Check the size of the maximal clique near the variable
     */
    public boolean biggerClique(int v, int maxSize) {

        // Checks the neighbour of v
        int[] neigh = neighbours(v);
        int N = neigh.length;

        // System.out.println(Arrays.toString(neigh));

        // enumerate all the subsets
        int allMasks = (1 << neigh.length);

        for (int i = 1; i < allMasks; i++) {

            // System.out.println(thread);

            // get new subset
            TIntArrayList s = new TIntArrayList();

            for (int j = 0; j < N; j++) {
                if ((i & (1 << j)) > 0) {
                    s.add(neigh[j]);
                    // System.out.println(s);
                }
            }

            boolean complete = true;

            // Check if every pair on the set are connected, we already know that each is already connected to v
            for (int i1 = 0; i1 < s.size() && complete; i1++) {
                for (int i2 = i1 + 1; i2 < s.size() && complete; i2++) {
                    if (!check(s.get(i1), s.get(i2))) {
                        complete = false;
                    }
                }
            }

            // if complete and the size is bigger than the allowed
            if (complete && (s.size() + 1) > maxSize) {
                s.add(v);
                p(s.toString());
                return true;
            }
        }

        return false;

        /*

         // Get all the subsets of the neighbours
         for (int size = neigh.length; size > 0; size--) {

         List<int[]> sets = RandomStuff.getSubsets(neigh, size);

         // For each set of the given size
         for (int[] s: sets) {
         boolean complete = true;

         // Check if every pair on the set are connected, we already know that each is already connected to v1 and v2
         for (int i1 = 0; i1 < s.length && complete; i1++) {
         for (int i2 = i1 + 1; i2 < s.length && complete; i2++) {
         if (!check(s[i1], s[i2])) {
         complete = false;
         }
         }
         }

         if (complete) {
         biggest = max(biggest, size);
         }
         }
         }*/

    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        Formatter fm = new Formatter(s);

        fm.format("graph Base {\n");
        for (int v1 = 0; v1 < n; v1++) {
            fm.format("%s \n", name(v1));
            for (int v2 = v1 + 1; v2 < n; v2++) {
                if (check(v1, v2)) {
                    fm.format("%s -- %s \n", name(v1), name(v2));
                }
            }
        }

        s.append("}");
        return s.toString();
    }

    public Undirected clone() {
        Undirected nw = new Undirected(n);

        cloneArray(this.arcs, nw.arcs);
        cloneArray(this.neigh, nw.neigh);
        return nw;
    }

    public int n_adj(int j) {
        int t = 0;

        for (int i = 0; i < n; i++) {
            if ((i != j) && (check(i, j))) {
                t++;
            }
        }
        return t;
    }

    public int[] adj(int j) {
        TIntArrayList t = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if ((i != j) && (check(i, j))) {
                t.add(i);
            }
        }
        return t.toArray();
    }

    public static Undirected read(String s) throws IOException {
        BufferedReader br = new BufferedReader(getReader(s));
        int n = Integer.valueOf(br.readLine().trim());
        Undirected u = new Undirected(n);
        String l;

        while ((l = br.readLine()) != null) {
            String[] aux = l.split("-");

            u.mark(Integer.valueOf(aux[0].trim()),
                    Integer.valueOf(aux[1].trim()));
        }
        return u;
    }

    public Undirected getSubUndirected(TIntSet s) {

        int[] ar = s.toArray();

        Arrays.sort(ar);
        Undirected n_u = new Undirected(ar.length);

        n_u.names = new String[ar.length];
        for (int i = 0; i < ar.length; i++) {
            int a = ar[i];

            n_u.names[i] = name(a);

            int[] adj = adj(a);

            for (int j = 0; j < adj.length; j++) {
                int n_adj = Arrays.binarySearch(ar, adj[j]);

                n_u.mark(i, n_adj);
            }
        }

        return n_u;
    }

    public void write(Writer w) throws IOException {
        wf(w, "graph Base {\n");
        // wf(w, "ratio=\"0.7\"; \n");
        // wf(w, "node[fontsize=20];\n");
        // wf(w, "overlap=false;\n");
        // wf(w, "splines=true;\n");

        int j = 0;

        for (int v1 = 0; v1 < n; v1++) {

            wf(w, "\"%s\";\n", name(v1));

            for (int v2 = v1 + 1; v2 < n; v2++) {
                if (check(v1, v2)) {
                    wf(w, "\"%s\" -- \"%s\"; \n", name(v1), name(v2));
                    j++;
                }
            }
            w.flush();
        }
        wf(w, "label=\"Nodes: %d, arcs: %d\"\n}\n", n, j);
        w.close();
    }

    public void write(String s) throws IOException {
        write(getWriter(s));
    }

    public int numEdges() {
        int t = 0;

        for (int i = 0; i < n; i++) {
            for (int n : neigh[i].toArray()) {
                if (n > i) {
                    t++;
                }
            }
        }
        return t;
    }
}
