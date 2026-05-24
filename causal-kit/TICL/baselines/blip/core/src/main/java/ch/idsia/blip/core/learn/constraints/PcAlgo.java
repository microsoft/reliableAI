package ch.idsia.blip.core.learn.constraints;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.io.dat.DatFileReader;
import ch.idsia.blip.core.learn.constraints.oracle.Oracle;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

import static ch.idsia.blip.core.utils.RandomStuff.f;
import static ch.idsia.blip.core.utils.RandomStuff.pf;


/**
 * Independence scorer
 * <p/>
 * Ashes to ashes. Dust to dust.
 */
public class PcAlgo {

    /**
     * Flag for verbose operations
     */
    public int verbose = 0;

    /**
     * Mutual Information threshold oracle
     */
    public Oracle oracle;

    private DataSet dat;

    public void execute(DatFileReader dr) throws IOException {
        execute(dr.read());
    }

    public void execute(DataSet dat) throws IOException {

        if (oracle == null) {
            return;
        }

        if (verbose > 1) {
            f("Starting Pc Algorithm. Oracle %s\n", oracle.toString());
        }

        this.dat = dat;

        Pair<Undirected, HashMap<Integer, int[]>> result = adjancencySearch();

        // Inference.Arcs graph = orientSkeleton(result.getFirst(), result.getSecond());

        // graph = orientArcs(graph);

    }

    public void printSkeleton(Undirected skeleton) {
        for (int i = 0; i < skeleton.size; i++) {
            if (skeleton.check(i)) {
                int[] v = skeleton.r_index(i);

                System.out.printf("%s - %s \n", dat.l_nm_var[v[0]],
                        dat.l_nm_var[v[1]]);
            }
        }

    }

    private Pair<Undirected, HashMap<Integer, int[]>> adjancencySearch() {

        // Prepare skeleton, separation set
        Undirected skeleton = new Undirected(dat.n_var, true);
        HashMap<Integer, int[]> separSet = new HashMap<Integer, int[]>();

        TIntArrayList[] adj = new TIntArrayList[dat.n_var];

        int l = -1;
        boolean first_cycle = true;

        while (first_cycle) {
            l = l + 1;

            int max_l = 3;

            if (l > max_l) {
                break;
            }

            first_cycle = false;

            // Prepare adjacency copy
            prepareAdjacency(skeleton, adj);

            // Check all the pairs
            int i = -1;
            int j = 0;

            while (true) {

                // Go to next pair
                i++;
                if (i == dat.n_var) {
                    i = 0;
                    j++;
                }
                if (j == dat.n_var) {
                    break;
                }

                if (i == j) {
                    continue;
                }

                // This pair has to be adjacent in C
                if (!skeleton.check(i, j)) {
                    continue;
                }

                // Check |a(X_i) \ {X_j}| >= l
                int d = adj[i].size();

                if (adj[i].binarySearch(j) >= 0) {
                    d -= 1;
                }

                if (d < l) {
                    continue;
                }

                first_cycle = true;

                if (verbose > 1) {
                    pf("Considering %s - %s on size %d \n", h(i), h(j), l);
                }

                // Consider off all the subsets S \in a(X_i) \ {X_j}, with |S| = l
                TIntArrayList cp = new TIntArrayList(adj[i].size() - 1);

                cp.addAll(adj[i]);
                cp.remove(j);

                List<int[]> res = ArrayUtils.getSubsets(cp.toArray(), l);
                boolean condIndep = false;

                for (int k = 0; k < res.size() && !condIndep; k++) {
                    int[] s = res.get(k);

                    if (oracle.condInd(i, j, s)) {
                        if (verbose > 0) {
                            pf("delete edge %s - %s on conditioning set %s \n",
                                    h(i), h(j), h(s));
                        }
                        // delete edge X_i - X_j
                        skeleton.empty(i, j);
                        // let sepset(X_i, X_j) = sepset(X_j, X_i) = S
                        separSet.put(skeleton.index(i, j), s);
                        condIndep = true;
                    }
                }

            }
        }

        return new Pair<Undirected, HashMap<Integer, int[]>>(skeleton, separSet);
    }

    /**
     * for all X_i in C do: a(X_i) = adj(C, X_i)
     */
    private void prepareAdjacency(Undirected skeleton, TIntArrayList[] adj) {
        for (int i = 0; i < dat.n_var; i++) {
            adj[i] = new TIntArrayList();
        }
        for (int i = 0; i < dat.n_var; i++) {
            for (int j = i + 1; j < dat.n_var; j++) {
                if (skeleton.check(i, j)) {
                    adj[i].add(j);
                    adj[j].add(i);
                }
            }
        }
    }

    private String h(int s) {
        return dat.l_nm_var[s];
    }

    private List<String> h(int[] s) {
        List<String> g = new ArrayList<String>();

        for (int e : s) {
            g.add(dat.l_nm_var[e]);
        }
        return g;
    }

    public Undirected skeleton(DatFileReader dr) throws IOException {
        return skeleton(dr.read());
    }

    public Undirected skeleton(DataSet dat) throws IOException {
        this.dat = dat;
        return adjancencySearch().getFirst();
    }
}
