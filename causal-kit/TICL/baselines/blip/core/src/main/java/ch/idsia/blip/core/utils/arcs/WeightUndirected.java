package ch.idsia.blip.core.utils.arcs;


import ch.idsia.blip.core.utils.data.set.TIntSet;

import java.io.IOException;
import java.io.Writer;
import java.util.Arrays;
import java.util.HashMap;

import static ch.idsia.blip.core.utils.RandomStuff.wf;


/**
 * Matrix of weighted, undirected arcs of a bayesian network
 */
public class WeightUndirected extends Undirected {

    public HashMap<Integer, Double> weight;

    public WeightUndirected(int n_var) {
        super(n_var);
    }

    @Override
    void prepare() {
        super.prepare();
        weight = new HashMap<Integer, Double>();
    }

    @Override
    public void mark(int v1, int v2) {
        mark(v1, v2, 1.0);
    }

    public void mark(int v1, int v2, double w) {
        super.mark(v1, v2);
        weight.put(index(v1, v2), w);
    }

    @Override
    public void mark(int i) {
        mark(i, 1.0);
    }

    public void mark(int i, double w) {
        super.mark(i);
        weight.put(i, w);
    }

    @Override
    public void empty(int v1, int v2) {
        super.empty(v1, v2);
        weight.remove(index(v1, v2));
    }

    @Override
    public void empty(int i) {
        super.empty(i);
        weight.remove(i);
    }

    public void write(Writer w) throws IOException {
        wf(w, "graph Base {\n");
        int j = 0;

        for (int v1 = 0; v1 < n; v1++) {
            for (int v2 = v1 + 1; v2 < n; v2++) {
                if (check(v1, v2)) {
                    double we = (weight.get(index(v1, v2)) * 3);

                    wf(w, "\"%s\" -- \"%s\" [penwidth=%.2f]\n", name(v1),
                            name(v2), we, weight.get(index(v1, v2)));
                    j++;
                }
            }
            w.flush();
        }
        wf(w, "label=\"Nodes: %d, arcs: %d\"\n}\n", n, j);
        w.close();
    }

    @Override
    public Undirected getSubUndirected(TIntSet s) {

        int[] ar = s.toArray();

        Arrays.sort(ar);
        WeightUndirected n_u = new WeightUndirected(ar.length);

        n_u.names = new String[ar.length];
        for (int i = 0; i < ar.length; i++) {
            int a = ar[i];

            n_u.names[i] = name(a);

            int[] adj = adj(a);

            for (int j = 0; j < adj.length; j++) {
                int n_adj = Arrays.binarySearch(ar, adj[j]);
                double w = weight.get(index(a, adj[j]));

                n_u.mark(i, n_adj, w);
            }
        }

        return n_u;
    }
}
