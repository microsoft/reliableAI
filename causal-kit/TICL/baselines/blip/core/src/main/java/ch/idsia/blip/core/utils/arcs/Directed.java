package ch.idsia.blip.core.utils.arcs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.Formatter;

import static ch.idsia.blip.core.utils.data.ArrayUtils.zeros;


/**
 * Matrix of undirected arcs of a bayesian network
 */
public class Directed extends Arcs {

    public Directed(BayesianNetwork bn) {
        super(bn);
    }

    public Directed(int n_var) {
        super(n_var, false);
    }

    Directed(int n_var, boolean b) {
        super(n_var, b);
    }

    public Directed(ParentSet[] str) {
        super(str.length);
        for (int i = 0; i < n; i++) {
            if (str[i] != null) {
                for (int p : str[i].parents) {
                    mark(p, i);
                }
            }
        }
    }

    @Override
    protected int getSize() {
        return n * n;
    }

    @Override
    protected int index(int v1, int v2) {
        return v1 * n + v2;
    }

    @Override
    public int[] r_index(int i) {
        int[] arr = zeros(2);

        arr[0] = i / n;
        arr[1] = i % n;
        return arr;
    }

    public int[] parents(int v) {
        TIntArrayList aux = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if (check(i, v)) {
                aux.add(i);
            }
        }
        aux.sort();
        return aux.toArray();
    }

    public int[] ancestors(int v) {
        TIntArrayList aux = new TIntArrayList();

        int[] ch = parents(v);

        aux.addAll(ch);

        int i = 0;

        while (i < aux.size()) {
            ch = parents(aux.get(i));

            for (int c : ch) {
                if (!aux.contains(c)) {
                    aux.add(c);
                }
            }

            i++;
        }

        aux.sort();
        return aux.toArray();
    }

    public int[] childrens(int v) {
        TIntArrayList aux = new TIntArrayList();

        for (int i = 0; i < n; i++) {
            if (check(v, i)) {
                aux.add(i);
            }
        }
        aux.sort();
        return aux.toArray();
    }

    public int[] descendants(int v) {
        TIntArrayList aux = new TIntArrayList();

        int[] ch = childrens(v);

        aux.addAll(ch);

        int i = 0;

        while (i < aux.size()) {
            ch = childrens(aux.get(i));

            for (int c : ch) {
                if (!aux.contains(c)) {
                    aux.add(c);
                }
            }

            i++;
        }

        aux.sort();
        return aux.toArray();
    }

    public Pair<int[], int[]> nonZeroIndex() {
        TIntArrayList a1 = new TIntArrayList();
        TIntArrayList a2 = new TIntArrayList();

        for (int i = 0; i < size; i++) {
            if (check(i)) {
                int[] b = r_index(i);

                a1.add(b[0]);
                a2.add(b[1]);
            }
        }
        return new Pair<int[], int[]>(a1.toArray(), a2.toArray());
    }

    public Undirected moralize() {
        Undirected U = new Undirected(n);

        for (int v = 0; v < n; v++) {
            int[] par = parents(v);

            for (int p : par) {
                U.mark(v, p);
            }
            for (int i1 = 0; i1 < par.length; i1++) {
                for (int i2 = i1 + 1; i2 < par.length; i2++) {
                    U.mark(par[i1], par[i2]);
                }
            }
        }
        return U;
    }

    public String toString() {
        StringBuilder s = new StringBuilder();
        Formatter fm = new Formatter(s);

        fm.format("digraph Base {\n");

        for (int v1 = 0; v1 < n; v1++) {
            // fm.format("%s \n", name(v1));
            for (int v2 = 0; v2 < n; v2++) {
                if (check(v1, v2)) {
                    fm.format(" \"%s\" -> \"%s\" \n", name(v1), name(v2));
                    // fm.format(" %s -> %s \n", name(v1), name(v2));
                }
            }
        }

        fm.format("}\n");
        return s.toString();
    }

}
