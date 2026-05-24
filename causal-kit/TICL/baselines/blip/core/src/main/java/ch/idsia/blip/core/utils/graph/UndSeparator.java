package ch.idsia.blip.core.utils.graph;


import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.data.set.TIntSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class UndSeparator {

    public static List<Und> go(Und u) {
        List<Und> l_U = new ArrayList<Und>();
        List<TIntSet> sep = getSeparated(u);

        for (TIntSet s : sep) {
            l_U.add(getSubUnd(u, s));
        }
        return l_U;
    }

    private static Und getSubUnd(Und u, TIntSet s) {
        int[] ar = s.toArray();

        Arrays.sort(ar);
        Und n_u = new Und(ar.length);

        n_u.names = new String[ar.length];
        for (int i = 0; i < ar.length; i++) {
            int a = ar[i];

            int[] ps = u.neigh[a];

            for (int j = 0; j < ps.length; j++) {
                int n_j = Arrays.binarySearch(ar, ps[j]);

                if (n_j > i) {
                    n_u.mark(i, n_j);
                }
            }

            n_u.names[i] = u.names[a];
        }

        return n_u;
    }

    private static List<TIntSet> getSeparated(Und u) {

        List<TIntSet> seps = new ArrayList<TIntSet>();

        // variabiles to process
        TIntSet proc = new TIntHashSet();

        for (int i = 0; i < u.n; i++) {
            proc.add(i);
        }

        while (!proc.isEmpty()) {

            // take first
            int v = pop(proc);

            // new set
            TIntHashSet n_set = new TIntHashSet();

            n_set.add(v);

            TIntSet todo = new TIntHashSet();

            todo.add(v);

            while (!todo.isEmpty()) {

                int t = pop(todo);

                proc.remove(t);

                for (int p : u.neigh[t]) {
                    if (!n_set.contains(p)) {
                        n_set.add(p);
                        todo.add(p);
                    }
                }

            }

            seps.add(n_set);
        }

        return seps;
    }

    private static int pop(TIntSet proc) {
        int v = proc.iterator().next();

        proc.remove(v);
        return v;
    }
}
