package ch.idsia.blip.core.utils.graph;


import ch.idsia.blip.core.utils.arcs.Undirected;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.data.set.TIntSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


public class UndirectedSeparator {

    public static List<Undirected> go(Undirected u) {
        List<Undirected> l_U = new ArrayList<Undirected>();
        List<TIntSet> sep = getSeparated(u);

        for (TIntSet s : sep) {
            l_U.add(getSubUndirected(u, s));
        }
        return l_U;
    }

    private static Undirected getSubUndirected(Undirected u, TIntSet s) {
        int[] ar = s.toArray();

        Arrays.sort(ar);
        Undirected n_u = new Undirected(ar.length);

        for (int i = 0; i < ar.length; i++) {
            int a = ar[i];

            int[] ps = u.neighbours(a);
            int[] n_ps = new int[ps.length];

            for (int j = 0; j < ps.length; j++) {
                n_u.mark(i, Arrays.binarySearch(ar, ps[j]));
            }

            n_u.names[i] = u.names[a];
        }

        return n_u;
    }

    private static List<TIntSet> getSeparated(Undirected u) {

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

                for (int p : u.neighbours(t)) {
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
