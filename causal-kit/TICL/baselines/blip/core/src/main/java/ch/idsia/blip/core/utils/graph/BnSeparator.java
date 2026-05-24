package ch.idsia.blip.core.utils.graph;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.data.set.TIntHashSet;
import ch.idsia.blip.core.utils.data.set.TIntSet;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;


/**
 * Separates a BN in subparts
 */
public class BnSeparator {

    public static List<BayesianNetwork> go(BayesianNetwork bn) {
        List<BayesianNetwork> l_bn = new ArrayList<BayesianNetwork>();
        List<TIntSet> sep = getSeparated(bn);

        for (TIntSet s : sep) {
            l_bn.add(getSubBn(bn, s));
        }
        return l_bn;
    }

    private static BayesianNetwork getSubBn(BayesianNetwork bn, TIntSet s) {
        int[] ar = s.toArray();

        Arrays.sort(ar);
        BayesianNetwork n_bn = new BayesianNetwork(ar.length);

        for (int i = 0; i < ar.length; i++) {
            int a = ar[i];

            n_bn.l_nm_var[i] = bn.name(a);
            n_bn.l_values_var[i] = bn.values(a);
            n_bn.l_ar_var[i] = bn.arity(a);

            int[] ps = bn.parents(a);
            int[] n_ps = new int[ps.length];

            for (int j = 0; j < ps.length; j++) {
                n_ps[j] = Arrays.binarySearch(ar, ps[j]);
            }
            n_bn.setParents(i, n_ps);
        }

        return n_bn;
    }

    private static List<TIntSet> getSeparated(BayesianNetwork bn) {

        List<TIntSet> seps = new ArrayList<TIntSet>();

        // variabiles to process
        TIntSet proc = new TIntHashSet();

        for (int i = 0; i < bn.n_var; i++) {
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

                for (int p : bn.parents(t)) {
                    if (!n_set.contains(p)) {
                        n_set.add(p);
                        todo.add(p);
                    }
                }
                for (int c : bn.childrens(t)) {
                    if (!n_set.contains(c)) {
                        n_set.add(c);
                        todo.add(c);
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
