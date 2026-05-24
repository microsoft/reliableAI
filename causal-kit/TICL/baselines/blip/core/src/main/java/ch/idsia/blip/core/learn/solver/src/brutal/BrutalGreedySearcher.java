package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.learn.solver.BaseSolver;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.other.Pair;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.RandomStuff;

import java.util.*;


/**
 * IMPROVES best parent - handle search
 * (may generate memory problem, in that case fall back to old searcher)
 */
public class BrutalGreedySearcher extends BrutalOldSearcher {

    public ArrayList<TreeSet<SIntSet>> assignments;

    public BrutalGreedySearcher(BaseSolver solver, int tw) {
        super(solver, tw);
    }

    @Override
    protected void clear() {
        super.clear();

        assignments = new ArrayList<TreeSet<SIntSet>>(solver.n_var);
        for (int i = 0; i < solver.n_var; i++) {
            assignments.add(new TreeSet<SIntSet>());
        }
    }

    @Override
    protected void addHandler(SIntSet sIntSet) {
        super.addHandler(sIntSet);

        for (int p : sIntSet.set) {
            assignments.get(p).add(sIntSet);
        }
    }

    @Override
    protected Pair<ParentSet, SIntSet> bestHandler(int v) {

        for (ParentSet p : m_scores[v]) {

            // for the empty parent set, choose a random handler set
            if (p.parents.length == 0) {
                // Take random
                SIntSet h = rand(handles);

                return new Pair<ParentSet, SIntSet>(p, h);
            }

            TreeSet<SIntSet> handles = evaluate(p);

            if (handles.size() > 0) {
                SIntSet h = rand(handles);

                return new Pair<ParentSet, SIntSet>(p, h);
            }
        }

        return new Pair<ParentSet, SIntSet>(new ParentSet(), new SIntSet());
    }

    protected TreeSet<SIntSet> evaluate(ParentSet p) {

        // check all the parents in the set have some handles assigned
        boolean stopNow = false;
        HashMap<TreeSet<SIntSet>, Integer> handlesAssigned = new HashMap<TreeSet<SIntSet>, Integer>();
        TreeSet<SIntSet> aux;

        // Make list of handles assigned to parents variables to make intersect
        for (int i = 0; i < p.parents.length && !stopNow; i++) {
            aux = assignments.get(p.parents[i]);
            if (aux.size() == 0) {
                stopNow = true;
            }
            handlesAssigned.put(aux, aux.size());
        }
        if (stopNow) {
            return new TreeSet<SIntSet>();
        }

        // Sort list of handles by value
        List<TreeSet<SIntSet>> handlesNew = RandomStuff.sortByValuesList(
                handlesAssigned);
        Iterator<TreeSet<SIntSet>> it = handlesNew.iterator();

        // Go on and compute the intersection!
        TreeSet<SIntSet> good = new TreeSet<SIntSet>();

        good.addAll(it.next());
        while (it.hasNext()) {
            good.retainAll(it.next());

            if (good.size() == 0) {
                break;
            }
        }

        return good;
    }

    protected SIntSet rand(TreeSet<SIntSet> h) {
        int v = solver.randInt(0, h.size() - 1);
        Iterator<SIntSet> i = h.iterator();

        while (v > 1) {
            i.next();
            v--;
        }
        return i.next();
    }

    public TreeSet<SIntSet> intersect(TreeSet<SIntSet> arr1, TreeSet<SIntSet> arr2) {

        arr1.retainAll(arr2);

        return arr1;

        /*
         Iterator<SIntSet> i = arr1.iterator();
         Iterator<SIntSet> j = arr2.iterator();
         int n1 = arr1.size();
         int n2 = arr2.size();

         TreeSet<SIntSet> aux = new TreeSet<SIntSet>();

         SIntSet a1 = null;
         boolean b1 = true;
         int i1 = -1;
         SIntSet a2 = null;
         boolean b2 = true;
         int i2 = -1;

         while ((i1 < n1) && (i2 < n2)) {

         if (b1) {
         a1 = i.next();
         b1 = false;
         i1++;
         }

         if (b2) {
         a2 = j.next();
         b2 = false;
         i2++;
         }

         int c = a1.compareTo(a2);

         if (c < 0) {
         b1 = true;
         } else if (c > 0) {
         b2 = true;
         } else {

         aux.add(a1);
         b1 = true;
         b2 = true;
         }
         }

         return aux;

         */
    }

}
