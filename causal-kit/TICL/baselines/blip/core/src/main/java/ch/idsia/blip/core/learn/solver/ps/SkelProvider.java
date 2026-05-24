package ch.idsia.blip.core.learn.solver.ps;


import ch.idsia.blip.core.utils.arcs.Directed;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.ParentSet;
import ch.idsia.blip.core.utils.other.SubsetIterator;

import java.util.Iterator;
import java.util.TreeSet;


// Build parent set based on skeleton
public class SkelProvider implements Provider {

    private final Directed skel;
    private ParentSet[][] m_scores;

    public SkelProvider(Directed skel) {
        this.skel = skel;
    }

    @Override
    public ParentSet[][] getParentSets() {
        if (m_scores == null) {
            computePS();
        }
        return m_scores;
    }

    private void computePS() {
        m_scores = new ParentSet[skel.n][];
        for (int i = 0; i < skel.n; i++) {
            TreeSet<ParentSet> ps = new TreeSet<ParentSet>();

            TIntArrayList parents = new TIntArrayList();

            for (int j = 0; j < skel.n; j++) {
                if (skel.check(j, i)) {
                    parents.add(j);
                }
            }

            SubsetIterator si = new SubsetIterator(parents.toArray());

            while (si.hasNext()) {
                int[] p = si.next();

                ps.add(new ParentSet(p.length, p));
            }

            ParentSet[] s = new ParentSet[ps.size()];
            Iterator<ParentSet> it = ps.iterator();
            int b = 0;

            while (it.hasNext()) {
                s[b++] = it.next();
            }

            m_scores[i] = s;
        }
    }
}
