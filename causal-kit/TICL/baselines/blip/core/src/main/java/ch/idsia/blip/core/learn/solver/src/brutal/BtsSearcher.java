package ch.idsia.blip.core.learn.solver.src.brutal;


import ch.idsia.blip.core.utils.arcs.Und;
import ch.idsia.blip.core.learn.solver.brtl.BrutalUndirectedSolver;
import ch.idsia.blip.core.utils.data.ArrayUtils;
import ch.idsia.blip.core.utils.data.SIntSet;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class BtsSearcher extends BrutalMaxUndirectedSearcher {


    private int counter;

    public BtsSearcher(BrutalUndirectedSolver solver, int tw, Und und) {
        super(solver, tw, und);
        counter = 0;
    }

    @Override
    public ParentSet[] search() {
        super.search();

        counter++;

        return null;
    }

    protected Result sampleWeighted() {

        int sel = -1;
        double max = -1;

        TIntIterator it = todo.iterator();

        while (it.hasNext()) {
            int v = it.next();

            double sk = bests[v].neigh.length;
            /*
            if (counter < 100)
                sk *= (1 + solver.randDouble()/(100-counter));
            else
                sk *= (1 + solver.randDouble());
                */

            if (sk > max) {
                sel = v;
                max = sk;
            }
        }

        return bests[sel];
    }

    protected void updateBests(ArrayList<SIntSet> l_a) {
        // Update best handlers
        TIntIterator it = todo.iterator();

        while (it.hasNext()) {
            int v = it.next();

            int sk = bests[v].neigh.length;

            for (SIntSet h : l_a) {
                int[] neigh = ArrayUtils.intersect(und.neigh[v], h.set);

                if (neigh.length > sk) {

                    Result c = new Result(v, neigh, h);

                    bests[v] = c;
                    sk = bests[v].neigh.length;

                }
            }
        }
    }
}
