package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;

import static ch.idsia.blip.core.utils.RandomStuff.p;


public class HardJointMissingSEM extends HardMissingSEM {

    protected Runnable getEmSearcher(int t) {
        return new HardEmJointSearcher(t);
    }


    private class HardEmJointSearcher extends  HardEmSearcher {

        public HardEmJointSearcher(int t) {
            super(t);
        }

        @Override
        protected void expect(VariableElimination vEl, int r, TIntArrayList qu, TIntIntHashMap e) {

            int nd = 0;

            TIntIntHashMap res = new TIntIntHashMap();

            while (nd < qu.size()) {

                int l = Math.min(15, qu.size() - nd);
                int[] q = new int[l];
                for (int j = 0; j < l; j++)
                    q[j] = qu.get(nd + j);



                BayesianFactor f = vEl.query(q, e);
                int row = f.mostProbable();
                for (int i = f.dom.length - 1; i >= 0; i--) {
                    int var = f.dom[i];
                    int val = row / f.stride[i];
                    row -= val * f.stride[i];
                    res.put(var, val);
                }

                nd += l;
            }


            addResult(r, res);
        }
    }
}
