package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.score.SoftMissingBDeu;
import ch.idsia.blip.core.inference.ve.BayesianFactor;
import ch.idsia.blip.core.inference.ve.VariableElimination;
import ch.idsia.blip.core.learn.scorer.IndependenceScorer;
import ch.idsia.blip.core.learn.scorer.concurrency.ThreadCompleteListener;
import ch.idsia.blip.core.utils.data.array.TDoubleArrayList;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import ch.idsia.blip.core.utils.data.hash.TIntIntHashMap;


public class SoftMissingSEM extends HardMissingSEM
        implements ThreadCompleteListener {
    protected TDoubleArrayList weights;

    protected void prepExpect() {
        super.prepExpect();

        this.weights.reset();
    }

    protected Runnable getEmSearcher(int t) {
        return new SoftEmSearcher(t);
    }

    protected void prepare() {
        super.prepare();

        this.weights = new TDoubleArrayList();
    }

    protected DataSet setScore(IndependenceScorer is) {
        is.score = new SoftMissingBDeu(this.cleanDat, this.completion,
                this.weights, 1.0D);
        return null;
    }

    private class SoftEmSearcher
            implements Runnable {
        private final int t;

        public SoftEmSearcher(int t) {
            this.t = t;
        }

        public void run() {
            VariableElimination vEl = new VariableElimination(
                    SoftMissingSEM.this.bn, false);

            for (;;) {
                int r = SoftMissingSEM.this.getNextMissingRow();

                if (r == -1) {
                    return;
                }
                short[] sample = SoftMissingSEM.this.samples[r];

                TIntArrayList q = new TIntArrayList();
                TIntIntHashMap e = new TIntIntHashMap();

                for (int n = 0; n < SoftMissingSEM.this.n_var; n++) {
                    if ((n >= sample.length) || (sample[n] == -1)) {
                        q.add(n);
                    } else {
                        e.put(n, sample[n]);
                    }
                }
                SoftMissingSEM.this.addResult(r, vEl.query(q.toArray(), e));
            }
        }
    }

    private void addResult(int r, BayesianFactor f) {
        synchronized (this.lock) {
            int s = this.n_compl;
            int size = 1;

            for (int c : f.card) {
                size *= c;
            }
            int e = s + size;

            for (int v = 0; v < this.dat.n_var; v++) {
                if (v < this.samples[0].length) {
                    int val = this.samples[r][v];

                    if (val >= 0) {
                        for (int j = s; j < e; j++) {
                            this.completion[v][val].add(j);
                        }
                    }
                }
            }
            for (int j = 0; j < f.potent.length; j++) {
                int w = j;

                for (int k = 0; k < f.dom.length; k++) {
                    int n = f.dom[k];

                    int ar = this.dat.l_n_arity[n];
                    int val = w % ar;

                    w /= ar;
                    this.completion[n][val].add(s + j);
                }
                this.weights.add(f.getPotent(j));
            }
            this.n_compl = e;

            this.done += 1;
        }
    }
}
