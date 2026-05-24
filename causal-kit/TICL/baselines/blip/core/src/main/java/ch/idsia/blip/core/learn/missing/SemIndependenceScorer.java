package ch.idsia.blip.core.learn.missing;


import ch.idsia.blip.core.learn.scorer.IndependenceScorer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;


public class SemIndependenceScorer extends IndependenceScorer {

    public List<Map<int[], Double>> scores;

    private int n_true;

    @Override
    public void prepare() {
        super.prepare();

        if (scores == null) {
            scores = new ArrayList<Map<int[], Double>>();
            for (int i = 0; i < n_var; i++) {
                scores.add(null);
            }
        }
    }

    @Override
    public IndependenceScorer.IndependenceSearcher getNewSearcher(int n) {
        return new SemIndependenceSearcher(n);
    }

    private void saveScores(int n, Map<int[], Double> sc) {
        synchronized (lock) {
            while (scores.size() <= n) {
                scores.add(null);
            }
            scores.set(n, sc);
        }
    }

    private double retrieveScore(int n, int[] s) {

        synchronized (lock) {
            if (scores.get(n).containsKey(s)) {
                return scores.get(n).get(s);
            }
        }
        return 0;
    }

    public void setScores(List<Map<int[], Double>> sc, int n_true) {
        this.scores = sc;
        this.n_true = n_true;
    }

    public class SemIndependenceSearcher extends IndependenceSearcher {

        public SemIndependenceSearcher(int in_n) {
            super(in_n);
        }

        @Override
        public void conclude() {
            saveScores(n, scores);
            super.conclude();
        }

        @Override
        protected double getScore(int[] set_p, int[][] p_values) {
            double s = 0;

            // Check if it does not contains hidden variables, in that case
            // simply retrieve old value
            if (n < n_true && boring(set_p)) {
                s = retrieveScore(n, set_p);
            }

            if (s != 0) {
                return s;
            } else {
                return score.computeScore(n, set_p, p_values);
            }
        }

        private boolean boring(int[] set_p) {
            for (int p : set_p) {
                if (p >= n_true) {
                    return false;
                }
            }
            return true;
        }
    }

}
