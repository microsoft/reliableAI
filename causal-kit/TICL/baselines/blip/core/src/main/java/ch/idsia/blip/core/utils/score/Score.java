package ch.idsia.blip.core.utils.score;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;
import ch.idsia.blip.core.utils.data.SIntSet;

import java.util.Map;
import java.util.TreeMap;


public abstract class Score extends Analyzer {

    protected TreeMap<SIntSet, Double>[] scores;

    /**
     * Num of evaluations made
     */
    public int numEvaluated = 0;
    public boolean debug = false;

    Score(DataSet dat) {
        super(dat);
    }

    /**
     * Compute the score with no parent set
     *
     * @return computed BDeu score
     */
    public abstract double computeScore(int n);

    /**
     * Compute the score for a parent set
     *
     * @param set_p parent set
     * @return computed BDeu score
     */
    public abstract double computeScore(int n, int[] set_p, int[][] p_values);

    public double computeScore(int n, int p) {
        return computeScore(n, new int[] { p});
    }

    public double computeScore(int n, int[] set_p) {
        if (set_p.length == 0)
            return computeScore(n);

        return computeScore(n, set_p, computeParentSetValues(set_p));
    }

    public abstract String descr();

    /**
     * I'm searching for a parent set with good score for old parent set and new variable,
     * and low score between each member of the old parent set and the new variable.
     *
     * @param n
     * @param p1     old parent set
     * @param p2     new variable to add
     * @param scores
     * @return heuristic skore
     */
    public abstract double computePrediction(int n, int[] p1, int p2, Map<int[], Double> scores);

}
