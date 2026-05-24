package ch.idsia.blip.core.learn.scorer.utils;


import java.util.Arrays;

/**
 * Entry of a parent set in the linked-list queue.
 */
public class OpenParentSet implements Comparable<OpenParentSet> {

    // Parent set to evaluate
    public final int[] s;

    // New parent variable added
    public final int new_p;

    // Prediction
    public final double sk;

    // Previous counts
    public final int[][] p_values;

    public OpenParentSet(int[] s, int new_p, double sk, int[][] p_values) {
        this.s = s;
        this.new_p = new_p;
        this.sk = sk;
        this.p_values = p_values;
    }

    public OpenParentSet(int n, int new_p, double sk, int[][] p_values) {
        this(new int[] { n}, new_p, sk, p_values);
    }

    @Override
    public int compareTo(OpenParentSet other) {
        if (sk < other.sk) {
            return 1;
        }
        return -1;
    }

    public String toString() {
     return String.format("(%s %d %.3f)", Arrays.toString(s), new_p, sk);
     }

}

