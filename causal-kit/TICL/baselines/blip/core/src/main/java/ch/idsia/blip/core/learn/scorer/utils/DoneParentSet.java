package ch.idsia.blip.core.learn.scorer.utils;


import java.util.Arrays;


/**
 * Entry of a parent set in the linked-list queue.
 */
class DoneParentSet implements Comparable<DoneParentSet> {

    /**
     * Score of the parent set.
     */
    private final double sk;

    /**
     * Parent set
     */
    private int[] pset;

    /**
     * Default constructor.
     *
     * @param sk score of the parent set
     */
    public DoneParentSet(double sk) {
        this.pset = pset.clone();
        this.sk = sk;
    }

    @Override
    public int compareTo(DoneParentSet o) {

        if (o.pset.length < pset.length) {
            return -1;
        } else if (o.pset.length > pset.length) {
            return 1;
        }

        for (int i = 0; i < pset.length; i++) {
            if (o.pset[i] < pset[i]) {
                return -1;
            } else if (o.pset[i] > pset[i]) {
                return 1;
            }
        }

        return 0;
    }

    public String toString() {
        return String.format("(%s %.3f)", Arrays.toString(pset), sk);
    }

}

