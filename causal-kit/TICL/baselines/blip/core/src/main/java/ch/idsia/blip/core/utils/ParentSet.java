package ch.idsia.blip.core.utils;


import java.util.Arrays;
import java.util.BitSet;


/**
 * Entry of a parent score
 */
public class ParentSet implements Comparable<ParentSet> {

    // Indexes of parent variables
    public final int[] parents;

    // Score of the parent set.
    public final double sk;

    public BitSet bs;

    /**
     * Constructor from local score line
     *
     * @param aux splitted line
     */
    public ParentSet(String[] aux) {
        aux[0] = aux[0].replace(",", ".");
        sk = Double.parseDouble(aux[0]);
        int s = Integer.parseInt(aux[1]);

        parents = new int[s];
        for (int i = 0; i < s; i++) {
            parents[i] = Integer.valueOf(aux[i + 2]);
        }
        Arrays.sort(parents);
    }

    public ParentSet(double i, int[] ints) {
        sk = i;
        parents = ints;
        Arrays.sort(parents);
    }

    public ParentSet() {
        this(0, new int[0]);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        ParentSet parentSet = (ParentSet) o;

        return Arrays.equals(parents, parentSet.parents);

    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(parents);
    }

    @Override
    public int compareTo(ParentSet other) {
        if (sk < other.sk) {
            return 1;
        } else if (other.sk < sk) {
            return -1;
        }

        if (parents.length < other.parents.length) {
            return 1;
        } else if (other.parents.length < parents.length) {
            return -1;
        }

        return 0;
    }

    public String toString() {
        return String.format("(%.2f %d %s)", sk, parents.length,
                Arrays.toString(parents));
    }

    /**
     * Print for scores files.
     *
     * @return pretty formatted parent set description
     */
    public String prettyPrint() {

        StringBuilder builder = new StringBuilder();

        builder.append(String.format("%.5f %d ", sk, parents.length));
        for (int v : parents) {
            builder.append(v);
            builder.append(" ");
        }
        return builder.toString();
    }
}
