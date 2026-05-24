package ch.idsia.blip.core.utils.data;


import java.util.Arrays;


public class SIntSet implements Comparable<SIntSet> {

    public final int[] set;

    public SIntSet(int[] p) {
        set = p;
        if (p != null) {
            Arrays.sort(set);
        }
    }

    public SIntSet() {
        this(new int[0]);
    }

    public SIntSet(int n) {
        this(new int[] { n});
    }

    @Override
    public int compareTo(SIntSet o) {
        if (o.set.length < set.length) {
            return 1;
        } else if (o.set.length > set.length) {
            return -1;
        }

        for (int i = 0; i < set.length; i++) {
            if (o.set[i] < set[i]) {
                return 1;
            } else if (o.set[i] > set[i]) {
                return -1;
            }
        }

        return 0;
    }

    @Override
    public boolean equals(Object other) {
        if (other == null) {
            return false;
        }
        if (other == this) {
            return true;
        }
        if (!(other instanceof SIntSet)) {
            return false;
        }
        SIntSet o = (SIntSet) other;

        return compareTo(o) == 0;
    }

    public int hashCode() {
        return Arrays.hashCode(set);
    }

    @Override
    public String toString() {
        return Arrays.toString(set);
    }
}
