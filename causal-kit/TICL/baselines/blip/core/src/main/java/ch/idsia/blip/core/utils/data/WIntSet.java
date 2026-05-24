package ch.idsia.blip.core.utils.data;


import java.util.Arrays;


public class WIntSet implements Comparable<WIntSet> {

    public final int[] set;

    public WIntSet(int[] p) {
        set = p;
    }

    @Override
    public int compareTo(WIntSet o) {
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
        if (!(other instanceof WIntSet)) {
            return false;
        }
        WIntSet o = (WIntSet) other;

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
