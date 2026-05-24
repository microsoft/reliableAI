package ch.idsia.blip.core.utils.data;


import java.util.Arrays;


public class SShortSet implements Comparable<SShortSet> {

    public final short[] set;

    public SShortSet(short[] p) {
        set = p;
        if (p != null) {
            Arrays.sort(set);
        }
    }

    public SShortSet() {
        this(new short[0]);
    }

    public SShortSet(short n) {
        this(new short[] { n});
    }

    @Override
    public int compareTo(SShortSet o) {
        if (o.set.length < set.length) {
            return -1;
        } else if (o.set.length > set.length) {
            return 1;
        }

        for (int i = 0; i < set.length; i++) {
            if (o.set[i] < set[i]) {
                return -1;
            } else if (o.set[i] > set[i]) {
                return 1;
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
        if (!(other instanceof SShortSet)) {
            return false;
        }
        SShortSet o = (SShortSet) other;

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
