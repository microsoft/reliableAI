package ch.idsia.blip.core.inference.sample;


import java.util.Arrays;


public class MpeSol implements Comparable<MpeSol> {

    public final short[] set;

    public MpeSol(short[] p) {
        set = new short[p.length];
        System.arraycopy(p, 0, set, 0, p.length);
    }

    public MpeSol() {
        this(new short[0]);
    }

    public MpeSol(short n) {
        this(new short[] { n});
    }

    @Override
    public int compareTo(MpeSol o) {
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
        if (!(other instanceof MpeSol)) {
            return false;
        }
        MpeSol o = (MpeSol) other;

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
