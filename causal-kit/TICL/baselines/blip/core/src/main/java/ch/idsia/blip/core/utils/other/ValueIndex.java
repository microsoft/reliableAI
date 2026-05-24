package ch.idsia.blip.core.utils.other;


public class ValueIndex implements Comparable<ValueIndex> {
    public final int value;
    public final int index;

    public ValueIndex(int value, int index) {
        this.value = value;
        this.index = index;
    }

    @Override
    public int compareTo(ValueIndex other) {
        // compare on value;
        if (this.value < other.value) {
            return -1;
        } else if (this.value > other.value) {
            return 1;
        } else {
            return 0;
        }
    }
}
