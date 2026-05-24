package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.data.array.TIntArrayList;


public class SubsetIterator {

    private final int[] set;

    private final int max;

    private int index;

    public SubsetIterator(int[] originalList) {
        set = originalList;
        max = (1 << set.length);
        index = 0;
    }

    public boolean hasNext() {
        return index < max;
    }

    public int[] next() {

        TIntArrayList newSet = new TIntArrayList();

        int flag = 1;

        for (int e : set) {
            if ((index & flag) != 0) {
                newSet.add(e);
            }

            flag <<= 1;
        }

        ++index;
        return newSet.toArray();

    }

}
