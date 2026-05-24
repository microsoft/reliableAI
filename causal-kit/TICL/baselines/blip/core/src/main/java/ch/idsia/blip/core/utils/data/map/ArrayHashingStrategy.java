package ch.idsia.blip.core.utils.data.map;


import java.util.Arrays;


public class ArrayHashingStrategy implements HashingStrategy<int[]> {

    public int computeHashCode(int[] array) {
        return Arrays.hashCode(array);
    }

    public boolean equals(int[] arr1, int[] arr2) {
        return Arrays.equals(arr1, arr2);
    }
}
