package ch.idsia.blip.core.utils.data.array;


import ch.idsia.blip.core.utils.data.common.TIntCollection;

import java.util.Random;


/**
 * Interface for Trove list implementations.
 */
public interface TIntList extends TIntCollection {

    /**
     * Returns the value that is used to represent null. The default
     * value is generally zero, but can be changed during construction
     * of the collection.
     *
     * @return the value that represents null
     */
    int getNoEntryValue();

    /**
     * Returns the number of values in the list.
     *
     * @return the number of values in the list.
     */
    int size();

    /**
     * Tests whether this list contains any values.
     *
     * @return true if the list is empty.
     */
    boolean isEmpty();

    /**
     * Adds <tt>val</tt> to the end of the list, growing as needed.
     *
     * @param val an <code>int</code> value
     * @return true if the list was modified by the add operation
     */
    boolean add(int val);

    /**
     * Adds the values in the array <tt>vals</tt> to the end of the
     * list, in order.
     *
     * @param vals an <code>int[]</code> value
     */
    void add(int[] vals);

    /**
     * Adds a subset of the values in the array <tt>vals</tt> to the
     * end of the list, in order.
     *
     * @param vals   an <code>int[]</code> value
     * @param offset the offset at which to start copying
     * @param length the number of values to copy.
     */
    void add(int[] vals, int offset, int length);

    /**
     * Inserts <tt>value</tt> into the list at <tt>offset</tt>.  All
     * values including and to the right of <tt>offset</tt> are shifted
     * to the right.
     *
     * @param offset an <code>int</code> value
     * @param value  an <code>int</code> value
     */
    void insert(int offset, int value);

    /**
     * Inserts the array of <tt>values</tt> into the list at
     * <tt>offset</tt>.  All values including and to the right of
     * <tt>offset</tt> are shifted to the right.
     *
     * @param offset an <code>int</code> value
     * @param values an <code>int[]</code> value
     */
    void insert(int offset, int[] values);

    /**
     * Inserts a slice of the array of <tt>values</tt> into the list
     * at <tt>offset</tt>.  All values including and to the right of
     * <tt>offset</tt> are shifted to the right.
     *
     * @param offset    an <code>int</code> value
     * @param values    an <code>int[]</code> value
     * @param valOffset the offset in the values array at which to
     *                  start copying.
     * @param len       the number of values to copy from the values array
     */
    void insert(int offset, int[] values, int valOffset, int len);

    /**
     * Returns the value at the specified offset.
     *
     * @param offset an <code>int</code> value
     * @return an <code>int</code> value
     */
    int get(int offset);

    /**
     * Sets the value at the specified offset.
     *
     * @param offset an <code>int</code> value
     * @param val    an <code>int</code> value
     * @return The value previously at the given index.
     */
    int set(int offset, int val);

    /**
     * Replace the values in the list starting at <tt>offset</tt> with
     * the contents of the <tt>values</tt> array.
     *
     * @param offset the first offset to replace
     * @param values the source of the new values
     */
    void set(int offset, int[] values);

    /**
     * Replace the values in the list starting at <tt>offset</tt> with
     * <tt>length</tt> values from the <tt>values</tt> array, starting
     * at valOffset.
     *
     * @param offset    the first offset to replace
     * @param values    the source of the new values
     * @param valOffset the first value to copy from the values array
     * @param length    the number of values to copy
     */
    void set(int offset, int[] values, int valOffset, int length);

    /**
     * Sets the value at the specified offset and returns the
     * previously stored value.
     *
     * @param offset an <code>int</code> value
     * @param val    an <code>int</code> value
     * @return the value previously stored at offset.
     */
    int replace(int offset, int val);

    /**
     * Flushes the internal state of the list, resetting the capacity
     * to the default.
     */
    void clear();

    /**
     * Removes <tt>value</tt> from the list.
     *
     * @param value an <code>int</code> value
     * @return true if the list was modified by the remove operation.
     */
    boolean remove(int value);

    /**
     * Removes <tt>value</tt> at a given offset from the list.
     *
     * @param offset an <code>int</code> value that represents
     *               the offset to the element to be removed
     * @return an <tt>int</tt> that is the value removed.
     */
    int removeAt(int offset);

    /**
     * Removes <tt>length</tt> values from the list, starting at
     * <tt>offset</tt>
     *
     * @param offset an <code>int</code> value
     * @param length an <code>int</code> value
     */
    void remove(int offset, int length);

    /**
     * Reverse the order of the elements in the list.
     */
    void reverse();

    /**
     * Reverse the order of the elements in the range of the list.
     *
     * @param from the inclusive index at which to start reversing
     * @param to   the exclusive index at which to stop reversing
     */
    void reverse(int from, int to);

    /**
     * Shuffle the elements of the list using the specified random
     * number generator.
     *
     * @param rand a <code>Random</code> value
     */
    void shuffle(Random rand);

    /**
     * Returns a sublist of this list.
     *
     * @param begin low endpoint (inclusive) of the subList.
     * @param end   high endpoint (exclusive) of the subList.
     * @return sublist of this list from begin, inclusive to end, exclusive.
     * @throws IndexOutOfBoundsException - endpoint out of range
     * @throws IllegalArgumentException  - endpoints out of order (end > begin)
     */
    TIntList subList(int begin, int end);

    /**
     * Copies the contents of the list into a native array.
     *
     * @return an <code>int[]</code> value
     */
    int[] toArray();

    /**
     * Copies a slice of the list into a native array.
     *
     * @param offset the offset at which to start copying
     * @param len    the number of values to copy.
     * @return an <code>int[]</code> value
     */
    int[] toArray(int offset, int len);

    /**
     * Copies a slice of the list into a native array.
     * <p>
     * <p>If the list fits in the specified array with room to spare (thread.e.,
     * the array has more elements than the list), the element in the array
     * immediately following the end of the list is set to
     * <tt>{@link #getNoEntryValue()}</tt>.
     * (This is useful in determining the length of the list <thread>only</thread> if
     * the caller knows that the list does not contain any "null" elements.)
     * <p>
     * <p>NOTE: Trove does not allocate a new array if the array passed in is
     * not large enough to hold all of the data elements.  It will instead fill
     * the array passed in.
     *
     * @param dest the array to copy into.
     * @return the array passed in.
     */
    int[] toArray(int[] dest);

    /**
     * Copies a slice of the list into a native array.
     *
     * @param dest   the array to copy into.
     * @param offset the offset where the first value should be copied
     * @param len    the number of values to copy.
     * @return the array passed in.
     */
    int[] toArray(int[] dest, int offset, int len);

    /**
     * Copies a slice of the list into a native array.
     *
     * @param dest       the array to copy into.
     * @param source_pos the offset of the first value to copy
     * @param dest_pos   the offset where the first value should be copied
     * @param len        the number of values to copy.
     * @return the array passed in.
     */
    int[] toArray(int[] dest, int source_pos, int dest_pos, int len);

    /**
     * Sort the values in the list (ascending) using the Sun quicksort
     * implementation.
     *
     * @see java.util.Arrays#sort
     */
    void sort();

    /**
     * Sort a slice of the list (ascending) using the Sun quicksort
     * implementation.
     *
     * @param fromIndex the index at which to start sorting (inclusive)
     * @param toIndex   the index at which to stop sorting (exclusive)
     * @see java.util.Arrays#sort
     */
    void sort(int fromIndex, int toIndex);

    /**
     * Fills every slot in the list with the specified value.
     *
     * @param val the value to use when filling
     */
    void fill(int val);

    /**
     * Fills a range in the list with the specified value.
     *
     * @param fromIndex the offset at which to start filling (inclusive)
     * @param toIndex   the offset at which to stop filling (exclusive)
     * @param val       the value to use when filling
     */
    void fill(int fromIndex, int toIndex, int val);

    /**
     * Performs a binary search for <tt>value</tt> in the entire list.
     * Note that you <b>must</b> @{link #sort sort} the list before
     * doing a search.
     *
     * @param value the value to search for
     * @return the absolute offset in the list of the value, or its
     * negative insertion point into the sorted list.
     */
    int binarySearch(int value);

    /**
     * Performs a binary search for <tt>value</tt> in the specified
     * range.  Note that you <b>must</b> @{link #sort sort} the list
     * or the range before doing a search.
     *
     * @param value     the value to search for
     * @param fromIndex the lower boundary of the range (inclusive)
     * @param toIndex   the upper boundary of the range (exclusive)
     * @return the absolute offset in the list of the value, or its
     * negative insertion point into the sorted list.
     */
    int binarySearch(int value, int fromIndex, int toIndex);

    /**
     * Searches the list front to back for the index of
     * <tt>value</tt>.
     *
     * @param value an <code>int</code> value
     * @return the first offset of the value, or -1 if it is not in
     * the list.
     * @see #binarySearch for faster searches on sorted lists
     */
    int indexOf(int value);

    /**
     * Searches the list front to back for the index of
     * <tt>value</tt>, starting at <tt>offset</tt>.
     *
     * @param offset the offset at which to start the linear search
     *               (inclusive)
     * @param value  an <code>int</code> value
     * @return the first offset of the value, or -1 if it is not in
     * the list.
     * @see #binarySearch for faster searches on sorted lists
     */
    int indexOf(int offset, int value);

    /**
     * Searches the list back to front for the last index of
     * <tt>value</tt>.
     *
     * @param value an <code>int</code> value
     * @return the last offset of the value, or -1 if it is not in
     * the list.
     * @see #binarySearch for faster searches on sorted lists
     */
    int lastIndexOf(int value);

    /**
     * Searches the list back to front for the last index of
     * <tt>value</tt>, starting at <tt>offset</tt>.
     *
     * @param offset the offset at which to start the linear search
     *               (exclusive)
     * @param value  an <code>int</code> value
     * @return the last offset of the value, or -1 if it is not in
     * the list.
     * @see #binarySearch for faster searches on sorted lists
     */
    int lastIndexOf(int offset, int value);

    /**
     * Searches the list for <tt>value</tt>
     *
     * @param value an <code>int</code> value
     * @return true if value is in the list.
     */
    boolean contains(int value);

    /**
     * Finds the maximum value in the list.
     *
     * @return the largest value in the list.
     * @throws IllegalStateException if the list is empty
     */
    int max();

    /**
     * Finds the minimum value in the list.
     *
     * @return the smallest value in the list.
     * @throws IllegalStateException if the list is empty
     */
    int min();

    /**
     * Calculates the sum of all the values in the list.
     *
     * @return the sum of the values in the list (zero if the list is empty).
     */
    int sum();
}
