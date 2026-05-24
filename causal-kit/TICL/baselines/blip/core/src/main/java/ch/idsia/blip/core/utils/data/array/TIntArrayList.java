package ch.idsia.blip.core.utils.data.array;


import ch.idsia.blip.core.utils.data.Constants;
import ch.idsia.blip.core.utils.data.HashFunctions;
import ch.idsia.blip.core.utils.data.common.TIntCollection;
import ch.idsia.blip.core.utils.data.common.TIntIterator;

import java.util.Arrays;
import java.util.Collection;
import java.util.Random;


public class TIntArrayList implements TIntList {

    static final long serialVersionUID = 1L;

    /**
     * the data of the list
     */
    public int[] _d;

    /**
     * the index after the last entry in the list
     */
    protected int _pos;

    /**
     * the default capacity for new lists
     */
    protected static final int DEFAULT_CAPACITY = Constants.DEFAULT_CAPACITY;

    /**
     * the int value that represents null
     */
    protected int no_entry_value;

    /**
     * Creates a new <code>SIntArray</code> instance with the
     * default capacity.
     */
    public TIntArrayList() {
        this(DEFAULT_CAPACITY, (int) 0);
    }

    /**
     * Creates a new <code>SIntArray</code> instance with the
     * specified capacity.
     *
     * @param capacity an <code>int</code> value
     */
    public TIntArrayList(int capacity) {
        this(capacity, (int) 0);
    }

    /**
     * Creates a new <code>SIntArray</code> instance with the
     * specified capacity.
     *
     * @param capacity       an <code>int</code> value
     * @param no_entry_value an <code>int</code> value that represents null.
     */
    public TIntArrayList(int capacity, int no_entry_value) {
        _d = new int[capacity];
        _pos = 0;
        this.no_entry_value = no_entry_value;
    }

    /**
     * Creates a new <code>SIntArray</code> instance that contains
     * a copy of the collection passed to us.
     *
     * @param collection the collection to copy
     */
    public TIntArrayList(TIntCollection collection) {
        this(collection.size());
        addAll(collection);
    }

    /**
     * Creates a new <code>SIntArray</code> instance whose
     * capacity is the length of <tt>values</tt> array and whose
     * initial contents are the specified values.
     * <p>
     * A defensive copy of the given values is held by the new instance.
     *
     * @param values an <code>int[]</code> value
     */
    public TIntArrayList(int[] values) {
        this(values.length);
        add(values);
    }

    protected TIntArrayList(int[] values, int no_entry_value, boolean wrap) {
        if (!wrap) {
            throw new IllegalStateException("Wrong call");
        }

        if (values == null) {
            throw new IllegalArgumentException("values can not be null");
        }

        _d = values;
        _pos = values.length;
        this.no_entry_value = no_entry_value;
    }

    /**
     * Returns a primitive List implementation that wraps around the given primitive array.
     * <p/>
     * NOTE: mutating operation are allowed as long as the List does not grow. In that case
     * an IllegalStateException will be thrown
     *
     * @param values
     * @return
     */
    public static TIntArrayList wrap(int[] values) {
        return wrap(values, (int) 0);
    }

    /**
     * Returns a primitive List implementation that wraps around the given primitive array.
     * <p/>
     * NOTE: mutating operation are allowed as long as the List does not grow. In that case
     * an IllegalStateException will be thrown
     *
     * @param values
     * @param no_entry_value
     * @return
     */
    public static TIntArrayList wrap(int[] values, int no_entry_value) {
        return new TIntArrayList(values, no_entry_value, true) {

            /**
             * Growing the wrapped external array is not allow
             */
            @Override
            public void ensureCapacity(int capacity) {
                if (capacity > _d.length) {
                    throw new IllegalStateException(
                            "Can not grow ArrayList wrapped external array");
                }
            }
        };
    }

    @Override
    public int getNoEntryValue() {
        return no_entry_value;
    }

    // sizing

    /**
     * Grow the internal array as needed to accommodate the specified number of elements.
     * The size of the array bytes on each resize unless capacity requires more than twice
     * the current capacity.
     */
    public void ensureCapacity(int capacity) {
        if (capacity > _d.length) {
            int newCap = Math.max(_d.length << 1, capacity);
            int[] tmp = new int[newCap];

            System.arraycopy(_d, 0, tmp, 0, _d.length);
            _d = tmp;
        }
    }

    @Override
    public int size() {
        return _pos;
    }

    @Override
    public boolean isEmpty() {
        return _pos == 0;
    }

    /**
     * Sheds any excess capacity above and beyond the current size of the list.
     */
    public void trimToSize() {
        if (_d.length > size()) {
            int[] tmp = new int[size()];

            toArray(tmp, 0, tmp.length);
            _d = tmp;
        }
    }

    // modifying


    @Override
    public boolean add(int val) {
        ensureCapacity(_pos + 1);
        _d[_pos++] = val;
        return true;
    }

    @Override
    public void add(int[] vals) {
        add(vals, 0, vals.length);
    }

    @Override
    public void add(int[] vals, int offset, int length) {
        ensureCapacity(_pos + length);
        System.arraycopy(vals, offset, _d, _pos, length);
        _pos += length;
    }

    @Override
    public void insert(int offset, int value) {
        if (offset == _pos) {
            add(value);
            return;
        }
        ensureCapacity(_pos + 1);
        // shift right
        System.arraycopy(_d, offset, _d, offset + 1, _pos - offset);
        // insert
        _d[offset] = value;
        _pos++;
    }

    @Override
    public void insert(int offset, int[] values) {
        insert(offset, values, 0, values.length);
    }

    @Override
    public void insert(int offset, int[] values, int valOffset, int len) {
        if (offset == _pos) {
            add(values, valOffset, len);
            return;
        }

        ensureCapacity(_pos + len);
        // shift right
        System.arraycopy(_d, offset, _d, offset + len, _pos - offset);
        // insert
        System.arraycopy(values, valOffset, _d, offset, len);
        _pos += len;
    }

    @Override
    public int get(int offset) {
        if (offset >= _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }
        return _d[offset];
    }

    /**
     * Returns the value at the specified offset without doing any bounds checking.
     */
    public int getQuick(int offset) {
        return _d[offset];
    }

    @Override
    public int set(int offset, int val) {
        if (offset >= _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }

        int prev_val = _d[offset];

        _d[offset] = val;
        return prev_val;
    }

    @Override
    public int replace(int offset, int val) {
        if (offset >= _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }
        int old = _d[offset];

        _d[offset] = val;
        return old;
    }

    @Override
    public void set(int offset, int[] values) {
        set(offset, values, 0, values.length);
    }

    @Override
    public void set(int offset, int[] values, int valOffset, int length) {
        if (offset < 0 || offset + length > _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }
        System.arraycopy(values, valOffset, _d, offset, length);
    }

    /**
     * Sets the value at the specified offset without doing any bounds checking.
     */
    public void setQuick(int offset, int val) {
        _d[offset] = val;
    }

    @Override
    public void clear() {
        clear(DEFAULT_CAPACITY);
    }

    /**
     * Flushes the internal state of the list, setting the capacity of the empty list to
     * <tt>capacity</tt>.
     */
    public void clear(int capacity) {
        _d = new int[capacity];
        _pos = 0;
    }

    /**
     * Sets the size of the list to 0, but does not change its capacity. This method can
     * be used as an alternative to the {@link #clear()} method if you want to recycle a
     * list without allocating new backing arrays.
     */
    public void reset() {
        _pos = 0;
        Arrays.fill(_d, no_entry_value);
    }

    /**
     * Sets the size of the list to 0, but does not change its capacity. This method can
     * be used as an alternative to the {@link #clear()} method if you want to recycle a
     * list without allocating new backing arrays. This method differs from
     * {@link #reset()} in that it does not clear the old values in the backing array.
     * Thus, it is possible for getQuick to return stale data if this method is used and
     * the caller is careless about bounds checking.
     */
    public void resetQuick() {
        _pos = 0;
    }

    @Override
    public boolean remove(int value) {
        for (int index = 0; index < _pos; index++) {
            if (value == _d[index]) {
                remove(index, 1);
                return true;
            }
        }
        return false;
    }

    @Override
    public int removeAt(int offset) {
        int old = get(offset);

        remove(offset, 1);
        return old;
    }

    @Override
    public void remove(int offset, int length) {
        if (length == 0) {
            return;
        }
        if (offset < 0 || offset >= _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }

        if (offset == 0) {
            // data at the front
            System.arraycopy(_d, length, _d, 0, _pos - length);
        } else if (_pos - length == offset) {// no copy to make, decrementing pos "deletes" values at
            // the end
        } else {
            // data in the middle
            System.arraycopy(_d, offset + length, _d, offset,
                    _pos - (offset + length));
        }
        _pos -= length;
        // no need to clear old values beyond _pos, because this is a
        // primitive collection and 0 takes as much room as any other
        // value
    }

    @Override
    public TIntIterator iterator() {
        return new SIntArrayIterator(0);
    }

    @Override
    public boolean containsAll(Collection<?> collection) {
        for (Object element : collection) {
            if (element instanceof Integer) {
                int c = ((Integer) element).intValue();

                if (!contains(c)) {
                    return false;
                }
            } else {
                return false;
            }

        }
        return true;
    }

    @Override
    public boolean containsAll(TIntCollection collection) {
        if (this == collection) {
            return true;
        }
        TIntIterator iter = collection.iterator();

        while (iter.hasNext()) {
            int element = iter.next();

            if (!contains(element)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean containsAll(int[] array) {
        for (int i = array.length; i-- > 0;) {
            if (!contains(array[i])) {
                return false;
            }
        }
        return true;
    }

    @Override
    public boolean addAll(Collection<? extends Integer> collection) {
        boolean changed = false;

        for (Integer element : collection) {
            int e = element.intValue();

            if (add(e)) {
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public boolean addAll(TIntCollection collection) {
        boolean changed = false;
        TIntIterator iter = collection.iterator();

        while (iter.hasNext()) {
            int element = iter.next();

            if (add(element)) {
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public boolean addAll(int[] array) {
        boolean changed = false;

        for (int element : array) {
            if (add(element)) {
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public boolean retainAll(Collection<?> collection) {
        boolean modified = false;
        TIntIterator iter = iterator();

        while (iter.hasNext()) {
            if (!collection.contains(Integer.valueOf(iter.next()))) {
                iter.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public boolean retainAll(TIntCollection collection) {
        if (this == collection) {
            return false;
        }
        boolean modified = false;
        TIntIterator iter = iterator();

        while (iter.hasNext()) {
            if (!collection.contains(iter.next())) {
                iter.remove();
                modified = true;
            }
        }
        return modified;
    }

    @Override
    public boolean retainAll(int[] array) {
        boolean changed = false;

        Arrays.sort(array);
        int[] data = _d;

        for (int i = _pos; i-- > 0;) {
            if (Arrays.binarySearch(array, data[i]) < 0) {
                remove(i, 1);
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public boolean removeAll(Collection<?> collection) {
        boolean changed = false;

        for (Object element : collection) {
            if (element instanceof Integer) {
                int c = ((Integer) element).intValue();

                if (remove(c)) {
                    changed = true;
                }
            }
        }
        return changed;
    }

    @Override
    public boolean removeAll(TIntCollection collection) {
        if (collection == this) {
            clear();
            return true;
        }
        boolean changed = false;
        TIntIterator iter = collection.iterator();

        while (iter.hasNext()) {
            int element = iter.next();

            if (remove(element)) {
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public boolean removeAll(int[] array) {
        boolean changed = false;

        for (int i = array.length; i-- > 0;) {
            if (remove(array[i])) {
                changed = true;
            }
        }
        return changed;
    }

    @Override
    public void reverse() {
        reverse(0, _pos);
    }

    @Override
    public void reverse(int from, int to) {
        if (from == to) {
            return; // nothing to do
        }
        if (from > to) {
            throw new IllegalArgumentException("from cannot be greater than to");
        }
        for (int i = from, j = to - 1; i < j; i++, j--) {
            swap(i, j);
        }
    }

    /**
     * Swap the values at offsets <tt>thread</tt> and <tt>j</tt>.
     *
     * @param i an offset into the data array
     * @param j an offset into the data array
     */
    private void swap(int i, int j) {
        int tmp = _d[i];

        _d[i] = _d[j];
        _d[j] = tmp;
    }

    // copying


    @Override
    public TIntList subList(int begin, int end) {
        if (end < begin) {
            throw new IllegalArgumentException(
                    "end index " + end + " greater than begin index " + begin);
        }
        if (begin < 0) {
            throw new IndexOutOfBoundsException("begin index can not be < 0");
        }
        if (end > _d.length) {
            throw new IndexOutOfBoundsException("end index < " + _d.length);
        }
        TIntArrayList list = new TIntArrayList(end - begin);

        for (int i = begin; i < end; i++) {
            list.add(_d[i]);
        }
        return list;
    }

    @Override
    public int[] toArray() {
        return toArray(0, _pos);
    }

    @Override
    public int[] toArray(int offset, int len) {
        int[] rv = new int[len];

        toArray(rv, offset, len);
        return rv;
    }

    @Override
    public int[] toArray(int[] dest) {
        int len = dest.length;

        if (dest.length > _pos) {
            len = _pos;
            dest[len] = no_entry_value;
        }
        toArray(dest, 0, len);
        return dest;
    }

    @Override
    public int[] toArray(int[] dest, int offset, int len) {
        if (len == 0) {
            return dest; // nothing to copy
        }
        if (offset < 0 || offset >= _pos) {
            throw new ArrayIndexOutOfBoundsException(offset);
        }
        System.arraycopy(_d, offset, dest, 0, len);
        return dest;
    }

    @Override
    public int[] toArray(int[] dest, int source_pos, int dest_pos, int len) {
        if (len == 0) {
            return dest; // nothing to copy
        }
        if (source_pos < 0 || source_pos >= _pos) {
            throw new ArrayIndexOutOfBoundsException(source_pos);
        }
        System.arraycopy(_d, source_pos, dest, dest_pos, len);
        return dest;
    }

    // comparing


    @Override
    public boolean equals(Object other) {
        if (other == this) {
            return true;
        } else if (other instanceof TIntArrayList) {
            TIntArrayList that = (TIntArrayList) other;

            if (that.size() != this.size()) {
                return false;
            } else {
                for (int i = _pos; i-- > 0;) {
                    if (this._d[i] != that._d[i]) {
                        return false;
                    }
                }
                return true;
            }
        } else {
            return false;
        }
    }

    @Override
    public int hashCode() {
        int h = 0;

        for (int i = _pos; i-- > 0;) {
            h += HashFunctions.hash(_d[i]);
        }
        return h;
    }

    // sorting


    @Override
    public void sort() {
        Arrays.sort(_d, 0, _pos);
    }

    @Override
    public void sort(int fromIndex, int toIndex) {
        Arrays.sort(_d, fromIndex, toIndex);
    }

    // filling


    @Override
    public void fill(int val) {
        Arrays.fill(_d, 0, _pos, val);
    }

    @Override
    public void fill(int fromIndex, int toIndex, int val) {
        if (toIndex > _pos) {
            ensureCapacity(toIndex);
            _pos = toIndex;
        }
        Arrays.fill(_d, fromIndex, toIndex, val);
    }

    // searching


    @Override
    public int binarySearch(int value) {
        return binarySearch(value, 0, _pos);
    }

    @Override
    public int binarySearch(int value, int fromIndex, int toIndex) {
        if (fromIndex < 0) {
            throw new ArrayIndexOutOfBoundsException(fromIndex);
        }
        if (toIndex > _pos) {
            throw new ArrayIndexOutOfBoundsException(toIndex);
        }

        int low = fromIndex;
        int high = toIndex - 1;

        while (low <= high) {
            int mid = (low + high) >>> 1;
            int midVal = _d[mid];

            if (midVal < value) {
                low = mid + 1;
            } else if (midVal > value) {
                high = mid - 1;
            } else {
                return mid; // value found
            }
        }
        return -(low + 1); // value not found.
    }

    @Override
    public int indexOf(int value) {
        return indexOf(0, value);
    }

    @Override
    public int indexOf(int offset, int value) {
        for (int i = offset; i < _pos; i++) {
            if (_d[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public int lastIndexOf(int value) {
        return lastIndexOf(_pos, value);
    }

    @Override
    public int lastIndexOf(int offset, int value) {
        for (int i = offset; i-- > 0;) {
            if (_d[i] == value) {
                return i;
            }
        }
        return -1;
    }

    @Override
    public boolean contains(int value) {
        return lastIndexOf(value) >= 0;
    }

    @Override
    public int max() {
        if (size() == 0) {
            throw new IllegalStateException(
                    "cannot find maximum of an empty list");
        }
        int max = Integer.MIN_VALUE;

        for (int i = 0; i < _pos; i++) {
            if (_d[i] > max) {
                max = _d[i];
            }
        }
        return max;
    }

    @Override
    public int min() {
        if (size() == 0) {
            throw new IllegalStateException(
                    "cannot find minimum of an empty list");
        }
        int min = Integer.MAX_VALUE;

        for (int i = 0; i < _pos; i++) {
            if (_d[i] < min) {
                min = _d[i];
            }
        }
        return min;
    }

    @Override
    public int sum() {
        int sum = 0;

        for (int i = 0; i < _pos; i++) {
            sum += _d[i];
        }
        return sum;
    }

    // stringification


    @Override
    public String toString() {
        final StringBuilder buf = new StringBuilder("{");

        for (int i = 0, end = _pos - 1; i < end; i++) {
            buf.append(_d[i]);
            buf.append(", ");
        }
        if (size() > 0) {
            buf.append(_d[_pos - 1]);
        }
        buf.append("}");
        return buf.toString();
    }

    public void copy(int[] nv) {
        System.arraycopy(_d, 0, nv, 0, nv.length);
    }

    public int last() {
        return _d[_pos - 1];
    }

    /**
     * SIntArray iterator
     */
    class SIntArrayIterator implements TIntIterator {

        /**
         * Index of element to be returned by subsequent call to next.
         */
        private int cursor = 0;

        /**
         * Index of element returned by most recent call to next or
         * previous.  Reset to -1 if this element is deleted by a call
         * to remove.
         */
        int lastRet = -1;

        SIntArrayIterator(int index) {
            cursor = index;
        }

        @Override
        public boolean hasNext() {
            return cursor < size();
        }

        @Override
        public int next() {
            int next = get(cursor);

            lastRet = cursor++;
            return next;
        }

        @Override
        public void remove() {
            if (lastRet == -1) {
                throw new IllegalStateException();
            }

            TIntArrayList.this.remove(lastRet, 1);
            if (lastRet < cursor) {
                cursor--;
            }
            lastRet = -1;

        }
    }

    public int[] data() {
        return _d;
    }

    @Override
    public void shuffle(Random rand) {
        for (int i = _pos; i-- > 1;) {
            swap(i, rand.nextInt(i));
        }
    }
}
