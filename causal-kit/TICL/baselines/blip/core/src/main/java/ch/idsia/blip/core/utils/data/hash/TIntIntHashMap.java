package ch.idsia.blip.core.utils.data.hash;


import ch.idsia.blip.core.utils.data.HashFunctions;
import ch.idsia.blip.core.utils.data.common.TIntCollection;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.THashPrimitiveIterator;
import ch.idsia.blip.core.utils.data.set.TIntSet;
import ch.idsia.blip.core.utils.data.set.TPrimitiveHash;

import java.util.Arrays;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.Map;

import static ch.idsia.blip.core.utils.RandomStuff.f;


/**
 * An open addressed Map implementation for int keys and int values.
 *
 * @author Eric D. Friedman
 * @author Rob Eden
 * @author Jeff Randall
 * @version $Id: _K__V_HashMap.template,v 1.1.2.16 2010/03/02 04:09:50 robeden Exp $
 */
public class TIntIntHashMap extends TIntIntHash implements TIntIntMap {
    static final long serialVersionUID = 1L;

    /**
     * the values of the map
     */
    private transient int[] _values;

    /**
     * Creates a new <code>TIntIntHashMap</code> instance with the default
     * capacity and load factor.
     */
    public TIntIntHashMap() {
        super();
    }

    /**
     * Creates a new <code>TIntIntHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the default load factor.
     *
     * @param initialCapacity an <code>int</code> value
     */
    public TIntIntHashMap(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Creates a new <code>TIntIntHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the specified load factor.
     *
     * @param initialCapacity an <code>int</code> value
     * @param loadFactor      a <code>float</code> value
     */
    public TIntIntHashMap(int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor);
    }

    /**
     * Creates a new <code>TIntIntHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the specified load factor.
     *
     * @param initialCapacity an <code>int</code> value
     * @param loadFactor      a <code>float</code> value
     * @param noEntryKey      a <code>int</code> value that represents
     *                        <tt>null</tt> for the Key set.
     * @param noEntryValue    a <code>int</code> value that represents
     *                        <tt>null</tt> for the Value set.
     */
    public TIntIntHashMap(int initialCapacity, float loadFactor,
            int noEntryKey, int noEntryValue) {
        super(initialCapacity, loadFactor, noEntryKey, noEntryValue);
    }

    /**
     * Creates a new <code>TIntIntHashMap</code> instance containing
     * all of the entries in the map passed in.
     *
     * @param keys   a <tt>int</tt> array containing the keys for the matching values.
     * @param values a <tt>int</tt> array containing the values.
     */
    public TIntIntHashMap(int[] keys, int[] values) {
        super(Math.max(keys.length, values.length));

        int size = Math.min(keys.length, values.length);

        for (int i = 0; i < size; i++) {
            this.put(keys[i], values[i]);
        }
    }

    /**
     * Creates a new <code>TIntIntHashMap</code> instance containing
     * all of the entries in the map passed in.
     *
     * @param map a <tt>TIntIntMap</tt> that will be duplicated.
     */
    public TIntIntHashMap(TIntIntMap map) {
        super(map.size());
        if (map instanceof TIntIntHashMap) {
            TIntIntHashMap hashmap = (TIntIntHashMap) map;

            this._loadFactor = hashmap._loadFactor;
            this.no_entry_key = hashmap.no_entry_key;
            this.no_entry_value = hashmap.no_entry_value;
            // noinspection RedundantCast
            if (this.no_entry_key != (int) 0) {
                Arrays.fill(_set, this.no_entry_key);
            }
            // noinspection RedundantCast
            if (this.no_entry_value != (int) 0) {
                Arrays.fill(_values, this.no_entry_value);
            }
            setUp((int) Math.ceil(DEFAULT_CAPACITY / _loadFactor));
        }
        putAll(map);
    }

    /**
     * initializes the hashtable to a prime capacity which is at least
     * <tt>initialCapacity + 1</tt>.
     *
     * @param initialCapacity an <code>int</code> value
     * @return the actual capacity chosen
     */
    protected int setUp(int initialCapacity) {
        int capacity;

        capacity = super.setUp(initialCapacity);
        _values = new int[capacity];
        return capacity;
    }

    /**
     * rehashes the map to the new capacity.
     *
     * @param newCapacity an <code>int</code> value
     */
    
    /**
     * {@inheritDoc}
     */
    protected void rehash(int newCapacity) {
        int oldCapacity = _set.length;

        int oldKeys[] = _set;
        int oldVals[] = _values;
        byte oldStates[] = _states;

        _set = new int[newCapacity];
        _values = new int[newCapacity];
        _states = new byte[newCapacity];

        for (int i = oldCapacity; i-- > 0;) {
            if (oldStates[i] == FULL) {
                int o = oldKeys[i];
                int index = insertKey(o);

                _values[index] = oldVals[i];
            }
        }
    }

    /**
     * {@inheritDoc}
     */
    public int put(int key, int value) {
        int index = insertKey(key);

        return doPut(value, index);
    }

    /**
     * {@inheritDoc}
     */
    public int putIfAbsent(int key, int value) {
        int index = insertKey(key);

        if (index < 0) {
            return _values[-index - 1];
        }
        return doPut(value, index);
    }

    private int doPut(int value, int index) {
        int previous = no_entry_value;
        boolean isNewMapping = true;

        if (index < 0) {
            index = -index - 1;
            previous = _values[index];
            isNewMapping = false;
        }
        _values[index] = value;

        if (isNewMapping) {
            postInsertHook(consumeFreeSlot);
        }

        return previous;
    }

    /**
     * {@inheritDoc}
     */
    public void putAll(Map<? extends Integer, ? extends Integer> map) {
        ensureCapacity(map.size());
        // could winasobs this for cases when map instanceof THashMap
        for (Map.Entry<? extends Integer, ? extends Integer> entry : map.entrySet()) {
            this.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * {@inheritDoc}
     */
    public void putAll(TIntIntMap map) {
        ensureCapacity(map.size());
        TIntIntIterator iter = map.iterator();

        while (iter.hasNext()) {
            iter.advance();
            this.put(iter.key(), iter.value());
        }
    }

    /**
     * {@inheritDoc}
     */
    public int get(int key) {
        int index = index(key);

        return index < 0 ? no_entry_value : _values[index];
    }

    /**
     * {@inheritDoc}
     */
    public void clear() {
        super.clear();
        Arrays.fill(_set, 0, _set.length, no_entry_key);
        Arrays.fill(_values, 0, _values.length, no_entry_value);
        Arrays.fill(_states, 0, _states.length, FREE);
    }

    /**
     * {@inheritDoc}
     */
    public boolean isEmpty() {
        return 0 == _size;
    }

    /**
     * {@inheritDoc}
     */
    public int remove(int key) {
        int prev = no_entry_value;
        int index = index(key);

        if (index >= 0) {
            prev = _values[index];
            removeAt(index); // clear key,state; adjust size
        }
        return prev;
    }

    /**
     * {@inheritDoc}
     */
    protected void removeAt(int index) {
        _values[index] = no_entry_value;
        super.removeAt(index); // clear key, state; adjust size
    }

    /**
     * {@inheritDoc}
     */
    public TIntSet keySet() {
        return new TKeyView();
    }

    /**
     * {@inheritDoc}
     */
    public int[] keys() {
        int[] keys = new int[size()];
        int[] k = _set;
        byte[] states = _states;

        for (int i = k.length, j = 0; i-- > 0;) {
            if (states[i] == FULL) {
                keys[j++] = k[i];
            }
        }
        return keys;
    }

    /**
     * {@inheritDoc}
     */
    public int[] keys(int[] array) {
        int size = size();

        if (array.length < size) {
            array = new int[size];
        }

        int[] keys = _set;
        byte[] states = _states;

        for (int i = keys.length, j = 0; i-- > 0;) {
            if (states[i] == FULL) {
                array[j++] = keys[i];
            }
        }
        return array;
    }

    /**
     * {@inheritDoc}
     */
    public TIntCollection valueCollection() {
        return new TValueView();
    }

    /**
     * {@inheritDoc}
     */
    public int[] values() {
        int[] vals = new int[size()];
        int[] v = _values;
        byte[] states = _states;

        for (int i = v.length, j = 0; i-- > 0;) {
            if (states[i] == FULL) {
                vals[j++] = v[i];
            }
        }
        return vals;
    }

    /**
     * {@inheritDoc}
     */
    public int[] values(int[] array) {
        int size = size();

        if (array.length < size) {
            array = new int[size];
        }

        int[] v = _values;
        byte[] states = _states;

        for (int i = v.length, j = 0; i-- > 0;) {
            if (states[i] == FULL) {
                array[j++] = v[i];
            }
        }
        return array;
    }

    /**
     * {@inheritDoc}
     */
    public boolean containsValue(int val) {
        byte[] states = _states;
        int[] vals = _values;

        for (int i = vals.length; i-- > 0;) {
            if (states[i] == FULL && val == vals[i]) {
                return true;
            }
        }
        return false;
    }

    /**
     * {@inheritDoc}
     */
    public boolean containsKey(int key) {
        return contains(key);
    }

    /**
     * {@inheritDoc}
     */
    public TIntIntIterator iterator() {
        return new TIntIntHashIterator(this);
    }

    /**
     * {@inheritDoc}
     */
    public boolean increment(int key) {
        return adjustValue(key, (int) 1);
    }

    /**
     * {@inheritDoc}
     */
    public boolean adjustValue(int key, int amount) {
        int index = index(key);

        if (index < 0) {
            return false;
        } else {
            _values[index] += amount;
            return true;
        }
    }

    /**
     * {@inheritDoc}
     */
    public int adjustOrPutValue(int key, int adjust_amount, int put_amount) {
        int index = insertKey(key);
        final boolean isNewMapping;
        final int newValue;

        if (index < 0) {
            index = -index - 1;
            newValue = (_values[index] += adjust_amount);
            isNewMapping = false;
        } else {
            newValue = (_values[index] = put_amount);
            isNewMapping = true;
        }

        byte previousState = _states[index];

        if (isNewMapping) {
            postInsertHook(consumeFreeSlot);
        }

        return newValue;
    }

    /**
     * a view onto the keys of the map.
     */
    protected class TKeyView implements TIntSet {

        /**
         * {@inheritDoc}
         */
        public TIntIterator iterator() {
            return new TIntIntKeyHashIterator(TIntIntHashMap.this);
        }

        /**
         * {@inheritDoc}
         */
        public int getNoEntryValue() {
            return no_entry_key;
        }

        /**
         * {@inheritDoc}
         */
        public int size() {
            return _size;
        }

        /**
         * {@inheritDoc}
         */
        public boolean isEmpty() {
            return 0 == _size;
        }

        /**
         * {@inheritDoc}
         */
        public boolean contains(int entry) {
            return TIntIntHashMap.this.contains(entry);
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray() {
            return TIntIntHashMap.this.keys();
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray(int[] dest) {
            return TIntIntHashMap.this.keys(dest);
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntIntMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean add(int entry) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        public boolean remove(int entry) {
            return no_entry_value != TIntIntHashMap.this.remove(entry);
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(Collection<?> collection) {
            for (Object element : collection) {
                if (element instanceof Integer) {
                    int ele = (Integer) element;

                    if (!TIntIntHashMap.this.containsKey(ele)) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(TIntCollection collection) {
            TIntIterator iter = collection.iterator();

            while (iter.hasNext()) {
                if (!TIntIntHashMap.this.containsKey(iter.next())) {
                    return false;
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(int[] array) {
            for (int element : array) {
                if (!TIntIntHashMap.this.contains(element)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntIntMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean addAll(Collection<? extends Integer> collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntIntMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean addAll(TIntCollection collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntIntMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean addAll(int[] array) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        @SuppressWarnings({ "SuspiciousMethodCalls"})
        public boolean retainAll(Collection<?> collection) {
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

        /**
         * {@inheritDoc}
         */
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

        /**
         * {@inheritDoc}
         */
        public boolean retainAll(int[] array) {
            boolean changed = false;

            Arrays.sort(array);
            int[] set = _set;
            byte[] states = _states;

            for (int i = set.length; i-- > 0;) {
                if (states[i] == FULL
                        && (Arrays.binarySearch(array, set[i]) < 0)) {
                    removeAt(i);
                    changed = true;
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(Collection<?> collection) {
            boolean changed = false;

            for (Object element : collection) {
                if (element instanceof Integer) {
                    int c = (Integer) element;

                    if (remove(c)) {
                        changed = true;
                    }
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(TIntCollection collection) {
            if (this == collection) {
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

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(int[] array) {
            boolean changed = false;

            for (int i = array.length; i-- > 0;) {
                if (remove(array[i])) {
                    changed = true;
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public void clear() {
            TIntIntHashMap.this.clear();
        }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof TIntSet)) {
                return false;
            }
            final TIntSet that = (TIntSet) other;

            if (that.size() != this.size()) {
                return false;
            }
            for (int i = _states.length; i-- > 0;) {
                if (_states[i] == FULL) {
                    if (!that.contains(_set[i])) {
                        return false;
                    }
                }
            }
            return true;
        }

        @Override
        public int hashCode() {
            int hashcode = 0;

            for (int i = _states.length; i-- > 0;) {
                if (_states[i] == FULL) {
                    hashcode += HashFunctions.hash(_set[i]);
                }
            }
            return hashcode;
        }
    }


    /**
     * a view onto the values of the map.
     */
    protected class TValueView implements TIntCollection {

        /**
         * {@inheritDoc}
         */
        public TIntIterator iterator() {
            return new TIntIntValueHashIterator(TIntIntHashMap.this);
        }

        /**
         * {@inheritDoc}
         */
        public int getNoEntryValue() {
            return no_entry_value;
        }

        /**
         * {@inheritDoc}
         */
        public int size() {
            return _size;
        }

        /**
         * {@inheritDoc}
         */
        public boolean isEmpty() {
            return 0 == _size;
        }

        /**
         * {@inheritDoc}
         */
        public boolean contains(int entry) {
            return TIntIntHashMap.this.containsValue(entry);
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray() {
            return TIntIntHashMap.this.values();
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray(int[] dest) {
            return TIntIntHashMap.this.values(dest);
        }

        public boolean add(int entry) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        public boolean remove(int entry) {
            int[] values = _values;
            int[] set = _set;

            for (int i = values.length; i-- > 0;) {
                if ((set[i] != FREE && set[i] != REMOVED) && entry == values[i]) {
                    removeAt(i);
                    return true;
                }
            }
            return false;
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(Collection<?> collection) {
            for (Object element : collection) {
                if (element instanceof Integer) {
                    int ele = (Integer) element;

                    if (!TIntIntHashMap.this.containsValue(ele)) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(TIntCollection collection) {
            TIntIterator iter = collection.iterator();

            while (iter.hasNext()) {
                if (!TIntIntHashMap.this.containsValue(iter.next())) {
                    return false;
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(int[] array) {
            for (int element : array) {
                if (!TIntIntHashMap.this.containsValue(element)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        public boolean addAll(Collection<? extends Integer> collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        public boolean addAll(TIntCollection collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        public boolean addAll(int[] array) {
            throw new UnsupportedOperationException();
        }

        /**
         * {@inheritDoc}
         */
        @SuppressWarnings({ "SuspiciousMethodCalls"})
        public boolean retainAll(Collection<?> collection) {
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

        /**
         * {@inheritDoc}
         */
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

        /**
         * {@inheritDoc}
         */
        public boolean retainAll(int[] array) {
            boolean changed = false;

            Arrays.sort(array);
            int[] values = _values;
            byte[] states = _states;

            for (int i = values.length; i-- > 0;) {
                if (states[i] == FULL
                        && (Arrays.binarySearch(array, values[i]) < 0)) {
                    removeAt(i);
                    changed = true;
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(Collection<?> collection) {
            boolean changed = false;

            for (Object element : collection) {
                if (element instanceof Integer) {
                    int c = (Integer) element;

                    if (remove(c)) {
                        changed = true;
                    }
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(TIntCollection collection) {
            if (this == collection) {
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

        /**
         * {@inheritDoc}
         */
        public boolean removeAll(int[] array) {
            boolean changed = false;

            for (int i = array.length; i-- > 0;) {
                if (remove(array[i])) {
                    changed = true;
                }
            }
            return changed;
        }

        /**
         * {@inheritDoc}
         */
        public void clear() {
            TIntIntHashMap.this.clear();
        }

        class TIntIntValueHashIterator extends THashPrimitiveIterator implements TIntIterator {

            /**
             * Creates an iterator over the specified map
             *
             * @param hash the <tt>TPrimitiveHash</tt> we will be iterating over.
             */
            TIntIntValueHashIterator(TPrimitiveHash hash) {
                super(hash);
            }

            /**
             * {@inheritDoc}
             */
            public int next() {
                moveToNextIndex();
                return _values[_index];
            }

            /**
             * @{inheritDoc}
             */
            public void remove() {
                if (_expectedSize != _hash.size()) {
                    throw new ConcurrentModificationException();
                }

                // Disable auto compaction during the remove. This is a workaround for bug 1642768.
                try {
                    _hash.tempDisableAutoCompaction();
                    TIntIntHashMap.this.removeAt(_index);
                } finally {
                    _hash.reenableAutoCompaction(false);
                }

                _expectedSize--;
            }
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public boolean equals(Object other) {
            if (!(other instanceof TIntIntMap)) {
                return false;
            }
            TIntIntMap that = (TIntIntMap) other;

            if (that.size() != this.size()) {
                return false;
            }
            int[] values = _values;
            byte[] states = _states;
            int this_no_entry_value = getNoEntryValue();
            int that_no_entry_value = that.getNoEntryValue();

            for (int i = values.length; i-- > 0;) {
                if (states[i] == FULL) {
                    int key = _set[i];
                    int that_value = that.get(key);
                    int this_value = values[i];

                    if ((this_value != that_value)
                            && (this_value != this_no_entry_value)
                            && (that_value != that_no_entry_value)) {
                        return false;
                    }
                }
            }
            return true;
        }

        /**
         * {@inheritDoc}
         */
        @Override
        public int hashCode() {
            int hashcode = 0;
            byte[] states = _states;

            for (int i = _values.length; i-- > 0;) {
                if (states[i] == FULL) {
                    hashcode += HashFunctions.hash(_set[i])
                            ^ HashFunctions.hash(_values[i]);
                }
            }
            return hashcode;
        }

    }


    class TIntIntHashIterator extends THashPrimitiveIterator implements TIntIntIterator {

        /**
         * Creates an iterator over the specified map
         *
         * @param map the <tt>TIntIntHashMap</tt> we will be iterating over.
         */
        TIntIntHashIterator(TIntIntHashMap map) {
            super(map);
        }

        /**
         * {@inheritDoc}
         */
        public void advance() {
            moveToNextIndex();
        }

        /**
         * {@inheritDoc}
         */
        public int key() {
            return _set[_index];
        }

        /**
         * {@inheritDoc}
         */
        public int value() {
            return _values[_index];
        }

        /**
         * {@inheritDoc}
         */
        public int setValue(int val) {
            int old = value();

            _values[_index] = val;
            return old;
        }

        /**
         * @{inheritDoc}
         */
        public void remove() {
            if (_expectedSize != _hash.size()) {
                throw new ConcurrentModificationException();
            }
            // Disable auto compaction during the remove. This is a workaround for bug 1642768.
            try {
                _hash.tempDisableAutoCompaction();
                TIntIntHashMap.this.removeAt(_index);
            } finally {
                _hash.reenableAutoCompaction(false);
            }
            _expectedSize--;
        }
    }


    class TIntIntKeyHashIterator extends THashPrimitiveIterator implements TIntIterator {

        /**
         * Creates an iterator over the specified map
         *
         * @param hash the <tt>TPrimitiveHash</tt> we will be iterating over.
         */
        TIntIntKeyHashIterator(TPrimitiveHash hash) {
            super(hash);
        }

        /**
         * {@inheritDoc}
         */
        public int next() {
            moveToNextIndex();
            return _set[_index];
        }

        /**
         * @{inheritDoc}
         */
        public void remove() {
            if (_expectedSize != _hash.size()) {
                throw new ConcurrentModificationException();
            }

            // Disable auto compaction during the remove. This is a workaround for bug 1642768.
            try {
                _hash.tempDisableAutoCompaction();
                TIntIntHashMap.this.removeAt(_index);
            } finally {
                _hash.reenableAutoCompaction(false);
            }

            _expectedSize--;
        }
    }

    public String toString() {
        StringBuilder r = new StringBuilder();

        r.append("{");
        TIntIterator it = keySet().iterator();
        boolean first = true;

        while (it.hasNext()) {
            if (first) {
                first = false;
            } else {
                r.append(", ");
            }
            int k = it.next();

            r.append(f("%d: %d", k, get(k)));
        }
        r.append("}");
        return r.toString();
    }
}

