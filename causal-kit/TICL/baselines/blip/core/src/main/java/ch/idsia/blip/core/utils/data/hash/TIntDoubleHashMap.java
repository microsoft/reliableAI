package ch.idsia.blip.core.utils.data.hash;


import ch.idsia.blip.core.utils.data.HashFunctions;
import ch.idsia.blip.core.utils.data.common.TDoubleCollection;
import ch.idsia.blip.core.utils.data.common.TDoubleIterator;
import ch.idsia.blip.core.utils.data.common.TIntCollection;
import ch.idsia.blip.core.utils.data.common.TIntIterator;
import ch.idsia.blip.core.utils.data.set.THashPrimitiveIterator;
import ch.idsia.blip.core.utils.data.set.TIntSet;
import ch.idsia.blip.core.utils.data.set.TPrimitiveHash;

import java.util.Arrays;
import java.util.Collection;
import java.util.ConcurrentModificationException;
import java.util.Map;


/**
 * An open addressed Map implementation for int keys and double values.
 *
 * @author Eric D. Friedman
 * @author Rob Eden
 * @author Jeff Randall
 * @version $Id: _K__V_HashMap.template,v 1.1.2.16 2010/03/02 04:09:50 robeden Exp $
 */
public class TIntDoubleHashMap extends TIntDoubleHash implements TIntDoubleMap {
    static final long serialVersionUID = 1L;

    /**
     * the values of the map
     */
    private transient double[] _values;

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance with the default
     * capacity and load factor.
     */
    public TIntDoubleHashMap() {
        super();
    }

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the default load factor.
     *
     * @param initialCapacity an <code>int</code> value
     */
    public TIntDoubleHashMap(int initialCapacity) {
        super(initialCapacity);
    }

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the specified load factor.
     *
     * @param initialCapacity an <code>int</code> value
     * @param loadFactor      a <code>float</code> value
     */
    public TIntDoubleHashMap(int initialCapacity, float loadFactor) {
        super(initialCapacity, loadFactor);
    }

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance with a prime
     * capacity equal to or greater than <tt>initialCapacity</tt> and
     * with the specified load factor.
     *
     * @param initialCapacity an <code>int</code> value
     * @param loadFactor      a <code>float</code> value
     * @param noEntryKey      a <code>int</code> value that represents
     *                        <tt>null</tt> for the Key set.
     * @param noEntryValue    a <code>double</code> value that represents
     *                        <tt>null</tt> for the Value set.
     */
    public TIntDoubleHashMap(int initialCapacity, float loadFactor,
            int noEntryKey, double noEntryValue) {
        super(initialCapacity, loadFactor, noEntryKey, noEntryValue);
    }

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance containing
     * all of the entries in the map passed in.
     *
     * @param keys   a <tt>int</tt> array containing the keys for the matching values.
     * @param values a <tt>double</tt> array containing the values.
     */
    public TIntDoubleHashMap(int[] keys, double[] values) {
        super(Math.max(keys.length, values.length));

        int size = Math.min(keys.length, values.length);

        for (int i = 0; i < size; i++) {
            this.put(keys[i], values[i]);
        }
    }

    /**
     * Creates a new <code>TIntDoubleHashMap</code> instance containing
     * all of the entries in the map passed in.
     *
     * @param map a <tt>TIntDoubleMap</tt> that will be duplicated.
     */
    public TIntDoubleHashMap(TIntDoubleMap map) {
        super(map.size());
        if (map instanceof TIntDoubleHashMap) {
            TIntDoubleHashMap hashmap = (TIntDoubleHashMap) map;

            this._loadFactor = hashmap._loadFactor;
            this.no_entry_key = hashmap.no_entry_key;
            this.no_entry_value = hashmap.no_entry_value;
            // noinspection RedundantCast
            if (this.no_entry_key != (int) 0) {
                Arrays.fill(_set, this.no_entry_key);
            }
            // noinspection RedundantCast
            if (this.no_entry_value != (double) 0) {
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
        _values = new double[capacity];
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
        double oldVals[] = _values;
        byte oldStates[] = _states;

        _set = new int[newCapacity];
        _values = new double[newCapacity];
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
    public double put(int key, double value) {
        int index = insertKey(key);

        return doPut(value, index);
    }

    /**
     * {@inheritDoc}
     */
    public double putIfAbsent(int key, double value) {
        int index = insertKey(key);

        if (index < 0) {
            return _values[-index - 1];
        }
        return doPut(value, index);
    }

    private double doPut(double value, int index) {
        double previous = no_entry_value;
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
    public void putAll(Map<? extends Integer, ? extends Double> map) {
        ensureCapacity(map.size());
        // could winasobs this for cases when map instanceof THashMap
        for (Map.Entry<? extends Integer, ? extends Double> entry : map.entrySet()) {
            this.put(entry.getKey(), entry.getValue());
        }
    }

    /**
     * {@inheritDoc}
     */
    public void putAll(TIntDoubleMap map) {
        ensureCapacity(map.size());
        TIntDoubleIterator iter = map.iterator();

        while (iter.hasNext()) {
            iter.advance();
            this.put(iter.key(), iter.value());
        }
    }

    /**
     * {@inheritDoc}
     */
    public double get(int key) {
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
    public double remove(int key) {
        double prev = no_entry_value;
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

    @Override
    public TDoubleCollection valueCollection() {
        // return new TValueView();
        return null;
    }

    /**
     * {@inheritDoc}
     */
    public double[] values() {
        double[] vals = new double[size()];
        double[] v = _values;
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
    public double[] values(double[] array) {
        int size = size();

        if (array.length < size) {
            array = new double[size];
        }

        double[] v = _values;
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
    public boolean containsValue(double val) {
        byte[] states = _states;
        double[] vals = _values;

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
    public TIntDoubleIterator iterator() {
        return new TIntDoubleHashIterator(this);
    }

    /**
     * {@inheritDoc}
     */
    public boolean increment(int key) {
        return adjustValue(key, (double) 1);
    }

    /**
     * {@inheritDoc}
     */
    public boolean adjustValue(int key, double amount) {
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
    public double adjustOrPutValue(int key, double adjust_amount, double put_amount) {
        int index = insertKey(key);
        final boolean isNewMapping;
        final double newValue;

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
            return new TIntDoubleKeyHashIterator(TIntDoubleHashMap.this);
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
            return TIntDoubleHashMap.this.contains(entry);
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray() {
            return TIntDoubleHashMap.this.keys();
        }

        /**
         * {@inheritDoc}
         */
        public int[] toArray(int[] dest) {
            return TIntDoubleHashMap.this.keys(dest);
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntDoubleMap
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
            return no_entry_value != TIntDoubleHashMap.this.remove(entry);
        }

        /**
         * {@inheritDoc}
         */
        public boolean containsAll(Collection<?> collection) {
            for (Object element : collection) {
                if (element instanceof Integer) {
                    int ele = (Integer) element;

                    if (!TIntDoubleHashMap.this.containsKey(ele)) {
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
                if (!TIntDoubleHashMap.this.containsKey(iter.next())) {
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
                if (!TIntDoubleHashMap.this.contains(element)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntDoubleMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean addAll(Collection<? extends Integer> collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntDoubleMap
         * <p/>
         * {@inheritDoc}
         */
        public boolean addAll(TIntCollection collection) {
            throw new UnsupportedOperationException();
        }

        /**
         * Unsupported when operating upon a Key Set view of a TIntDoubleMap
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
            TIntDoubleHashMap.this.clear();
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

        /**
         * a view onto the values of the map.
         */
        protected class TValueView implements TDoubleCollection {

            /**
             * {@inheritDoc}
             */
            public TDoubleIterator iterator() {
                return new TIntDoubleValueHashIterator(TIntDoubleHashMap.this);
            }

            /**
             * {@inheritDoc}
             */
            public double getNoEntryValue() {
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
            public boolean contains(double entry) {
                return TIntDoubleHashMap.this.containsValue(entry);
            }

            /**
             * {@inheritDoc}
             */
            public double[] toArray() {
                return TIntDoubleHashMap.this.values();
            }

            /**
             * {@inheritDoc}
             */
            public double[] toArray(double[] dest) {
                return TIntDoubleHashMap.this.values(dest);
            }

            public boolean add(double entry) {
                throw new UnsupportedOperationException();
            }

            /**
             * {@inheritDoc}
             */
            public boolean remove(double entry) {
                double[] values = _values;
                int[] set = _set;

                for (int i = values.length; i-- > 0;) {
                    if ((set[i] != FREE && set[i] != REMOVED)
                            && entry == values[i]) {
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
                    if (element instanceof Double) {
                        double ele = (Double) element;

                        if (!TIntDoubleHashMap.this.containsValue(ele)) {
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
            public boolean containsAll(TDoubleCollection collection) {
                TDoubleIterator iter = collection.iterator();

                while (iter.hasNext()) {
                    if (!TIntDoubleHashMap.this.containsValue(iter.next())) {
                        return false;
                    }
                }
                return true;
            }

            /**
             * {@inheritDoc}
             */
            public boolean containsAll(double[] array) {
                for (double element : array) {
                    if (!TIntDoubleHashMap.this.containsValue(element)) {
                        return false;
                    }
                }
                return true;
            }

            /**
             * {@inheritDoc}
             */
            public boolean addAll(Collection<? extends Double> collection) {
                throw new UnsupportedOperationException();
            }

            /**
             * {@inheritDoc}
             */
            public boolean addAll(TDoubleCollection collection) {
                throw new UnsupportedOperationException();
            }

            /**
             * {@inheritDoc}
             */
            public boolean addAll(double[] array) {
                throw new UnsupportedOperationException();
            }

            /**
             * {@inheritDoc}
             */
            @SuppressWarnings({ "SuspiciousMethodCalls"})
            public boolean retainAll(Collection<?> collection) {
                boolean modified = false;
                TDoubleIterator iter = iterator();

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
            public boolean retainAll(TDoubleCollection collection) {
                if (this == collection) {
                    return false;
                }
                boolean modified = false;
                TDoubleIterator iter = iterator();

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
            public boolean retainAll(double[] array) {
                boolean changed = false;

                Arrays.sort(array);
                double[] values = _values;
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
                    if (element instanceof Double) {
                        double c = (Double) element;

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
            public boolean removeAll(TDoubleCollection collection) {
                if (this == collection) {
                    clear();
                    return true;
                }
                boolean changed = false;
                TDoubleIterator iter = collection.iterator();

                while (iter.hasNext()) {
                    double element = iter.next();

                    if (remove(element)) {
                        changed = true;
                    }
                }
                return changed;
            }

            /**
             * {@inheritDoc}
             */
            public boolean removeAll(double[] array) {
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
                TIntDoubleHashMap.this.clear();
            }
        }


        class TIntDoubleKeyHashIterator extends THashPrimitiveIterator implements TIntIterator {

            /**
             * Creates an iterator over the specified map
             *
             * @param hash the <tt>TPrimitiveHash</tt> we will be iterating over.
             */
            TIntDoubleKeyHashIterator(TPrimitiveHash hash) {
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
                    TIntDoubleHashMap.this.removeAt(_index);
                } finally {
                    _hash.reenableAutoCompaction(false);
                }

                _expectedSize--;
            }
        }


        class TIntDoubleValueHashIterator extends THashPrimitiveIterator implements TDoubleIterator {

            /**
             * Creates an iterator over the specified map
             *
             * @param hash the <tt>TPrimitiveHash</tt> we will be iterating over.
             */
            TIntDoubleValueHashIterator(TPrimitiveHash hash) {
                super(hash);
            }

            /**
             * {@inheritDoc}
             */
            public double next() {
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
                    TIntDoubleHashMap.this.removeAt(_index);
                } finally {
                    _hash.reenableAutoCompaction(false);
                }

                _expectedSize--;
            }
        }

    }


    class TIntDoubleHashIterator extends THashPrimitiveIterator implements TIntDoubleIterator {

        /**
         * Creates an iterator over the specified map
         *
         * @param map the <tt>TIntDoubleHashMap</tt> we will be iterating over.
         */
        TIntDoubleHashIterator(TIntDoubleHashMap map) {
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
        public double value() {
            return _values[_index];
        }

        /**
         * {@inheritDoc}
         */
        public double setValue(double val) {
            double old = value();

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
                TIntDoubleHashMap.this.removeAt(_index);
            } finally {
                _hash.reenableAutoCompaction(false);
            }
            _expectedSize--;
        }
    }
}
