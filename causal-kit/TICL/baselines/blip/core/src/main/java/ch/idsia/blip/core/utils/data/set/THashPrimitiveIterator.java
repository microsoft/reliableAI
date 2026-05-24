package ch.idsia.blip.core.utils.data.set;


import ch.idsia.blip.core.utils.data.common.TPrimitiveIterator;

import java.util.ConcurrentModificationException;
import java.util.NoSuchElementException;


/**
 * Implements all iterator functions for the hashed object set.
 * Subclasses may override objectAtIndex to vary the object
 * returned by calls to next() (e.g. for values, and Map.Entry
 * objects).
 * <p/>
 * <p> Note that iteration is fastest if you forego the calls to
 * <tt>hasNext</tt> in favor of checking the size of the map
 * yourself and then call next() that many times:
 * <p/>
 * <pre>
 * Iterator thread = collection.iterator();
 * for (int size = collection.size(); size-- > 0;) {
 *   Object o = thread.next();
 * }
 * </pre>
 * <p/>
 * <p>You may, of course, use the hasNext(), next() idiom too if
 * you aren't in a performance critical spot.</p>
 */
public abstract class THashPrimitiveIterator implements TPrimitiveIterator {

    /**
     * the data map this iterator traverses
     */
    protected final TPrimitiveHash _hash;

    /**
     * the number of elements this iterator believes are in the
     * data map it accesses.
     */
    protected int _expectedSize;

    /**
     * the index used for iteration.
     */
    protected int _index;

    /**
     * Creates a <tt>TPrimitiveIterator</tt> for the specified collection.
     *
     * @param hash the <tt>TPrimitiveHash</tt> we want to iterate over.
     */
    public THashPrimitiveIterator(TPrimitiveHash hash) {
        _hash = hash;
        _expectedSize = _hash.size();
        _index = _hash.capacity();
    }

    /**
     * Returns the index of the next value in the data map
     * or a negative value if the iterator is exhausted.
     *
     * @return an <code>int</code> value
     * @throws java.util.ConcurrentModificationException if the underlying collection's
     *                                                   size has been modified since the iterator was created.
     */
    private int nextIndex() {
        if (_expectedSize != _hash.size()) {
            throw new ConcurrentModificationException();
        }

        byte[] states = _hash._states;
        int i = _index;

        while (i-- > 0 && (states[i] != TPrimitiveHash.FULL)) {}
        return i;
    }

    /**
     * Returns true if the iterator can be advanced past its current
     * location.
     *
     * @return a <code>boolean</code> value
     */
    public boolean hasNext() {
        return nextIndex() >= 0;
    }

    /**
     * Removes the last entry returned by the iterator.
     * Invoking this method more than once for a single entry
     * will leave the underlying data map in a confused
     * state.
     */
    public void remove() {
        if (_expectedSize != _hash.size()) {
            throw new ConcurrentModificationException();
        }

        // Disable auto compaction during the remove. This is a workaround for bug 1642768.
        try {
            _hash.tempDisableAutoCompaction();
            _hash.removeAt(_index);
        } finally {
            _hash.reenableAutoCompaction(false);
        }

        _expectedSize--;
    }

    /**
     * Sets the internal <tt>index</tt> so that the `next' object
     * can be returned.
     */
    protected final void moveToNextIndex() {
        // doing the assignment && < 0 in one line shaves
        // 3 opcodes...
        if ((_index = nextIndex()) < 0) {
            throw new NoSuchElementException();
        }
    }

} // TPrimitiveIterator
