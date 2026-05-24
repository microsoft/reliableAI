package ch.idsia.blip.core.utils.data.common;


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
public interface TPrimitiveIterator extends TIterator {

    /**
     * Returns true if the iterator can be advanced past its current
     * location.
     *
     * @return a <code>boolean</code> value
     */
    boolean hasNext();

    /**
     * Removes the last entry returned by the iterator.
     * Invoking this method more than once for a single entry
     * will leave the underlying data map in a confused
     * state.
     */
    void remove();

} // TPrimitiveIterator
