package ch.idsia.blip.core.utils.data.common;


/**
 * Iterator for int collections.
 */
public interface TIntIterator extends TIterator {

    /**
     * Advances the iterator to the next element in the underlying collection
     * and returns it.
     *
     * @return the next int in the collection
     */
    int next();
}
