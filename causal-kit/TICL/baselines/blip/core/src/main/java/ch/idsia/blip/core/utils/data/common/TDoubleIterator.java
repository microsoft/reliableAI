package ch.idsia.blip.core.utils.data.common;


/**
 * Iterator for double collections.
 */
public interface TDoubleIterator extends TIterator {

    /**
     * Advances the iterator to the next element in the underlying collection
     * and returns it.
     *
     * @return the next double in the collection
     */
    double next();
}
