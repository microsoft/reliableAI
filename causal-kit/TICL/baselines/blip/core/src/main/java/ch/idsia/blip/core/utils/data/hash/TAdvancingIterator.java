package ch.idsia.blip.core.utils.data.hash;


import ch.idsia.blip.core.utils.data.common.TIterator;


/**
 * Common interface for iterators that operate via the "advance" method for moving the
 * cursor to the next element.
 */
interface TAdvancingIterator extends TIterator {

    /**
     * Moves the iterator forward to the next entry.
     *
     * @throws java.util.NoSuchElementException if the iterator is already exhausted
     */
    void advance();
}
