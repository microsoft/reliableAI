package ch.idsia.blip.core.utils.other;


import ch.idsia.blip.core.utils.data.SIntSet;

import java.util.HashSet;
import java.util.Set;
import java.util.TreeMap;


/**
 * Cache of parent set row_values
 */
public class ParentSetCache {

    /**
     * Maximum cache size
     */
    private final long max_size;

    /**
     * Map of cached values (hash of parent set -> value)
     */
    private final TreeMap<SIntSet, int[][]> m_values;

    /**
     * Map of retention counter (hash of parent set -> counter)
     */
    private final TreeMap<SIntSet, Integer> m_cnt;

    /**
     * Default constructor
     *
     * @param in_max_size maximum cache size
     */
    public ParentSetCache(long in_max_size) {
        max_size = in_max_size;
        m_values = new TreeMap<SIntSet, int[][]>();
        m_cnt = new TreeMap<SIntSet, Integer>();
    }

    /**
     * @return current size of cache
     */
    private int size() {
        return m_values.size();
    }

    /**
     * @return if the cache is full
     */
    private boolean isFull() {
        return size() >= max_size;
    }

    /**
     * Increase retention counter on a parent set
     */
    private void increaseCnt(SIntSet pset) {
        m_cnt.put(pset, getCnt(pset) + 1);
    }

    /**
     * Decrease retention counter on a parent set
     *
     * @param pset hash of the parent set
     */
    private void decreaseCnt(SIntSet pset) {
        if (m_cnt.containsKey(pset)) {
            int old = m_cnt.get(pset);

            if (old > 1) {
                m_cnt.put(pset, old - 1);
            }
        }
    }

    /**
     * @param pset hash of a parent set
     * @return retention counter of given parent set
     */
    private int getCnt(SIntSet pset) {
        if (m_cnt.containsKey(pset)) {
            return m_cnt.get(pset);
        }
        return -1;
    }

    /**
     * Free the values cached for the given parent set
     *
     * @param pset hash of the parent set
     */
    private void free(SIntSet pset) {
        m_values.remove(pset);
        m_cnt.remove(pset);
    }

    /**
     * If there is space left, save the values of a parent set in cache
     *
     * @param pset     hash of the parent set
     * @param p_values values already computed
     */
    public void save(SIntSet pset, int[][] p_values) {
        if (isFull()) {
            makeSpace();
        }

        if (!isFull()) {
            m_values.put(pset, p_values);
            m_cnt.put(pset, 0);
        }
    }

    /**
     * Free from the cache all the entry with the lowest counter
     */

    private void makeSpace() {
        int least_cnt = 1000;

        Set<SIntSet> l = new HashSet<SIntSet>();

        for (SIntSet parentSet : m_cnt.keySet()) {
            int s = getCnt(parentSet);

            if (s < least_cnt) {
                least_cnt = s;
                l = new HashSet<SIntSet>();
            }

            if (s == least_cnt) {
                l.add(parentSet);
            }

            decreaseCnt(parentSet);
        }

        for (SIntSet hash : l) {
            free(hash);
        }

        // log.conclude(String.format("Make space: freed counter: %s, there are %s of them. Now size are %s (values) / %s (cnt) on %s max, ",
        // least_cnt, l.size(), m_values.size(), m_cnt.size(), max_size));

        // If there wasn't enough new space, repeat
        if (((m_values.size() * 1.0) / max_size) > 0.90) {
            makeSpace();
        }
    }

    /**
     * @param hash_p hash of the parent set
     * @return values cached for the parent set
     */
    public int[][] retrieve(SIntSet hash_p) {
        if (m_values.containsKey(hash_p)) {
            increaseCnt(hash_p);
            return m_values.get(hash_p);
        }

        return new int[0][];
    }

}
