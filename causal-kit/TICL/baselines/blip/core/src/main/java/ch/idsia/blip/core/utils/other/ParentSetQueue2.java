package ch.idsia.blip.core.utils.other;


/**
 * Queue for parent set to evaluate.
 */
public class ParentSetQueue2 {

    /**
     * Maximum size for the queue
     */
    private final int max_size;

    /**
     * Current size for the queue
     */
    private int size;

    /**
     * Pointer to the first entry in the linked-list
     */
    private ParentSetEntry first;

    /**
     * Pointer to the last entry in the linked-list
     */
    private ParentSetEntry last;

    /**
     * Default constructor.
     *
     * @param in_max_size Maximum queue size.
     */
    public ParentSetQueue2(int in_max_size) {
        max_size = in_max_size;

        first = null;
        last = null;

        size = 0;
    }

    /**
     * @return if the queue is already full.
     */
    public boolean isFull() {
        return size >= max_size;
    }

    /**
     * @return if the queue is empty.
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Add a new parent set to the score. Find the correct position in the linked list map.
     *
     * @param hash_p hash of parent set base
     * @param p2     last variable added to parent set
     * @param score  score of the parent set
     */
    public void add(long hash_p, int p2, double score) {

        ParentSetEntry p = new ParentSetEntry(hash_p, p2, score);

        size++;

        // If list is empty
        if (first == null) {
            first = p;
            last = p;
            return;
        }

        ParentSetEntry c = first;

        while ((c != null) && (c.sk < score)) {
            c = c.next;
        }

        // If thread'm at the end of the list
        if (c == null) {
            last.next = p;
            p.prev = last;
            last = p;
            return;
        }

        // If thread'm at the initCl of the list
        if (first.equals(c)) {
            first = p;
            c.prev = p;
            p.next = c;
            return;
        }

        // Else we are in the middle in list
        c.prev.next = p;
        p.prev = c.prev;
        c.prev = p;
        p.next = c;
    }

    /**
     * @return the best parent set present in queue.
     */
    public ParentSetEntry popBest() {
        ParentSetEntry p = first;

        size--;

        if (first != null) {
            first = first.next;
        }

        if (first != null) {
            first.prev = null;
        }

        if (last.equals(p)) {
            last = null;
        }

        return p;
    }

    /**
     * @return the score of the best parent set present in queue.
     */
    public double getBestScore() {
        if (first != null) {
            return first.sk;
        }
        return 0;
    }

    /**
     * @return the worst parent set present in queue.
     */
    public ParentSetEntry popWorst() {

        ParentSetEntry p = last;

        size--;

        if (last != null) {
            last = last.prev;
        }

        if (last != null) {
            last.next = null;
        }

        if (first.equals(p)) {
            first = null;
        }

        return p;
    }

    /**
     * @return the score of the worst parent set present in queue.
     */
    public double getWorstScore() {
        if (last != null) {
            return last.sk;
        }
        return 0;
    }

    /**
     * @return current size of the queue.
     */
    public int size() {
        return size;
    }

    public String toString() {
        StringBuilder s = new StringBuilder();

        s.append("Size: ");
        s.append(size);
        s.append(" First: ");
        if (first != null) {
            s.append(first);
        } else {
            s.append("none");
        }

        s.append(". Last: ");
        if (last != null) {
            s.append(last);
        } else {
            s.append("none");
        }

        s.append(". \n Next: ");

        ParentSetEntry c = first;
        int i = 0;
        StringBuilder next = new StringBuilder();

        while (c != null) {
            next.append(c);
            next.append(" ");
            c = c.next;
            i++;
        }

        s.append(i);
        s.append(" ");
        s.append(next);
        s.append("\n Prev: ");

        c = last;
        i = 0;
        StringBuilder prev = new StringBuilder();

        while (c != null) {
            prev.append(c);
            prev.append(" ");
            c = c.prev;
            i++;
        }

        s.append(i);
        s.append(" ");
        s.append(prev);
        s.append("\n");

        return s.toString();
    }

    /**
     * Entry of a parent set in the linked-list queue.
     */
    public class ParentSetEntry implements Comparable<ParentSetEntry> {

        /**
         * Hash of the parent set base.
         */
        public long hash_p;

        /**
         * Last variable added to parent set.
         */
        public int p2;

        /**
         * Score of the parent set.
         */
        public double sk;

        /**
         * Next entry in queue.
         */
        public ParentSetEntry next;

        /**
         * Previous entry in queue.
         */
        public ParentSetEntry prev;

        /**
         * Default constructor.
         *
         * @param in_hash_p hash of the parent set base
         * @param in_p2     last variable added to parent set
         * @param in_sk     score of the parent set
         */
        public ParentSetEntry(long in_hash_p, int in_p2, double in_sk) {
            hash_p = in_hash_p;
            p2 = in_p2;
            sk = in_sk;
        }

        @Override
        public int compareTo(ParentSetEntry other) {
            if (sk > other.sk) {
                return 1;
            }
            return -1;
        }

        public String toString() {
            return String.format("(%d %d %.3f)", hash_p, p2, sk);
        }

    }
}
