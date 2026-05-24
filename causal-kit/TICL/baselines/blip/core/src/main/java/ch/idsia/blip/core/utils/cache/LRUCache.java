package ch.idsia.blip.core.utils.cache;


import java.util.Hashtable;


/**
 * This class implements a Generic LRU Cache
 *
 */

public class LRUCache<T1, T2> {
    class CacheNode {

        CacheNode prev;
        CacheNode next;
        T2 value;
        T1 key;

        CacheNode() {}
    }

    public LRUCache(int i) {
        currentSize = 0;
        cacheSize = i;
        nodes = new Hashtable<T1, CacheNode>(i);
    }

    public T2 get(T1 key) {
        CacheNode node = nodes.get(key);

        if (node != null) {
            moveToHead(node);
            return node.value;
        } else {
            return null;
        }
    }

    public void put(T1 key, T2 value) {
        CacheNode node = nodes.get(key);

        if (node == null) {
            if (currentSize >= cacheSize) {
                if (last != null) {
                    nodes.remove(last.key);
                }
                removeLast();
            } else {
                currentSize++;
            }
            node = new CacheNode();
        }
        node.value = value;
        node.key = key;
        moveToHead(node);
        nodes.put(key, node);
    }

    public T2 remove(T1 key) {
        CacheNode node = nodes.get(key);

        if (node != null) {
            if (node.prev != null) {
                node.prev.next = node.next;
            }
            if (node.next != null) {
                node.next.prev = node.prev;
            }
            if (last == node) {
                last = node.prev;
            }
            if (first == node) {
                first = node.next;
            }
        }
        return node.value;
    }

    public void clear() {
        first = null;
        last = null;
    }

    private void removeLast() {
        if (last != null) {
            if (last.prev != null) {
                last.prev.next = null;
            } else {
                first = null;
            }
            last = last.prev;
        }
    }

    private void moveToHead(CacheNode node) {
        if (node == first) {
            return;
        }
        if (node.prev != null) {
            node.prev.next = node.next;
        }
        if (node.next != null) {
            node.next.prev = node.prev;
        }
        if (last == node) {
            last = node.prev;
        }
        if (first != null) {
            node.next = first;
            first.prev = node;
        }
        first = node;
        node.prev = null;
        if (last == null) {
            last = first;
        }
    }

    private int cacheSize;
    private Hashtable<T1, CacheNode> nodes;
    private int currentSize;
    private CacheNode first;
    private CacheNode last;
}
