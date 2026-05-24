package ch.idsia.blip.core.utils.cache;


class Node<T1, T2> {
    T1 key;
    T2 value;

    Node pre;
    Node next;

    public Node(T1 key, T2 value) {
        this.key = key;
        this.value = value;
    }
}
