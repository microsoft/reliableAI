package ch.idsia.blip.core.utils.other;


/**
 * Entry of a parent score
 */
public class Clique {

    public int[] variables;

    public Clique parent;

    public Clique(int[] v, Clique p) {
        variables = v;
        parent = p;
    }
}
