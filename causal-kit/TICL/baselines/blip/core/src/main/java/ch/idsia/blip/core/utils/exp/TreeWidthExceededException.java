package ch.idsia.blip.core.utils.exp;


public class TreeWidthExceededException extends Exception {
    private final int k;
    private final int max;

    public TreeWidthExceededException(int k, int max) {
        super(String.format("Exceeded maximum treewidth of %d with %d ", max, k));
        this.max = max;
        this.k = k;
    }
}
