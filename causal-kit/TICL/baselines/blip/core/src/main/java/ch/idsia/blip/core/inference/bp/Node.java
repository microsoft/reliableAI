package ch.idsia.blip.core.inference.bp;


import java.util.ArrayList;

import static ch.idsia.blip.core.utils.data.ArrayUtils.cloneArray;


public class Node {

    protected ArrayList<Node> nbrs;

    protected ArrayList<double[]> oldoutgoing;

    protected final ArrayList<double[]> outgoing;

    protected final ArrayList<double[]> incoming;

    protected int nid;

    protected int dim;

    protected boolean enabled;

    double epsilon = Math.pow(10, -4);

    public Node(int nid) {
        this(nid, 0);
    }

    public Node(int nid, int dim) {
        this.enabled = true;
        this.nid = nid;
        this.dim = dim;
        this.nbrs = new ArrayList<Node>();
        this.incoming = new ArrayList<double[]>();
        this.outgoing = new ArrayList<double[]>();
        this.oldoutgoing = new ArrayList<double[]>();
    }

    public void reset() {
        this.enabled = true;
    }

    public void disable() {
        this.enabled = false;
    }

    public void enable() {
        this.enabled = true;
        for (Node n : this.nbrs) {
            // don't call enable() as it will recursively enable entire graph
            n.enabled = true;
        }
    }

    public void nextStep() {
        oldoutgoing = new ArrayList<double[]>();
        for (int i = 0; i < outgoing.size(); i++) {
            oldoutgoing.set(i, cloneArray(outgoing.get(i)));
        }
    }

    public void normalizeMessages() {// Normalize to sum to 1
        /* for (this.outgoing)
         this.outgoing = [x / np.sum(x) for x in this.outgoing] */}

    public void receiveMessage(Node f, double[] m) {
        // Places new message into correct location in new message list

        if (this.enabled) {
            int i = this.nbrs.indexOf(f);

            this.incoming.set(i, m);
        }
    }

    public void sendMessages() {
        // Sends all outgoing messages

        for (int i = 0; i < this.outgoing.size(); i++) {
            nbrs.get(i).receiveMessage(this, this.outgoing.get(i));
        }
    }

    public boolean checkConvergence() {
        // Check if any messages have changed

        if (!enabled) {
            return true;
        }

        for (int i = 0; i < outgoing.size(); i++) {
            // check messages have same shape
            // this.oldoutgoing[i].shape = this.outgoing[i].shape
            for (int j = 0; j < outgoing.get(i).length; j++) {
                double delta = (Math.abs(
                        outgoing.get(i)[j] - oldoutgoing.get(i)[j]));

                if (delta > epsilon) {
                    return false;
                }
            }
        }

        return true;
    }
}
