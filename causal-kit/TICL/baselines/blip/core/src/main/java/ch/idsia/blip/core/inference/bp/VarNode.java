package ch.idsia.blip.core.inference.bp;


// Variable node in factor graph
public class VarNode extends Node {

    private final String name;

    public VarNode(String name, int dim, int nid) {
        super(nid, dim);
        this.name = name;
    }

    public void reset() {
        super.reset();

        /*
         size = range(0, len(self.incoming))
         self.incoming = [np.ones((self.dim,1)) for i in size]
         self.outgoing = [np.ones((self.dim,1)) for i in size]
         self.oldoutgoing = [np.ones((self.dim,1)) for i in size]
         self.observed = -1
         */
    }

    public void condition() { // observation
        // Condition on observing certain value

        enable();

        /*
         this.observed = observation;
         // set messages (won't change)
         for i in range(0, len(self.outgoing)):
         self.outgoing[i] = np.zeros((self.dim, 1))
         self.outgoing[i][self.observed] = 1.
         nextStep(); // copy into oldoutgoing
         */
    }

    // Multiplies together incoming messages to make new outgoing
    public void prepMessages() {/*
         // compute new messages if no observation has been made
         if (enabled && observed< 0 && nbrs.size() > 1) {
         // switch reference for old messages
         nextStep();
         for (int i = 0; i < this.incoming[i]; i++) {
         // multiply together all excluding message at current index
         curr = self.incoming[:]
         del curr[ i]
         self.outgoing[i] = reduce(np.multiply, curr);
         }

         // normalize once finished with all messages
         normalizeMessages();
         }
         */}

}
