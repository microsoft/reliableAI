package ch.idsia.blip.core.inference.bp;


import java.util.ArrayList;

import static ch.idsia.blip.core.utils.data.ArrayUtils.onesD;


// Factor node in factor graph
public class FactNode extends Node {

    private int numNbrs;

    public FactNode(int nid, ArrayList<VarNode> nbrs) {
        super(nid);
        // this.P = P;

        // list storing refs to variable nodes
        this.nbrs = new ArrayList<Node>();
        for (Node n : nbrs) {
            this.nbrs.add(n);
        }

        // num of edges
        numNbrs = this.nbrs.size();
        // numDependencies = P.squeeze().ndim;

        // init messages
        for (int i = 0; i < numNbrs; i++) {
            VarNode v = nbrs.get(i);
            int vdim = v.dim;

            // init for factor
            this.incoming.add(onesD(vdim));
            this.outgoing.add(onesD(vdim));
            this.oldoutgoing.add(onesD(vdim));

            // init for variable
            v.nbrs.add(this);
            v.incoming.add(onesD(vdim));
            v.outgoing.add(onesD(vdim));
            v.oldoutgoing.add(onesD(vdim));
        }

        // Factor dimensions does not match size of domain."
        /* if (numNbrs != numDependencies)
         p("ERRORE")
         */
    }

    public void reset() {
        super.reset();

        for (int i = 0; i < incoming.size(); i++) {
            incoming.set(i, onesD(nbrs.get(i).dim));
            outgoing.set(i, onesD(nbrs.get(i).dim));
            oldoutgoing.set(i, onesD(nbrs.get(i).dim));
        }
    }

    // Multiplies incoming messages w/ P to make new outgoing
    public void prepMessages() {
        if (!enabled) {
            return;
        }

        // switch references for old messages
        this.nextStep();

        int mnum = incoming.size();

        // do tiling in advance
        // roll axes to match shape of newMessage after
        for (int i = 0; i < mnum; i++) {// find tiling size
            /*
             nextShape = list(self.P.shape)
             del nextShape[ i]
             nextShape.insert(0, 1)
             #need to expand incoming message to correct num of dims to tile properly
             prepShape = [1 for x in nextShape]
             prepShape[0] = self.incoming[i].shape[0]
             self.incoming[i].shape = prepShape
             #tile and roll
             self.incoming[i] = np.tile(self.incoming[i], nextShape)
             self.incoming[i] = np.rollaxis(self.incoming[i], 0, i + 1)
             */}

        /*
         #loop over subsets
         for i in range(0, mnum):
         curr = self.incoming[:]
         del curr[ i]
         newMessage = reduce(np.multiply, curr, self.P)

         #sum over all vars except i !
         #roll axis i to front then sum over all other axes
         newMessage = np.rollaxis(newMessage, i, 0)
         newMessage = np.sum(newMessage, tuple(range(1, mnum)))
         newMessage.shape = (newMessage.shape[0], 1)

         #store new message
         self.outgoing[i] = newMessage

         #normalize once finished with all messages
         self.normalizeMessages()
         */
    }

}
