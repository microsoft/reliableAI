package ch.idsia.blip.core.utils.arcs;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.utils.ParentSet;

import java.util.List;


/**
 * Matrix of directed arcs, with named nodes, of a bayesian network
 */
public class NamedDirected extends Directed {

    public String[] names;

    public NamedDirected(BayesianNetwork bn) {
        super(bn);
    }

    public NamedDirected(int n_var) {
        super(n_var, false);
    }

    public NamedDirected(int n_var, boolean b) {
        super(n_var, b);
    }

    public NamedDirected(ParentSet[] str) {
        super(str);
    }

    @Override
    public String name(int v) {
        return names[v];
    }

    public void setNames(List<String> l) {
        names = new String[l.size()];
        for (int i = 0; i < l.size(); i++) {
            names[i] = l.get(i);
        }
    }

    public void setNames(String[] l_nm_var) {
        names = l_nm_var;
    }
}
