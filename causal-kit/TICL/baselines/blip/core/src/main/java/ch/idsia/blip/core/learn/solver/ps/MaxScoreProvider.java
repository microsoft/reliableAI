package ch.idsia.blip.core.learn.solver.ps;


import ch.idsia.blip.core.utils.ParentSet;

import java.util.ArrayList;
import java.util.List;


public class MaxScoreProvider extends SimpleProvider {

    private final int max_size;

    public MaxScoreProvider(ParentSet[][] sc, int tw) {
        super(sc);
        this.max_size = tw;
    }

    @Override
    public ParentSet[][] getParentSets() {
        ParentSet[][] newPSets = new ParentSet[sc.length][];

        for (int i = 0; i < sc.length; i++) {
            List<ParentSet> aux = new ArrayList<ParentSet>();

            for (ParentSet ps : sc[i]) {
                if (ps.parents.length <= max_size) {
                    aux.add(ps);
                }
            }

            newPSets[i] = new ParentSet[aux.size()];
            for (int j = 0; j < aux.size(); j++) {
                newPSets[i][j] = aux.get(j);
            }
        }

        sc = newPSets;
        return newPSets;
    }
}
