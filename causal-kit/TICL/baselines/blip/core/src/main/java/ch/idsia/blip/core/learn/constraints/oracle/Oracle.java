package ch.idsia.blip.core.learn.constraints.oracle;


import ch.idsia.blip.core.utils.DataSet;
import ch.idsia.blip.core.utils.analyze.Analyzer;


public abstract class Oracle extends Analyzer {

    protected Oracle(DataSet dat) {
        super(dat);
    }

    public abstract boolean condInd(int x, int y, int[] z);
}
