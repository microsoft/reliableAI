package ch.idsia.blip.core.learn.solver.samp;


/* Does nothing */
public class NullSampler implements Sampler {

    @Override
    public int[] sample() {
        return new int[0];
    }

    @Override
    public void init() {}
}
