package ch.idsia.blip.core.learn.solver.samp;


import java.util.Random;


public class SamplerUtils {

    static public Sampler getAdvSampler(String sampler, String dat_path, int n_var, Random rand) {
        if ("ent".equals(sampler)) {
            return new EntropySampler(dat_path, n_var, rand);
        }
        if ("ent_b".equals(sampler)) {
            return new EntropyBSampler(dat_path, n_var, rand);
        }
        if ("mi_b".equals(sampler)) {
            return new MIBSampler(dat_path, n_var, rand);
        } else if ("r_ent".equals(sampler)) {
            return new EntropyRSampler(dat_path, n_var, rand);
        } else if ("mi".equals(sampler)) {
            return new MISampler(dat_path, n_var, rand);
        } else if ("r_mi".equals(sampler)) {
            return new MIRSampler(dat_path, n_var, rand);
        } else {
            // return new SimpleSampler(sc.n_var);
            return new SimpleSampler(n_var, rand);
        }
    }

}
