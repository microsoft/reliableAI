package ch.idsia.blip.api.measure;


import ch.idsia.blip.api.common.HammingDist;
import org.junit.Test;


public class HammingDistTest {

    @Test
    public void test() {
        HammingDist.main(
                new String[] {
            "hmdist", "-n1", "../experiments/sampler/BN_N_20_0.uai", "-n2",
            "../experiments/sampler/BN_N_20_2.uai",
        });
    }
}
