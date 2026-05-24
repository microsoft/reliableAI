package ch.idsia.blip.api.common;


import org.junit.Test;


public class SamGeApiTest {

    @Test
    public void test() {
        SamGeApi.main(
                new String[] {
            "samge", "-n", "../experiments/sampler/cancer.uai", "-d",
            "../experiments/sampler/cancer-5000.dat", "-set", "500"
        });
    }
}
