package ch.idsia.blip.core;


import ch.idsia.TheTest;
import ch.idsia.blip.core.utils.other.Gamma;
import org.junit.Test;


public class GammaTest extends TheTest {

    @Test
    public void gammaTest() {
        System.out.println(Gamma.lgamma(1.0));
        System.out.println(Gamma.lgamma(10.0));
        System.out.println(Gamma.lgamma(100.0));
    }
}
