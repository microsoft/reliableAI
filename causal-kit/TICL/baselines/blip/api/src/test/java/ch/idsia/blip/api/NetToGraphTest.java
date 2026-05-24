package ch.idsia.blip.api;


import ch.idsia.blip.api.common.NetToGraphApi;
import org.junit.Test;


public class NetToGraphTest {

    @Test
    public void testSupreme() {
        String path = "/home/loskana/Desktop/genetics/12.05/result/";

        String[] args = {
            "", "-n", path + "500a.net", "-t", "100", "-h", path + "selected"
        };

        NetToGraphApi.main(args);
    }
}
