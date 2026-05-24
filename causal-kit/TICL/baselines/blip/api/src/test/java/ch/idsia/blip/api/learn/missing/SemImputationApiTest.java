package ch.idsia.blip.api.learn.missing;

import org.junit.Test;

import static org.junit.Assert.*;

public class SemImputationApiTest {

    @Test
    public void testMain () {

        String path = "/home/loskana/Desktop/blip/blip/";

        SemImputationApi.main(new String[] {"",
                "-d", path + "data/child-5000-missing.dat",
                "-o", path + "data/child-5000-imputed.dat",
                "-r", path + "data/child.res",
                "-t", "1",
                "-tmp", path + "data/tmp"});
    }

}