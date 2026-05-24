
import ch.idsia.blip.api.common.LLEvalApi;
import org.junit.Test;


public class ExpSPN {

    String path = "/home/loskana/Desktop/SPN/kmax/";

    private int time = 60;

    @Test
    public void go() throws Exception {
        LLEvalApi.main(
                new String[] {
            "", "-d", path + "data/nltcs.valid.dat", "-n",
            path + "work/nltcs/res.uai", "-l"

        });
    }
}
