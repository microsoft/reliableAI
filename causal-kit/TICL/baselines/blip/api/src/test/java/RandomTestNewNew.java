import ch.idsia.blip.api.common.LLEvalApi;
import ch.idsia.blip.api.learn.solver.tw.QuietMcSolverApi;
import org.junit.Test;

import static ch.idsia.blip.core.utils.RandomStuff.getDataSet;
import static ch.idsia.blip.core.utils.RandomStuff.p;


public class RandomTestNewNew {

    @Test
    public void testest(){
        LLEvalApi.main(new String[] {
                "",
                "-d", "/home/loskana/Desktop/test/child/dat",
                "-b", "1",
                "-u", "0",
                "-t", "2",
                "-j", "/home/loskana/Desktop/test/child/her"
        });
    }


    @Test
    public void testest3(){
        QuietMcSolverApi.main(new String[] {
                "",
                "-j", "/home/loskana/Desktop/child-5000.jkl",
                "-b", "1",
                "-t", "2",
                "-r", "h",
                "-w", "5",
                "-v", "1"
        });
    }
}
