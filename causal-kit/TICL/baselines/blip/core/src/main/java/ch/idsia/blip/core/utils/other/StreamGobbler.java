package ch.idsia.blip.core.utils.other;


import java.io.*;


public class StreamGobbler extends Thread {

    private final PrintWriter w;
    InputStream is;

    public StreamGobbler(InputStream is, String file) throws FileNotFoundException {
        this.is = is;
        this.w = new PrintWriter(file);
    }

    @Override
    public void run() {
        try {
            InputStreamReader isr = new InputStreamReader(is);
            BufferedReader br = new BufferedReader(isr);
            String line = null;

            while ((line = br.readLine()) != null) {
                w.append(line + "\n");
            }
        } catch (IOException ioe) {
            ioe.printStackTrace();
        } finally {
            w.close();
        }
    }
}
