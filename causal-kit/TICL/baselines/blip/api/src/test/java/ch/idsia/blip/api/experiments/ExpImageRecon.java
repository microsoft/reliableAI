package ch.idsia.blip.api.experiments;


import ch.idsia.blip.core.utils.BayesianNetwork;
import ch.idsia.blip.core.io.bn.BnErgWriter;
import ch.idsia.blip.core.io.bn.BnNetReader;
import ch.idsia.blip.core.io.bn.BnUaiWriter;
import ch.idsia.blip.core.learn.param.ParLe;
import ch.idsia.blip.core.utils.data.array.TIntArrayList;
import org.junit.Test;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.*;
import java.util.ArrayList;

import static ch.idsia.blip.core.utils.RandomStuff.*;


public class ExpImageRecon {

    int white = Color.white.getRGB();
    int black = Color.black.getRGB();
    int grey = Color.red.getRGB();

    @Test
    public void gen() throws IOException {
        String h = "/home/loskana/Desktop/image/";
        BayesianNetwork bn = BnNetReader.ex(h + "1.net");
        BayesianNetwork bnNew = ParLe.ex(bn, getDataSet(h + "train-file1.dat"));

        BnErgWriter.ex(h + "1", bnNew);
        BnUaiWriter.ex(h + "1", bnNew);
    }

    @Test
    public void exp() throws IOException, InterruptedException {
        String h = "/home/loskana/Desktop/image/";

        int q_s = 400;

        for (int i = 0; i < 10; i++) {

            int s = 28 * 28;
            // Create random query
            TIntArrayList query = new TIntArrayList();

            for (int j = 0; j < q_s; j++) {
                int d = rand(s);

                while (query.contains(d)) {
                    d = rand(s);
                }
                query.add(d);
            }

            test(h, String.valueOf(q_s), i, query);
        }

    }

    @Test
    public void exp2() throws IOException, InterruptedException {
        String h = "/home/loskana/Desktop/image/";

        for (int i = 0; i < 10; i++) {

            int s = 28 * 28;
            // Create random query
            int r = (int) (Math.random() * 28);
            TIntArrayList query = new TIntArrayList();

            for (int y = r; y < r + 5; y++) {
                for (int x = 0; x < 28; x++) {
                    query.add(y * 28 + x);
                }
            }

            test(h, "new", i, query);
        }

    }

    @Test
    public void exp3() throws IOException, InterruptedException {
        String h = "/home/loskana/Desktop/image/";

        int s = 28 * 28;
        // Create random query
        int r = (int) (Math.random() * 28);
        TIntArrayList query = new TIntArrayList();

        for (int y = 0; y < 14; y++) {
            for (int x = 0; x < 28; x++) {
                query.add(y * 28 + x);
            }
        }

        test(h, "ciao", 1, query);

        query = new TIntArrayList();
        for (int y = 15; y < 28; y++) {
            for (int x = 0; x < 28; x++) {
                query.add(y * 28 + x);
            }
        }

        test(h, "ciao2", 1, query);

    }

    private void test(String h, String q_s, int u, TIntArrayList query) throws IOException, InterruptedException {
        // Read original image
        BufferedImage img = ImageIO.read(new File(h + "test.png"));
        int width = img.getWidth();
        int height = img.getHeight();

        ImageIO.write(resizeImage(img, 600, 600), "png",
                new File(h + "res/original.png"));

        int s = width * height;

        boolean[] mtx = new boolean[s];
        int i = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = grey(img, x, y);

                mtx[i++] = rgb > 250;
            }
        }

        // Write to stdout
        /*
         i = 0;
         for (int y = 0; y < height; y++) {
         for (int x = 0; x < width; x++)
         pf("%d ", (mtx[i++] ? 1 : 0));
         p("\n");
         } */

        BufferedImage img2 = writeQuery(h, width, height, mtx, query);

        // Write evidence
        PrintWriter w = new PrintWriter(h + "evid", "UTF-8");

        // wf(w, "/* Evidence */ \n");
        wf(w, "%d ", s - query.size());
        for (int j = 0; j < s; j++) {
            if (query.contains(j)) {
                continue;
            }
            wf(w, "%d %d ", j, mtx[j] ? 1 : 0);
        }
        w.close();

        // Execute MPE
        String r = "./daoopt -f 1.uai -e evid";
        Process proc = Runtime.getRuntime().exec(r, new String[0], new File(h));
        ArrayList<String> out = exec(proc);
        String res = out.get(out.size() - 1);
        String[] aux = res.split(" ");

        BufferedImage img3 = writeRecovered(h, width, height, aux);

        BufferedImage join = joinBufferedImage(img, img2, img3);
        BufferedImage result = resizeImage(join, 1200, 400);

        ImageIO.write(result, "png",
                new File(f("%s/res/%s-result%d.png", h, q_s, u)));
    }

    public static BufferedImage joinBufferedImage(BufferedImage img1, BufferedImage img2, BufferedImage img3) {

        // do some calculate first
        int offset = 5;
        int wid = img1.getWidth() + img2.getWidth() + img3.getWidth()
                + offset * 2;
        int height = Math.max(Math.max(img1.getHeight(), img2.getHeight()),
                img3.getHeight())
                + offset;
        // create pa new buffer and draw two image into the new image
        BufferedImage newImage = new BufferedImage(wid, height,
                BufferedImage.TYPE_INT_ARGB);
        Graphics2D g2 = newImage.createGraphics();
        Color oldColor = g2.getColor();

        // fill background
        g2.setPaint(Color.WHITE);
        g2.fillRect(0, 0, wid, height);
        // draw image
        g2.setColor(oldColor);
        g2.drawImage(img1, null, 0, 0);
        g2.drawImage(img2, null, img1.getWidth() + offset, 0);
        g2.drawImage(img3, null,
                img1.getWidth() + offset + img2.getWidth() + offset, 0);
        g2.dispose();
        return newImage;
    }

    private BufferedImage writeQuery(String h, int width, int height, boolean[] mtx, TIntArrayList query) throws IOException {
        int i; // Write image
        BufferedImage img2 = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_RGB);

        i = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int c;

                if (query.contains(i)) {
                    c = grey;
                } else if (mtx[i]) {
                    c = white;
                } else {
                    c = black;
                }

                img2.setRGB(x, y, c);
                i++;
            }
        }

        ImageIO.write(resizeImage(img2, 600, 600), "png",
                new File(h + "res/query.png"));

        return img2;
    }

    private BufferedImage writeRecovered(String h, int width, int height, String[] aux) throws IOException {
        int i; // Write final image
        BufferedImage img3 = new BufferedImage(width, height,
                BufferedImage.TYPE_INT_RGB);

        i = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int c;

                if (aux[i + 3].equals("1")) {
                    c = white;
                } else {
                    c = black;
                }

                img3.setRGB(x, y, c);
                i++;
            }
        }

        ImageIO.write(resizeImage(img3, 600, 600), "png",
                new File(h + "res/final.png"));

        return img3;
    }

    private static BufferedImage resizeImage(BufferedImage originalImage, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height,
                originalImage.getType());
        Graphics2D g = resizedImage.createGraphics();

        g.drawImage(originalImage, 0, 0, width, height, null);
        g.dispose();

        return resizedImage;
    }

    private int rand(int s) {
        return (int) (Math.random() * s);
    }

    private int grey(BufferedImage img, int x, int y) {
        int rgb = img.getRGB(x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = (rgb & 0xFF);

        return (r + g + b) / 3;
    }

    private int[] px(boolean b) {
        if (b) {
            return new int[] { 255, 255, 255};
        } else {
            return new int[] { 0, 0, 0};
        }
    }
}
