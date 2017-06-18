import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Main {
    public static void main(String[] args) throws IOException {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        if (args.length == 0) {
            System.out.println("No args detected, reading from stdin: k files...");
            BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
            args = br.readLine().split(" ");
            br.close();
        } else if (args.length == 1) {
            System.out.println("Usage: quantize k files...");
            System.exit(1);
        }
        int k = Integer.parseInt(args[0]);
        for (int i = 1; i < args.length; i++) {
            String fileName = args[i];
            if (fileName.startsWith(".\\")) {
                fileName = fileName.substring(2);
            }
            System.out.println("Processing " + args[i]);
            quantizeImage(k, args[i]);
        }
        System.out.println("Done!");
    }

    static void quantizeImage(int k, String file) {
        Mat img = Imgcodecs.imread(file);
        List<Mat> clusters = cluster(img, k);
        Mat res = clusters.get(0);
        // Combine clusters into one image
        for (int i = 1; i < clusters.size(); i++) {
            Mat cluster = clusters.get(i);
            for (int y = 0; y < cluster.rows(); y++) {
                for (int x = 0; x < cluster.cols(); x++) {
                    double[] resRGB = res.get(y, x);
                    double[] clusterRGB = cluster.get(y, x);
                    assert(resRGB.length == clusterRGB.length);
                    for (int z = 0; z < resRGB.length; z++) {
                        resRGB[z] += clusterRGB[z];
                    }
                    res.put(y, x, resRGB);
                }
            }
        }
        String desFile = file.replaceAll("\\.JPG|\\.jpg", String.format("_k-%d_result.jpg", k));
        System.out.println("Writing result to " + desFile);
        Imgcodecs.imwrite(desFile, res);
    }

    public static List<Mat> cluster(Mat cutout, int k) {
        Mat samples = cutout.reshape(1, cutout.cols() * cutout.rows());
        Mat samples32f = new Mat();
        samples.convertTo(samples32f, CvType.CV_32F, 1.0 / 255.0);
        Mat labels = new Mat();
        TermCriteria criteria = new TermCriteria(TermCriteria.EPS + TermCriteria.MAX_ITER, 10, 1);
        Mat centers = new Mat();
        Core.kmeans(samples32f, k, labels, criteria, 10, Core.KMEANS_RANDOM_CENTERS, centers);
        return showClusters(cutout, labels, centers);
    }

    private static List<Mat> showClusters (Mat cutout, Mat labels, Mat centers) {
        centers.convertTo(centers, CvType.CV_8UC1, 255.0);
        centers.reshape(3);
        List<Mat> clusters = new ArrayList<>();
        for(int i = 0; i < centers.rows(); i++) {
            clusters.add(Mat.zeros(cutout.size(), cutout.type()));
        }
        Map<Integer, Integer> counts = new HashMap<>();
        for(int i = 0; i < centers.rows(); i++) counts.put(i, 0);
        int rows = 0;
        for(int y = 0; y < cutout.rows(); y++) {
            for(int x = 0; x < cutout.cols(); x++) {
                int label = (int)labels.get(rows, 0)[0];
                int r = (int)centers.get(label, 2)[0];
                int g = (int)centers.get(label, 1)[0];
                int b = (int)centers.get(label, 0)[0];
                clusters.get(label).put(y, x, b, g, r);
                counts.put(label, counts.get(label) + 1);
                rows++;
            }
        }
        return clusters;
    }
}
