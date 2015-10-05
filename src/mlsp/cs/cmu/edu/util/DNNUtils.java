package mlsp.cs.cmu.edu.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import training.DataInstance;

public class DNNUtils {

  public static String printVector(double[] vec) {
    DecimalFormat f = new DecimalFormat("###.###");
    StringBuilder sb = new StringBuilder("[  ");
    for (Double d : vec)
      sb.append(f.format(d) + "  ");
    return sb.toString() + "]";
  }

  public static List<double[]> getInputsFromFile(String file) throws IOException {
    List<double[]> inputs = new ArrayList<double[]>();
    Scanner scn = new Scanner(new File(file));
    while (scn.hasNextLine()) {
      String line = scn.nextLine();
      String[] arr = line.split("\\,");
      double[] vector = new double[arr.length];
      for (int i = 0; i < arr.length; i++)
        vector[i] = Double.valueOf(arr[i]);
      inputs.add(vector);
    }
    scn.close();
    return inputs;
  }

  public static List<DataInstance> getTrainingInstances(String file, int inputDimension,
          int outputDimension) throws IOException {
    List<double[]> data = getInputsFromFile(file);
    List<DataInstance> instances = new ArrayList<DataInstance>();
    for (double[] instance : data)
      instances.add(new DataInstance(inputDimension, outputDimension, instance));
    return instances;
  }

}
