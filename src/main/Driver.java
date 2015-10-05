package main;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;

public class Driver {

  static double learningRate = 0.05;
  static double squaredError = 0;
  
  static Input[] inputs; 
  static Edge[] w1; 
  static Neuron[] layer;
  static Edge[] w2; 
  static Output[] output;
  
  static DecimalFormat f = new DecimalFormat("##.####");
  
  public static void main(String[] args) {
    Input x1 = new Input();
    Input x2 = new Input();
    Neuron z1 = new Neuron();
    Neuron z2 = new Neuron();
    Bias b = new Bias();
    Output o = new Output();

    Edge e1, e2, e3, e4, e5, e6, e7, e8, e9;
    e1 = new Edge();
    e2 = new Edge();
    e3 = new Edge();
    e4 = new Edge();
    e5 = new Edge();
    e6 = new Edge();
    e7 = new Edge();
    e8 = new Edge();
    e9 = new Edge();
    
    connect(x1, e1, z1);
    connect(x1, e2, z2);
    connect(x2, e3, z1);
    connect(x2, e4, z2);
    
    connect(z1, e5, o);
    connect(z2, e6, o);
    
    connect(b, e7, z1);
    connect(b, e8, z2);
    connect(b, e9, o);
    
    inputs = new Input[] { x1, x2 };
    w1 = new Edge[] { e1, e2, e3, e4, e7, e8 };
    layer = new Neuron[] { z1, z2 };
    w2 = new Edge[] { e5, e6, e9 };
    output = new Output[] { o };
    
    Random r = new Random();
    List<double[]> data = new ArrayList<double[]>();
    for(int i = 0; i < 100000; i++) {
      double x, y, z;
      x = r.nextInt(100);
      y = r.nextInt(100); 
      z = x + y;
      double[] d = new double[] {z, x, y};
      data.add(d);
    }
    data = standardize(data, 1, 2);
    data = normalize(data, 0, 1, 0);
    
    DecimalFormat f = new DecimalFormat("##.#####");
    double prevSqError = Double.POSITIVE_INFINITY;
    while(true) {
      squaredError = 0;
      for(double[] x : data)
        train(x[0], x[1], x[2]);
      double diff = Math.abs(prevSqError - squaredError);
      System.out.println("Squared Error: " + f.format(squaredError) + "\tDiff: " + diff);
      prevSqError = squaredError;
      if(diff < 1.0e-7) 
        break;
    }
    System.out.println("Converged.");
    System.out.print("Weights w1: ");
    for(Edge d : w1) {
      System.out.print(d.getWeight() + " ");
    }
    System.out.println();
    System.out.print("Weights w2: ");
    for(Edge d : w2) {
      System.out.print(d.getWeight() + " ");
    }

  }
  
  private static List<double[]> standardize(List<double[]> data, int... cols) {
    for (int k = 0; k < cols.length; k++) {
      int j = cols[k];
      double sum = 0;
      double N = data.size();
      /* calc avg */
      for (int i = 0; i < data.size(); i++) {
        double[] x = data.get(i);
        sum += x[j];
      }
      double mean = sum / N;
      /* calc variance */
      sum = 0;
      for (int i = 0; i < data.size(); i++) {
        double[] x = data.get(i);
        sum += Math.pow((x[j] - mean), 2);
      }
      double variance = sum / (N-1);
      double stdev = Math.sqrt(variance);
      for (int i = 0; i < data.size(); i++) {
        double[] x = data.get(i);
        x[j] = (x[j] - mean) / stdev;
        data.set(i, x);
      }
    }
    return data;
  }
  
  private static List<double[]> normalize(List<double[]> data, double low, double high, int... cols) {
    for (int k = 0; k < cols.length; k++) {
      int j = cols[k];
      double min = data.get(0)[j];
      double max = min;
      /* calc range */
      for (int i = 0; i < data.size(); i++) {
        double[] x = data.get(i);
        min = Math.min(min, x[j]);
        max = Math.max(max, x[j]);
      }
      /* normalize on range */
      for (int i = 0; i < data.size(); i++) {
        double[] x = data.get(i);
        x[j] = (x[j] - min) / (max - min);
        x[j] = x[j] * (high - low) + low;
        data.set(i, x);
      }
    }
    return data;
  }

  private static void connect(Neuron in, Edge w, Neuron out) {
    w.setIncomingElement(in);
    w.setOutgoingElement(out);
    in.addOutgoingElement(w);
    out.addIncomingElement(w);
  }
  
  private static void train(double trueOutput, double... input) {
    forwardPropagate(input);
    backPropagate(trueOutput);
  }
  
  private static void forwardPropagate(double[] input) {
    /* set input */
    for (int i = 0; i < input.length; i++)
      inputs[i].setInputValue(input[i]);
    forward(inputs);
    forward(w1);
    forward(layer);
    forward(w2);
    forward(output);
  }
  
  private static void backPropagate(double trueOutput) {
    double t, o;
    t = trueOutput;
    o = output[0].getOutput();
    output[0].setTruthValue(t);
    squaredError += Math.pow((t - o), 2);
//    System.out.println("Output: " + f.format(o) + "\t" + "Truth: " + f.format(t));
    backward(output);
    backward(w2);
    backward(layer);
    backward(w1);
    backward(inputs);
  }
  
  private static void forward(NetworkElement[] elements) {
    for(NetworkElement e : elements)
      e.forward();
  }
  
  private static void backward(NetworkElement[] elements) {
    for(NetworkElement e : elements)
      e.backward();
  }

}
