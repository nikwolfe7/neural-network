package main;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.LinearOutput;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.util.DNNUtils;

public class Driver {

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
    Neuron z3 = new Neuron();
    Bias b = new Bias();
    Output o = new LinearOutput();

    Edge e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13;
    e1 = new Edge();
    e2 = new Edge();
    e3 = new Edge();
    e4 = new Edge();
    e5 = new Edge();
    e6 = new Edge();
    e7 = new Edge();
    e8 = new Edge();
    e9 = new Edge();
    e10 = new Edge();
    e11 = new Edge();
    e12 = new Edge();
    e13 = new Edge();
    
    connect(x1, e1, z1);
    connect(x1, e2, z2);
    connect(x1, e10, z3);
    connect(x2, e3, z1);
    connect(x2, e4, z2);
    connect(x2, e11, z3);
    
    connect(z1, e5, o);
    connect(z2, e6, o);
    connect(z3, e12, o);
    
    connect(b, e7, z1);
    connect(b, e8, z2);
    connect(b, e13, z3);
    connect(b, e9, o);
    
    inputs = new Input[] { x1, x2 };
    w1 = new Edge[] { e1, e2, e3, e4, e7, e8, e10, e11, e13 };
    layer = new Neuron[] { z1, z2, z3 };
    w2 = new Edge[] { e5, e6, e9, e12 };
    output = new Output[] { o };
    
    /* Do training */
    List<double[]> data = getData(1000000);
    DecimalFormat f = new DecimalFormat("##.#####");
    double prevSqError = Double.POSITIVE_INFINITY;
    while(true) {
      squaredError = 0;
      for(double[] x : data)
        train(x[0], x[1], x[2]);
      double diff = Math.abs(prevSqError - squaredError);
      System.out.println("Squared Error: " + f.format(squaredError) + "\tDiff: " + diff);
      prevSqError = squaredError;
      if(diff < 0.001) 
        break;
    }
    
    /* Print convergence report */
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
    
    System.out.println("\nTesting...");
    data = getData(10);
    squaredError = 0;
    for(double[] x : data)
      test(x[0], x[1], x[2]);
    System.out.println("Error on Test Set: " + squaredError);

  }
  
  private static List<double[]> getData(int numInstances) {
    Random r = new Random();
    List<double[]> data = new ArrayList<double[]>();
    for(int i = 0; i < numInstances; i++) {
      double x, y, z;
      x = r.nextInt(10);
      y = r.nextInt(10); 
      z = x + y;
      double[] d = new double[] {z, x, y};
      data.add(d);
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
  
  private static void test(double trueOutput, double... input) {
    forwardPropagate(input);
    double t, o;
    t = trueOutput;
    o = output[0].getOutput();
    squaredError += Math.pow((t - o), 2);
    System.out.println("Output: " + f.format(o) + "\t\t" + "Truth: " + f.format(t));
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
