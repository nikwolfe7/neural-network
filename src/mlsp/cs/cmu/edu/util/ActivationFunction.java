package mlsp.cs.cmu.edu.util;

public class ActivationFunction {
  
  /*================================================*/
  /* Public functions to change
  /*================================================*/
  public static double getActivation(double input) {
//    return sigmoid(input);
    return tanh(input);
  }
  
  public static double getActivationDerivative(double output) {
//    return getSigmoidActivationDerivative(output);
    return getTanhActivationDerivative(output);
  }
  
  /*================================================*/
  /* Private functions
  /*================================================*/
  private static double sigmoid(double input) {
    return 1.0 / (1.0 + Math.exp(-input));
  }
  
  private static double tanh(double input) {
    return 1.7159 * Math.tanh(0.66666666666 * input);
  }
  
  private static double getSigmoidActivationDerivative(double output) {
    return output * (1 - output);
  }
  
  private static double getTanhActivationDerivative(double output) {
    return 1.14393333333 / Math.pow(Math.cosh(0.66666666666 * output), 2);
  }

  public static void main(String[] args) {
    double i = -5;
    while (i < 5) {
      System.out.println(i + "," + ActivationFunction.getActivation(i));
      i += 0.1;
    }
  }

}
